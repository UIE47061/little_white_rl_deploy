import './style.css';
import type { Config, State } from './types';
import { Joystick } from './joystick';
import { RobotViewer } from './viewer';

const root=document.querySelector<HTMLDivElement>('#app')!;
root.innerHTML=`
<header><div class="brand"><span class="brand-icon">白</span><div>小白<span class="subtitle">ROBOT CONTROL</span></div></div>
<div class="telemetry"><span id="connection" class="badge">● 連線中</span><span id="mode" class="badge">STOP</span><span id="ownership">觀看模式</span></div>
<div class="header-actions"><button id="claim" class="primary">取得控制</button><button id="advanced-toggle" aria-expanded="false" aria-controls="advanced">調整參數 ☷</button></div></header>
<main>
<section class="stage"><div class="stage-caption"><span class="eyebrow">ROBOT STATUS</span><h1 class="live-title"><span class="live-dot" aria-hidden="true"></span>LIVE</h1><p id="pose-status">等待機器人狀態</p></div>
<div id="viewer"></div><button id="view-reset" class="view-reset">↺ 重置視角</button>
<div id="waiting" class="waiting">等待 DDS 狀態</div><div class="stage-foot"><span id="model-status">載入模型中…</span><span>機身置中 · 非世界位置</span></div></section>
<section class="console">
<div class="mode-row"><div class="mode-buttons"><button data-mode="sit">Sit <span>坐下</span></button><button data-mode="stand">Stand <span>站立</span></button><button data-mode="move">Move <span>策略控制</span></button></div>
<button data-mode="stop" class="stop">■ 停止／阻尼</button></div>
<div class="progress-track"><div id="progress"></div></div>
<div class="parameter-header"><div class="gait-row"><span class="gait-label">步態</span><div id="gait" class="gait-buttons" role="group" aria-label="快速切換步態"></div></div><span id="parameter-hint">Move 時生效</span></div><div id="basic" class="basic"></div>
<div class="sticks"><div class="stick-unit"><div id="translation" class="stick" role="group" aria-label="前後與側移菇菇頭"><span class="axis up">前</span><span class="axis down">後</span><span class="axis left">左</span><span class="axis right">右</span><div class="knob"></div></div><div class="stick-label">前後／側移 <output id="linear-readout">0.00 / 0.00 m/s</output></div></div>
<div class="drive-status"><span class="eyebrow">DRIVE</span><strong id="drive-label">等待控制</strong><span>放開歸零<br/>斷線返回站姿</span></div>
<div class="stick-unit"><div id="rotation" class="stick" role="group" aria-label="左右轉向菇菇頭"><span class="axis left">↶</span><span class="axis right">↷</span><div class="knob"></div></div><div class="stick-label">左右轉向 <output id="yaw-readout">0.00 rad/s</output></div></div></div>
</section></main>
<div id="advanced-backdrop" class="drawer-backdrop" hidden></div>
<aside id="advanced" role="dialog" aria-modal="true" aria-labelledby="advanced-title" hidden><div class="drawer-title"><div><span class="eyebrow">FINE TUNE</span><h2 id="advanced-title">進階調整</h2></div><button id="advanced-close" aria-label="關閉進階參數">✕</button></div><div class="drawer-content"><p>設定會保留；進入 Move 後生效。</p><div id="extra"></div><label id="idle-row" class="idle" hidden><input id="idle-step" type="checkbox"/>零速度時原地踏步</label><button id="defaults" class="wide">恢復參數預設值</button></div></aside>
<footer><span id="reason">正在連接控制器…</span><span id="age">— ms</span></footer><div id="toast" role="status" aria-live="polite"></div>`;
const el=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
let cfg:Config, state:State|null=null, socket:WebSocket|null=null, client='', viewer:RobotViewer|null=null;
let receivedAt=0, requestId=0, foreground=!document.hidden, toastTimer=0, reconnectTimer=0;
let vx=0,vy=0,yaw=0,resetting=false;
const pending=new Map<number,{at:number,action:string}>();
const isFresh=()=>!!state && state.fresh && performance.now()-receivedAt < Math.max(0,(cfg.state_timeout-state.state_age)*1000);
const owns=()=>!!client && state?.owner===client && socket?.readyState===WebSocket.OPEN && isFresh() && foreground;
const canDrive=()=>owns() && state?.mode==='move';
const sticks=[new Joystick(el('translation'),false,canDrive,(x,y)=> {
  if(cfg) {vx=-y*(y<0?cfg.speeds.forward:cfg.speeds.backward);vy=-x*cfg.speeds.lateral;}
  if(x===0 && y===0) sendControl();
}),new Joystick(el('rotation'),true,canDrive,(x)=>{if(cfg) yaw=-x*cfg.speeds.yaw;if(x===0) sendControl();})];
function resetSticks(){resetting=true;sticks.forEach(stick=>stick.reset());vx=vy=yaw=0;resetting=false;}
function toast(text:string){el('toast').textContent=text;el('toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=window.setTimeout(()=>el('toast').classList.remove('visible'),3500);}
function send(message:Record<string,unknown>,ack=true) {
  if(!socket || socket.readyState!==WebSocket.OPEN) return;
  if(socket.bufferedAmount>8192) {socket.close(); return;}
  if(ack) {const id=++requestId;message.id=id;pending.set(id,{at:performance.now(),action:String(message.type)});}
  socket.send(JSON.stringify(message));
}
function sendControl(){if(!resetting && owns() && state) send({type:'control',epoch:state.epoch,vx,vy,yaw_rate:yaw},false);}
function relinquish(){
  foreground=false;resetSticks();if(state?.owner===client) send({type:'release'});
  updateUI();
}
window.addEventListener('blur',relinquish);
window.addEventListener('focus',()=>{foreground=!document.hidden;updateUI();});
document.addEventListener('visibilitychange',()=>{if(document.hidden) relinquish();else {foreground=true;updateUI();}});
window.addEventListener('pagehide',()=>{relinquish();socket?.close();});
window.addEventListener('pageshow',()=>{foreground=!document.hidden;if(cfg && socket?.readyState===WebSocket.CLOSED) connect();});
window.addEventListener('resize',resetSticks);

function acceptState(next:State) {
  if(state && next.time<state.time) return;
  if(!state || state.epoch!==next.epoch || state.owner!==next.owner || state.mode!==next.mode) resetSticks();
  state=next;receivedAt=performance.now();
  if(isFresh()) viewer?.update(next);
  updateUI();
}
function updateUI(){
  if(!cfg) return;
  const connected=socket?.readyState===WebSocket.OPEN,fresh=isFresh(),owner=owns();
  el('connection').textContent=connected?(fresh?'● 已連線':'● 狀態過期'):'● 已斷線';
  el('connection').className=`badge ${connected&&fresh?'online':'offline'}`;
  el('mode').textContent=(state?.mode||'stop').toUpperCase();
  el('ownership').textContent=owner?'你正在控制':state?.owner?'其他裝置控制中':'觀看模式';
  el<HTMLButtonElement>('claim').textContent=owner?'釋放控制':'取得控制';
  el<HTMLButtonElement>('claim').disabled=!connected||!fresh||!foreground||(!owner&&!!state?.owner)||!state?.running;
  document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(button=>{
    const mode=button.dataset.mode;
    button.disabled=!owner || !state?.running || (mode==='move'&&(state.mode!=='stand'||state.transition<1));
    button.classList.toggle('active',mode===state?.mode);
  });
  document.querySelectorAll<HTMLInputElement>('input[type=range]').forEach(input=>{
    input.disabled=!owner;
    const name=input.dataset.param!,spec=cfg.parameters[name];
    const value=state?.commands[name]??spec.default;
    const editing=input.dataset.editing==='true';
    if(!editing) input.value=String(value);
    el(`value-${name}`).textContent=`${(editing?Number(input.value):value).toFixed(spec.step<0.01?3:2)} ${spec.unit}`;
    const fill=(Number(input.value)-spec.min)/(spec.max-spec.min)*100;
    input.style.setProperty('--fill',`${fill}%`);
  });
  document.querySelectorAll<HTMLButtonElement>('[data-gait]').forEach(button=>{
    const selected=button.dataset.gait===state?.gait;
    button.disabled=!owner||!state?.running;
    button.classList.toggle('active',selected);
    button.setAttribute('aria-pressed',String(selected));
  });
  el<HTMLInputElement>('idle-step').disabled=!owner;
  el<HTMLInputElement>('idle-step').checked=!!state?.commands.idle_step;
  el<HTMLButtonElement>('defaults').disabled=!owner;
  el('parameter-hint').textContent=state?.mode==='move'?'Move 中 · 即時套用':'Move 時生效';
  el('reason').textContent=state?.error||state?.reason||'等待控制器狀態';
  el('pose-status').textContent=fresh?'實際關節角度與機身姿態':'資料暫停 · 保留最後姿態';
  el('progress').style.width=`${(state?.transition??0)*100}%`;
  el('progress').classList.toggle('transitioning',!!state&&state.transition<1);
  el('waiting').hidden=fresh;
  el('waiting').textContent=state?.q?'資料過期 · 最後姿態':'等待 DDS 狀態';
  el('drive-label').textContent=canDrive()?'可以操控':owner?'按 Stand → Move':'等待控制';
  root.classList.toggle('driving',canDrive());
  const age=state?state.state_age*1000+performance.now()-receivedAt:0;
  el('age').textContent=state?`${Math.round(age)} ms`:'— ms';
  el('linear-readout').textContent=`${(state?.commands.vx??0).toFixed(2)} / ${(state?.commands.vy??0).toFixed(2)} m/s`;
  el('yaw-readout').textContent=`${(state?.commands.yaw_rate??0).toFixed(2)} rad/s`;
}
function connect(){
  clearTimeout(reconnectTimer);
  socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
  socket.onmessage=event=>{
    const message=JSON.parse(event.data);
    if(message.type==='hello') {client=message.client;updateUI();}
    else if(message.type==='state') acceptState(message);
    else if(message.type==='result') {
      pending.delete(message.id);
      if(message.state) acceptState(message.state);
      if(!message.ok) toast(message.error||'操作失敗');
    }
  };
  socket.onclose=()=>{client='';pending.clear();resetSticks();state=null;updateUI();reconnectTimer=window.setTimeout(connect,1500);};
  socket.onerror=()=>socket?.close();
}
function slider(name:string){
  const spec=cfg.parameters[name];
  const box=document.createElement('label');box.className='parameter';
  box.innerHTML=`<span>${spec.label}<output id="value-${name}"></output></span><input aria-label="${spec.label}" data-param="${name}" type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${spec.default}" disabled/>`;
  const input=box.querySelector('input')!;
  let lastSent=0;
  input.addEventListener('pointerdown',()=>{input.dataset.editing='true';});
  for(const type of ['pointerup','pointercancel','blur']) input.addEventListener(type,()=>{input.dataset.editing='false';});
  // Live updates are limited to 10 Hz; change sends the final value as well.
  input.addEventListener('input',()=>{
    el(`value-${name}`).textContent=`${Number(input.value).toFixed(spec.step<0.01?3:2)} ${spec.unit}`;
    input.style.setProperty('--fill',`${(Number(input.value)-spec.min)/(spec.max-spec.min)*100}%`);
    if(owns() && performance.now()-lastSent>100) {
      send({type:'parameters',values:{[name]:Number(input.value)}});lastSent=performance.now();
    }
  });
  input.addEventListener('change',()=>{if(owns()) send({type:'parameters',values:{[name]:Number(input.value)}});});
  return box;
}
el('claim').onclick=()=>{resetSticks();send({type:owns()?'release':'acquire'});};
document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(button=>button.onclick=()=>{resetSticks();send({type:'mode',mode:button.dataset.mode});});
function drawer(open:boolean){
  if(open){resetSticks();sendControl();}
  el('advanced').hidden=!open;
  el('advanced-backdrop').hidden=!open;
  document.body.classList.toggle('drawer-open',open);
  el('advanced-toggle').setAttribute('aria-expanded',String(open));
  (open?el('advanced-close'):el('advanced-toggle')).focus({preventScroll:true});
}
el('advanced-toggle').onclick=()=>drawer(el('advanced').hidden);
el('advanced-close').onclick=()=>drawer(false);
el('advanced-backdrop').onclick=()=>drawer(false);
document.addEventListener('keydown',e=>{if(e.key==='Escape') drawer(false);});
el('idle-step').onchange=()=>send({type:'parameters',values:{idle_step:Number(el<HTMLInputElement>('idle-step').checked)}});
el('defaults').onclick=()=>{resetSticks();send({type:'reset'});};
el('view-reset').onclick=()=>viewer?.reset();
async function start(){
  try {
    const response=await fetch('/api/config');if(!response.ok) throw new Error('無法讀取控制設定');cfg=await response.json();
    for(const name of ['height','frequency','swing_height']) el('basic').append(slider(name));
    for(const name of ['pitch','roll','width','length','duration']) el('extra').append(slider(name));
    for(const gait of cfg.gaits) {
      const button=document.createElement('button');
      button.type='button';button.dataset.gait=gait;button.textContent=gait.toUpperCase();
      button.disabled=true;button.setAttribute('aria-pressed','false');
      button.onclick=()=>{if(owns()) send({type:'gait',gait});};
      el('gait').append(button);
    }
    el('idle-row').hidden=!cfg.idle_control;
    try {viewer=new RobotViewer(el('viewer'),cfg,text=>{el('model-status').textContent=text;});}
    catch {el('model-status').textContent='3D 無法啟動，請檢查瀏覽器 WebGL 支援';}
    connect();updateUI();
    setInterval(()=>{
      if(!isFresh()) resetSticks();
      sendControl();updateUI();
      for(const [id,request] of pending) if(performance.now()-request.at>2000) {
        pending.delete(id);toast('操作未確認，請檢查連線');
      }
    },50);
  } catch(error) {toast(String(error));el('reason').textContent='連線失敗，請重新整理';}
}
start();
