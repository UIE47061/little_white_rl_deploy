"""Real DDS/MuJoCo + HTTP/WebSocket + Chromium acceptance test.

Run from the deploy directory with the robot environment and optional test deps:
  python tests/run_web_smoke.py --domain-id 71 --browser /opt/google/chrome/chrome
Uses an isolated DDS domain by default. Never runs against a physical robot.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def simulator(domain):
    import numpy as np
    import mujoco
    import yaml
    sim_dir=ROOT.parent/"csl_mujoco/simulate_python"
    sys.path.insert(0,str(sim_dir))
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge
    cfg=yaml.safe_load((ROOT/"little_white_wtw.yaml").read_text())
    model=mujoco.MjModel.from_xml_path(str(ROOT.parent/"csl_mujoco/unitree_robots/little_white_v3/scene.xml"))
    data=mujoco.MjData(model)
    for joint in range(model.nu):
        data.qpos[model.jnt_qposadr[model.actuator_trnid[joint,0]]]=cfg["default_angles"][joint]
    data.qpos[2]=0.34
    mujoco.mj_forward(model,data)
    ChannelFactoryInitialize(domain,"lo")
    bridge=UnitreeSdk2Bridge(model,data)
    started=False
    while True:
        start=time.monotonic()
        # Keep the initial fixture pose until the test explicitly presses Stand.
        # After that, run ordinary gravity/contact physics even in damping mode.
        if bridge.motor_command is not None and np.any(bridge.motor_command[:,1]>0):
            started=True
        if started:
            bridge.UpdateMotorControl()
            mujoco.mj_step(model,data)
        time.sleep(max(0,model.opt.timestep-(time.monotonic()-start)))


class Client:
    def __init__(self):
        self.state=None;self.client=None;self.futures={};self.serial=0
        self.values=[0.,0.,0.];self.heartbeats=True;self.states=[]

    async def start(self,url):
        from websockets.asyncio.client import connect
        self.ws=await connect(url,proxy=None)
        self.reader=asyncio.create_task(self.read())
        self.heartbeat=asyncio.create_task(self.beat())
        await self.until(lambda:self.state is not None)
        return self

    async def read(self):
        async for raw in self.ws:
            message=json.loads(raw)
            if message["type"]=="hello":self.client=message["client"]
            elif message["type"]=="state":
                if self.state is None or message['time']>=self.state['time']:
                    self.state=message
                self.states.append((time.monotonic(),message["cycles"]))
            elif message["type"]=="result":
                if "state" in message and (self.state is None or message['state']['time']>=self.state['time']):
                    self.state=message["state"]
                future=self.futures.pop(message.get("id"),None)
                if future and not future.done():future.set_result(message)

    async def beat(self):
        while True:
            if self.heartbeats and self.state and self.state["owner"]==self.client:
                await self.ws.send(json.dumps(dict(type="control",epoch=self.state["epoch"],
                                                   vx=self.values[0],vy=self.values[1],yaw_rate=self.values[2])))
            await asyncio.sleep(.05)

    async def request(self,kind,ok=True,**data):
        self.serial+=1
        future=asyncio.get_running_loop().create_future()
        self.futures[self.serial]=future
        await self.ws.send(json.dumps(dict(type=kind,id=self.serial,**data)))
        result=await asyncio.wait_for(future,3)
        assert result["ok"]==ok,result
        return result

    async def until(self,predicate,timeout=6):
        deadline=time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>deadline:raise AssertionError(f"Timed out; state={self.state}")
            await asyncio.sleep(.03)

    async def close(self):
        await self.ws.close()
        self.reader.cancel();self.heartbeat.cancel()
        await asyncio.gather(self.reader,self.heartbeat,return_exceptions=True)


async def browser_test(base, observer, executable, output):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=executable,headless=True,
                    args=['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        context=await browser.new_context(viewport={"width":844,"height":390},has_touch=True,is_mobile=True)
        page=await context.new_page();errors=[];requests=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.on("request",lambda request:requests.append(request.url))
        await page.goto(base)
        await page.wait_for_selector('#viewer[data-model-loaded=true]')
        await page.get_by_role('button',name='取得控制',exact=True).click()
        await page.get_by_role('button',name='釋放控制',exact=True).wait_for()
        await page.locator('[data-mode=stand]').click()
        await page.locator('[data-mode=move]').wait_for(state='visible')
        await page.wait_for_function("!document.querySelector('[data-mode=move]').disabled")
        await page.locator('[data-mode=move]').click()
        await page.wait_for_function("document.querySelector('#app').classList.contains('driving')")
        await page.locator('input[data-param=frequency]').evaluate("e=>{e.value='2.9';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));}")
        await observer.until(lambda:abs(observer.state['commands']['frequency']-2.9)<1e-5)
        await page.screenshot(path=str(output/'phone-landscape.png'))
        left=await page.locator('#translation').bounding_box()
        right=await page.locator('#rotation').bounding_box()
        session=await context.new_cdp_session(page)
        touches=[dict(x=left['x']+left['width']/2+15,y=left['y']+left['height']/2-20,id=1),
                 dict(x=right['x']+right['width']/2+25,y=right['y']+right['height']/2,id=2)]
        await session.send('Input.dispatchTouchEvent',dict(type='touchStart',touchPoints=touches))
        await observer.until(lambda:observer.state['commands']['vx']>.05 and
                              observer.state['commands']['vy']<-.01 and
                              observer.state['commands']['yaw_rate']<-.05)
        await session.send('Input.dispatchTouchEvent',dict(type='touchCancel',touchPoints=[]))
        await observer.until(lambda:all(abs(observer.state['commands'][key])<1e-6 for key in ('vx','vy','yaw_rate')))
        await page.locator('#advanced-toggle').click()
        await page.locator('#advanced').wait_for(state='visible')
        await page.locator('#advanced-close').click()
        # Browser visibility/background release is also enforced server-side.
        await page.evaluate("document.dispatchEvent(new Event('visibilitychange')); window.dispatchEvent(new Event('blur'))")
        await observer.until(lambda:observer.state['owner'] is None and observer.state['mode']=='stand')
        await page.set_viewport_size(dict(width=390,height=844))
        await page.screenshot(path=str(output/'phone-portrait.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Horizontal overflow'
        await page.set_viewport_size(dict(width=1440,height=900))
        await page.screenshot(path=str(output/'desktop.png'),full_page=True)
        await page.close()
        assert not errors,errors
        assert all(url.startswith(base) for url in requests),requests
        await browser.close()
    print('PASS browser: local URDF, dual touch, cancellation, background release, responsive layout')


async def acceptance(args,sim,deploy,output):
    base=f'http://127.0.0.1:{args.port}'
    first=await Client().start(base.replace('http','ws')+'/ws')
    second=await Client().start(base.replace('http','ws')+'/ws')
    assert first.state['mode']=='stop' and first.state['fresh']
    await first.request('acquire')
    await second.request('acquire',ok=False)
    await second.request('mode',mode='stop',ok=False)
    await first.request('mode',mode='move',ok=False)
    await first.request('parameters',values={'height':.02,'frequency':2.8})
    assert first.state['mode']=='stop'
    await first.request('mode',mode='sit')
    await first.until(lambda:first.state['mode']=='sit' and first.state['transition']>=1)
    sit_q=first.state['q'][:]
    await first.request('mode',mode='stand')
    await first.until(lambda:first.state['mode']=='stand' and first.state['transition']>=1)
    assert max(abs(a-b) for a,b in zip(sit_q,first.state['q']))>.1
    await first.request('mode',mode='move')
    assert abs(first.state['commands']['height']-.02)<1e-6
    first.values=[.15,.05,.12]
    await first.until(lambda:abs(first.state['commands']['vx']-.15)<1e-6)
    before=first.state['commands'].copy()
    await first.request('parameters',values={'height':.01,'frequency':99},ok=False)
    assert first.state['commands']==before
    await first.request('reset')
    first.values=[0,0,0]
    await first.until(lambda:abs(first.state['commands']['vx'])<1e-6)
    await asyncio.sleep(.5)
    first.heartbeats=False
    await first.until(lambda:first.state['owner'] is None and first.state['mode']=='stand')
    await first.request('control',epoch=first.state['epoch'],vx=.1,vy=0,yaw_rate=0,ok=False)
    first.heartbeats=True
    await first.request('acquire')
    assert first.state['mode']=='stand'
    await first.request('release')
    await first.until(lambda:first.state['owner'] is None)
    print('PASS DDS + WebSocket: sit/stand/move, feedback, parameters, ownership, invalid inputs, timeout')
    if args.browser:
        await browser_test(base,second,args.browser,output)
    # Reacquire and kill the HTTP process; inspect actual DDS output independently.
    await first.request('acquire')
    await first.until(lambda:first.state['transition']>=1)
    await first.request('mode',mode='move')
    first.values=[.1,0,0]
    await first.until(lambda:first.state['commands']['vx']>.05)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    import numpy as np
    import yaml
    ChannelFactoryInitialize(args.domain_id,'lo')
    commands=[]
    subscriber=ChannelSubscriber('rt/lowcmd',LowCmd_)
    subscriber.Init(lambda msg:commands.append((time.monotonic(),
                    [m.q for m in msg.motor_cmd[:12]],[m.kp for m in msg.motor_cmd[:12]])),10)
    await asyncio.sleep(.2)
    children=Path(f'/proc/{deploy.pid}/task/{deploy.pid}/children').read_text().split()
    web_pid=next(int(pid) for pid in children if b'spawn_main' in Path(f'/proc/{pid}/cmdline').read_bytes())
    os.kill(web_pid,signal.SIGKILL)
    await asyncio.sleep(3.8)
    assert deploy.poll() is None and commands
    defaults=yaml.safe_load((ROOT/'little_white_wtw.yaml').read_text())['default_angles']
    assert np.allclose(commands[-1][1],defaults,atol=1e-6) and min(commands[-1][2])>0,commands[-1]
    print('PASS HTTP process SIGKILL: parent watchdog returns to fixed Stand without server cleanup')
    # Killing the simulator must override the return-to-stand path, even with no HTTP process.
    sim.terminate();sim.wait(timeout=3)
    await asyncio.sleep(.8)
    assert commands[-1][0]>time.monotonic()-.1 and max(commands[-1][2])==0
    subscriber.Close()
    print('PASS simulator/DDS loss: damping overrides return to Stand')
    await first.close();await second.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain-id',type=int,default=71)
    parser.add_argument('--port',type=int,default=18888);parser.add_argument('--browser')
    parser.add_argument('--simulator',action='store_true');args=parser.parse_args()
    if args.simulator:simulator(args.domain_id);return
    output=ROOT/'web/test-results';output.mkdir(exist_ok=True)
    env={**os.environ,'PYTHONUNBUFFERED':'1','SDL_VIDEODRIVER':'dummy'}
    processes=[]
    with (output/'simulator.log').open('w') as slog,(output/'controller.log').open('w') as clog:
        try:
            sim=subprocess.Popen([sys.executable,__file__,'--simulator','--domain-id',str(args.domain_id)],stdout=slog,stderr=subprocess.STDOUT,env=env)
            processes.append(sim)
            deploy=subprocess.Popen([sys.executable,str(ROOT/'little_white_wtw.py'),'lo','--domain-id',str(args.domain_id),'--web','--web-host','127.0.0.1','--web-port',str(args.port)],stdout=clog,stderr=subprocess.STDOUT,env=env)
            processes.append(deploy)
            deadline=time.monotonic()+30
            while True:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/config',timeout=1) as response:
                        assert json.load(response)['motor_joints'][0]=='FL_hip_joint'
                    break
                except (OSError,ValueError):
                    if deploy.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError((output/'controller.log').read_text()+'\n'+(output/'simulator.log').read_text())
                    time.sleep(.1)
            asyncio.run(acceptance(args,sim,deploy,output))
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
    print('Screenshots and logs:',output)

if __name__=='__main__':main()
