import { deadzone } from './math.mjs';
export class Joystick {
  private pointer:number|null = null;
  private knob:HTMLElement;
  constructor(private el:HTMLElement, private horizontal:boolean,
              private enabled:()=>boolean, private change:(x:number,y:number)=>void) {
    this.knob = el.querySelector('.knob')!;
    el.addEventListener('pointerdown', e => {
      if (this.pointer !== null || !this.enabled()) return;
      e.preventDefault(); this.pointer=e.pointerId;
      el.setPointerCapture(e.pointerId); this.update(e);
    });
    el.addEventListener('pointermove', e => {if (e.pointerId===this.pointer) this.update(e);});
    for (const type of ['pointerup','pointercancel','lostpointercapture']) {
      el.addEventListener(type, (e) => {if ((e as PointerEvent).pointerId===this.pointer) this.reset();});
    }
  }
  private update(e:PointerEvent) {
    if (!this.enabled()) {this.reset(); return;}
    const rect=this.el.getBoundingClientRect(); const radius=rect.width*0.34;
    let x=(e.clientX-rect.left-rect.width/2)/radius;
    let y=this.horizontal ? 0 : (e.clientY-rect.top-rect.height/2)/radius;
    const length=Math.max(1,Math.hypot(x,y)); x/=length; y/=length;
    this.knob.style.transform=`translate(${x*radius}px,${y*radius}px)`;
    const [dx,dy]=deadzone(x,y); this.change(dx,dy);
  }
  reset() {
    const id=this.pointer; this.pointer=null;
    if(id!==null && this.el.hasPointerCapture(id)) this.el.releasePointerCapture(id);
    this.knob.style.transform='translate(0,0)'; this.change(0,0);
  }
}
