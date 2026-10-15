import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import URDFLoader, { URDFRobot } from 'urdf-loader';
import { imuQuaternion, jointValues } from './math.mjs';
import type { Config, State } from './types';

export class RobotViewer {
  private robot:URDFRobot|null=null;
  private pending:State|null=null;
  private renderer:THREE.WebGLRenderer;
  private camera=new THREE.PerspectiveCamera(40,1,0.01,30);
  private controls:OrbitControls;
  private scene=new THREE.Scene();
  constructor(private el:HTMLElement, private cfg:Config, status:(text:string)=>void) {
    this.scene.background=new THREE.Color('#10191d');
    this.scene.fog=new THREE.Fog('#10191d',2,5);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute('aria-label','小白實際姿態 3D 模型');
    el.append(this.renderer.domElement);
    this.camera.up.set(0,0,1);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.enableDamping=true; this.controls.enablePan=false;
    this.controls.minDistance=0.5; this.controls.maxDistance=3;
    this.controls.maxPolarAngle=Math.PI*0.86;
    this.reset();
    this.scene.add(new THREE.HemisphereLight(0xe0f5ff,0x45505a,3));
    const light=new THREE.DirectionalLight(0xffffff,3); light.position.set(2,-3,4); this.scene.add(light);
    const rim=new THREE.DirectionalLight(0x79dfc4,1.7); rim.position.set(-2,2,2); this.scene.add(rim);
    const grid=new THREE.GridHelper(6,60,0x36504c,0x20312f); grid.rotation.x=Math.PI/2;
    this.scene.add(grid);
    const ring=new THREE.Mesh(new THREE.RingGeometry(0.47,0.474,96),new THREE.MeshBasicMaterial({color:0x4d8d7c,side:THREE.DoubleSide}));
    ring.position.z=0.001; this.scene.add(ring);
    const manager=new THREE.LoadingManager();
    const loader=new URDFLoader(manager);
    loader.loadMeshCb=(path,_manager,_material,done)=> {
      new STLLoader(manager).load(path, geometry=> {
        geometry.computeVertexNormals();
        done(new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xe2e7e4,metalness:0.2,roughness:0.6})));
      },undefined,err=>done(new THREE.Object3D(),err as Error));
    };
    manager.onError=()=>status('模型資源載入失敗，請重新整理');
    manager.onLoad=()=>status('模型就緒 · 依回傳狀態同步');
    loader.load(cfg.model_url, robot=> {
      this.robot=robot; robot.position.set(0,0,0.34); robot.visible=false;
      // Display feedback beyond URDF limits too, rather than hiding a mismatch.
      for (const joint of Object.values(robot.joints)) joint.ignoreLimits=true;
      this.scene.add(robot);
      if(this.pending) this.update(this.pending);
      this.el.dataset.modelLoaded='true';
    },undefined,()=>status('無法載入小白 URDF'));
    new ResizeObserver(()=>this.resize()).observe(el); this.resize();
    this.renderer.setAnimationLoop(()=>{this.controls.update();this.renderer.render(this.scene,this.camera);});
  }
  reset() {this.camera.position.set(0.95,-1.05,0.83);this.controls.target.set(0,0,0.23);this.controls.update();}
  private resize() {
    const w=this.el.clientWidth,h=this.el.clientHeight;
    if(!w||!h) return;
    this.camera.aspect=w/h; this.camera.updateProjectionMatrix();this.renderer.setSize(w,h);
  }
  update(state:State) {
    this.pending=state;
    if(!this.robot || !state.fresh || !state.q || !state.quaternion) return;
    const quaternion=imuQuaternion(state.quaternion);
    this.robot.quaternion.set(quaternion[0],quaternion[1],quaternion[2],quaternion[3]);
    this.robot.setJointValues(jointValues(this.cfg.motor_joints,state.q));
    this.robot.visible=true;
  }
}
