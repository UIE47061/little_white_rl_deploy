export interface Parameter {index:number; min:number; max:number; step:number; unit:string; label:string; default:number}
export interface Config {
  parameters:Record<string,Parameter>; defaults:Record<string,number>;
  speeds:{forward:number; backward:number; lateral:number; yaw:number};
  gaits:string[]; idle_control:boolean; motor_joints:string[]; quaternion:string;
  model_url:string; command_timeout:number; state_timeout:number; transition_seconds:number;
}
export interface State {
  type:'state'; time:number; state_age:number; fresh:boolean; running:boolean; mode:string;
  transition:number; owner:string|null; epoch:number; reason:string; error:string|null;
  q:number[]|null; dq:number[]|null; quaternion:number[]|null;
  commands:Record<string,number>; gait:string; cycles:number; bridge_age?:number;
}
