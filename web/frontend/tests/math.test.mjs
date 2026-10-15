import test from 'node:test';
import assert from 'node:assert/strict';
import { deadzone, imuQuaternion, jointValues } from '../src/math.mjs';
test('Z-up IMU identity and rotations map wxyz to xyzw without axis swapping',()=>{
  assert.deepEqual(imuQuaternion([1,0,0,0]),[0,0,0,1]);
  for(let axis=0;axis<3;axis++) {
    const input=[Math.SQRT1_2,0,0,0];input[axis+1]=Math.SQRT1_2;
    const output=imuQuaternion(input);
    assert.ok(Math.abs(output[axis]-Math.SQRT1_2)<1e-12);
    assert.ok(Math.abs(output[3]-Math.SQRT1_2)<1e-12);
  }
  assert.throws(()=>imuQuaternion([0,0,0,0]));
});
test('dead zone removes jitter and clamps diagonal velocity',()=>{
  assert.deepEqual(deadzone(0.03,0.03),[0,0]);
  assert.deepEqual(deadzone(1,0),[1,0]);
  assert.ok(Math.abs(Math.hypot(...deadzone(1,1))-1)<1e-12);
});
test('joint mapping follows configured motor names rather than policy order',()=>{
  const names=['FL_hip_joint','FR_hip_joint','RL_hip_joint','RR_hip_joint'];
  assert.deepEqual(jointValues(names,[.1,.2,.3,.4]),{FL_hip_joint:.1,FR_hip_joint:.2,RL_hip_joint:.3,RR_hip_joint:.4});
});
