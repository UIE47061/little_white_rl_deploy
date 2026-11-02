"""Little White MJLab deployment: one Controller, YAML and TorchScript policy.

Structure follows the original little_white_isaaclab.py Controller.
"""
import argparse
import os
import threading
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result


DEFAULT_CONFIG = Path(__file__).with_suffix('.yaml')


class Controller:
    def __init__(self, config_path=DEFAULT_CONFIG, use_gui=True, use_gamepad=False):
        path = Path(config_path).resolve()
        self.config = cfg = yaml.safe_load(path.read_text())
        self.policy_path = (path.parent / cfg['policy_path']).resolve()
        self.metadata = meta = json.loads((path.parent / cfg['metadata_path']).read_text())
        expected = {'schema_version': 3,
                    'input_shape': ['batch', 48], 'output_shape': ['batch', 12],
                    'history_length': 1,
                    'term_names': ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos',
                                   'joint_vel', 'actions', 'command'],
                    'term_dims': [3, 3, 3, 12, 12, 12, 3], 'quaternion_order': 'wxyz',
                    'linear_velocity_frame': 'body',
                    'joint_position_mode': 'relative',
                    'observation_units': 'unscaled_physical_units',
                    'action_clip': None,
                    'normalization': 'included_in_policy_do_not_normalize_again'}
        for name, value in expected.items():
            if meta.get(name) != value:
                raise ValueError(f'Unsupported MJLab metadata: {name}')
        digest = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        if digest != cfg['policy_sha256'] or digest != meta['policy_sha256']:
            raise ValueError('Model SHA256 differs from configuration')
        self.default_angles = vector(cfg['default_angles'], 12, 'default_angles')
        self.kps = vector(cfg['kps'], 12, 'kps')
        self.kds = vector(cfg['kds'], 12, 'kds')
        self.action_scale = vector(cfg['action_scale'], 12, 'action_scale')
        self.sit_angles = vector(cfg['sit_angles'], 12, 'sit_angles')
        for actual, expected in ((self.default_angles, meta['default_joint_pos']),
                                 (self.kps, meta['kp']), (self.kds, meta['kd']),
                                 (self.action_scale, meta['action_scale'])):
            if not np.allclose(actual, expected, rtol=0, atol=1e-7):
                raise ValueError('YAML pose, PD or action scale differs from training metadata')
        if cfg['num_obs'] != 48 or cfg['num_actions'] != 12:
            raise ValueError('This policy requires 48 observations and 12 actions')
        self.dt = cfg['dt']
        self.decimation = cfg['control_decimation']
        self.control_dt = 1 / float(meta['policy_hz'])
        if not np.isclose(cfg['dt'], meta['physics_dt']) or not np.isclose(
                cfg['dt'] * cfg['control_decimation'], self.control_dt):
            raise ValueError('Control timing differs from metadata')
        if not np.isclose(cfg['torque_limit'], meta['effort_limit_nm']):
            raise ValueError('Torque limit differs from metadata')
        self.motor_joints = [f'{leg}_{part}_joint' for leg in cfg['motor_leg_order']
                             for part in ('hip', 'thigh', 'calf')]
        names = [name[:2].upper() + name[2:].lower() for name in meta['joint_names']]
        if len(set(self.motor_joints)) != 12 or set(names) != set(self.motor_joints):
            raise ValueError('Policy and DDS joints must be matching permutations')
        self.policy_to_motor = np.array([self.motor_joints.index(name) for name in names])
        self.motor_to_policy = np.argsort(self.policy_to_motor)
        self.commands = vector(cfg['cmd_init'], 3, 'commands')
        self.validate_commands(self.commands)
        self.scales = vector(cfg['observation_scales'], 48, 'observation_scales')
        if (not np.array_equal(self.scales, np.ones(48)) or
                cfg['joint_position_mode'] != 'relative' or cfg.get('action_clip') is not None):
            raise ValueError('Preprocessing differs from the training contract')
        if cfg['linear_velocity_frame'] not in ('world', 'body'):
            raise ValueError('linear_velocity_frame must be world or body')
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.linear_velocity_time = 0.0
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        torch.set_num_threads(1)
        self.policy = torch.jit.load(str(self.policy_path), map_location='cpu').eval()
        self.reset()
        self.lock = threading.RLock()
        self.qpos = np.zeros(12, dtype=np.float32)
        self.qvel = np.zeros(12, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.obs = np.zeros(cfg['num_obs'], dtype=np.float32)
        self.low_state = None
        self.state_time = 0.0
        self.mode = 'stop'
        self.is_running = False
        self.lowCmdWriteThreadPtr = None
        self.teleop = None
        self.use_gui, self.use_gamepad = use_gui, use_gamepad
        self.error = None
        self.counter = 0
        self.next_inference = 0.0
        self.transition_start = 0.0
        self.transition_q = self.qpos.copy()
        self.target_dof_pos = self.default_angles[self.motor_to_policy].copy()

    @staticmethod
    def gravity(quaternion):
        quat = vector(quaternion, 4, 'wxyz quaternion')
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            raise ValueError('IMU quaternion is not initialized')
        w, x, y, z = quat / norm
        return np.array([2*(w*y-x*z), -2*(y*z+w*x), 1-2*(w*w+z*z)], dtype=np.float32)

    def validate_commands(self, commands):
        values = vector(commands, 3, 'commands')
        ranges = self.metadata['training_command_ranges']
        for value, name in zip(values, ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')):
            lower, upper = ranges[name]
            if value < lower - 1e-7 or value > upper + 1e-7:
                raise ValueError(f'{name} must be in [{lower}, {upper}]')

    def reset(self):
        self.last_action = np.zeros(12, dtype=np.float32)

    def update_linear_velocity(self, velocity):
        self.linear_velocity = vector(velocity, 3, 'linear velocity').copy()
        self.linear_velocity_time = time.monotonic()

    def require_linear_velocity(self):
        if time.monotonic() - self.linear_velocity_time > self.config['state_timeout']:
            raise RuntimeError('No recent rt/sportmodestate linear velocity')

    @staticmethod
    def world_to_body(velocity, quaternion):
        quat = vector(quaternion, 4, 'wxyz quaternion')
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            raise ValueError('IMU quaternion is not initialized')
        w, x, y, z = quat / norm
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
        return rotation.T @ velocity

    def observe(self, q, dq, quaternion):
        self.require_linear_velocity()
        self.validate_commands(self.commands)
        pos = vector(q, 12, 'joint positions')[self.policy_to_motor]
        if self.config['joint_position_mode'] == 'relative':
            pos = pos - self.default_angles
        elif self.config['joint_position_mode'] != 'absolute':
            raise ValueError('joint_position_mode must be relative or absolute')
        velocity = self.linear_velocity
        if self.config['linear_velocity_frame'] == 'world':
            velocity = self.world_to_body(velocity, quaternion)
        return np.concatenate((velocity, vector(self.angular_velocity, 3, 'gyro'),
                               self.gravity(quaternion), pos,
                               vector(dq, 12, 'joint velocities')[self.policy_to_motor],
                               self.last_action, self.commands)) * self.scales

    def infer(self, q, dq, quaternion):
        self.obs = obs = self.observe(q, dq, quaternion)
        with torch.inference_mode():
            output = self.policy(torch.from_numpy(obs).unsqueeze(0))
        if tuple(output.shape) != (1, 12):
            raise ValueError('Expected policy output [1,12]')
        action = vector(output[0].cpu().numpy(), 12, 'policy action')
        self.last_action = action.copy()
        clip = self.config.get('action_clip')
        if clip is not None:
            action = np.clip(action, -float(clip), float(clip))
        target = self.default_angles + self.action_scale * action
        return target[self.motor_to_policy]

    def check_mujoco_mapping(self, scene_path):
        """Check actual actuator/sensor joint IDs, rather than XML label order."""
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
        if model.nu != 12:
            raise ValueError("Expected 12 MuJoCo actuators")
        for slot, joint_name in enumerate(self.motor_joints):
            actuator_joint = model.joint(int(model.actuator_trnid[slot, 0])).name
            if actuator_joint != joint_name:
                raise ValueError(f"DDS slot {slot}: actuator {actuator_joint}, configured {joint_name}")
            for sensor_id, kind in ((slot, mujoco.mjtSensor.mjSENS_JOINTPOS),
                                    (slot+12, mujoco.mjtSensor.mjSENS_JOINTVEL)):
                if (model.sensor_type[sensor_id] != kind or
                        model.joint(int(model.sensor_objid[sensor_id])).name != joint_name or
                        model.sensor_adr[sensor_id] != sensor_id):
                    raise ValueError(f"DDS slot {slot}: {joint_name} sensor mapping mismatch")
            if (not model.actuator_ctrllimited[slot] or
                    not np.allclose(model.actuator_ctrlrange[slot],
                                    [-self.config["torque_limit"], self.config["torque_limit"]])):
                raise ValueError(f"{joint_name}: actuator torque limits differ from policy")
        print("Policy index -> DDS / actuator index -> joint")
        for index, slot in enumerate(self.policy_to_motor):
            print(f"{index:2d} -> {slot:2d} -> {self.motor_joints[slot]}")
        print("Actuator, position sensor, velocity sensor and torque limits: matched")


    def InitLowCmd(self):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_cmd.head[:] = [0xFE, 0xEF]
        self.low_cmd.level_flag = 0xFF
        for motor in self.low_cmd.motor_cmd:
            motor.mode = 0x01
            motor.q = motor.dq = motor.kp = motor.kd = motor.tau = 0.0
        self.crc = CRC()

    def Init(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
        self.InitLowCmd()
        self.lowcmd_publisher = ChannelPublisher('rt/lowcmd', LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber('rt/lowstate', LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        self.velocity_subscriber = ChannelSubscriber(
            self.config['linear_velocity_topic'], SportModeState_)
        self.velocity_subscriber.Init(self.HighStateMessageHandler, 10)
        deadline = time.monotonic() + self.config['startup_timeout']
        while not self.state_time:
            if time.monotonic() >= deadline:
                raise TimeoutError('No valid rt/lowstate received; start the simulator first')
            time.sleep(.01)
        if self.use_gamepad:
            from headless_teleop import HeadlessTeleop
            self.teleop = HeadlessTeleop(self.commands, self.config, self.metadata)
        elif self.use_gui:
            if (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or
                    os.environ.get('SDL_VIDEODRIVER') == 'dummy'):
                from mjlab_teleop import MJLabTeleop
                self.teleop = MJLabTeleop(
                    self.commands, self.metadata['training_command_ranges'], self.config)
            else:
                print('No desktop display; use console commands.')
        self.Start()

    def Start(self):
        self.is_running = True
        self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite, name='mjlab_lowcmd')
        self.lowCmdWriteThreadPtr.start()

    def ShutDown(self):
        self.is_running = False
        if self.lowCmdWriteThreadPtr is not None:
            self.lowCmdWriteThreadPtr.join(timeout=2)
        if self.teleop is not None:
            self.teleop.close()
        if hasattr(self, 'lowcmd_publisher'):
            self._publish()
        for name in ('lowstate_subscriber', 'velocity_subscriber', 'lowcmd_publisher'):
            channel = getattr(self, name, None)
            if channel is not None:
                channel.Close()

    def LowStateMessageHandler(self, msg):
        try:
            qpos = vector([m.q for m in msg.motor_state[:12]], 12, 'joint positions')
            qvel = vector([m.dq for m in msg.motor_state[:12]], 12, 'joint velocities')
            quat = vector(msg.imu_state.quaternion, 4, 'quaternion')
            ang_vel = vector(msg.imu_state.gyroscope, 3, 'gyro')
            self.gravity(quat)
        except ValueError:
            return
        with self.lock:
            self.low_state = msg
            self.qpos, self.qvel, self.quat, self.angular_velocity = qpos, qvel, quat, ang_vel
            self.state_time = time.monotonic()

    def HighStateMessageHandler(self, msg):
        try:
            velocity = vector(msg.velocity, 3, 'linear velocity')
        except ValueError:
            return
        with self.lock:
            self.update_linear_velocity(velocity)

    def get_current_state(self):
        return self.qpos.copy(), self.qvel.copy(), self.angular_velocity.copy(), self.quat.copy()

    def reset_timer(self):
        self.transition_start = time.monotonic()
        self.transition_q = self.qpos.copy()
        self.counter = 0
        self.reset()

    def set_mode(self, mode):
        if mode not in ('stand', 'sit', 'move', 'stop'):
            raise ValueError('Unknown mode')
        with self.lock:
            if mode != 'stop':
                if not self.state_time or time.monotonic()-self.state_time > self.config['state_timeout']:
                    raise RuntimeError('No recent robot state')
                if mode == 'move':
                    if (self.mode != 'stand' or time.monotonic()-self.transition_start <
                            self.config['transition_seconds']):
                        raise ValueError('Use stand and wait for the transition before move')
                    self.require_linear_velocity()
            self.mode = mode
            self.commands[:] = 0
            self.reset_timer()
            if self.teleop is not None:
                self.teleop.set_command(self.commands, velocity_override=True)

    def stand_up(self):
        self.set_mode('stand')

    def sit_down(self):
        self.set_mode('sit')

    def move_rl(self):
        self.set_mode('move')

    def _transition(self, goal):
        fraction = min(1.0, (time.monotonic()-self.transition_start) /
                       self.config['transition_seconds'])
        self.target_dof_pos = ((1-fraction)*self.transition_q +
                               fraction*goal[self.motor_to_policy])
        self._publish(self.target_dof_pos)

    def stand(self):
        self._transition(self.default_angles)

    def sit(self):
        self._transition(self.sit_angles)

    def step(self):
        # Training order: lin_vel, ang_vel, gravity, q-relative, dq, action, cmd.
        qpos, qvel, _, quat = self.get_current_state()
        self.target_dof_pos = self.infer(qpos, qvel, quat)
        return self.last_action.copy()

    def move(self):
        now = time.monotonic()
        if self.counter == 0 or now >= self.next_inference:
            if self.teleop is not None:
                self.commands[:] = self.teleop.get_command()
            self.step()
            self.next_inference = (now if self.counter == 0 else self.next_inference) + self.control_dt
            if self.next_inference <= now:
                self.next_inference = now + self.control_dt
        self._publish(self.target_dof_pos)
        self.counter += 1

    def _publish(self, target=None):
        # Send targets and gains; the simulator/robot closes its local PD loop.
        kps, kds = self.kps[self.motor_to_policy], self.kds[self.motor_to_policy]
        for i, motor in enumerate(self.low_cmd.motor_cmd[:12]):
            motor.q = float(target[i]) if target is not None else 0.0
            motor.dq = motor.tau = 0.0
            motor.kp = float(kps[i]) if target is not None else 0.0
            motor.kd = float(kds[i]) if target is not None else 2.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def LowCmdWrite(self):
        torch.set_num_threads(1)
        next_publish = time.monotonic()
        try:
            while self.is_running:
                with self.lock:
                    if time.monotonic()-self.state_time > self.config['state_timeout']:
                        if self.mode != 'stop':
                            print('Robot state timed out; switched to damping')
                            self.set_mode('stop')
                    if self.mode == 'stand':
                        self.stand()
                    elif self.mode == 'sit':
                        self.sit()
                    elif self.mode == 'move':
                        self.move()
                    else:
                        self._publish()
                next_publish += self.dt
                now = time.monotonic()
                if next_publish <= now:
                    next_publish = now
                time.sleep(max(0, next_publish-now))
        except Exception as exc:
            self.error = exc
            print(f'MJLab controller stopped: {exc}')
        finally:
            self.is_running = False
            self._publish()

    def command(self, text):
        words = text.split()
        if not words:
            return
        with self.lock:
            command_dict = {'stand': self.stand_up, 'sit': self.sit_down,
                            'move': self.move_rl, 'stop': lambda: self.set_mode('stop')}
            if len(words) == 1 and words[0] in command_dict:
                if not self.is_running:
                    raise RuntimeError(f'Controller is not running: {self.error}')
                command_dict[words[0]]()
            elif len(words) == 4 and words[0] == 'cmd':
                if self.use_gamepad:
                    raise ValueError('Velocity is controlled by the gamepad')
                if self.mode != 'move':
                    raise ValueError('Use move before sending velocity commands')
                commands = vector([float(v) for v in words[1:]], 3, 'commands')
                self.validate_commands(commands)
                self.commands[:] = commands
                if self.teleop is not None:
                    self.teleop.set_command(commands, velocity_override=True)
            elif words == ['status']:
                print(f'mode={self.mode}, running={self.is_running}, error={self.error}')
                print(f'vx, vy, yaw = {self.commands.tolist()}')
            else:
                raise ValueError('Use stand/sit/move/stop, cmd <vx> <vy> <yaw>, status, exit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('interface', nargs='?')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--domain-id', type=int)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-mapping', type=Path)
    parser.add_argument('--no-gui', action='store_true', help='Disable the keyboard panel')
    parser.add_argument('--gamepad', action='store_true', help='Use a physical gamepad instead of the keyboard panel')
    args = parser.parse_args()
    controller = Controller(args.config, use_gui=not args.no_gui, use_gamepad=args.gamepad)
    if args.check or args.check_mapping:
        q = controller.default_angles[controller.motor_to_policy]
        controller.update_linear_velocity(np.zeros(3))
        target = controller.infer(q, np.zeros(12), [1, 0, 0, 0])
        print(f'Policy: {controller.policy_path}')
        print(f'Input: [1,48]; output: {target.shape}; finite: {np.isfinite(target).all()}')
        print(f'Policy -> DDS: {controller.policy_to_motor.tolist()}')
        print('Preprocessing: training-confirmed 48 values, relative joints / physical units; no history or height scan.')
        print(f'PD: KP={controller.kps[0]}, KD={controller.kds[0]}; control={controller.control_dt}s')
        if args.check_mapping:
            controller.check_mujoco_mapping(args.check_mapping)
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    domain = args.domain_id if args.domain_id is not None else controller.config['domain_id']
    interface = args.interface or controller.config['interface']
    ChannelFactoryInitialize(domain, interface)
    try:
        controller.Init()
        print(f'MJLab ready on domain {domain}, interface {interface}; mode=damping.')
        print('stand -> wait 3 seconds -> move')
        print('Commands: stand, sit, move, stop, cmd <vx> <vy> <yaw>, status, exit')
        if args.gamepad:
            print('Gamepad: left stick=forward/lateral, right stick=yaw; button 0=zero velocity. '
                  'Center sticks after stand/move. Console stop=damping.')
        elif controller.teleop is not None:
            print(f"Keyboard mode: {controller.config.get('keyboard_mode', 'incremental')}; "
                  'W/S, A/D, Q/E; Space/Esc=zero velocity. Focus the panel first.')
        while controller.is_running:
            text = input('MJLab> ').strip()
            if text == 'exit':
                break
            try:
                controller.command(text)
            except (ValueError, RuntimeError) as exc:
                print(exc)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        controller.ShutDown()


if __name__ == '__main__':
    main()
