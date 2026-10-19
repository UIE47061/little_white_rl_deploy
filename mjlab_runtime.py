"""MJLab omni TorchScript observation/history and DDS joint adapter."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result


class MJLabPolicy:
    def __init__(self, config_path):
        path = Path(config_path).resolve()
        self.config = cfg = yaml.safe_load(path.read_text())
        self.policy_path = (path.parent / cfg['policy_path']).resolve()
        self.metadata = meta = json.loads((path.parent / cfg['metadata_path']).read_text())
        expected = {'input_shape': ['batch', 20, 45], 'output_shape': ['batch', 12],
                    'history_order': 'oldest_to_newest',
                    'reset_history': 'repeat_first_observation',
                    'term_names': ['base_ang_vel', 'projected_gravity', 'joint_pos',
                                   'joint_vel', 'actions', 'command'],
                    'term_dims': [3, 3, 12, 12, 12, 3], 'quaternion_order': 'wxyz',
                    'normalization': 'included_in_policy_do_not_normalize_again'}
        for name, value in expected.items():
            if meta.get(name) != value:
                raise ValueError(f'Unsupported MJLab metadata: {name}')
        if hashlib.sha256(self.policy_path.read_bytes()).hexdigest() != cfg['policy_sha256']:
            raise ValueError('Model SHA256 differs from configuration')
        self.default = vector(meta['default_joint_pos'], 12, 'default pose')
        self.kp = np.full(12, meta['kp'], dtype=np.float32)
        self.kd = np.full(12, meta['kd'], dtype=np.float32)
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
        self.scales = vector(cfg['observation_scales'], 45, 'observation_scales')
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        torch.set_num_threads(1)
        self.policy = torch.jit.load(str(self.policy_path), map_location='cpu').eval()
        self.reset()

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
        self.history = np.zeros((20, 45), dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.history_initialized = False

    def observe(self, q, dq, quaternion):
        self.validate_commands(self.commands)
        pos = vector(q, 12, 'joint positions')[self.policy_to_motor]
        if self.config['joint_position_mode'] == 'relative':
            pos = pos - self.default
        elif self.config['joint_position_mode'] != 'absolute':
            raise ValueError('joint_position_mode must be relative or absolute')
        return np.concatenate((vector(self.angular_velocity, 3, 'gyro'),
                               self.gravity(quaternion), pos,
                               vector(dq, 12, 'joint velocities')[self.policy_to_motor],
                               self.last_action, self.commands)) * self.scales

    def infer(self, q, dq, quaternion):
        obs = self.observe(q, dq, quaternion)
        if not self.history_initialized:
            self.history[:] = obs
            self.history_initialized = True
        else:
            self.history[:-1] = self.history[1:].copy()
            self.history[-1] = obs
        with torch.inference_mode():
            # Export packs terms internally; supply chronological frames unchanged.
            output = self.policy(torch.from_numpy(self.history.copy()).unsqueeze(0))
        if tuple(output.shape) != (1, 12):
            raise ValueError('Expected policy output [1,12]')
        action = vector(output[0].cpu().numpy(), 12, 'policy action')
        self.last_action = action.copy()
        clip = self.config.get('action_clip')
        if clip is not None:
            action = np.clip(action, -float(clip), float(clip))
        target = self.default + float(self.metadata['action_scale']) * action
        return target[self.motor_to_policy]

    def check_mujoco_mapping(self, scene_path):
        # Shared check inspects actuator AND position/velocity sensor order.
        from little_white_wtw import WTWPolicy
        WTWPolicy.check_mujoco_mapping(self, scene_path)
