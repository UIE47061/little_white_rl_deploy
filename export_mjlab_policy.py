"""Export the 48-observation Little White actor using its training environment.

Run from ../mjlab with uv run --no-sync ../little_white_rl_deploy/export_mjlab_policy.py
--checkpoint PATH --output PATH. Runtime deployment does not require mjlab.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='Mjlab-Velocity-Blind-Easy-Little-White-V3-Gait-V4')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = load_env_cfg(args.task, play=True)
    terms = list(cfg.observations['actor'].terms)
    expected = ['base_lin_vel', 'base_ang_vel', 'projected_gravity',
                'joint_pos', 'joint_vel', 'actions', 'command']
    if terms != expected:
        raise ValueError(f'Unsupported actor terms: {terms}')
    rl_cfg = load_rl_cfg(args.task)
    if rl_cfg.clip_actions is not None:
        raise ValueError('This export expects unclipped training actions')
    actor_cfg = asdict(rl_cfg.actor)
    for key in ('class_name', 'cnn_cfg', 'rnn_type', 'rnn_hidden_dim', 'rnn_num_layers'):
        actor_cfg.pop(key, None)
    obs = TensorDict({'actor': torch.zeros(1, 48)}, batch_size=[1])
    actor = MLPModel(obs, {'actor': ['actor']}, 'actor', 12, **actor_cfg).eval()
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    exported = torch.jit.script(actor.as_jit().eval())
    generator = torch.Generator().manual_seed(46)
    inputs = torch.randn(256, 48, generator=generator)
    with torch.inference_mode():
        reference = actor(TensorDict({'actor': inputs}, batch_size=[256]))
        actual = exported(inputs)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    policy_path = args.output / 'policy.pt'
    exported.save(str(policy_path))
    pose = cfg.scene.entities['robot'].init_state.joint_pos
    actuator = cfg.scene.entities['robot'].articulation.actuators[0]
    command = cfg.commands['twist'].ranges
    metadata = {
        'schema_version': 3,
        'task': args.task,
        'source_checkpoint': str(checkpoint),
        'source_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        'input_shape': ['batch', 48], 'output_shape': ['batch', 12],
        'history_length': 1, 'term_names': terms,
        'term_dims': [3, 3, 3, 12, 12, 12, 3],
        'joint_names': [f'{leg}_{part}_joint' for leg in ('FR', 'FL', 'RL', 'RR')
                        for part in ('hip', 'thigh', 'calf')],
        'default_joint_pos': [pose[f'.*_{part}_joint'] for _ in range(4)
                              for part in ('hip', 'thigh', 'calf')],
        'kp': actuator.stiffness, 'kd': actuator.damping,
        'effort_limit_nm': actuator.effort_limit,
        'action_scale': cfg.actions['joint_pos'].scale,
        'action_clip': None,
        'physics_dt': cfg.sim.mujoco.timestep,
        'policy_hz': 1 / (cfg.sim.mujoco.timestep * cfg.decimation),
        'training_command_ranges': {
            name: list(getattr(command, name))
            for name in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')},
        'quaternion_order': 'wxyz',
        'linear_velocity_frame': 'body',
        'normalization': 'included_in_policy_do_not_normalize_again',
        'joint_position_mode': 'relative',
        'observation_units': 'unscaled_physical_units',
        'terrain_height_scan': False,
        'deployment_status': 'export_and_interface_checked_walking_not_validated',
        'export_max_absolute_error': float((actual - reference).abs().max()),
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported {policy_path}; SHA256 {metadata["policy_sha256"]}')
    print(f'Actor parity: max absolute error {metadata["export_max_absolute_error"]}')


if __name__ == '__main__':
    main()
