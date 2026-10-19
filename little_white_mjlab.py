"""MJLab omni deployment over DDS. Use --check before starting a simulator."""
import argparse
import os
from pathlib import Path
import time

import numpy as np

from mjlab_runtime import MJLabPolicy, vector
from little_white_wtw import Controller as DDSController

DEFAULT_CONFIG = Path(__file__).with_suffix('.yaml')


class Controller(DDSController):
    """Reuse DDS transport/timing; observations and actions belong to MJLab."""
    def __init__(self, runtime, use_gui=False, use_gamepad=False):
        super().__init__(runtime, use_gui=False, use_gamepad=use_gamepad)
        self.keyboard_enabled = use_gui and not use_gamepad

    def Init(self):
        super().Init()
        if self.keyboard_enabled:
            if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
                    or os.environ.get('SDL_VIDEODRIVER') == 'dummy'):
                print('No desktop display; keyboard disabled. Use console commands.')
                return
            from mjlab_teleop import MJLabTeleop
            self.teleop = MJLabTeleop(self.runtime.commands,
                self.runtime.metadata['training_command_ranges'], self.config)

    def LowStateMessageHandler(self, msg):
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)
        try:
            gyro = vector(gyro, 3, 'gyro')
        except ValueError:
            return
        with self.lock:
            old_time = self.state_time
            super().LowStateMessageHandler(msg)
            if self.state_time != old_time:
                self.runtime.angular_velocity = gyro.copy()

    def set_mode(self, mode):
        if mode not in ('stand', 'sit', 'move', 'stop'):
            raise ValueError('Unknown mode')
        with self.lock:
            if mode == 'move' and (self.mode != 'stand' or
                    time.monotonic() - self.transition_start < self.config['transition_seconds']):
                raise ValueError('Use stand and wait for the transition before move')
            self.runtime.commands[:] = 0
            super().set_mode(mode)
            if self.teleop is not None:
                self.teleop.set_command(self.runtime.commands, velocity_override=True)

    def command(self, text):
        words = text.split()
        if not words:
            return
        with self.lock:
            if len(words) == 1 and words[0] in ('stand', 'sit', 'move', 'stop'):
                if not self.running:
                    raise RuntimeError(f'Controller is not running: {self.error}')
                self.set_mode(words[0])
            elif len(words) == 4 and words[0] == 'cmd':
                if self.use_gamepad:
                    raise ValueError('Velocity is controlled by the gamepad; use stop for damping')
                if self.mode != 'move':
                    raise ValueError('Use move before sending velocity commands')
                commands = vector([float(v) for v in words[1:]], 3, 'velocity commands')
                self.runtime.validate_commands(commands)
                self.runtime.commands[:] = commands
                if self.teleop is not None:
                    self.teleop.set_command(commands, velocity_override=True)
            elif words == ['status']:
                print(f'mode={self.mode}, running={self.running}, error={self.error}')
                print(f'vx, vy, yaw = {self.runtime.commands.tolist()}')
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
    runtime = MJLabPolicy(args.config)
    if args.check or args.check_mapping:
        q = runtime.default[runtime.motor_to_policy]
        target = runtime.infer(q, np.zeros(12), [1, 0, 0, 0])
        print(f'Policy: {runtime.policy_path}')
        print(f'Input: [1,20,45]; output: {target.shape}; finite: {np.isfinite(target).all()}')
        print(f'Policy -> DDS: {runtime.policy_to_motor.tolist()}')
        print('Preprocessing: relative joints / physical units; training-source confirmation pending.')
        if args.check_mapping:
            runtime.check_mujoco_mapping(args.check_mapping)
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    domain = args.domain_id if args.domain_id is not None else runtime.config['domain_id']
    interface = args.interface or runtime.config['interface']
    ChannelFactoryInitialize(domain, interface)
    controller = Controller(runtime, use_gui=not args.no_gui, use_gamepad=args.gamepad)
    try:
        controller.Init()
        print(f'MJLab ready on domain {domain}, interface {interface}; mode=damping.')
        print('stand -> wait 3 seconds -> move')
        print('Commands: stand, sit, move, stop, cmd <vx> <vy> <yaw>, status, exit')
        if args.gamepad:
            print('Gamepad: left stick=forward/lateral, right stick=yaw; button 0=zero velocity. '
                  'Center sticks after stand/move. Console stop=damping.')
        elif controller.teleop is not None:
            print(f"Keyboard mode: {runtime.config.get('keyboard_mode', 'incremental')}; "
                  'W/S, A/D, Q/E; Space/Esc=zero velocity. Focus the panel first.')
        while controller.running:
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
