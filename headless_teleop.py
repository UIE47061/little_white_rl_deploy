"""Shared pygame gamepad input for MJLab and WTW, without a GUI window.

Axis convention follows morrisx28/big_reddog_rl_deploy/headless_teleop.py.
This adapter supplies velocity only; robot modes remain console commands.
"""
import argparse
import multiprocessing as mp
import os
from queue import Empty
import time

import numpy as np


DEFAULTS = dict(device_index=0, forward_axis=1, lateral_axis=0, yaw_axis=3,
                forward_sign=-1, lateral_sign=-1, yaw_sign=-1,
                zero_button=0, deadzone=.1, timeout=.25)


def gamepad_config(config):
    cfg = {**DEFAULTS, **config.get('gamepad', {})}
    for name in ('device_index', 'forward_axis', 'lateral_axis', 'yaw_axis', 'zero_button'):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f'gamepad.{name} must be a nonnegative integer')
    for name in ('forward_sign', 'lateral_sign', 'yaw_sign'):
        if cfg[name] not in (-1, 1):
            raise ValueError(f'gamepad.{name} must be -1 or 1')
    if not np.isfinite([cfg['deadzone'], cfg['timeout']]).all():
        raise ValueError('Gamepad deadzone and timeout must be finite')
    if not 0 <= cfg['deadzone'] < 1 or cfg['timeout'] <= .02:
        raise ValueError('Gamepad deadzone must be [0,1); timeout must exceed 0.02s')
    return cfg


def velocity_limits(config, metadata):
    limits = np.array([config['max_lin'], config.get('max_backward', .2),
                       config.get('max_lateral', .2), config['max_ang']], dtype=np.float32)
    if not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError('Gamepad velocity limits must be finite and positive')
    ranges = metadata.get('training_command_ranges')
    if ranges is not None:
        limits = np.minimum(limits, [ranges['lin_vel_x'][1], -ranges['lin_vel_x'][0],
                            min(ranges['lin_vel_y'][1], -ranges['lin_vel_y'][0]),
                            min(ranges['ang_vel_z'][1], -ranges['ang_vel_z'][0])])
    if (limits <= 0).any():
        raise ValueError('Gamepad expects bidirectional velocity ranges')
    return limits


def axis_velocity(axes, limits, deadzone):
    axes = np.asarray(axes, dtype=np.float32)
    if axes.shape != (3,) or not np.isfinite(axes).all():
        raise ValueError('Invalid gamepad axes')
    axes = np.clip(axes, -1, 1)
    axes[np.abs(axes) < deadzone] = 0
    forward, backward, lateral, yaw = limits
    return axes * np.array([forward if axes[0] >= 0 else backward, lateral, yaw], dtype=np.float32)


def _pygame():
    # The event pump needs SDL video initialization, but no window is created.
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    pygame.display.init()
    pygame.joystick.init()
    return pygame


def _open_joystick(pg, cfg):
    index = cfg['device_index']
    if pg.joystick.get_count() <= index:
        raise RuntimeError(f'Gamepad index {index} not found; connect a controller and run headless_teleop.py --list')
    joystick = pg.joystick.Joystick(index)
    joystick.init()
    if max(cfg['forward_axis'],cfg['lateral_axis'],cfg['yaw_axis']) >= joystick.get_numaxes():
        joystick.quit()
        raise ValueError('Configured gamepad axis unavailable; check gamepad.*_axis in YAML')
    if cfg['zero_button'] >= joystick.get_numbuttons():
        joystick.quit()
        raise ValueError('Configured zero_button unavailable on this gamepad')
    return joystick


def gamepad_worker(values, lock, running, startup, heartbeat, reset_serial, cfg, limits):
    pg = joystick = None
    announced = False
    commands = np.frombuffer(values, dtype=np.float32)
    armed = False
    serial = -1
    reconnect_at = 0.
    try:
        pg = _pygame()
        joystick = _open_joystick(pg, cfg)
        startup.put((True, joystick.get_name()))
        announced = True
        clock = pg.time.Clock()
        while running.is_set():
            events = pg.event.get()
            if joystick is not None and any(e.type == pg.JOYDEVICEREMOVED and
                    e.instance_id == joystick.get_instance_id() for e in events):
                joystick.quit()
                joystick = None
                armed = False
                print('Gamepad disconnected; velocity command zeroed.', flush=True)
            if joystick is None and time.monotonic() >= reconnect_at:
                reconnect_at = time.monotonic()+.5
                try:
                    joystick = _open_joystick(pg, cfg)
                    armed = False
                    print('Gamepad reconnected; center sticks to resume.', flush=True)
                except (RuntimeError, ValueError, pg.error):
                    pass
            velocity = np.zeros(3, dtype=np.float32)
            if joystick is not None:
                axes = [joystick.get_axis(cfg[f'{axis}_axis'])*cfg[f'{axis}_sign']
                        for axis in ('forward', 'lateral', 'yaw')]
                zero_pressed = bool(joystick.get_button(cfg['zero_button']))
                velocity = axis_velocity(axes, limits, cfg['deadzone'])
                with lock:
                    if serial != reset_serial.value:
                        serial = reset_serial.value
                        armed = False
                    if zero_pressed:
                        armed = False
                    elif not np.any(velocity):
                        armed = True
                    if not armed:
                        velocity[:] = 0
                    commands[:3] = velocity
                    heartbeat.value = time.monotonic()
            else:
                with lock:
                    commands[:3] = 0
                    heartbeat.value = time.monotonic()
            clock.tick(50)
    except Exception as exc:
        if not announced:
            startup.put((False, str(exc)))
        else:
            print(f'Gamepad stopped: {exc}; velocity command zeroed.', flush=True)
    finally:
        with lock:
            commands[:3] = 0
        running.clear()
        if joystick is not None:
            joystick.quit()
        if pg is not None:
            pg.quit()


class HeadlessTeleop:
    def __init__(self, config_init, config, metadata):
        command = np.asarray(config_init, dtype=np.float32)
        if command.shape not in ((3,), (15,)) or not np.isfinite(command).all():
            raise ValueError('Expected 3 MJLab or 15 WTW command values')
        cfg = gamepad_config(config)
        limits = velocity_limits(config, metadata)
        context = mp.get_context('spawn')
        self.lock = context.RLock()
        self.values = context.Array('f', command.tolist(), lock=False)
        self.running = context.Event()
        self.heartbeat = context.Value('d', 0., lock=False)
        self.reset_serial = context.Value('i', 0, lock=False)
        self.timeout = cfg['timeout']
        self.startup = context.Queue(maxsize=1)
        self.running.set()
        self.process = context.Process(target=gamepad_worker, daemon=True,
            args=(self.values,self.lock,self.running,self.startup,self.heartbeat,
                  self.reset_serial,cfg,limits))
        self.process.start()
        try:
            ok, message = self.startup.get(timeout=5)
        except Empty:
            self.close()
            raise RuntimeError('Gamepad initialization timed out') from None
        if not ok:
            self.close()
            raise RuntimeError(message)
        self.name = message
        print(f'Gamepad: {self.name}; vx +{limits[0]:g}/-{limits[1]:g}, '
              f'vy +/-{limits[2]:g}, yaw +/-{limits[3]:g}. Center sticks to enable.', flush=True)

    def get_command(self):
        with self.lock:
            command = np.frombuffer(self.values, dtype=np.float32).copy()
            if (not self.running.is_set() or not self.process.is_alive() or
                    time.monotonic()-self.heartbeat.value > self.timeout):
                command[:3] = 0
                self.reset_serial.value += 1
        return command

    def set_command(self, command, velocity_override=False):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (len(self.values),) or not np.isfinite(command).all():
            raise ValueError('Invalid gamepad command vector')
        with self.lock:
            shared = np.frombuffer(self.values, dtype=np.float32)
            shared[3:] = command[3:]  # Preserve WTW gait/height parameters.
            if velocity_override:
                shared[:3] = 0
                self.reset_serial.value += 1

    def close(self):
        self.running.clear()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.startup.close()
        self.startup.join_thread()


def main():
    parser = argparse.ArgumentParser(description='List pygame gamepad indices and raw axis readings; no DDS.')
    parser.add_argument('--list', action='store_true', help='List devices once and exit')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    pg = _pygame()
    try:
        count = pg.joystick.get_count()
        print(f'Gamepads detected: {count}')
        for i in range(count):
            joy = pg.joystick.Joystick(i)
            joy.init()
            print(f'{i}: {joy.get_name()} | axes={joy.get_numaxes()} buttons={joy.get_numbuttons()}')
        if args.list or count == 0:
            return
        if not 0 <= args.device < count:
            parser.error('Invalid device index')
        joy = pg.joystick.Joystick(args.device)
        joy.init()
        while True:
            pg.event.pump()
            axes = [round(joy.get_axis(i),3) for i in range(joy.get_numaxes())]
            buttons = [i for i in range(joy.get_numbuttons()) if joy.get_button(i)]
            print(f'axes={axes} buttons={buttons}     ', end='\r',flush=True)
            time.sleep(.1)
    except KeyboardInterrupt:
        pass
    finally:
        pg.quit()


if __name__ == '__main__':
    main()
