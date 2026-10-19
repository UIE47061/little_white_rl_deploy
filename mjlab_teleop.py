"""Three-axis MJLab keyboard panel, in a separate process."""
import multiprocessing as mp
import time

import numpy as np


def held_command(held, speeds):
    forward, backward, lateral, yaw = speeds
    if 'space' in held:
        return np.zeros(3, dtype=np.float32)
    return np.array([forward*('w' in held)-backward*('s' in held),
                     lateral*(('a' in held)-('d' in held)),
                     yaw*(('q' in held)-('e' in held))], dtype=np.float32)


def increment_command(command, key, speeds, linear_step=.2, angular_step=.2):
    result = np.asarray(command, dtype=np.float32).copy()
    if key in ('space', 'escape'):
        result[:] = 0
    elif key in ('w', 's', 'a', 'd', 'q', 'e'):
        axis, sign = {'w': (0, 1), 's': (0, -1), 'a': (1, 1),
                      'd': (1, -1), 'q': (2, 1), 'e': (2, -1)}[key]
        result[axis] += sign * (angular_step if axis == 2 else linear_step)
        result = np.clip(result, [-speeds[1], -speeds[2], -speeds[3]],
                         [speeds[0], speeds[2], speeds[3]]).astype(np.float32)
    return result


def panel_worker(values, lock, manual, running, ready, speeds, mode='hold', steps=(.2,.2)):
    import pygame
    command = np.frombuffer(values, dtype=np.float32)
    held = set()
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((720, 340))
        pygame.display.set_caption('MJLab Keyboard Controller')
        font = pygame.font.SysFont('DejaVu Sans', 19)
        if mode == 'incremental':
            pygame.key.set_repeat(250, 100)
        clock = pygame.time.Clock()
        ready.set()
        while running.is_set():
            events = pygame.event.get()
            with lock:
                for event in events:
                    if event.type == pygame.QUIT:
                        running.clear()
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        held.clear()
                        manual.value = False
                        command[:] = 0
                    elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                        key = pygame.key.name(event.key)
                        if event.type == pygame.KEYDOWN:
                            if key == 'escape':
                                command[:] = 0
                                held.clear()
                                manual.value = False
                            elif key in ('w', 's', 'a', 'd', 'q', 'e', 'space'):
                                held.add(key)
                                manual.value = False
                                if mode == 'incremental':
                                    command[:] = increment_command(command, key, speeds, *steps)
                        else:
                            held.discard(key)
                if mode == 'hold' and not manual.value:
                    command[:] = held_command(held, speeds)
                if not running.is_set():
                    command[:] = 0
                current = command.copy()
            screen.fill((23, 27, 35))
            lines = ['MJLAB KEYBOARD CONTROLLER',
                     'Terminal: stand -> wait 3 seconds -> move',
                     ('Press keys to increase/decrease and retain velocity.' if mode == 'incremental'
                      else 'Click this window, then hold movement keys.'),
                     'W / S: forward / backward',
                     'A / D: left / right     Q / E: turn left / right',
                     'Space / Esc: zero velocity',
                     ('Key release retains velocity; lose focus: zero.' if mode == 'incremental'
                      else 'Release keys / lose focus: zero velocity'),
                     'Terminal stop: damping; exit: quit controller',
                     f'vx {current[0]:+.2f} m/s    vy {current[1]:+.2f} m/s    yaw {current[2]:+.2f} rad/s']
            for row, line in enumerate(lines):
                screen.blit(font.render(line, True, (221, 226, 234)), (20, 18 + row*34))
            pygame.display.flip()
            clock.tick(50)
    finally:
        with lock:
            command[:] = 0
        running.clear()
        pygame.quit()


class MJLabTeleop:
    def __init__(self, commands, ranges, config=None):
        config = config or {}
        mode = config.get('keyboard_mode', 'incremental')
        if mode not in ('incremental', 'hold'):
            raise ValueError('keyboard_mode must be incremental or hold')
        steps = (float(config.get('keyboard_linear_step', .2)),
                 float(config.get('keyboard_angular_step', .2)))
        caps = [float(config.get(name, default)) for name, default in
                [('max_lin',1.0), ('max_backward',.6), ('max_lateral',.6), ('max_ang',.6)]]
        if not np.isfinite([*steps,*caps]).all() or min(*steps,*caps) <= 0:
            raise ValueError('Keyboard steps and speed limits must be finite and positive')
        context = mp.get_context('spawn')
        self.lock = context.RLock()
        self.values = context.Array('f', list(commands), lock=False)
        self.manual = context.Value('b', True, lock=False)
        self.running, self.ready = context.Event(), context.Event()
        self.running.set()
        speeds = (min(caps[0], ranges['lin_vel_x'][1]), min(caps[1], -ranges['lin_vel_x'][0]),
                  min(caps[2], ranges['lin_vel_y'][1], -ranges['lin_vel_y'][0]),
                  min(caps[3], ranges['ang_vel_z'][1], -ranges['ang_vel_z'][0]))
        self.process = context.Process(target=panel_worker, daemon=True,
            args=(self.values, self.lock, self.manual, self.running, self.ready, speeds, mode, steps))
        self.process.start()
        deadline = time.monotonic() + 5
        while not self.ready.wait(.05):
            if not self.process.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError('Keyboard panel could not start; use --no-gui')

    def get_command(self):
        with self.lock:
            result = np.frombuffer(self.values, dtype=np.float32).copy()
        if not self.running.is_set() or not self.process.is_alive():
            result[:] = 0
        return result

    def set_command(self, command, velocity_override=False):
        with self.lock:
            np.frombuffer(self.values, dtype=np.float32)[:] = command
            if velocity_override:
                self.manual.value = True

    def close(self):
        self.running.clear()
        self.process.join(2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)
