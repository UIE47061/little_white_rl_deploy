"""WTW keyboard panel, matching sim2sim_mujoco's command keys and limits."""

import multiprocessing as mp
import time

import numpy as np

from wtw_commands import GAITS, WTWCommandState, validate_command


def panel_worker(values, lock, manual_velocity, running, ready, defaults, limits, speeds, idle_control):
    import pygame

    state = WTWCommandState(defaults, limits)
    command = np.frombuffer(values, dtype=np.float32)
    held = set()
    movement_keys = {"w", "s", "a", "d", "q", "e"}
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((780, 650))
        pygame.display.set_caption("WTW Control Panel")
        font = pygame.font.SysFont("DejaVu Sans", 18)
        title = pygame.font.SysFont("DejaVu Sans", 24, bold=True)
        pygame.key.set_repeat(250, 100)
        clock = pygame.time.Clock()
        ready.set()
        while running.is_set():
            events = pygame.event.get()
            with lock:
                state.command[:] = command
                for event in events:
                    if event.type == pygame.QUIT:
                        running.clear()
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        held.clear()
                        manual_velocity.value = False
                    elif event.type == pygame.KEYDOWN:
                        key = pygame.key.name(event.key)
                        if key == "escape":
                            running.clear()
                        elif key in movement_keys:
                            held.add(key)
                            manual_velocity.value = False
                        elif key == "space":
                            held.add(key)
                            manual_velocity.value = False
                        elif key == "c":
                            if idle_control and key not in held:
                                state.handle(key)
                            held.add(key)
                        else:
                            state.handle(key)
                            if key == "0":
                                held.clear()
                                manual_velocity.value = False
                    elif event.type == pygame.KEYUP:
                        held.discard(pygame.key.name(event.key))
                if not manual_velocity.value:
                    vx, backward, vy, yaw = speeds
                    state.command[:3] = [
                        vx*int("w" in held)-backward*int("s" in held),
                        vy*(int("a" in held)-int("d" in held)),
                        yaw*(int("q" in held)-int("e" in held)),
                    ]
                if "space" in held or not running.is_set():
                    state.command[:3] = 0
                command[:] = state.command
                c = command.copy()
            gait = next((name for name, value in GAITS.items()
                         if np.allclose(c[5:8], value)), "custom")
            screen.fill((23, 27, 35))
            screen.blit(title.render("WTW CONTROL", True, (98, 204, 234)), (24, 18))
            lines = [
                "Click this window to control. Use stand / move in the terminal.",
                "W / S : forward / backward (hold; release to stop)",
                "A / D : left / right        Q / E : turn left / right",
                "Space : zero velocity      Esc : zero velocity and close panel",
                "1 trot    2 pace    3 bound    4 pronk",
                "R / F : frequency +/-       T / G : body height +/-",
                "Y / H : swing height +/-    U / J : pitch +/-",
                "I / K : roll +/-            O / L : stance width +/-",
                "P / ; : stance length +/-   ] / [ : duty factor +/-",
                ("C : idle STAND / STEP      0 : restore defaults" if idle_control else
                 "Idle switch requires the new trained policy.  0 : restore defaults"),
                "",
                f"Gait: {gait.upper()}    vx {c[0]:+.2f}    vy {c[1]:+.2f}    yaw {c[2]:+.2f}",
                f"Frequency: {c[4]:.2f} Hz    Duty: {c[8]:.3f}    Idle: {'STEP' if c[14] else 'STAND'}",
                f"Body height: {c[3]:+.3f} m     Swing height: {c[9]:.3f} m",
                f"Pitch: {c[10]:+.3f} rad       Roll: {c[11]:+.3f} rad",
                f"Stance width: {c[12]:.3f} m    Length: {c[13]:.3f} m",
                "Gait/body changes preserve policy history and gait phase.",
                "Displayed values are commands; actual tracking may differ.",
            ]
            for index, line in enumerate(lines):
                color = (135, 218, 166) if 11 <= index <= 15 else (221, 226, 234)
                screen.blit(font.render(line, True, color), (24, 64+index*31))
            pygame.display.flip()
            clock.tick(50)
    finally:
        with lock:
            command[:3] = 0
        running.clear()
        pygame.quit()


class WTWTeleop:
    """A separate GUI process with atomic shared 15-dimensional commands."""

    def __init__(self, config_init, max_lin=0.3, max_ang=0.5,
                 max_lateral=0.15, max_backward=0.2, control_limits=None, idle_control=False):
        defaults = validate_command(config_init).copy()
        WTWCommandState(defaults, control_limits)  # Validate before spawning.
        context = mp.get_context("spawn")
        self.lock = context.RLock()
        self.values = context.Array("f", defaults.tolist(), lock=False)
        self.manual_velocity = context.Value("b", True, lock=False)
        self.running = context.Event()
        self.ready = context.Event()
        self.running.set()
        self.process = context.Process(
            target=panel_worker, daemon=True,
            args=(self.values, self.lock, self.manual_velocity, self.running,
                  self.ready, defaults, control_limits,
                  (max_lin, max_backward, max_lateral, max_ang), idle_control))
        self.process.start()
        deadline = time.monotonic()+5
        while not self.ready.wait(0.05):
            if not self.process.is_alive() or time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("WTW keyboard window could not start; use --no-gui")

    def get_command(self):
        with self.lock:
            command = np.frombuffer(self.values, dtype=np.float32).copy()
        if not self.process.is_alive() or not self.running.is_set():
            command[:3] = 0
        return command

    def set_command(self, command, velocity_override=False):
        command = validate_command(command)
        with self.lock:
            np.frombuffer(self.values, dtype=np.float32)[:] = command
            if velocity_override:
                self.manual_velocity.value = True

    def close(self):
        self.running.clear()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
