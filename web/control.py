"""Controller-side web protocol. All methods run under Controller.lock.

The controller loop calls watchdog independently of the HTTP process. Receipt
clock values are server-side monotonic timestamps, never browser-supplied times.
"""
import math
import time

from wtw_commands import COMMAND_NAMES, GAITS, control_parameters

COMMAND_TIMEOUT = 0.3


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("Expected a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def make_config(runtime):
    cfg = runtime.config
    return {
        "parameters": control_parameters(runtime.commands, cfg.get("wtw_keyboard_limits")),
        "defaults": dict(zip(COMMAND_NAMES, map(float, runtime.commands))),
        "speeds": {"forward": cfg["max_lin"], "backward": cfg.get("max_backward", 0.2),
                   "lateral": cfg.get("max_lateral", 0.15), "yaw": cfg["max_ang"]},
        "gaits": list(GAITS), "idle_control": cfg.get("idle_control_enabled", False),
        "motor_joints": list(runtime.motor_joints), "quaternion": "wxyz",
        "model_url": "/robot/urdf/little_white_v3.urdf", "command_timeout": COMMAND_TIMEOUT,
        "state_timeout": cfg["state_timeout"], "transition_seconds": cfg["transition_seconds"],
    }


class WebControl:
    def __init__(self, controller):
        self.controller = controller
        self.config = make_config(controller.runtime)
        self.defaults = controller.runtime.commands.copy()
        self.owner = None
        self.last_heartbeat = 0.0
        self.reason = "尚未取得控制"
        self.epoch = 0  # Invalidates held joystick touches on every mode change.

    def fresh(self, now):
        c = self.controller
        return c.state is not None and now-c.state_time <= c.config["state_timeout"]

    def progress(self, now):
        c = self.controller
        if c.mode not in ("stand", "sit"):
            return 1.0
        return min(1.0, max(0.0, (now-c.transition_start)/c.config["transition_seconds"]))

    def release(self, reason, now):
        c = self.controller
        c.runtime.commands[:3] = 0
        if self.owner is not None:
            self.epoch += 1
            if c.mode == "move" and self.fresh(now) and c.running:
                c.set_mode("stand")
        self.owner = None
        self.reason = reason

    def watchdog(self, now=None):
        now = time.monotonic() if now is None else now
        if self.owner is not None:
            if not self.fresh(now):
                self.controller.set_mode("stop")
                self.release("DDS 狀態過期，已切阻尼", now)
            elif now-self.last_heartbeat > COMMAND_TIMEOUT:
                self.release("操控逾時，已清除移動命令", now)

    def handle(self, client, message, received, now=None):
        now = time.monotonic() if now is None else now
        self.watchdog(now)
        if not isinstance(message, dict):
            raise ValueError("Expected an object")
        action = message.get("type")
        if not isinstance(action, str):
            raise ValueError("Missing message type")
        # A delayed release can only release its own session. All other queued
        # requests expire, so a backlog cannot restart the robot after timeout.
        if action == "release":
            if self.owner == client:
                self.release("已釋放控制", now)
            return
        if now-received > COMMAND_TIMEOUT or received > now:
            raise ValueError("Expired command; acquire control again")
        c = self.controller
        if action == "acquire":
            if self.owner is not None and self.owner != client:
                raise ValueError("另一個裝置正在控制")
            if not c.running or not self.fresh(now):
                raise ValueError("控制器尚未就緒或 DDS 狀態過期")
            if self.owner is None:
                self.owner = client
                self.epoch += 1
                c.runtime.commands[:3] = 0
            self.last_heartbeat = received
            self.reason = "已取得控制"
            return
        if self.owner != client:
            raise ValueError("請先取得控制")
        if not c.running:
            raise ValueError("Controller is not running")
        if action == "control":
            speeds = self.config["speeds"]
            values = [number(message.get(k)) for k in ("vx", "vy", "yaw_rate")]
            limits = [(-speeds["backward"], speeds["forward"]),
                      (-speeds["lateral"], speeds["lateral"]), (-speeds["yaw"], speeds["yaw"])]
            if any(not lo <= value <= hi for value, (lo, hi) in zip(values, limits)):
                raise ValueError("Velocity outside configured limits")
            if message.get("epoch") != self.epoch:
                raise ValueError("Mode changed; reset joystick")
            # Heartbeats never arm motion; only a new mode request can do that.
            self.last_heartbeat = received
            if c.mode == "move":
                c.runtime.commands[:3] = values
            else:
                c.runtime.commands[:3] = 0
        elif action == "mode":
            mode = message.get("mode")
            if mode not in ("stand", "sit", "move", "stop"):
                raise ValueError("Unknown mode")
            if mode == "move" and (c.mode != "stand" or self.progress(now) < 1):
                raise ValueError("請先完成 Stand，再啟動 Move")
            if mode != "stop" and not self.fresh(now):
                raise ValueError("DDS state is stale")
            c.runtime.commands[:3] = 0
            c.set_mode(mode)
            self.epoch += 1
        elif action in ("parameters", "gait", "reset"):
            candidate = c.runtime.commands.copy()
            if action == "reset":
                candidate[:] = self.defaults
                candidate[:3] = 0
            elif action == "gait":
                gait = message.get("gait")
                if not isinstance(gait, str) or gait not in GAITS:
                    raise ValueError("Unknown gait")
                candidate[5:8] = GAITS[gait]
            else:
                parameters = message.get("values")
                if not isinstance(parameters, dict) or not parameters:
                    raise ValueError("Expected parameter values")
                for name, raw in parameters.items():
                    value = number(raw)
                    if name == "idle_step" and self.config["idle_control"]:
                        if value not in (0, 1):
                            raise ValueError("idle_step must be 0 or 1")
                        candidate[14] = value
                    else:
                        spec = self.config["parameters"].get(name)
                        if spec is None or not spec["min"]-1e-7 <= value <= spec["max"]+1e-7:
                            raise ValueError(f"Invalid parameter: {name}")
                        candidate[spec["index"]] = value
            c.runtime.validate_commands(candidate)
            c.runtime.commands[:] = candidate
        else:
            raise ValueError("Unknown request type")

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        c = self.controller
        q, dq, quat = c.state if c.state is not None else (None, None, None)
        command = c.runtime.commands
        gait = next((name for name, values in GAITS.items()
                     if all(abs(float(command[i+5])-v) < 1e-5 for i, v in enumerate(values))), "custom")
        return {
            "type": "state", "time": now, "state_age": max(0.0, now-c.state_time),
            "fresh": self.fresh(now), "running": c.running, "mode": c.mode,
            "transition": self.progress(now), "owner": self.owner, "epoch": self.epoch,
            "reason": self.reason, "error": str(c.error) if c.error else None,
            "q": q.tolist() if q is not None else None,
            "dq": dq.tolist() if dq is not None else None,
            "quaternion": quat.tolist() if quat is not None else None,
            "commands": dict(zip(COMMAND_NAMES, map(float, command))), "gait": gait,
            "preset": getattr(c.runtime, "preset", None),
            "effective_commands": dict(zip(COMMAND_NAMES, map(float, getattr(c.runtime, "effective_commands", command)))),
            "cycles": c.counter,
        }
