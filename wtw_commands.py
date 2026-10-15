"""WTW command contract and keyboard state, independent of the training repo."""

import numpy as np

COMMAND_NAMES = (
    "vx",
    "vy",
    "yaw_rate",
    "height",
    "frequency",
    "phase",
    "offset",
    "bound",
    "duration",
    "swing_height",
    "pitch",
    "roll",
    "width",
    "length",
    "idle_step",
)
COMMAND_SCALE = (
    2.0,
    2.0,
    0.25,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.15,
    0.3,
    0.3,
    1.0,
    1.0,
    1.0,
)
GAITS = {
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.0),
    "pronk": (0.0, 0.0, 0.0),
}


def control_parameters(command, control_limits=None):
    """Single source of ranges for the keyboard panel and web controls."""
    defaults = validate_command(command)
    limits = control_limits or {}
    if set(limits) - {"height"}:
        raise ValueError("Unsupported WTW keyboard control limit")
    height = np.asarray(limits.get("height", [-0.04, 0.04]), dtype=float)
    if height.shape != (2,) or not np.isfinite(height).all() or height[0] > height[1]:
        raise ValueError("WTW height limits require finite MIN <= MAX")
    width, length = map(float, defaults[12:14])
    specs = (
        ("height", 3, *height, 0.01, "m", "身體高度偏移"),
        ("frequency", 4, 1.8, 3.2, 0.1, "Hz", "踏頻"),
        ("swing_height", 9, 0.03, 0.09, 0.01, "m", "抬腳高度"),
        ("pitch", 10, -0.15, 0.15, 0.025, "rad", "俯仰"),
        ("roll", 11, -0.15, 0.15, 0.025, "rad", "側傾"),
        ("width", 12, width-0.025, width+0.025, 0.005, "m", "站寬"),
        ("length", 13, length-0.025, length+0.025, 0.005, "m", "站長"),
        ("duration", 8, 0.4, 0.6, 0.025, "", "支撐相比例"),
    )
    return {name: dict(index=index, min=float(low), max=float(high), step=step,
                       unit=unit, label=label, default=float(defaults[index]))
            for name, index, low, high, step, unit, label in specs}


def validate_command(command):
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (15,) or not np.isfinite(command).all():
        raise ValueError("WTW requires 15 finite commands")
    if command[4] <= 0 or not 0.1 <= command[8] <= 0.9:
        raise ValueError(
            "WTW frequency must be positive; duty factor must be in [0.1, 0.9]"
        )
    if command[9] < 0 or np.any(command[12:14] <= 0) or command[14] not in (0, 1):
        raise ValueError("Invalid WTW swing height, stance dimensions, or idle_step (0/1)")
    return command


def initial_command(config, geometry):
    settings = config.get("wtw", {})
    command = np.zeros(15, dtype=np.float32)
    command[:3] = config.get("cmd_init", [0.0, 0.0, 0.0])
    command[5:8] = GAITS[settings.get("gait", "trot")]
    for index, name, default in (
        (3, "height", 0.0),
        (4, "frequency", 2.5),
        (8, "duration", 0.5),
        (9, "swing_height", 0.06),
        (10, "pitch", 0.0),
        (11, "roll", 0.0),
        (12, "width", geometry["stance_width"]),
        (13, "length", geometry["stance_length"]),
    ):
        command[index] = settings.get(name, default)
    return validate_command(command)


class WTWCommandState:
    """Persistent gait/body settings; held movement keys are handled separately."""

    def __init__(self, command, control_limits=None):
        self.defaults = validate_command(command).copy()
        self.command = self.defaults.copy()
        self.gait = next(
            (k for k, v in GAITS.items() if np.allclose(self.command[5:8], v)), "custom"
        )
        parameters = control_parameters(self.defaults, control_limits)
        self.adjustments = {}
        for name, keys in zip(parameters, ("tg", "rf", "yh", "uj", "ik", "ol", "p;", "][")):
            spec = parameters[name]
            for key, sign in zip(keys, (1, -1)):
                self.adjustments[key] = (spec["index"], sign*spec["step"], spec["min"], spec["max"])

    def handle(self, key):
        if key == "c":
            self.command[14] = 1 - self.command[14]
        elif key in ("1", "2", "3", "4"):
            self.gait = tuple(GAITS)[int(key) - 1]
            self.command[5:8] = GAITS[self.gait]
        elif key == "0":
            self.command[:] = self.defaults
            self.gait = next(
                (k for k, v in GAITS.items() if np.allclose(self.defaults[5:8], v)),
                "custom",
            )
        elif key in self.adjustments:
            index, step, low, high = self.adjustments[key]
            self.command[index] = np.clip(float(self.command[index]) + step, low, high)
        elif key == "space":
            self.command[:3] = 0
        validate_command(self.command)
