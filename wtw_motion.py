"""Shared operator presets and 50 Hz command shaping (no simulator dependencies)."""
import math
import torch

PRESETS = {
    "trot": {"frequency": 2.5, "duration": .5, "swing_height": .06},
    "slippery_trot": {"frequency": 2.5, "duration": .6, "swing_height": .04},
}
LIMITS = dict(planar_acceleration=.4, planar_deceleration=.6, yaw_acceleration=.8,
              straight_speed=.6, turning_speed=.3, turning_yaw_threshold=.05)


def command_limits(overrides=None):
    if overrides is not None and (not isinstance(overrides, dict) or set(overrides) - set(LIMITS)):
        raise ValueError("Unknown command limiter settings")
    values = {**LIMITS, **(overrides or {})}
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("Command limits must be finite and positive")
    if values["turning_speed"] > values["straight_speed"]:
        raise ValueError("Turning speed must not exceed straight speed")
    return values


def apply_preset(command, preset):
    """Preserve velocity, pose, stance and idle selector; explicitly select trot."""
    result = command.clone()
    values = PRESETS[preset]
    result[..., 4] = values["frequency"]
    result[..., 5:8] = result.new_tensor([.5, 0, 0])
    result[..., 8] = values["duration"]
    result[..., 9] = values["swing_height"]
    return result


def limit_velocity(current, target, dt=.02, *, emergency=False, limits=None):
    """Vector (not per-axis) slew limits. Brake before entering a fast turn.

    Inputs end in (vx, vy, yaw). dt may be a scalar or one value per world.
    Emergency bypasses all slew limits. Ordinary zero requests decelerate.
    """
    p = LIMITS if limits is None else {**LIMITS, **limits}
    dt = torch.as_tensor(dt, dtype=current.dtype, device=current.device)
    while dt.ndim < current.ndim:
        dt = dt.unsqueeze(-1)
    turning = ((target[..., 2:3].abs() > p["turning_yaw_threshold"])
               | (current[..., 2:3].abs() > p["turning_yaw_threshold"]))
    cap = torch.where(turning, p["turning_speed"], p["straight_speed"])
    desired = target[..., :2]
    desired = desired * (cap / desired.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
    delta = desired - current[..., :2]
    braking = (delta * current[..., :2]).sum(-1, keepdim=True) < 0
    rate = torch.where(braking, p["planar_deceleration"], p["planar_acceleration"])
    planar = current[..., :2] + delta * (rate * dt / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
    yaw_target = torch.where(planar.norm(dim=-1, keepdim=True) > p["turning_speed"] + 1e-6,
                             torch.zeros_like(target[..., 2:3]), target[..., 2:3])
    yaw = current[..., 2:3] + (yaw_target - current[..., 2:3]).clamp(
        min=-p["yaw_acceleration"] * dt, max=p["yaw_acceleration"] * dt)
    result = torch.cat((planar, yaw), -1)
    emergency = torch.as_tensor(emergency, device=current.device, dtype=torch.bool)
    if emergency.ndim == result.ndim - 1:
        emergency = emergency.unsqueeze(-1)
    return torch.where(emergency, torch.zeros_like(result), result)
