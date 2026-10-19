"""WTW TorchScript deployment over Unitree DDS, with a standalone policy check.

Run: python little_white_wtw.py [interface] [--no-gui | --web]
Check without DDS/GUI: python little_white_wtw.py --check
Console: stand, move, sit, stop, gait trot/pace/bound/pronk,
         preset trot/slippery_trot, cmd <name> <value>, status, exit.
The controller sends joint targets and PD gains. The simulator evaluates PD
against current joint state on every physics step and enforces actuator limits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch
import yaml

from wtw_motion import command_limits, limit_velocity


DEFAULT_CONFIG = Path(__file__).with_suffix(".yaml")


def create_teleop(config, commands):
    # WTW uses a pygame panel so held keys and key releases match sim2sim.
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY"):
            print("No desktop display; use console commands for teleoperation.")
            return None
    from wtw_teleop import WTWTeleop
    return WTWTeleop(config_init=commands.tolist(),
                     max_lin=config["max_lin"], max_ang=config["max_ang"],
                     max_lateral=config.get("max_lateral", 0.15),
                     max_backward=config.get("max_backward", 0.2),
                     control_limits=config.get("wtw_keyboard_limits"),
                     idle_control=config.get("idle_control_enabled", False))


def joint_vector(value, name):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (12,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain 12 finite values")
    return vector


class WTWPolicy:
    """Own the WTW observation history and policy/DDS joint permutations."""

    def __init__(self, config_path=DEFAULT_CONFIG):
        config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(config_path.read_text())
        cfg = self.config
        self.policy_path = (config_path.parent / cfg["policy_path"]).resolve()
        metadata_path = (config_path.parent / cfg["metadata_path"]).resolve()
        self.metadata = json.loads(metadata_path.read_text())
        meta = self.metadata
        self.config["idle_control_enabled"] = meta.get("idle_control", {}).get("version") == 1
        expected_fields = [["gravity", 3], ["commands", 15], ["q_relative", 12],
                           ["dq", 12], ["last_action", 12], ["previous_action", 12], ["clock", 4]]
        expected_joints = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RL", "RR")
                           for part in ("hip", "thigh", "calf")]
        for key, expected in {"schema_version": 2, "observation_version": 1,
                              "gait_version": 2, "control_version": 1,
                              "obs_fields": expected_fields, "joints": expected_joints,
                              "history_order": "oldest_to_newest", "quaternion": "wxyz"}.items():
            if meta.get(key) != expected:
                raise ValueError(f"Unsupported WTW metadata: {key}")
        for key, meta_key in {"dt": "physics_dt", "control_decimation": "decimation",
                              "num_obs": "obs_dim", "history_length": "history",
                              "num_actions": "action_dim", "action_scale": "action_scale",
                              "action_clip": "action_clip", "torque_limit": "torque_limit",
                              "default_angles": "default", "cmd_scale": "command_scale"}.items():
            if not np.allclose(cfg[key], meta[meta_key], rtol=0, atol=1e-7):
                raise ValueError(f"{key} differs from the exported model contract")
        digest = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        if digest != meta["policy_sha256"]:
            raise ValueError("policy.pt SHA256 does not match metadata.json")
        self.default = joint_vector(cfg["default_angles"], "default_angles")
        self.kp = joint_vector(cfg["kps"], "kps")
        self.kd = joint_vector(cfg["kds"], "kds")
        if not np.allclose(self.kp, meta["kp"]) or not np.allclose(self.kd, meta["kd"]):
            raise ValueError("PD gains differ from the exported model contract")
        if cfg["dof_vel_scale"] != 0.05 or cfg["observation_clip"] != 100.0:
            raise ValueError("WTW requires dq scale 0.05 and observation clip 100")
        legs = cfg["motor_leg_order"]
        if len(legs) != 4 or set(legs) != {"FR", "FL", "RL", "RR"}:
            raise ValueError("motor_leg_order must be a permutation of FR, FL, RL, RR")
        motor_joints = [f"{leg}_{part}_joint" for leg in legs for part in ("hip", "thigh", "calf")]
        self.motor_joints = motor_joints
        self.policy_to_motor = np.array([motor_joints.index(j) for j in expected_joints])
        self.motor_to_policy = np.argsort(self.policy_to_motor)
        self.commands = np.asarray(cfg["cmd_init"], dtype=np.float32)
        self.command_scale = np.asarray(cfg["cmd_scale"], dtype=np.float32)
        self.validate_commands(self.commands)
        self.preset = cfg.get("motion_preset")
        self.motion_limits = None
        if meta.get("slippery_command_limits") is not None:
            if self.preset not in meta.get("operator_presets", {}) or "command_limits" not in cfg:
                raise ValueError("This policy requires an explicit motion_preset and command_limits")
            self.motion_limits = command_limits(cfg["command_limits"])
            if self.motion_limits != meta["slippery_command_limits"]:
                raise ValueError("Command limits differ from the evaluated slippery policy")
            expected_preset = meta["operator_presets"][self.preset]
            if not np.allclose(self.commands[[4, 8, 9]], [expected_preset[k] for k in
                               ("frequency", "duration", "swing_height")], rtol=0, atol=1e-7):
                raise ValueError("cmd_init does not match the selected motion preset")
        elif self.preset is not None or cfg.get("command_limits") is not None:
            raise ValueError("This policy does not declare slippery presets or limits")
        self.device = torch.device(cfg["device"])
        torch.set_num_threads(1)
        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device).eval()
        self.control_dt = cfg["dt"] * cfg["control_decimation"]
        self.reset()

    def validate_commands(self, commands):
        if commands.shape != (15,) or not np.isfinite(commands).all():
            raise ValueError("Commands must contain 15 finite values")
        if commands[4] <= 0 or not 0.1 <= commands[8] <= 0.9:
            raise ValueError("Frequency must be positive and duty must be in [0.1, 0.9]")
        if commands[9] < 0 or np.any(commands[12:14] <= 0) or commands[14] not in (0, 1):
            raise ValueError("Invalid swing height, stance dimensions or idle_step (0/1)")
        if commands[14] and self.metadata.get("idle_control", {}).get("version") != 1:
            raise ValueError("This policy does not support idle_step")
        if self.metadata.get("slippery_command_limits") is not None:
            cfg = self.config
            limits = [(-cfg.get("max_backward", .3), cfg["max_lin"]),
                      (-cfg.get("max_lateral", .25), cfg.get("max_lateral", .25)),
                      (-cfg["max_ang"], cfg["max_ang"])]
            if any(not lo-1e-6 <= value <= hi+1e-6 for value, (lo, hi) in zip(commands[:3], limits)):
                raise ValueError("Velocity outside configured policy limits")

    def reset(self):
        cfg = self.config
        self.history = np.zeros((cfg["history_length"], cfg["num_obs"]), dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.phase = np.float32(0.0)
        self.effective_commands = self.commands.copy()
        if self.preset == "slippery_trot":
            self.effective_commands[:3] = 0

    def set_preset(self, preset):
        if preset not in self.metadata.get("operator_presets", {}):
            raise ValueError("This policy does not support the requested preset")
        values = self.metadata["operator_presets"][preset]
        self.commands[4] = values["frequency"]
        self.commands[5:8] = [.5, 0, 0]
        self.commands[8] = values["duration"]
        self.commands[9] = values["swing_height"]
        self.preset = preset  # preserve history, last actions and gait phase

    def prepare_command(self):
        """Shape target velocity once per policy step, before writing observations."""
        self.validate_commands(self.commands)
        # Keyboard reset and web parameter controls carry public WTW parameters,
        # so recognize the two exact preset pairs without an extra control bit.
        for name, values in self.metadata.get("operator_presets", {}).items():
            if np.allclose(self.commands[[8, 9]], [values["duration"], values["swing_height"]],
                           rtol=0, atol=1e-7):
                self.preset = name
                break
        self.effective_commands[3:] = self.commands[3:]
        if self.preset == "slippery_trot":
            current = torch.from_numpy(self.effective_commands[:3].copy()).unsqueeze(0)
            target = torch.from_numpy(self.commands[:3].copy()).unsqueeze(0)
            self.effective_commands[:3] = limit_velocity(current, target, self.control_dt,
                                                       limits=self.motion_limits)[0].numpy()
        else:
            self.effective_commands[:3] = self.commands[:3]

    @staticmethod
    def gravity(quaternion):
        quaternion = np.asarray(quaternion, dtype=np.float32)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("Expected a finite wxyz quaternion")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-6:
            raise ValueError("IMU quaternion is not initialized")
        w, x, y, z = quaternion / norm
        return np.array([2 * (w*y - x*z), -2 * (y*z + w*x),
                         1 - 2 * (w*w + z*z)], dtype=np.float32)

    def observe(self, q, dq, quaternion):
        c = self.effective_commands
        self.validate_commands(c)
        p, o, b = c[5:8]
        raw = np.remainder(self.phase + np.array([o, p+o+b, b, p], dtype=np.float32), 1.0)
        duty = c[8]
        warped = np.where(raw < duty, raw * 0.5 / duty,
                          0.5 + (raw-duty) * 0.5 / (1-duty))
        clock = np.sin(2 * np.pi * warped)
        q = joint_vector(q, "q")[self.policy_to_motor]
        dq = joint_vector(dq, "dq")[self.policy_to_motor]
        obs = np.concatenate((self.gravity(quaternion), c * self.command_scale,
                              q-self.default, dq*0.05, self.last_action,
                              self.previous_action, clock))
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def infer(self, q, dq, quaternion):
        self.prepare_command()
        obs = self.observe(q, dq, quaternion)
        self.history[:-1] = self.history[1:].copy()
        self.history[-1] = obs
        tensor = torch.from_numpy(self.history.reshape(1, -1)).to(self.device)
        with torch.inference_mode():
            action = self.policy(tensor).cpu().numpy().reshape(-1)
        action = joint_vector(action, "policy output")
        # Keep raw actions in observations, as in training; clip only for PD targets.
        self.previous_action = self.last_action.copy()
        self.last_action = action.copy()
        self.phase = np.float32(np.remainder(
            self.phase + np.float32(self.control_dt) * self.effective_commands[4], np.float32(1.0)))
        target = self.default + self.config["action_scale"] * np.clip(
            action, -self.config["action_clip"], self.config["action_clip"])
        return target[self.motor_to_policy]

    def torque(self, target, q, dq):
        kp, kd = self.kp[self.motor_to_policy], self.kd[self.motor_to_policy]
        return np.clip(kp * (target-q) - kd*dq,
                       -self.config["torque_limit"], self.config["torque_limit"])

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


class Controller:
    def __init__(self, runtime, use_gui=True, use_gamepad=False):
        self.runtime = runtime
        self.config = runtime.config
        self.lock = threading.RLock()
        self.state = None
        self.state_time = 0.0
        self.mode = "stop"
        self.target = runtime.default[runtime.motor_to_policy].copy()
        self.counter = 0
        self.error = None
        self.running = False
        self.thread = None
        self.teleop = None
        self.use_gui = use_gui
        self.use_gamepad = use_gamepad
        self.web_control = None

    def Init(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_cmd.head[:] = [0xFE, 0xEF]
        self.low_cmd.level_flag = 0xFF
        for motor in self.low_cmd.motor_cmd:
            motor.mode, motor.q, motor.dq = 0x01, 0.0, 0.0
            motor.kp, motor.kd, motor.tau = 0.0, 0.0, 0.0
        self.crc = CRC()
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.LowStateMessageHandler, 10)
        deadline = time.monotonic() + self.config["startup_timeout"]
        while self.state is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("No valid rt/lowstate received; start the simulator first")
            time.sleep(0.01)
        if self.use_gamepad:
            from headless_teleop import HeadlessTeleop
            self.teleop = HeadlessTeleop(self.runtime.commands, self.config, self.runtime.metadata)
        elif self.use_gui:
            self.teleop = create_teleop(self.config, self.runtime.commands)
        self.running = True
        self.thread = threading.Thread(target=self.LowCmdWrite, name="wtw_lowcmd")
        self.thread.start()

    def LowStateMessageHandler(self, msg):
        q = np.array([m.q for m in msg.motor_state[:12]], dtype=np.float32)
        dq = np.array([m.dq for m in msg.motor_state[:12]], dtype=np.float32)
        quat = np.array(msg.imu_state.quaternion, dtype=np.float32)
        try:
            joint_vector(q, "q")
            joint_vector(dq, "dq")
            self.runtime.gravity(quat)
        except ValueError:
            return
        with self.lock:
            self.state = q, dq, quat
            self.state_time = time.monotonic()

    def set_mode(self, mode):
        with self.lock:
            if mode == "stop":
                self.mode = "stop"
                self.counter = 0
                self.runtime.commands[:3] = 0
                self.runtime.reset()
                if self.teleop is not None:
                    self.teleop.set_command(self.runtime.commands, velocity_override=True)
                return  # Damping needs no fresh state and bypasses the command ramp.
            if self.state is None or time.monotonic()-self.state_time > self.config["state_timeout"]:
                raise RuntimeError("No recent robot state")
            self.mode = mode
            self.counter = 0
            self.error = None
            self.transition_start = time.monotonic()
            self.transition_q = self.state[0].copy()
            self.runtime.reset()
            self.target = self.runtime.default[self.runtime.motor_to_policy].copy()
            if self.use_gamepad and self.teleop is not None:
                self.runtime.commands[:3] = 0
                self.teleop.set_command(self.runtime.commands, velocity_override=True)

    def _publish(self, target=None):
        # Do not close the PD loop through delayed DDS state feedback. Send
        # targets/gains so the robot (or simulator) evaluates its local PD loop.
        kp = self.runtime.kp[self.runtime.motor_to_policy]
        kd = self.runtime.kd[self.runtime.motor_to_policy]
        for i, motor in enumerate(self.low_cmd.motor_cmd[:12]):
            motor.q = float(target[i]) if target is not None else 0.0
            motor.kp = float(kp[i]) if target is not None else 0.0
            motor.kd = float(kd[i]) if target is not None else 2.0
            motor.tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def LowCmdWrite(self):
        # Initialize the CPU backend on the actual inference thread too.
        # Setting this only while loading the model left live DDS inference
        # taking >30 ms on this host, despite get_num_threads() reporting 1.
        torch.set_num_threads(1)
        next_publish = time.monotonic()
        next_inference = next_publish
        try:
            while self.running:
                start = time.monotonic()
                with self.lock:
                    if start-self.state_time > self.config["state_timeout"]:
                        if self.mode != "stop":
                            print("Robot state timed out; switched to damping")
                        self.mode = "stop"
                    if self.web_control is not None:
                        self.web_control.watchdog(start)
                    if self.mode == "stop":
                        self._publish()
                    else:
                        q, dq, quat = self.state
                        if self.mode in ("stand", "sit"):
                            goal = (self.runtime.default if self.mode == "stand" else
                                    joint_vector(self.config["sit_angles"], "sit_angles"))
                            fraction = min(1.0, (start-self.transition_start) /
                                           self.config["transition_seconds"])
                            self.target = (1-fraction)*self.transition_q + fraction*goal[self.runtime.motor_to_policy]
                        elif self.counter == 0 or start >= next_inference:
                            if self.teleop is not None:
                                self.runtime.commands[:] = self.teleop.get_command()
                            self.target = self.runtime.infer(q, dq, quat)
                            # DDS serialization and OS sleeps do not have a fixed
                            # duration. Schedule the policy independently of packet count.
                            if self.counter == 0:
                                next_inference = start
                            next_inference += self.runtime.control_dt
                            if next_inference <= start:
                                next_inference = start + self.runtime.control_dt
                        self._publish(self.target)
                        self.counter += 1
                next_publish += self.config["dt"]
                now = time.monotonic()
                if next_publish <= now:
                    next_publish = now
                time.sleep(max(0.0, next_publish-now))
        except Exception as exc:
            self.error = exc
            print(f"WTW controller stopped: {exc}")
        finally:
            self.running = False
            self._publish()

    def command(self, text):
        words = text.split()
        if not words:
            return
        with self.lock:
            if self.teleop is not None:
                self.runtime.commands[:] = self.teleop.get_command()
            if len(words) == 1 and words[0] in ("stand", "sit", "move", "stop"):
                if not self.running:
                    raise RuntimeError(f"Controller is not running: {self.error}")
                self.set_mode(words[0])
            elif len(words) == 2 and words[0] == "gait":
                self.runtime.commands[5:8] = self.runtime.metadata["gait_presets"][words[1]]
                if self.teleop is not None:
                    self.teleop.set_command(self.runtime.commands)
            elif len(words) == 2 and words[0] == "preset":
                self.runtime.set_preset(words[1])
                if self.teleop is not None:
                    self.teleop.set_command(self.runtime.commands)
            elif len(words) == 3 and words[0] in ("cmd", "ctl"):
                index = 14 if words[1] == "idle_step" else self.runtime.metadata["command_names"].index(words[1])
                if self.use_gamepad and index < 3:
                    raise ValueError('Velocity is controlled by the gamepad; use stop for damping')
                candidate = self.runtime.commands.copy()
                candidate[index] = float(words[2])
                self.runtime.validate_commands(candidate)
                self.runtime.commands = candidate
                if self.teleop is not None:
                    self.teleop.set_command(candidate, velocity_override=index < 3)
            elif words == ["status"]:
                print(f"mode={self.mode}, running={self.running}, error={self.error}")
                print(dict(zip(self.runtime.metadata["command_names"], self.runtime.commands.tolist())))
                print(f"idle_step={int(self.runtime.commands[14])} (0=stand, 1=step at zero velocity)")
                print(f"preset={self.runtime.preset}; effective vx/vy/yaw={self.runtime.effective_commands[:3].tolist()}")
            else:
                raise ValueError("Use stand/sit/move/stop, preset trot/slippery_trot, gait <name>, cmd <name> <value>, ctl idle_step <0|1>, status or exit")

    def ShutDown(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.teleop is not None:
            self.teleop.close()
        if hasattr(self, "publisher"):
            self._publish()
        if hasattr(self, "subscriber"):
            self.subscriber.Close()
        if hasattr(self, "publisher"):
            self.publisher.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", nargs="?", help="Override YAML DDS interface")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--gamepad", action="store_true", help="Use a physical gamepad instead of the keyboard panel")
    parser.add_argument("--web", action="store_true", help="Use the phone dashboard instead of keyboard/console")
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Check model inference without DDS or GUI")
    parser.add_argument("--check-mapping", type=Path, metavar="SCENE_XML",
                        help="Verify DDS actuator/sensor order against a MuJoCo scene; no DDS or GUI")
    args = parser.parse_args()
    if args.gamepad and args.web:
        parser.error('--gamepad and --web select different control sources; choose one')
    runtime = WTWPolicy(args.config)
    if args.check or args.check_mapping:
        q = runtime.default[runtime.motor_to_policy].copy()
        target = runtime.infer(q, np.zeros(12, dtype=np.float32), [1, 0, 0, 0])
        print(f"Policy: {runtime.policy_path}")
        print(f"Input: (1, {runtime.history.size}); output: {target.shape}; finite: {np.isfinite(target).all()}")
        print(f"DDS motor order: {runtime.config['motor_leg_order']}")
        print(f"Preset: {runtime.preset}; command limits: {runtime.motion_limits}")
        if args.check_mapping:
            runtime.check_mujoco_mapping(args.check_mapping)
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    interface = args.interface or runtime.config["interface"]
    domain = args.domain_id if args.domain_id is not None else runtime.config["domain_id"]
    ChannelFactoryInitialize(domain, interface)
    if not 1 <= args.web_port <= 65535:
        parser.error("--web-port must be in 1..65535")
    if args.web:
        from web.server import check_dependencies
        check_dependencies()
        if not (Path(__file__).parent / "web/dist/index.html").is_file():
            parser.error("Web assets missing; run npm ci && npm run build in web/frontend")
    controller = Controller(runtime, runtime.config["use_gui"] and not args.no_gui and not args.web,
                            use_gamepad=args.gamepad)
    bridge = None
    try:
        controller.Init()
        print(f"WTW ready on DDS domain {domain}, interface {interface}. Mode: damping.")
        if args.web:
            from web.bridge import WebBridge
            bridge = WebBridge(controller, args.web_host, args.web_port)
            print(f"Dashboard: http://<this computer's LAN IP>:{args.web_port}")
            print("Acquire control in the browser, then Stand -> Move. Ctrl+C to exit.")
            while controller.running:
                time.sleep(0.1)
            return
        print(f"Policy: {runtime.policy_path}; preset: {runtime.preset}")
        if args.gamepad:
            print('Gamepad: left stick=forward/lateral, right stick=yaw; button 0=zero velocity. '
                  'Center sticks after stand/move. Console stop=damping.')
        print("Commands: stand, move, sit, stop, preset trot/slippery_trot, gait <name>, cmd <name> <value>, ctl idle_step <0|1>, status, exit")
        while controller.running:
            text = input("CMD: ").strip()
            if text == "exit":
                break
            try:
                controller.command(text)
            except (ValueError, KeyError, RuntimeError) as exc:
                print(exc)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if bridge is not None:
            bridge.close()
        controller.ShutDown()


if __name__ == "__main__":
    main()
