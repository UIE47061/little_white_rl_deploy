"""Record received Unitree lowcmd/lowstate for WTW versus mjlab support diagnosis."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import threading
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", nargs="?", help="Same DDS interface used by the controller")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("little_white_wtw.yaml"))
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate config and fields offline")
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error("seconds must be in (0, 300]")
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    interface = args.interface or cfg["interface"]
    domain = cfg["domain_id"] if args.domain_id is None else args.domain_id
    joints = [f"{leg}_{part}" for leg in cfg["motor_leg_order"] for part in ("hip", "thigh", "calf")]
    fields = ["elapsed_s", "state_age_s", "command_age_s", "state_packet_interval_s",
              "command_packet_interval_s", "target_changed",
              *[f"imu_quat_{i}" for i in range(4)],
              *[f"gyro_{i}" for i in range(3)]]
    motor_fields = ("q", "q_des", "q_error", "dq", "dq_des", "kp", "kd",
                    "tau_est", "tau_ff", "pd_requested")
    fields += [f"{joint}_{field}" for joint in joints for field in motor_fields]
    if args.check:
        print(json.dumps(dict(interface=interface, domain_id=domain, joints=joints, columns=len(fields))))
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

    ChannelFactoryInitialize(domain, interface)
    lock = threading.Lock()
    received = {}

    def on_command(msg):
        now = time.monotonic()
        data = [[float(getattr(motor, attr)) for attr in ("q", "dq", "kp", "kd", "tau")]
                for motor in msg.motor_cmd[:12]]
        with lock:
            old = received.get("command")
            interval = now - old[0] if old else None
            received["command"] = (now, interval, data)

    def on_state(msg):
        now = time.monotonic()
        data = [[float(motor.q), float(motor.dq), float(getattr(motor, "tau_est", float("nan")))]
                for motor in msg.motor_state[:12]]
        imu = list(msg.imu_state.quaternion) + list(msg.imu_state.gyroscope)
        with lock:
            old = received.get("state")
            interval = now - old[0] if old else None
            received["state"] = (now, interval, data, imu)

    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    command_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    state_sub.Init(on_state, 10)
    command_sub.Init(on_command, 10)
    metadata = dict(config=str(config_path), interface=interface, domain_id=domain,
                    seconds_requested=args.seconds, sampling_hz=100, motor_order=joints,
                    torque_note="pd_requested is calculated from received commands; tau_est is firmware telemetry")
    policy = (config_path.parent / cfg["policy_path"]).resolve()
    if policy.is_file():
        metadata["policy_sha256"] = hashlib.sha256(policy.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    try:
        print("Waiting for rt/lowstate and rt/lowcmd. Start Stand, then Move and forward.")
        deadline = time.monotonic() + 30
        while True:
            with lock:
                ready = "state" in received and "command" in received
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("No matching state and command. Check interface/domain.")
            time.sleep(.01)
        # Exclusive creation protects prior recordings.
        with args.output.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            started = time.monotonic()
            next_sample = started
            previous_target = None
            while time.monotonic() - started < args.seconds:
                with lock:
                    state, command = received["state"], received["command"]
                now = time.monotonic()
                target = tuple(m[0] for m in command[2])
                row = [now-started, now-state[0], now-command[0], state[1], command[1],
                       int(target != previous_target), *state[3]]
                for measured, requested in zip(state[2], command[2]):
                    q, dq, tau_est = measured
                    q_des, dq_des, kp, kd, tau_ff = requested
                    row.extend((q, q_des, q_des-q, dq, dq_des, kp, kd, tau_est, tau_ff,
                                kp*(q_des-q)+kd*(dq_des-dq)+tau_ff))
                writer.writerow(row)
                rows += 1
                previous_target = target
                next_sample += .01
                time.sleep(max(0, next_sample-time.monotonic()))
    finally:
        state_sub.Close()
        command_sub.Close()
        metadata["rows"] = rows
        if rows:
            args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        print(f"Recorded {rows} rows to {args.output}")


if __name__ == "__main__":
    main()
