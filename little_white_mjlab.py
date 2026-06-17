import time
import sys
import numpy as np
import threading
import traceback
import torch
import yaml
import argparse
import matplotlib.pyplot as plt
import csv
import pathlib
import gui_teleop

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import struct


NUM_MOTORS = 12


def build_joint_mappings(sdk_joint_order, policy_joint_order):
    if len(sdk_joint_order) != NUM_MOTORS or len(policy_joint_order) != NUM_MOTORS:
        raise ValueError(f"Both joint orders must contain {NUM_MOTORS} joints")
    if len(set(sdk_joint_order)) != NUM_MOTORS:
        raise ValueError("sdk_joint_order contains duplicate joints")
    if set(sdk_joint_order) != set(policy_joint_order):
        raise ValueError("sdk_joint_order and policy_joint_order must contain the same joints")

    policy_from_sdk = np.array(
        [sdk_joint_order.index(name) for name in policy_joint_order], dtype=np.int64
    )
    sdk_from_policy = np.array(
        [policy_joint_order.index(name) for name in sdk_joint_order], dtype=np.int64
    )
    return policy_from_sdk, sdk_from_policy


class Controller:
    def __init__(self):


        config_file = pathlib.Path(__file__).with_name("little_white_mjlab.yaml")
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.dt = config["dt"]
            self.ang_vel_scale = config["ang_vel_scale"]
            policy_path = pathlib.Path(config["policy_path"]).expanduser()
            self.onnx_path = pathlib.Path(config["onnx_path"]).expanduser()
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.decimation = config["control_decimation"]
            self.cmd_scale = config["cmd_scale"]
            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.action_scale = np.array(config["action_scale"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.sit_angles = np.array(config["sit_angles"], dtype=np.float32)
            self.cmd_init = np.array(config["cmd_init"], dtype=np.float32)
            self.action_clip = float(config.get("action_clip", 1.0))
            self.max_target_step = float(config.get("max_target_step", 0.02))
            self.min_upright_gravity_z = float(
                config.get("min_upright_gravity_z", -0.7)
            )
            self.max_joint_velocity = float(config.get("max_joint_velocity", 15.0))
            self.sdk_joint_order = config["sdk_joint_order"]
            self.policy_joint_order = config["policy_joint_order"]
            self.policy_from_sdk, self.sdk_from_policy = build_joint_mappings(
                self.sdk_joint_order, self.policy_joint_order
            )
            if num_actions != NUM_MOTORS:
                raise ValueError(f"num_actions must be {NUM_MOTORS}, got {num_actions}")
            if num_obs < 45:
                raise ValueError(f"num_obs must be at least 45, got {num_obs}")
            if not policy_path.is_file():
                raise FileNotFoundError(f"Policy file not found: {policy_path}")
            if not self.onnx_path.is_file():
                raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.controller_rt = 0.0
        self.is_running = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        # state
        self.target_dof_pos = self.default_angles.copy()
        self.target_dof_vel = np.zeros(NUM_MOTORS)
        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.quat = np.zeros(4) # q_w q_x q_y q_z
        self.ang_vel = np.zeros(3)
        self.lin_vel = np.zeros(3)

        self.mode = ''

        # Record
        self.ang_vel_data_list = []
        self.gravity_b_list = []
        self.joint_vel_list = []
        self.joint_pos_list = []

        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        self.teleop = gui_teleop.GUITeleop(
            config_init=config["cmd_init"], max_lin=0.8, max_ang=0.5
        )
        self.counter = 0

        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)

        # Chirp data collection
        self.min_freq = 0.1   
        self.max_freq = 2.0  
        self.duration = 20.0  
        self.chirp_amplitude = 1.0
        self.chirp_counter = 0

        self.log_time = []
        self.log_dof_pos = []
        self.log_des_dof_pos = []
        
        self.generate_chirp_profile()

        self.crc = CRC()

    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.highstate_subscriber.Init(self.HighStateMessageHandler, 10)

        # Init default pos #
        self.Start()

        print("Initial Sucess !!!")

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation


    def Start(self):
        self.is_running = True
        self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite)
        self.lowCmdWriteThreadPtr.start()

    def ShutDown(self):
        self.is_running = False
        self.teleop.close()
        self.lowCmdWriteThreadPtr.join()

    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= self.sit_angles[i]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.update_state()

    def HighStateMessageHandler(self, msg: SportModeState_):
        self.lin_vel[:] = msg.velocity
    

    def stand(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.default_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = 25
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.5
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def reset_timer(self):
        self.controller_rt = 0.0
    
    def sit(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.sit_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = 25
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.5
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def move(self):
        if self.counter % self.decimation == 0 and self.counter > 0:
            qpos, qvel, _, quat, _ = self.get_current_state()
            gravity_b = self.get_gravity_orientation(quat)
            invalid_state = not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel))
            tilted = gravity_b[2] > self.min_upright_gravity_z
            excessive_velocity = np.max(np.abs(qvel)) > self.max_joint_velocity
            if invalid_state or tilted or excessive_velocity:
                if invalid_state:
                    print("Safety stop: invalid state detected (qpos/qvel contains NaN or inf).")
                if tilted:
                    print(
                        f"Safety stop: robot tilted. gravity_b[2]={gravity_b[2]:.4f} > min_upright_gravity_z={self.min_upright_gravity_z:.4f}"
                    )
                if excessive_velocity:
                    print(
                        f"Safety stop: excessive joint velocity. max(|qvel|)={np.max(np.abs(qvel)):.4f} > max_joint_velocity={self.max_joint_velocity:.4f}"
                    )
                self.target_dof_pos = qpos.copy()
                self.action.fill(0.0)
                self.mode = ''
                return

            raw_action = self.step()
            self.action = np.clip(raw_action, -self.action_clip, self.action_clip)
            default_policy = self.default_angles[self.policy_from_sdk]
            target_policy = default_policy + self.action * self.action_scale
            desired_sdk = target_policy[self.sdk_from_policy]
            target_step = np.clip(
                desired_sdk - self.target_dof_pos,
                -self.max_target_step,
                self.max_target_step,
            )
            self.target_dof_pos += target_step

        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[i].kp = self.kps[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.counter += 1

    def step(self):
        qpos, qvel, ang_vel, quat, lin_vel = self.get_current_state()
        gravity_b = self.get_gravity_orientation(quat)
        cmd = self.teleop.get_command()
        qpos_policy = qpos[self.policy_from_sdk]
        qvel_policy = qvel[self.policy_from_sdk]
        default_policy = self.default_angles[self.policy_from_sdk]

        self.obs[:3] = lin_vel
        self.obs[3:6] = ang_vel * self.ang_vel_scale
        self.obs[6:9] = gravity_b
        self.obs[9:21] = (qpos_policy - default_policy) * self.dof_pos_scale
        self.obs[21:33] = qvel_policy * self.dof_vel_scale
        self.obs[33:45] = self.action
        self.obs[45:48] = cmd

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        with torch.inference_mode():
            self.action = self.policy(obs_tensor).cpu().numpy().squeeze()

        return self.action

    def inspect_policy(self):
        if self.low_state is None:
            print("No low-state message received yet.")
            return

        qpos, qvel, ang_vel, quat, lin_vel = self.get_current_state()
        gravity_b = self.get_gravity_orientation(quat)
        qpos_policy = qpos[self.policy_from_sdk]
        qvel_policy = qvel[self.policy_from_sdk]
        default_policy = self.default_angles[self.policy_from_sdk]

        obs = np.zeros_like(self.obs)
        obs[:3] = lin_vel
        obs[3:6] = ang_vel * self.ang_vel_scale
        obs[6:9] = gravity_b
        obs[9:21] = (qpos_policy - default_policy) * self.dof_pos_scale
        obs[21:33] = qvel_policy * self.dof_vel_scale
        obs[33:45] = self.action
        obs[45:48] = 0.0

        with torch.inference_mode():
            raw_action_policy = (
                self.policy(torch.from_numpy(obs).unsqueeze(0))
                .cpu()
                .numpy()
                .squeeze()
            )

        action_policy = np.clip(
            raw_action_policy, -self.action_clip, self.action_clip
        )
        target_policy = default_policy + action_policy * self.action_scale
        target_sdk = target_policy[self.sdk_from_policy]
        target_delta = target_sdk - qpos
        limited_target_sdk = qpos + np.clip(
            target_delta, -self.max_target_step, self.max_target_step
        )

        np.set_printoptions(precision=4, suppress=True)
        print("\nPOLICY INSPECTION (no policy command was applied)")
        print(f"quaternion [w,x,y,z]: {quat}")
        print(f"linear velocity:      {lin_vel}")
        print(f"projected gravity:     {gravity_b}")
        print(f"angular velocity:      {ang_vel}")
        print(f"SDK qpos:              {qpos}")
        print(f"policy qpos:           {qpos_policy}")
        print(f"policy default:        {default_policy}")
        print(f"joint-pos observation: {obs[9:21]}")
        print(f"raw policy action:     {raw_action_policy}")
        print(f"clipped policy action: {action_policy}")
        print(f"SDK target:            {target_sdk}")
        print(f"target - current:      {target_delta}")
        print(f"max target jump:       {np.max(np.abs(target_delta)):.4f} rad\n")
        print("SDK joint targets after one limited update:")
        for index, joint_name in enumerate(self.sdk_joint_order):
            print(
                f"{index:2d} {joint_name:15s} "
                f"q={qpos[index]: .4f} desired={target_sdk[index]: .4f} "
                f"limited={limited_target_sdk[index]: .4f}"
            )
        print()
    
    ## Chirp data collection ##
    def generate_chirp_profile(self):
        sample_rate = 1.0 / self.dt
        num_steps = int(self.duration * sample_rate)
        t = np.linspace(0, self.duration, num_steps)
        f0 = self.min_freq; f1 = self.max_freq
        phase = 2 * np.pi * (f0 * t + ((f1 - f0) / (2 * self.duration)) * t ** 2)
        chirp_signal = np.sin(phase)

        self.chirp_traj = np.zeros((num_steps, 12), dtype=np.float32)
        scales = np.array([0.1, 0.25, 0.5] * 4) * self.chirp_amplitude 
        directions = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0])

        for i in range(12):
            self.chirp_traj[:, i] = self.default_angles[i] + chirp_signal * scales[i] * directions[i]
        self.num_chirp_steps = num_steps
    
    def chrip_proccess(self):
        # Reset to default pos
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.default_angles[i]
            self.low_cmd.motor_cmd[i].kp = 25 
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.5
            self.low_cmd.motor_cmd[i].tau = 0.0
        
        if self.chirp_counter < self.num_chirp_steps:
            target_q = self.chirp_traj[self.chirp_counter]
                
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = target_q[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i] 
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0

            # Data Collection (Only during Chirp)
            if self.low_state is not None:
                current_q = np.array([self.low_state.motor_state[i].q for i in range(12)])
                # 記錄相對於 Chirp 開始的時間，方便對齊
                self.log_time.append(self.chirp_counter * self.dt) 
                self.log_dof_pos.append(current_q)
                self.log_des_dof_pos.append(target_q.copy())

            self.chirp_counter += 1
            if self.chirp_counter % 500 == 0:
                print(f"Chirp Progress: {self.chirp_counter}/{self.num_chirp_steps}")
        else:
            print("Chirp finished.")
            self.sit_down()
            
    
    def save_data(self):
        if len(self.log_time) == 0:
            return

        print("Saving data...")
        time_tensor = torch.tensor(self.log_time, dtype=torch.float32)
        dof_pos_tensor = torch.tensor(np.array(self.log_dof_pos), dtype=torch.float32)
        des_dof_pos_tensor = torch.tensor(np.array(self.log_des_dof_pos), dtype=torch.float32)

        save_dict = {
            "time": time_tensor,
            "dof_pos": dof_pos_tensor,
            "des_dof_pos": des_dof_pos_tensor
        }

        output_dir = pathlib.Path("data/big_red_dog")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "chirp_data.pt"
        
        torch.save(save_dict, file_path)
        print(f"Saved to {file_path}")

    def stand_up(self):
        self.mode = 'stand'
        self.reset_timer()

    def sit_down(self):
        self.mode = 'sit'
        self.reset_timer()
    
    def move_rl(self):
        self.mode = 'move'
        self.reset_timer()
        self.counter = 0
        self.action.fill(0.0)
        self.target_dof_pos = self.qpos.copy()
    
    def trigger_chirp(self):
        self.mode = 'chirp'
        self.reset_timer()
    
    
    def update_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq

        
        for i in range(3):
            self.ang_vel[i] = self.low_state.imu_state.gyroscope[i]

        for i in range(4):
            self.quat[i] = self.low_state.imu_state.quaternion[i]

    def get_current_state(self):
        return self.qpos, self.qvel, self.ang_vel, self.quat, self.lin_vel

    


    def LowCmdWrite(self):
        
        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'stand':
                self.stand()
            elif self.mode == 'sit':
                self.sit()
            elif self.mode == 'move':
                self.move()
            elif self.mode == 'chirp':
                self.chrip_proccess()
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
    def plot(self):
        # Plot the collected data after the simulation ends
        plt.figure(figsize=(18, 20))

        plt.subplot(4, 2, 1)
        for i in range(3): 
            plt.plot([step[i] for step in self.ang_vel_data_list], label=f"Angular Velocity {i}")
        plt.title(f"History Angular Velocity", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.subplot(4, 2, 2)
        for i in range(3):
            plt.plot([step[i] for step in self.gravity_b_list], label=f"Project Gravity {i}")
        plt.title(f"History Project Gravity", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.subplot(4, 2, 3)
        for i in range(NUM_MOTORS):
            plt.plot([step[i] for step in self.joint_pos_list], label=f"Joint Position {i}")
        plt.title(f"History Joint Position", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        
    def ResetParam(self):
        self.controller_rt = 0
        self.chirp_counter = 0
        self.is_running = False


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(1, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo") # default DDS port for pineapple

    controller = Controller()
    controller.Init()

    command_dict = {
        "stand": controller.stand_up,
        "sit": controller.sit_down,
        "move": controller.move_rl,
        "plot": controller.plot,
        "save": controller.save_data,
        "chirp": controller.trigger_chirp,
        "inspect": controller.inspect_policy,
    }

    while True:        
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.ShutDown()
                break

        except Exception as e:
            traceback.print_exc()
            break
    sys.exit(0)     
