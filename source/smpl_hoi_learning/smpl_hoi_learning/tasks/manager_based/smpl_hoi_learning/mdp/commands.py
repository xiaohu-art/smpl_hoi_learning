from omegaconf import MISSING
import torch
import numpy as np
import os
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm,CommandTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file, allow_pickle=True)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        
        print(self.joint_pos.shape,self.joint_vel.shape,self._body_pos_w.shape,self._body_quat_w.shape,self._body_lin_vel_w.shape,self._body_ang_vel_w.shape)

        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]


    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

class MotionCommand(CommandTerm):
    cfg: CommandTermCfg

    def __init__(self, cfg:CommandTermCfg,env:ManagerBasedRLEnv):
        super().__init__(cfg,env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_joint_indexes, robot_joint_names = self.robot.find_joints(self.cfg.joint_names, preserve_order=True)
        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
    
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        robot_all_joint_names = self.robot.joint_names
    
        # 打印前 5 个关节看看对不对得上
        print(f"数据预期的前5个关节: {self.cfg.joint_names[:5]}")
        print(f"机器人实际的前5个关节: {robot_all_joint_names[:5]}")

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[:, self.robot_joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[:, self.robot_joint_indexes]

    def _update_metrics(self):
        return 

    def _resample_command(self, env_ids: Sequence[int]):
        return 
    
    def _update_command(self):
        self.time_steps += 1
        env_ids_to_reset = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if len(env_ids_to_reset) > 0:
            self.time_steps[env_ids_to_reset]=0
        self.time_steps %= self.motion.time_step_total

        # set root state
        root_states = self.robot.data.default_root_state.clone()
        root_states[:, :3] = self.motion._body_pos_w[self.time_steps, 0]
        root_states[:, :2] += self._env.scene.env_origins[:, :2]
        root_states[:, 3:7] = self.motion._body_quat_w[self.time_steps, 0]
        root_states[:, 7:10] = self.motion._body_lin_vel_w[self.time_steps, 0]
        root_states[:, 10:] = self.motion._body_ang_vel_w[self.time_steps, 0]
        self.robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        joint_pos[:, self.robot_joint_indexes] = self.joint_pos[self.time_steps]
        joint_vel[:, self.robot_joint_indexes] = self.joint_vel[self.time_steps]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel,joint_ids=self.robot_joint_indexes)

        

SMPLH_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = "robot"
    body_names: Sequence[str] = (".*",)
    motion_file: str = "./output/output.npz"
    def __post_init__(self):
        AXES = ["x", "y", "z"]
        JOINT_ORDER = [n for n in SMPLH_BONE_ORDER_NAMES if n != "Pelvis"]
        self.joint_names = [f"{base}_{a}" for base in JOINT_ORDER for a in AXES]