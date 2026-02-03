# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# rewards.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.assets import Articulation, RigidObject  
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from SkillsBlender.source.skillsblender.skillsblender.tasks.path.mdp.commands.flat_path_command import PathCommand

import isaaclab.utils.math as math_utils
import isaaclab.utils.warp as warp_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



"""
General Rewards
"""

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.dones).float()

def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()



"""
Robot Rewards(Regularization) 
"""

#惩罚基座倾斜
def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def base_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for base acceleration"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    base_acc_lin = asset.data.body_acc_w[:, 0, :3]  # (num_envs, 3)
    base_acc_ang = asset.data.body_acc_w[:, 0, 3:]  # (num_envs, 3)
    reward = torch.square(torch.norm(base_acc_lin, dim=-1)) + 0.02 * torch.square(torch.norm(base_acc_ang, dim=-1))  # (num_envs,)
    return reward

# def base_height_l1(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     sensor_cfg: SceneEntityCfg | None = None,
# ) -> torch.Tensor:
#     """Penalize asset height from its target using L1 norm.

#     Note:
#         For flat terrain, target height is in the world frame. For rough terrain,
#         sensor readings can adjust the target height to account for the terrain.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     robot: Articulation = env.scene[asset_cfg.name]
#     target_height = robot.data.default_root_state[0, 2]
#     if sensor_cfg is not None:
#         sensor: RayCaster = env.scene[sensor_cfg.name]
#         # Adjust the target height using the sensor data
#         ray_hits = sensor.data.ray_hits_w[..., 2]
#         if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
#             adjusted_target_height = asset.data.root_pos_w[:, 2]  # fallback to current height if sensor data is invalid
#         else:
#             adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
#     else:
#         # Use the provided target height directly for flat terrain
#         adjusted_target_height = target_height
#     # Compute the L1 squared penalty
#     reward = torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)
#     return reward

def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel."""
    #Unitree Go2 默认基座高度为0.27m
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)

# def base_height_l2(
#     env: ManagerBasedRLEnv,
#     target_height: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     sensor_cfg: SceneEntityCfg | None = None,
# ) -> torch.Tensor:
#     """Penalize asset height from its target using L2 squared kernel.

#     Note:
#         For flat terrain, target height is in the world frame. For rough terrain,
#         sensor readings can adjust the target height to account for the terrain.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     if sensor_cfg is not None:
#         sensor: RayCaster = env.scene[sensor_cfg.name]
#         # Adjust the target height using the sensor data
#         adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
#     else:
#         # Use the provided target height directly for flat terrain
#         adjusted_target_height = target_height
#     # Compute the L2 squared penalty
#     return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)

#配置里面没有高度传感器，奖励函数里面有没事吧？？？？？



"""
Action penalties.
"""

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)



"""
Joint penalties.
"""

def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


"""
Contact sensor.
"""

def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)

def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact



"""
Gait Quality Rewards
"""
def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for high feet acceleration"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]  # (num_envs, num_feet, 3)
    penalty = torch.norm(feet_acc, dim=-1)  # (num_envs, num_feet)
    reward = torch.sum(torch.square(penalty), dim=-1)  # (num_envs,)
    return reward

def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for variance in feet air time"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # (num_envs, num_feet)
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]  # (num_envs, num_feet)
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)  # (num_envs,)
    
def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
    
def feet_air_time_1(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float,
    dis_threshold: float = 0.25, heading_threshold: float = 0.5
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    command = env.command_manager.get_term(command_name)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)
    heading_error = torch.abs(command.target_heading_b)  # (num_envs,)
    condition = (distance < dis_threshold) & (heading_error < heading_threshold)
    reward = torch.where(condition, 0.0, reward)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7 #(num_envs,)
    return reward

class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        # synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        # synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        body_ids_0, _ = self.contact_sensor.find_bodies(synced_feet_pair_names[0], preserve_order=True)
        body_ids_1, _ = self.contact_sensor.find_bodies(synced_feet_pair_names[1], preserve_order=True)
        if len(body_ids_0) != 2 or len(body_ids_1) != 2:
            raise ValueError(
                "Each synced feet pair must resolve to exactly two bodies. "
                f"Got: {body_ids_0} and {body_ids_1}."
            )
        synced_feet_pair_0 = [int(body_ids_0[0]), int(body_ids_0[1])]
        synced_feet_pair_1 = [int(body_ids_1[0]), int(body_ids_1[1])]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]
# class GaitReward(ManagerTermBase):
#     """Gait enforcing reward term for quadrupeds."""

#     def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)

#         # store params
#         self.std: float = cfg.params["std"]
#         self.command_name: str = cfg.params["command_name"]
#         self.max_err: float = cfg.params["max_err"]
#         self.velocity_threshold: float = cfg.params["velocity_threshold"]
#         self.command_threshold: float = cfg.params["command_threshold"]

#         # sensors / assets
#         self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
#         self.asset_name: str = cfg.params["asset_cfg"].name
#         self.asset: Articulation = env.scene[self.asset_name]

#         # --- resolve synced feet pairs ---
#         synced_feet_pair_names = cfg.params["synced_feet_pair_names"]

#         # validate structure: [[a,b],[c,d]]
#         if (
#             not isinstance(synced_feet_pair_names, (list, tuple))
#             or len(synced_feet_pair_names) != 2
#             or any((not isinstance(p, (list, tuple)) or len(p) != 2) for p in synced_feet_pair_names)
#         ):
#             raise ValueError(
#                 "GaitReward expects cfg.params['synced_feet_pair_names'] to be like "
#                 "[['FL_foot','FR_foot'], ['RL_foot','RR_foot']] (two pairs, each pair has 2 feet). "
#                 f"Got: {synced_feet_pair_names}"
#             )

#         # normalize names (LF/RF/LH/RH -> FL/FR/RL/RR) if needed
#         pair0_names = [self._normalize_foot_name(n) for n in synced_feet_pair_names[0]]
#         pair1_names = [self._normalize_foot_name(n) for n in synced_feet_pair_names[1]]

#         # find body ids for each pair (must resolve to exactly 2 ids)
#         pair0_ids = self._resolve_two_feet_ids(pair0_names)
#         pair1_ids = self._resolve_two_feet_ids(pair1_names)

#         # store as [[id,id],[id,id]]
#         self.synced_feet_pairs = [pair0_ids, pair1_ids]



    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    # def _normalize_foot_name(self, name: str) -> str:
    #     """Map common alias conventions to the asset's naming."""
    #     if not isinstance(name, str):
    #         return name

    #     # common quadruped alias -> Unitree-style
    #     mapping = {
    #         "LF_": "FL_",
    #         "RF_": "FR_",
    #         "LH_": "RL_",
    #         "RH_": "RR_",
    #     }
    #     for k, v in mapping.items():
    #         if name.startswith(k):
    #             return v + name[len(k):]

    #     return name

    # def _resolve_two_feet_ids(self, foot_names: list[str]) -> list[int]:
    #     """Resolve exactly two foot body ids from names/regex, with clear error messages."""
    #     try:
    #         # preserve_order=True keeps [left, right] order as provided
    #         ids = self.contact_sensor.find_bodies(foot_names, preserve_order=True)
    #     except Exception as e:
    #         available = getattr(self.contact_sensor, "body_names", None)
    #         raise ValueError(
    #             f"Failed to resolve feet names {foot_names} in ContactSensor. "
    #             f"Available bodies: {available}"
    #         ) from e

    #     if len(ids) != 2:
    #         available = getattr(self.contact_sensor, "body_names", None)
    #         raise ValueError(
    #             f"Feet name list {foot_names} should resolve to exactly 2 bodies, but got {len(ids)}: {ids}. "
    #             f"Available bodies: {available}"
    #         )
    #     return ids
    
    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    dis_threshold: float = 0.25,
    heading_threshold: float = 0.5,
    jump_only: bool = False,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    
    # Calculate height error: only penalize if foot is LOWER than target height
    # error = max(0, target_height - current_height)
    foot_z_target_error = torch.clamp(target_height - footpos_in_body_frame[:, :, 2], min=0.0) # (num_envs, num_feet)
    
    # Identify swing phase: foot has significant horizontal velocity
    is_swing = torch.norm(footvel_in_body_frame[:, :, :2], dim=2) > 0.1 # (num_envs, num_feet)

    # We sum over all feet
    reward = torch.sum(foot_z_target_error * is_swing, dim=1) # (num_envs,)
    
    command: PathCommand = env.command_manager.get_term(command_name)
    if jump_only and hasattr(command, "is_in_jump_phase"):
        reward = torch.where(command.is_in_jump_phase, reward, torch.zeros_like(reward))
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)
    heading_error = torch.abs(command.target_heading_b)  # (num_envs,)
    condition = (distance < dis_threshold) & (heading_error < heading_threshold)
    reward = torch.where(condition, 0.0, reward)
    
    # Scale with gravity projection (optional, but good for stability)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

    
# def feet_stride_width_penalty(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     target_width: float = 0.3,
#     tolerance: float = 0.05
# ) -> torch.Tensor:
#     """
#     Penalize deviation from target stride width between left and right feet.
#     Encourages the robot to maintain a consistent lateral distance between feet.
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     body_pos_w = asset.data.body_pos_w
#     # Resolve body_ids if not already resolved
#     if sensor_cfg.body_ids is None:
#         sensor_cfg.resolve(env.scene)
    
#     if sensor_cfg.body_ids is None or len(sensor_cfg.body_ids) < 4:
#         return torch.zeros(env.num_envs, device=env.device)

#     # 获取足部位置（假设 body_ids 按照 FL, FR, RL, RR 顺序）
#     feet_pos_w = body_pos_w[:, sensor_cfg.body_ids, :]
    
#     # 前腿横向距离 (Y轴): FR - FL
#     front_width = torch.abs(feet_pos_w[:, 1, 1] - feet_pos_w[:, 0, 1])
#     # 后腿横向距离 (Y轴): RR - RL
#     rear_width = torch.abs(feet_pos_w[:, 3, 1] - feet_pos_w[:, 2, 1])

#     # 计算偏离目标宽度的误差
#     front_error = torch.abs(front_width - target_width)
#     rear_error = torch.abs(rear_width - target_width)

#     # 只有当偏差超过容忍度时才惩罚
#     front_penalty = torch.clamp(front_error - tolerance, min=0.0)
#     rear_penalty = torch.clamp(rear_error - tolerance, min=0.0)

#     return torch.square(front_penalty) + torch.square(rear_penalty)




"""
Task Rewards 
"""

def track_velocity_along_path_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "path_tracking",
    desired_speed: float = 1.0,
) -> torch.Tensor:
    """
    Reward for matching the commanded travel speed along the path direction.

    The reward uses the vector from the robot base to the next waypoint in world coordinates
    to define the forward direction and compares the base velocity projected onto this axis
    against the desired walking speed.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command: PathCommand = env.command_manager.get_term(command_name)

    # Resolve the per-environment target waypoint in world coordinates.
    env_ids = torch.arange(env.num_envs, device=env.device)
    next_indices = torch.clamp(command.current_waypoints_index, max=command.num_waypoints - 1)
    target_pos_w = command.pos_path_w[env_ids, next_indices]

    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_vel_w = asset.data.root_lin_vel_w[:, :3]

    direction_w = target_pos_w - robot_pos_w
    direction_norm = torch.norm(direction_w, dim=-1)
    valid_mask = direction_norm > 1e-6

    # Avoid division by zero when the waypoint coincides with the robot base.
    direction_unit = torch.zeros_like(direction_w)
    direction_unit[valid_mask] = direction_w[valid_mask] / direction_norm[valid_mask].unsqueeze(-1)

    projected_speed = torch.sum(robot_vel_w * direction_unit, dim=-1)
    reward = torch.exp(-torch.square(projected_speed - desired_speed) / (std ** 2))

    # Zero out the reward for degenerate directions to keep the output 1-D.
    reward = torch.where(valid_mask, reward, torch.zeros_like(reward))
    return reward

def stalling_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the stalling penalty based on the robot's velocity.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.

    Returns:
        torch.Tensor: The computed penalty tensor of shape (num_envs,).
    """
    command: PathCommand = env.command_manager.get_term(command_name)
    speed = torch.norm(command.robot_velocity_w, dim=-1)  # (num_envs,)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = (speed < 0.2) & (distance > 0.3)
    
    # Calculate reward using torch.where for vectorized operation
    reward = torch.where(condition, 1.0, 0.0)
    return reward

def track_path_pos_xy_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """Reward for position tracking in XY plane (exponential kernel)."""
    command = env.command_manager.get_term(command_name)
    error_sq = torch.square(command.metrics["error_pos_xy"])
    return torch.exp(-error_sq / std**2)

def track_path_heading_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """Reward for heading tracking (exponential kernel)."""
    command = env.command_manager.get_term(command_name)
    error_sq = torch.square(command.metrics["error_heading"])
    return torch.exp(-error_sq / std**2)

# def exploration_reward(env: ManagerBasedRLEnv, command_name: str, Tr: float = 1.0) -> torch.Tensor:
#     """Compute the exploration reward based on the orientation of the robot.

#     Args:
#         env (ManagerBasedRLEnv): The environment instance.
#         command_name (str): The name of the command to retrieve target positions.
#         Tr (float): The time window before the end of the episode to start rewarding.

#     Returns:
#         torch.Tensor: The computed reward tensor of shape (num_envs,).
#     """
#     command: PathCommand = env.command_manager.get_term(command_name)
#     robot_vel = command.robot_velocity_w[:, :3]  # (num_envs, 3)
#     target_vec = command.target_pos_w - command.robot_pos_w # (num_envs, 3)
#     distance = torch.norm(target_vec, dim=-1)  # (num_envs,)

#     # Condition for when to apply the reward
#     condition = (env.episode_length_buf * env.step_dt >= env.max_episode_length_s - Tr) & (distance <= 1.0) & (torch.norm(robot_vel, dim=-1) < 0.2)
    
#     # Calculate cosine similarity
#     # Dot product for the numerator
#     dot_product = torch.sum(robot_vel * target_vec, dim=-1)
#     # Norms for the denominator
#     robot_vel_norm = torch.norm(robot_vel, dim=-1)
#     target_vec_norm = torch.norm(target_vec, dim=-1)
    
#     # Calculate reward using torch.where for vectorized operation
#     cosine_sim = dot_product / (robot_vel_norm * target_vec_norm + 1e-8)
    
#     reward = torch.where(condition, 0.0, cosine_sim)
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # scale with gravity projection
#     return reward

#flat需要这个奖励函数吗
# def track_path_height_exp(
#     env: ManagerBasedRLEnv, 
#     std: float, 
#     command_name: str = "path_tracking"
# ) -> torch.Tensor:
#     """高度保持 (指数核)"""
#     command = env.command_manager.get_term(command_name)
#     error_sq = torch.square(command.metrics["error_pos_z"])
#     return torch.exp(-error_sq / std**2)



# ==============================================================================
# Goal-Related Rewards
# ==============================================================================

# def near_goal_velocity_penalty(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     distance_threshold: float,
#     max_velocity: float
# ) -> torch.Tensor:
#     """
#     接近终点时惩罚高速度，鼓励机器人减速。
#     """
#     command = env.command_manager.get_term(command_name)
#     target_pos_w = command.command[:, :3]
#     robot_pos_w = env.scene["robot"].data.root_pos_w[:, :3]

#     # 计算到目标点的距离
#     distance = torch.norm(target_pos_w - robot_pos_w, dim=-1)

#     # 获取机器人速度
#     velocity = torch.norm(env.scene["robot"].data.root_lin_vel_w[:, :3], dim=-1)

#     # 只在接近目标时应用惩罚
#     near_goal = distance < distance_threshold
#     velocity_excess = torch.clamp(velocity - max_velocity, min=0.0)

#     penalty = near_goal.float() * torch.square(velocity_excess)
#     return penalty

# def goal_reached_stability_reward(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     distance_threshold: float,
#     velocity_threshold: float
# ) -> torch.Tensor:
#     """
#     到达终点后奖励稳定站立（低速度 + 低姿态变化）。
#     """
#     command = env.command_manager.get_term(command_name)
#     target_pos_w = command.command[:, :3]
#     robot_pos_w = env.scene["robot"].data.root_pos_w[:, :3]

#     # 计算到目标点的距离
#     distance = torch.norm(target_pos_w - robot_pos_w, dim=-1)

#     # 获取机器人状态
#     velocity = torch.norm(env.scene["robot"].data.root_lin_vel_w[:, :3], dim=-1)
#     ang_vel = torch.norm(env.scene["robot"].data.root_ang_vel_w, dim=-1)

#     # 判断是否到达目标
#     at_goal = distance < distance_threshold

#     # 判断是否稳定（速度低且角速度低）
#     is_stable = (velocity < velocity_threshold) & (ang_vel < 0.5)

#     # 只有在目标点且稳定时才给予奖励
#     reward = (at_goal & is_stable).float()
#     return reward



# def gait_symmetry_reward(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """
#     奖励左右脚对称运动，鼓励对角步态模式。
#     通过比较左右侧关节的位置和速度对称性。
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     joint_pos = asset.data.joint_pos
#     joint_vel = asset.data.joint_vel

#     # 假设关节顺序为: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#     #                  RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
#     # 对于 GO2，每条腿3个关节
#     num_joints_per_leg = 3
#     if joint_pos.shape[1] >= 12:
#         # 左前腿 vs 右前腿
#         fl_pos = joint_pos[:, 0:num_joints_per_leg]
#         fr_pos = joint_pos[:, num_joints_per_leg:2*num_joints_per_leg]
#         # 左后腿 vs 右后腿
#         rl_pos = joint_pos[:, 2*num_joints_per_leg:3*num_joints_per_leg]
#         rr_pos = joint_pos[:, 3*num_joints_per_leg:4*num_joints_per_leg]

#         # 对角线对称：FL vs RR, FR vs RL (对角步态)
#         diagonal_pos_error = torch.sum(torch.square(fl_pos - rr_pos), dim=1) + \
#                             torch.sum(torch.square(fr_pos - rl_pos), dim=1)

#         # 速度对称性
#         fl_vel = joint_vel[:, 0:num_joints_per_leg]
#         fr_vel = joint_vel[:, num_joints_per_leg:2*num_joints_per_leg]
#         rl_vel = joint_vel[:, 2*num_joints_per_leg:3*num_joints_per_leg]
#         rr_vel = joint_vel[:, 3*num_joints_per_leg:4*num_joints_per_leg]

#         diagonal_vel_error = torch.sum(torch.square(fl_vel - rr_vel), dim=1) + \
#                             torch.sum(torch.square(fr_vel - rl_vel), dim=1)

#         # 使用指数核奖励对称性
#         symmetry_score = torch.exp(-(diagonal_pos_error + diagonal_vel_error) / 0.5)
#         return symmetry_score
#     else:
#         return torch.zeros(env.num_envs, device=env.device)