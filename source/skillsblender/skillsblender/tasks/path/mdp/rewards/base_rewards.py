from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

from skillsblender.tasks.path.mdp.commands.base_path_command import SegmentPathCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gravity_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Common stability gate used by gait-related rewards/penalties."""
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def skill_phase_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    skill_name: str,
) -> torch.Tensor:
    """Return a bool mask for the environments currently inside a given skill phase."""
    command = env.command_manager.get_term(command_name)
    if hasattr(command, "is_in_skill_phase"):
        mask = command.is_in_skill_phase(skill_name)
        if isinstance(mask, torch.Tensor):
            return mask.bool()
    prop_name = f"is_in_{skill_name}_phase"
    if hasattr(command, prop_name):
        mask = getattr(command, prop_name)
        if isinstance(mask, torch.Tensor):
            return mask.bool()
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


# ==============================================================================
# 1) Task Tracking Rewards
# ==============================================================================

def track_velocity_along_path_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "path_tracking",
    desired_speed: float = 1.0,
) -> torch.Tensor:
    """Reward matching travel speed along the path direction."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command: SegmentPathCommand = env.command_manager.get_term(command_name)

    env_ids = torch.arange(env.num_envs, device=env.device)
    next_indices = torch.clamp(command.current_waypoints_index, max=command.num_waypoints - 1)
    target_pos_w = command.pos_path_w[env_ids, next_indices]

    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_vel_w = asset.data.root_lin_vel_w[:, :3]

    direction_w = target_pos_w - robot_pos_w
    direction_norm = torch.norm(direction_w, dim=-1)
    valid_mask = direction_norm > 1e-6

    direction_unit = torch.zeros_like(direction_w)
    direction_unit[valid_mask] = direction_w[valid_mask] / direction_norm[valid_mask].unsqueeze(-1)

    projected_speed = torch.sum(robot_vel_w * direction_unit, dim=-1)
    reward = torch.exp(-torch.square(projected_speed - desired_speed) / (std**2))
    return torch.where(valid_mask, reward, torch.zeros_like(reward))


def track_path_pos_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "path_tracking",
) -> torch.Tensor:
    """Reward XY path tracking using an exponential kernel."""
    command = env.command_manager.get_term(command_name)
    error_sq = torch.square(command.metrics["error_pos_xy"])
    return torch.exp(-error_sq / std**2)

def track_path_pos_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    command_name: str = "path_tracking",
) -> torch.Tensor:
    """Track virtual Z target from command waypoints with an exponential kernel."""
    command = env.command_manager.get_term(command_name)
    err = command.metrics.get("error_pos_z", torch.zeros(env.num_envs, device=env.device))
    return torch.exp(-torch.square(err) / (std**2))

def track_path_heading_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "path_tracking",
) -> torch.Tensor:
    """Reward heading tracking using an exponential kernel."""
    command = env.command_manager.get_term(command_name)
    error_sq = torch.square(command.metrics["error_heading"])
    return torch.exp(-error_sq / std**2)


def stalling_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalty when the robot is far from target but nearly stalled."""
    command: SegmentPathCommand = env.command_manager.get_term(command_name)
    speed = torch.norm(command.robot_velocity_w, dim=-1)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)
    return torch.where((speed < 0.2) & (distance > 0.3), 1.0, 0.0)


# ==============================================================================
# 2) Survival and Base Regularization
# ==============================================================================

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.dones).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that are not episodic timeouts."""
    return env.termination_manager.terminated.float()


def flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat base orientation (projected gravity XY components)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize z-axis base linear velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def base_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for base acceleration."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_acc_lin = asset.data.body_acc_w[:, 0, :3]
    base_acc_ang = asset.data.body_acc_w[:, 0, 3:]
    return torch.square(torch.norm(base_acc_lin, dim=-1)) + 0.02 * torch.square(torch.norm(base_acc_ang, dim=-1))


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from target base height."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize rate of action change."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize articulation joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize articulation joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
) -> torch.Tensor:
    """Penalize left-right joint asymmetry."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= _gravity_gate(env)
    return reward


# ==============================================================================
# 3) Contact and Gait Rewards
# ==============================================================================

def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contacts that exceed the threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def desired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize when none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    return (~contacts).all(dim=1).float()


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding while in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def feet_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for large feet acceleration."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]
    penalty = torch.norm(feet_acc, dim=-1)
    return torch.sum(torch.square(penalty), dim=-1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for variance in feet air/contact time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= _gravity_gate(env)
    return reward


def feet_contact_balance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    left_feet: tuple[str, str] = ("FL_foot", "RL_foot"),
    right_feet: tuple[str, str] = ("FR_foot", "RR_foot"),
    front_feet: tuple[str, str] = ("FL_foot", "FR_foot"),
    rear_feet: tuple[str, str] = ("RL_foot", "RR_foot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for long-term contact imbalance across left/right and front/rear."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cache_key = (sensor_cfg.name, tuple(left_feet), tuple(right_feet), tuple(front_feet), tuple(rear_feet))
    cache_name = "_feet_contact_balance_ids_cache"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, {})
    cache: dict = getattr(env, cache_name)

    if cache_key not in cache:
        left_ids, _ = contact_sensor.find_bodies(list(left_feet), preserve_order=True)
        right_ids, _ = contact_sensor.find_bodies(list(right_feet), preserve_order=True)
        front_ids, _ = contact_sensor.find_bodies(list(front_feet), preserve_order=True)
        rear_ids, _ = contact_sensor.find_bodies(list(rear_feet), preserve_order=True)
        cache[cache_key] = (
            [int(i) for i in left_ids],
            [int(i) for i in right_ids],
            [int(i) for i in front_ids],
            [int(i) for i in rear_ids],
        )

    left_ids, right_ids, front_ids, rear_ids = cache[cache_key]
    if min(len(left_ids), len(right_ids), len(front_ids), len(rear_ids)) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    contacts = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    contacts_f = contacts.float()

    left_ratio = contacts_f[:, left_ids].mean(dim=1)
    right_ratio = contacts_f[:, right_ids].mean(dim=1)
    front_ratio = contacts_f[:, front_ids].mean(dim=1)
    rear_ratio = contacts_f[:, rear_ids].mean(dim=1)

    imbalance_lr = torch.square(left_ratio - right_ratio)
    imbalance_fb = torch.square(front_ratio - rear_ratio)
    reward = 0.5 * (imbalance_lr + imbalance_fb)
    reward *= _gravity_gate(env)
    return reward


def feet_air_time_1(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    dis_threshold: float = 0.25,
    heading_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward long foot swing time on first contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    command = env.command_manager.get_term(command_name)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)
    heading_error = torch.abs(command.target_heading_b)
    reward = torch.where((distance < dis_threshold) & (heading_error < heading_threshold), 0.0, reward)
    reward *= _gravity_gate(env)
    return reward


class GaitReward(ManagerTermBase):
    """Trotting-style gait reward for quadrupeds with two synchronized pairs."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports two synchronized feet pairs.")

        body_ids_0, _ = self.contact_sensor.find_bodies(synced_feet_pair_names[0], preserve_order=True)
        body_ids_1, _ = self.contact_sensor.find_bodies(synced_feet_pair_names[1], preserve_order=True)
        if len(body_ids_0) != 2 or len(body_ids_1) != 2:
            raise ValueError(f"Each pair must resolve to exactly two bodies. Got: {body_ids_0} and {body_ids_1}.")

        self.synced_feet_pairs = [[int(body_ids_0[0]), int(body_ids_0[1])], [int(body_ids_1[0]), int(body_ids_1[1])]]

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
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1

        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3

        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= _gravity_gate(env)
        return reward

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
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
    """Foot swing-height shaping reward in body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)

    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])

    foot_z_target_error = torch.clamp(target_height - footpos_in_body_frame[:, :, 2], min=0.0)
    is_swing = torch.norm(footvel_in_body_frame[:, :, :2], dim=2) > 0.1
    reward = torch.sum(foot_z_target_error * is_swing, dim=1)

    command: SegmentPathCommand = env.command_manager.get_term(command_name)
    if jump_only and hasattr(command, "is_in_jump_phase"):
        reward = torch.where(command.is_in_jump_phase, reward, torch.zeros_like(reward))

    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)
    heading_error = torch.abs(command.target_heading_b)
    reward = torch.where((distance < dis_threshold) & (heading_error < heading_threshold), 0.0, reward)
    reward *= _gravity_gate(env)
    return reward
