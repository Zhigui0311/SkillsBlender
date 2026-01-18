# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# rewards.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation  # [新增] 导入 Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ==============================================================================
# Task Rewards 
# ==============================================================================

def track_velocity_along_path_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """
    奖励沿着路径方向的速度 (核心动力来源)。
    鼓励机器人不仅要离路径近，还要顺着路径跑。
    """
    command = env.command_manager.get_term(command_name)
    # 1. 获取目标点方向向量 (世界坐标)
    target_pos_w = command.command[:, :3]
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :3]
    target_vec_w = target_pos_w - robot_pos_w
    target_dir_w = target_vec_w / (torch.norm(target_vec_w, dim=-1, keepdim=True) + 1e-6)
    # 2. 获取机器人当前线速度 (世界坐标)
    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :3]

    # 3. 计算投影速度：实际速度在目标方向上的投影
    projection_vel = torch.sum(vel_w * target_dir_w, dim=-1)
    
    # 4. 指数奖励：鼓励投影速度接近某个理想值（比如 1.0 m/s），或者直接取正值
    return torch.exp(-torch.square(projection_vel - 1.0) / std**2)


def track_path_pos_xy_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """XY 平面位置追踪 (指数核)"""
    command = env.command_manager.get_term(command_name)
    error_sq = torch.square(command.metrics["error_pos_xy"])
    return torch.exp(-error_sq / std**2)

def track_path_heading_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """航向追踪 (指数核)"""
    command = env.command_manager.get_term(command_name)
    error_sq = torch.square(command.metrics["error_heading"])
    return torch.exp(-error_sq / std**2)

#flat需要这个奖励函数吗
def track_path_height_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """高度保持 (指数核)"""
    command = env.command_manager.get_term(command_name)
    error_sq = torch.square(command.metrics["error_pos_z"])
    return torch.exp(-error_sq / std**2)

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活奖励"""
    return (~env.termination_manager.dones).float()

# ==============================================================================
# Regularization (正则化 - 防止四肢乱动)
# ==============================================================================

#flat的奖励函数
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚垂直速度"""
    # [修改] 显式标注 asset 类型为 Articulation
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚躯干晃动"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚躯干倾斜"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """惩罚动作变化率"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚力矩消耗"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)

def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚关节偏离默认位置"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

def undesired_contacts(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """惩罚非脚部接触"""
    # 这里使用的是 Sensor，不是 Articulation
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(torch.norm(net_contact_forces, dim=-1) > threshold, dim=1).float()

# ---新加---    
def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float
) -> torch.Tensor:
    """奖励合理的足部腾空时间"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # 获取首次接触时间
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # 奖励腾空时间在合理范围内
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return torch.clamp(reward, min=0.0)

def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚机器人高度偏离目标高度"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)

def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚关节加速度，鼓励平滑运动"""
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算关节加速度 = (当前速度 - 上一步速度) / dt
    # 注意：这里假设 last_joint_vel 存在，如果不存在需要在环境中维护
    # 为了简单起见，我们使用 joint_vel 的变化作为加速度的近似
    if not hasattr(env, '_last_joint_vel'):
        env._last_joint_vel = asset.data.joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    joint_acc = (asset.data.joint_vel - env._last_joint_vel) / env.step_dt
    env._last_joint_vel = asset.data.joint_vel.clone()
    return torch.sum(torch.square(joint_acc), dim=1)

# ==============================================================================
# Gait Quality Rewards
# ==============================================================================


def feet_stride_width_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_width: float = 0.3,
    tolerance: float = 0.05
) -> torch.Tensor:
    """
    惩罚足部横向距离过大或过小。
    适用于四足机器人，约束步宽在合理范围内。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 从 Articulation 获取刚体位置
    # body_pos_w 的形状为 (num_envs, num_bodies, 3)
    body_pos_w = asset.data.body_pos_w
    
    # Resolve body_ids if not already resolved
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)
    
    if sensor_cfg.body_ids is None or len(sensor_cfg.body_ids) < 4:
        return torch.zeros(env.num_envs, device=env.device)

    # 获取足部位置（假设 body_ids 按照 FL, FR, RL, RR 顺序）
    feet_pos_w = body_pos_w[:, sensor_cfg.body_ids, :]
    
    # 前腿横向距离 (Y轴): FR - FL
    front_width = torch.abs(feet_pos_w[:, 1, 1] - feet_pos_w[:, 0, 1])
    # 后腿横向距离 (Y轴): RR - RL
    rear_width = torch.abs(feet_pos_w[:, 3, 1] - feet_pos_w[:, 2, 1])

    # 计算偏离目标宽度的误差
    front_error = torch.abs(front_width - target_width)
    rear_error = torch.abs(rear_width - target_width)

    # 只有当偏差超过容忍度时才惩罚
    front_penalty = torch.clamp(front_error - tolerance, min=0.0)
    rear_penalty = torch.clamp(rear_error - tolerance, min=0.0)

    return torch.square(front_penalty) + torch.square(rear_penalty)


def gait_symmetry_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    奖励左右脚对称运动，鼓励对角步态模式。
    通过比较左右侧关节的位置和速度对称性。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel

    # 假设关节顺序为: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    #                  RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
    # 对于 GO2，每条腿3个关节
    num_joints_per_leg = 3
    if joint_pos.shape[1] >= 12:
        # 左前腿 vs 右前腿
        fl_pos = joint_pos[:, 0:num_joints_per_leg]
        fr_pos = joint_pos[:, num_joints_per_leg:2*num_joints_per_leg]
        # 左后腿 vs 右后腿
        rl_pos = joint_pos[:, 2*num_joints_per_leg:3*num_joints_per_leg]
        rr_pos = joint_pos[:, 3*num_joints_per_leg:4*num_joints_per_leg]

        # 对角线对称：FL vs RR, FR vs RL (对角步态)
        diagonal_pos_error = torch.sum(torch.square(fl_pos - rr_pos), dim=1) + \
                            torch.sum(torch.square(fr_pos - rl_pos), dim=1)

        # 速度对称性
        fl_vel = joint_vel[:, 0:num_joints_per_leg]
        fr_vel = joint_vel[:, num_joints_per_leg:2*num_joints_per_leg]
        rl_vel = joint_vel[:, 2*num_joints_per_leg:3*num_joints_per_leg]
        rr_vel = joint_vel[:, 3*num_joints_per_leg:4*num_joints_per_leg]

        diagonal_vel_error = torch.sum(torch.square(fl_vel - rr_vel), dim=1) + \
                            torch.sum(torch.square(fr_vel - rl_vel), dim=1)

        # 使用指数核奖励对称性
        symmetry_score = torch.exp(-(diagonal_pos_error + diagonal_vel_error) / 0.5)
        return symmetry_score
    else:
        return torch.zeros(env.num_envs, device=env.device)

# ==============================================================================
# Goal-Related Rewards
# ==============================================================================

def near_goal_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float,
    max_velocity: float
) -> torch.Tensor:
    """
    接近终点时惩罚高速度，鼓励机器人减速。
    """
    command = env.command_manager.get_term(command_name)
    target_pos_w = command.command[:, :3]
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :3]

    # 计算到目标点的距离
    distance = torch.norm(target_pos_w - robot_pos_w, dim=-1)

    # 获取机器人速度
    velocity = torch.norm(env.scene["robot"].data.root_lin_vel_w[:, :3], dim=-1)

    # 只在接近目标时应用惩罚
    near_goal = distance < distance_threshold
    velocity_excess = torch.clamp(velocity - max_velocity, min=0.0)

    penalty = near_goal.float() * torch.square(velocity_excess)
    return penalty

def goal_reached_stability_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float,
    velocity_threshold: float
) -> torch.Tensor:
    """
    到达终点后奖励稳定站立（低速度 + 低姿态变化）。
    """
    command = env.command_manager.get_term(command_name)
    target_pos_w = command.command[:, :3]
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :3]

    # 计算到目标点的距离
    distance = torch.norm(target_pos_w - robot_pos_w, dim=-1)

    # 获取机器人状态
    velocity = torch.norm(env.scene["robot"].data.root_lin_vel_w[:, :3], dim=-1)
    ang_vel = torch.norm(env.scene["robot"].data.root_ang_vel_w, dim=-1)

    # 判断是否到达目标
    at_goal = distance < distance_threshold

    # 判断是否稳定（速度低且角速度低）
    is_stable = (velocity < velocity_threshold) & (ang_vel < 0.5)

    # 只有在目标点且稳定时才给予奖励
    reward = (at_goal & is_stable).float()

    return reward

# ==============================================================================
# Jump-Specific Rewards (跳跃专用奖励函数)
# ==============================================================================

def jump_height_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_tolerance: float = 0.15
) -> torch.Tensor:
    """
    跳跃高度追踪奖励。
    当机器人处于跳跃阶段时，奖励机器人跟随目标抛物线轨迹。
    """
    command = env.command_manager.get_term(command_name)

    # 获取目标高度和当前高度
    target_pos_w = command.command[:, 2::4]  # 提取所有航点的Z坐标
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, 2]  # 当前Z坐标

    # 使用最近航点的高度作为目标
    target_height = target_pos_w[:, 0]

    # 计算高度误差
    height_error = torch.abs(robot_pos_w - target_height)

    # 使用指数核奖励高度跟踪
    reward = torch.exp(-torch.square(height_error) / (height_tolerance ** 2))

    return reward

def jump_clearance_reward(
    env: ManagerBasedRLEnv,
    min_clearance: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跳跃离地高度奖励。
    奖励机器人在跳跃时离地面足够高，避免碰到沟壑边缘。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取机器人base高度
    base_height = asset.data.root_pos_w[:, 2]

    # 获取机器人下方地面高度（通过地形查询）
    robot_xy = asset.data.root_pos_w[:, :2]
    ground_height = env.scene.terrain.terrain_generator.height_field_raw.sample(robot_xy)

    # 计算离地间隙
    clearance = base_height - ground_height

    # 只有当离地间隙超过最小值时才给予奖励
    reward = torch.clamp((clearance - min_clearance) / min_clearance, min=0.0, max=1.0)

    return reward

def jump_forward_velocity(
    env: ManagerBasedRLEnv,
    target_velocity: float = 1.5,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跳跃前向速度奖励。
    鼓励机器人在跳跃时保持足够的前向速度。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取机器人在body坐标系下的线速度
    vel_b = asset.data.root_lin_vel_b[:, :3]

    # 前向速度 (X轴)
    forward_vel = vel_b[:, 0]

    # 使用指数核奖励接近目标速度
    reward = torch.exp(-torch.square(forward_vel - target_velocity) / (std ** 2))

    return reward

def jump_landing_stability(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    landing_time_threshold: float = 0.2,
    velocity_threshold: float = 0.5
) -> torch.Tensor:
    """
    跳跃落地稳定性奖励。
    奖励机器人在落地后快速稳定，减少晃动。
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Resolve body_ids if needed
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    # 检测是否所有脚都着地
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    all_feet_contact = torch.all(torch.norm(contact_forces, dim=-1) > 1.0, dim=1)

    # 检测着地时间
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    recently_landed = torch.all(last_air_time < landing_time_threshold, dim=1)

    # 检测速度是否稳定
    robot = env.scene["robot"]
    velocity = torch.norm(robot.data.root_lin_vel_w[:, :3], dim=-1)
    is_stable = velocity < velocity_threshold

    # 只有满足所有条件时才给予奖励
    reward = (all_feet_contact & recently_landed & is_stable).float()

    return reward

def jump_pitch_control(
    env: ManagerBasedRLEnv,
    target_pitch_range: tuple[float, float] = (-0.2, 0.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跳跃俯仰角控制奖励。
    鼓励机器人在跳跃过程中保持合理的俯仰角，避免前倾或后仰过度。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 从四元数中提取俯仰角
    quat_w = asset.data.root_quat_w  # (N, 4) wxyz格式
    # 计算俯仰角 (pitch)
    # pitch = arcsin(2*(w*y - z*x))
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))

    # 检查俯仰角是否在合理范围内
    in_range = (pitch > target_pitch_range[0]) & (pitch < target_pitch_range[1])

    # 计算偏离程度
    lower_error = torch.clamp(target_pitch_range[0] - pitch, min=0.0)
    upper_error = torch.clamp(pitch - target_pitch_range[1], min=0.0)
    total_error = lower_error + upper_error

    # 使用指数核惩罚偏离
    reward = torch.exp(-total_error / 0.2)

    return reward

def jump_air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float = 0.3,
    max_air_time: float = 1.5
) -> torch.Tensor:
    """
    跳跃腾空时间奖励。
    奖励合理的腾空时间，太短说明没跳起来，太长说明失控了。
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Resolve body_ids if needed
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    # 获取脚部腾空时间
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # 计算平均腾空时间
    avg_air_time = torch.mean(last_air_time, dim=1)

    # 检测首次接触（刚落地）
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    just_landed = torch.any(first_contact, dim=1)

    # 只有刚落地时才计算奖励
    # 检查腾空时间是否在合理范围内
    in_range = (avg_air_time > min_air_time) & (avg_air_time < max_air_time)

    reward = (just_landed & in_range).float()

    return reward

