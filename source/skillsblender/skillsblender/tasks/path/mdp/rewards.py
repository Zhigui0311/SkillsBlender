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
