# terminations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation # [新增] 备用
from isaaclab.sensors import ContactSensor # [新增] 用于类型提示
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """达到最大步数时终止"""
    return env.episode_length_buf >= env.max_episode_length

def illegal_contact(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """检测非法接触 (Sensor-based)"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(torch.norm(net_contact_forces, dim=-1) > threshold, dim=1)

def path_deviation(
    env: ManagerBasedRLEnv, 
    min_threshold: float = 0.5, 
    max_threshold: float = 10.0,
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """According to progress, dynamically adjust the deviation threshold for path tracking termination."""
    command = env.command_manager.get_term(command_name)
    alpha = command.current_alpha.squeeze(-1) # 获取进度 [0, 1]
    
    # 进度越小（刚开始），阈值越大；进度越大（快到终点），阈值越小
    # 阈值从 max_threshold 线性下降到 min_threshold
    current_max_dist = max_threshold - (max_threshold - min_threshold) * alpha
    
    return command.metrics["error_pos_xy"] > current_max_dist

# def path_deviation_with_grace(
#     env: ManagerBasedRLEnv, 
#     max_deviation: float, 
#     grace_steps: int = 50, # 给予 50 步宽限期
#     command_name: str = "path_tracking"
# ) -> torch.Tensor:
#     """With a grace period, terminate if path deviation exceeds max_deviation after grace_steps."""
#     command = env.command_manager.get_term(command_name)
#     # 只有当步数超过 grace_steps 且误差超过阈值时才终止
#     deviation_trigger = command.metrics["error_pos_xy"] > max_deviation
#     time_trigger = env.episode_length_buf > grace_steps
    
#     return torch.logical_and(deviation_trigger, time_trigger)

#  如果想用这个作为摔倒判定，就需要 Articulation
def base_height_below_threshold(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    当 Base 高度低于阈值时终止。
    (比 ContactSensor 更通用的摔倒判定)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height