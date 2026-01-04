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
    max_deviation: float, 
    command_name: str 
) -> torch.Tensor:
    """偏离路径终止 (Command-based)"""
    command = env.command_manager.get_term(command_name)
    return command.metrics["error_pos_xy"] > max_deviation

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