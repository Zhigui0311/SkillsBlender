# observations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation # [新增]
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from .commands.path_command import PathCommand  
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def current_alpha(
    env: ManagerBasedRLEnv, 
    command_name: str 
    ) -> torch.Tensor:
    command: PathCommand = env.command_manager.get_term(command_name)
    return command.current_alpha

def path_slice_obs(
    env: ManagerBasedRLEnv, 
    command_name: str
    ) -> torch.Tensor:
    """obtain path slice observation"""
    command: PathCommand = env.command_manager.get_term(command_name)
    return command._command


def base_lin_vel(env: ManagerBasedRLEnv, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b

def base_ang_vel(env: ManagerBasedRLEnv, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b

def projected_gravity(env: ManagerBasedRLEnv, 
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b

def joint_pos_rel(env: ManagerBasedRLEnv, 
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos

def joint_vel(env: ManagerBasedRLEnv, 
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel

def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.action