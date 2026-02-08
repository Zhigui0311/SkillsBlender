from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .base_rewards import skill_phase_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def crouch_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str = "path_tracking",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base-height penalty gated to crouch segments only."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    mask = skill_phase_mask(env, command_name, "crouch")
    return error * mask.float()


def crouch_body_collision_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "path_tracking",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh"]),
    threshold: float = 1.0,
    penalty_scale: float = 1.0,
) -> torch.Tensor:
    """
    Crouch body collision penalty.

    Penalizes collisions of the robot body (base, hips, thighs) with obstacles
    during crouch phases. This encourages the robot to maintain proper clearance
    and avoid scraping against low obstacles.

    Args:
        command_name: Name of the path tracking command
        sensor_cfg: Contact sensor configuration for body parts
        threshold: Contact force threshold to detect collision (N)
        penalty_scale: Scaling factor for the penalty

    Returns:
        Penalty tensor for each environment (negative values)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Resolve body_ids if needed
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    # Get contact forces on body parts
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, num_bodies, 3)
    contact_magnitudes = torch.norm(contact_forces, dim=-1)  # (N, num_bodies)

    # Detect collisions (any body part in contact)
    has_collision = torch.any(contact_magnitudes > threshold, dim=1)  # (N,)

    # Apply penalty only during crouch phase
    mask = skill_phase_mask(env, command_name, "crouch")
    penalty = has_collision.float() * mask.float() * penalty_scale

    return penalty


def crouch_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "path_tracking",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_clearance: float = 0.05,
    max_clearance: float = 0.15,
) -> torch.Tensor:
    """
    Crouch clearance reward.

    Rewards maintaining optimal clearance above obstacles during crouch.
    Too little clearance risks collision, too much wastes energy.

    Args:
        command_name: Name of the path tracking command
        asset_cfg: Robot asset configuration
        min_clearance: Minimum safe clearance (m)
        max_clearance: Maximum efficient clearance (m)

    Returns:
        Reward tensor for each environment
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    # Get robot base height
    base_height = asset.data.root_pos_w[:, 2]

    # Get ground/obstacle height from height scanner if available
    if not hasattr(command, "height_scanner") or command.height_scanner is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(command.height_scanner, "data"):
        return torch.zeros(env.num_envs, device=env.device)

    # Get obstacle height below robot
    robot_xy = asset.data.root_pos_w[:, :2]
    ray_hits_w = command.height_scanner.data.ray_hits_w  # (N, R, 3)

    # Find closest ray hit to robot position
    deltas = ray_hits_w[..., :2] - robot_xy[:, None, :]
    dist = torch.norm(deltas, dim=-1)
    min_idx = torch.argmin(dist, dim=1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    obstacle_height = ray_hits_w[env_ids, min_idx, 2]
    obstacle_height = torch.where(torch.isfinite(obstacle_height), obstacle_height, base_height - 0.3)

    # Calculate clearance
    clearance = base_height - obstacle_height

    # Reward clearance in optimal range
    # Below min_clearance: risk of collision
    # Above max_clearance: inefficient
    in_optimal_range = (clearance >= min_clearance) & (clearance <= max_clearance)
    reward = in_optimal_range.float()

    # Apply only during crouch phase
    mask = skill_phase_mask(env, command_name, "crouch")
    reward = reward * mask.float()

    return reward


def crouch_forward_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "path_tracking",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_velocity: float = 0.3,
) -> torch.Tensor:
    """
    Crouch forward progress reward.

    Encourages the robot to maintain forward progress during crouch,
    preventing it from stopping or moving too slowly.

    Args:
        command_name: Name of the path tracking command
        asset_cfg: Robot asset configuration
        min_velocity: Minimum forward velocity to maintain (m/s)

    Returns:
        Reward tensor for each environment
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get forward velocity in body frame
    forward_vel = asset.data.root_lin_vel_b[:, 0]

    # Reward maintaining minimum velocity
    maintains_progress = forward_vel > min_velocity
    reward = maintains_progress.float()

    # Apply only during crouch phase
    mask = skill_phase_mask(env, command_name, "crouch")
    reward = reward * mask.float()

    return reward
