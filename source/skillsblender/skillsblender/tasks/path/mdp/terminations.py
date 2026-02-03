# terminations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation 
from isaaclab.sensors import ContactSensor 
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

#---xinjia
# def bad_orientation(
#     env: ManagerBasedRLEnv,
#     limit_angle: float = 0.5,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """当机体倾斜超过阈值时终止"""
#     asset = env.scene[asset_cfg.name]
#     # 投影重力的 z 分量：站立时约为 -1，倾斜时绝对值减小
#     return torch.abs(asset.data.projected_gravity_b[:, 2]) < torch.cos(limit_angle)
def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当机体倾斜超过阈值时终止
    
    通过四元数计算机体 z 轴与世界 z 轴的夹角
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w  # (N, 4) wxyz 格式
    
    # 从四元数提取机体 z 轴在世界坐标系中的方向
    # 对于 wxyz 格式: w=quat[:,0], x=quat[:,1], y=quat[:,2], z=quat[:,3]
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    
    # 机体 z 轴在世界坐标系中的 z 分量（即与世界 z 轴的点积）
    # 公式: body_z_world_z = 1 - 2*(x^2 + y^2)
    body_z_in_world_z = 1.0 - 2.0 * (x * x + y * y)
    
    # 站立时 body_z_in_world_z ≈ 1，倾斜时减小
    # 当该值小于 cos(limit_angle) 时终止
    return body_z_in_world_z < torch.cos(torch.tensor(limit_angle, device=quat_w.device))

# ==============================================================================
# Jump-Specific Terminations (跳跃专用终止条件)
# ==============================================================================

def jump_gap_fall(
    env: ManagerBasedRLEnv,
    height_threshold: float = -0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跳跃过程中掉入沟壑判定。
    当机器人高度低于地形基准高度阈值时终止。

    参数:
        env: 环境实例
        height_threshold: 相对于起始高度的阈值 (米)
        asset_cfg: 机器人资产配置
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取机器人当前高度
    current_height = asset.data.root_pos_w[:, 2]

    # 获取机器人下方的地形高度
    robot_xy = asset.data.root_pos_w[:, :2]

    # 简单判定：如果机器人高度低于某个绝对值，认为掉入沟壑
    # 这里假设平台高度在0附近，沟壑深度约-0.5m
    has_fallen = current_height < height_threshold

    return has_fallen

def jump_excessive_rotation(
    env: ManagerBasedRLEnv,
    max_roll: float = 1.2,  # 约70度
    max_pitch: float = 1.2,  # 约70度
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跳跃过程中过度旋转判定。
    当机器人的roll或pitch角超过阈值时终止，防止翻滚。

    参数:
        env: 环境实例
        max_roll: 最大允许的roll角 (弧度)
        max_pitch: 最大允许的pitch角 (弧度)
        asset_cfg: 机器人资产配置
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w  # (N, 4) wxyz格式

    # 提取roll和pitch角
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]

    # Roll: arctan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    # Pitch: arcsin(2*(w*y - z*x))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))

    # 检查是否超过阈值
    excessive_roll = torch.abs(roll) > max_roll
    excessive_pitch = torch.abs(pitch) > max_pitch

    return excessive_roll | excessive_pitch

def jump_landing_failure(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contact_feet: int = 2,
    check_after_air_time: float = 0.5,
    velocity_threshold: float = 2.0
) -> torch.Tensor:
    """
    跳跃落地失败判定。
    如果腾空时间足够长（说明已经跳起来了），但落地时接触脚数量不足，
    或者落地速度过快，则判定为落地失败。

    参数:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        min_contact_feet: 最少需要接触的脚数
        check_after_air_time: 只有腾空时间超过此值才检查 (秒)
        velocity_threshold: 落地时的最大允许速度 (m/s)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Resolve body_ids if needed
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    # 获取腾空时间
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    max_air_time = torch.max(last_air_time, dim=1)[0]

    # 只检查腾空时间足够长的情况（说明已经跳起来了）
    has_jumped = max_air_time > check_after_air_time

    # 检测当前接触脚数
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact_mask = torch.norm(contact_forces, dim=-1) > 1.0
    num_contact_feet = torch.sum(contact_mask, dim=1)

    # 检测首次接触（刚落地）
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    just_landed = torch.any(first_contact, dim=1)

    # 检查落地速度
    robot = env.scene["robot"]
    landing_velocity = torch.norm(robot.data.root_lin_vel_w[:, :3], dim=-1)

    # 落地失败条件：
    # 1. 已经跳起来了 (has_jumped)
    # 2. 刚落地 (just_landed)
    # 3. 且 (接触脚数不足 或 落地速度过快)
    bad_landing = just_landed & has_jumped & (
        (num_contact_feet < min_contact_feet) | (landing_velocity > velocity_threshold)
    )

    return bad_landing

def jump_timeout_no_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    time_threshold: float = 5.0,
    min_progress: float = 0.1
) -> torch.Tensor:
    """
    跳跃任务长时间无进展判定。
    如果在一定时间内，路径进度几乎没有推进，则判定为失败。

    参数:
        env: 环境实例
        command_name: 命令管理器中路径命令的名称
        time_threshold: 判定时间阈值 (秒)
        min_progress: 最小进度要求 (0-1)
    """
    command = env.command_manager.get_term(command_name)

    # 获取当前进度
    current_alpha = command.current_alpha.squeeze(-1)  # [0, 1]

    # 计算已经过去的时间
    elapsed_time = env.episode_length_buf * env.step_dt

    # 如果时间超过阈值，但进度不足，则终止
    no_progress = (elapsed_time > time_threshold) & (current_alpha < min_progress)

    return no_progress

def jump_stuck_in_gap(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    time_threshold: float = 2.0,
    velocity_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跳跃过程中卡在沟壑里判定。
    如果机器人在低高度位置且速度很小，持续一定时间，则判定为卡住。

    参数:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        time_threshold: 判定时间阈值 (秒)
        velocity_threshold: 判定速度阈值 (m/s)
        asset_cfg: 机器人资产配置
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Resolve body_ids if needed
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    # 检查是否有脚接触地面（排除完全悬空的情况）
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    has_contact = torch.any(torch.norm(contact_forces, dim=-1) > 1.0, dim=1)

    # 检查速度是否很小
    velocity = torch.norm(asset.data.root_lin_vel_w[:, :3], dim=-1)
    is_stuck = velocity < velocity_threshold

    # 检查高度是否异常低（可能在沟壑中）
    current_height = asset.data.root_pos_w[:, 2]
    is_low = current_height < 0.2  # 正常站立高度约0.34m

    # 维护一个卡住计时器（需要环境状态）
    if not hasattr(env, "_stuck_timer"):
        env._stuck_timer = torch.zeros(env.num_envs, device=env.device)

    # 更新计时器
    stuck_condition = has_contact & is_stuck & is_low
    env._stuck_timer = torch.where(
        stuck_condition,
        env._stuck_timer + env.step_dt,
        torch.zeros_like(env._stuck_timer)
    )

    # 判定卡住
    is_stuck_too_long = env._stuck_timer > time_threshold

    return is_stuck_too_long
