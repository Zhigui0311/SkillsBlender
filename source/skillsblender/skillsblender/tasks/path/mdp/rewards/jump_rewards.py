from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject  
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.utils.math import wrap_to_pi
import isaaclab.utils.warp as warp_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "path_tracking"
) -> torch.Tensor:
    """
    跳跃离地高度奖励。
    奖励机器人在跳跃时离地面足够高，避免碰到沟壑边缘。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取机器人base高度和XY位置
    robot_pos = asset.data.root_pos_w[:, :3]
    base_height = robot_pos[:, 2]
    robot_xy = robot_pos[:, :2]

    # 获取 JumpPathCommand 以访问 warp_mesh
    command = env.command_manager.get_term(command_name)

    # 使用 raycast 查询机器人下方的地面高度
    # 从机器人位置上方 10m 处向下发射射线
    ray_starts = torch.cat([
        robot_xy,
        base_height.unsqueeze(-1) + 10.0  # 从base上方10m处开始
    ], dim=-1)

    ray_directions = torch.tensor(
        [[0.0, 0.0, -1.0]],
        device=robot_xy.device
    ).expand(len(robot_xy), -1)

    # 执行 raycast
    ray_hits, _, _, _ = warp_utils.raycast_mesh(
        ray_starts=ray_starts,
        ray_directions=ray_directions,
        mesh=command.warp_mesh,
        max_dist=20.0
    )

    # 提取地面高度（Z坐标）
    # 如果射线未命中（返回inf），使用base_height作为fallback
    ground_height = torch.where(
        torch.isinf(ray_hits[:, 2]),
        base_height,  # Fallback
        ray_hits[:, 2]
    )

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

def hip_joint_angle_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_hip_angle: float = 0.15
) -> torch.Tensor:
    """
    惩罚hip关节角度过大，防止腿外八字。
    hip关节控制腿的横向展开，限制其角度可以防止腿张太开。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取所有关节位置
    joint_pos = asset.data.joint_pos

    # GO2的关节顺序: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    #                RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
    # hip关节索引: 0(FR), 3(FL), 6(RR), 9(RL)
    # FR和RR的hip应该是负值（向内），FL和RL应该是正值（向内）

    if joint_pos.shape[1] >= 12:
        fr_hip = joint_pos[:, 0]  # FR_hip_joint
        fl_hip = joint_pos[:, 3]  # FL_hip_joint
        rr_hip = joint_pos[:, 6]  # RR_hip_joint
        rl_hip = joint_pos[:, 9]  # RL_hip_joint

        # 计算hip角度的绝对值（偏离中立位置的程度）
        # 正常站立时，hip应该接近0或略微向内
        fr_penalty = torch.clamp(torch.abs(fr_hip) - max_hip_angle, min=0.0)
        fl_penalty = torch.clamp(torch.abs(fl_hip) - max_hip_angle, min=0.0)
        rr_penalty = torch.clamp(torch.abs(rr_hip) - max_hip_angle, min=0.0)
        rl_penalty = torch.clamp(torch.abs(rl_hip) - max_hip_angle, min=0.0)

        return torch.square(fr_penalty + fl_penalty + rr_penalty + rl_penalty)
    else:
        return torch.zeros(env.num_envs, device=env.device)

def spinning_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    angular_vel_threshold: float = 1.0,
    forward_vel_threshold: float = 0.3
) -> torch.Tensor:
    """
    惩罚原地打转行为。
    当角速度高而前向速度低时，说明机器人在原地打转而不是前进。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取角速度和线速度
    ang_vel_b = asset.data.root_ang_vel_b[:, :3]
    lin_vel_b = asset.data.root_lin_vel_b[:, :3]

    # 计算角速度大小（主要关注Z轴旋转）
    angular_vel_magnitude = torch.abs(ang_vel_b[:, 2])

    # 前向速度
    forward_vel = lin_vel_b[:, 0]

    # 检测打转行为：高角速度 + 低前向速度
    is_spinning = (angular_vel_magnitude > angular_vel_threshold) & (forward_vel < forward_vel_threshold)

    # 返回惩罚（负值）
    penalty = is_spinning.float()

    return penalty

def approach_momentum_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    approach_distance: float = 3.0,
    min_velocity: float = 1.0
) -> torch.Tensor:
    """
    接近阶段动量奖励。
    当机器人接近间隙时保持足够的前向速度，鼓励提前建立动量。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    # 获取机器人位置和前向速度
    robot_pos = asset.data.root_pos_w[:, :3]
    forward_vel = asset.data.root_lin_vel_b[:, 0]

    # 获取当前目标航点位置
    target_pos = command.pos_path_w[torch.arange(env.num_envs), command.current_waypoints_index]

    # 计算到目标的距离
    distance_to_target = torch.norm(target_pos[:, :2] - robot_pos[:, :2], dim=-1)

    # 检测是否在接近阶段（距离目标在approach_distance内）
    in_approach_phase = distance_to_target < approach_distance

    # 检测速度是否足够
    has_momentum = forward_vel > min_velocity

    # 只有在接近阶段且有足够速度时才给予奖励
    reward = (in_approach_phase & has_momentum).float()

    return reward

def consistency_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_velocity: float = 0.8,
    time_threshold: float = 2.0
) -> torch.Tensor:
    """
    一致性奖励。
    奖励机器人持续保持前向速度，避免走走停停。

    注意：这需要在环境中维护一个计数器来跟踪连续满足条件的时间。
    作为简化，我们只检查当前速度是否满足条件。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取前向速度
    forward_vel = asset.data.root_lin_vel_b[:, 0]

    # 检测速度是否持续满足条件
    # 简化版本：只检查当前速度
    maintains_velocity = forward_vel > min_velocity

    reward = maintains_velocity.float()

    return reward

