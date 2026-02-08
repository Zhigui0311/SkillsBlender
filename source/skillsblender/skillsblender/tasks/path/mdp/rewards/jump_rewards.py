from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# Jump Task-Specific Rewards
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
    env_ids = torch.arange(env.num_envs, device=env.device)
    idx = torch.clamp(command.current_waypoints_index, max=command.num_waypoints - 1)
    target_height = command.pos_path_w[env_ids, idx, 2]
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, 2]

    # 计算高度误差
    height_error = torch.abs(robot_pos_w - target_height)

    # 使用指数核奖励高度跟踪
    reward = torch.exp(-torch.square(height_error) / (height_tolerance ** 2))

    if hasattr(command, "is_in_jump_phase"):
        reward = torch.where(command.is_in_jump_phase, reward, torch.zeros_like(reward))

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

    # 获取 JumpPathCommand 以访问 height_scanner（如果有）
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "height_scanner") or command.height_scanner is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(command.height_scanner, "data"):
        return torch.zeros(env.num_envs, device=env.device)

    ray_hits_w = command.height_scanner.data.ray_hits_w  # (N, R, 3)
    deltas = ray_hits_w[..., :2] - robot_xy[:, None, :]
    dist = torch.norm(deltas, dim=-1)
    min_idx = torch.argmin(dist, dim=1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    ground_height = ray_hits_w[env_ids, min_idx, 2]
    ground_height = torch.where(torch.isfinite(ground_height), ground_height, base_height)

    # 计算离地间隙
    clearance = base_height - ground_height

    # 只有当离地间隙超过最小值时才给予奖励
    reward = torch.clamp((clearance - min_clearance) / min_clearance, min=0.0, max=1.0)

    if hasattr(command, "is_in_jump_phase"):
        reward = torch.where(command.is_in_jump_phase, reward, torch.zeros_like(reward))

    return reward

def jump_forward_velocity(
    env: ManagerBasedRLEnv,
    target_velocity: float = 1.5,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "path_tracking",
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
    command = env.command_manager.get_term(command_name)
    if hasattr(command, "is_in_jump_approach_phase") and hasattr(command, "is_in_jump_phase"):
        gate = command.is_in_jump_approach_phase | command.is_in_jump_phase
        reward = torch.where(gate, reward, torch.zeros_like(reward))
    elif hasattr(command, "is_in_jump_phase"):
        reward = torch.where(command.is_in_jump_phase, reward, torch.zeros_like(reward))

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

    forward_vel = asset.data.root_lin_vel_b[:, 0]
    if hasattr(command, "is_in_jump_approach_phase"):
        in_approach_phase = command.is_in_jump_approach_phase
    else:
        # fallback for legacy commands: use distance to jump-start if available.
        if hasattr(command, "_jump_start_dist") and hasattr(command, "_dist_along_planned"):
            s = command._dist_along_planned()
            dist_to_takeoff = command._jump_start_dist - s
            in_approach_phase = (dist_to_takeoff >= 0.0) & (dist_to_takeoff <= approach_distance)
        else:
            robot_pos = asset.data.root_pos_w[:, :3]
            target_pos = command.pos_path_w[torch.arange(env.num_envs), command.current_waypoints_index]
            distance_to_target = torch.norm(target_pos[:, :2] - robot_pos[:, :2], dim=-1)
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


def takeoff_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_vertical_velocity: float = 0.6,
    vertical_tolerance: float = 0.3,
    target_forward_velocity: float = 1.5,
    forward_tolerance: float = 0.3,
    takeoff_window: float = 0.5,
) -> torch.Tensor:
    """
    起跳速度奖励（带安全范围）。

    鼓励机器人在gap边缘达到合适的起跳速度：
    - 垂直速度：目标0.6 m/s（范围0.3-0.9）
    - 前向速度：目标1.5 m/s（范围1.2-1.8）

    使用高斯奖励函数，在目标值附近奖励最高，偏离越多奖励越低。

    Args:
        target_vertical_velocity: 目标向上速度 (m/s)
        vertical_tolerance: 垂直速度容差 (m/s)
        target_forward_velocity: 目标前向速度 (m/s)
        forward_tolerance: 前向速度容差 (m/s)
        takeoff_window: 起跳窗口距离 (m)，在gap前这个距离内检测起跳
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    # 获取速度
    vertical_vel = asset.data.root_lin_vel_w[:, 2]  # 世界坐标系Z轴速度
    forward_vel = asset.data.root_lin_vel_b[:, 0]   # 机体坐标系前向速度

    # 判断是否在起跳窗口内（gap前0.5m范围）
    in_takeoff_window = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if hasattr(command, "_seg_s0") and hasattr(command, "_seg_skill_id"):
        # 获取当前位置
        robot_pos = asset.data.root_pos_w[:, :3]

        # 找到jump技能段
        for env_idx in range(env.num_envs):
            # 查找jump段（skill_id=1）
            jump_segs = (command._seg_skill_id[env_idx] == 1)
            if torch.any(jump_segs):
                # 获取第一个jump段的起点
                jump_seg_idx = torch.nonzero(jump_segs, as_tuple=False)[0].item()
                gap_start = command._seg_s0[env_idx, jump_seg_idx].item()

                # 计算沿路径的距离
                path_start = command.pos_path_w[env_idx, 0]
                path_dir = command._planned_forward_dir[env_idx]
                robot_rel = robot_pos[env_idx, :2] - path_start[:2]
                dist_along_path = torch.dot(robot_rel, path_dir[:2])

                # 检查是否在起跳窗口内（gap前takeoff_window米）
                dist_to_gap = gap_start - dist_along_path
                in_takeoff_window[env_idx] = (dist_to_gap >= 0.0) & (dist_to_gap <= takeoff_window)

    # 计算高斯奖励：在目标值附近奖励最高
    vertical_error = (vertical_vel - target_vertical_velocity) / vertical_tolerance
    vertical_reward = torch.exp(-torch.square(vertical_error))

    forward_error = (forward_vel - target_forward_velocity) / forward_tolerance
    forward_reward = torch.exp(-torch.square(forward_error))

    # 综合奖励：只在起跳窗口内给奖励
    reward = in_takeoff_window.float() * vertical_reward * forward_reward

    return reward


def feet_edge_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "path_tracking",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    edge_distance_threshold: float = 0.3,
    reward_scale: float = 1.0,
) -> torch.Tensor:
    """
    Blind jump edge detection reward using path_slice data.

    Encourages the robot to detect gap edges and lift feet appropriately
    before jumping, even without direct visual feedback. Uses the path_slice
    observation to detect upcoming terrain changes.

    This reward helps the robot learn to:
    1. Detect gap edges from path_slice height changes
    2. Lift feet preemptively when approaching a gap
    3. Time the jump takeoff based on terrain features

    Args:
        command_name: Name of the path tracking command
        sensor_cfg: Contact sensor configuration for feet
        edge_distance_threshold: Distance ahead to check for edges (m)
        reward_scale: Scaling factor for the reward

    Returns:
        Reward tensor for each environment
    """
    command = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Resolve body_ids if needed
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    # Get path_slice observation if available
    if not hasattr(command, "path_slice") or command.path_slice is None:
        return torch.zeros(env.num_envs, device=env.device)

    path_slice = command.path_slice  # Shape: (N, num_points, 3) - XYZ positions

    # Detect edges in path_slice by looking for height discontinuities
    # Compare consecutive points to find sudden height drops (gaps)
    height_diffs = path_slice[:, 1:, 2] - path_slice[:, :-1, 2]  # (N, num_points-1)

    # Find significant height drops (potential gap edges)
    # Negative values indicate drops
    edge_detected = height_diffs < -0.15  # 15cm drop threshold

    # Find the closest edge within threshold distance
    # Calculate distance along path for each point
    xy_diffs = path_slice[:, 1:, :2] - path_slice[:, :-1, :2]  # (N, num_points-1, 2)
    distances = torch.norm(xy_diffs, dim=-1)  # (N, num_points-1)
    cumulative_dist = torch.cumsum(distances, dim=1)  # (N, num_points-1)

    # Check if edge is within threshold distance
    edge_nearby = edge_detected & (cumulative_dist < edge_distance_threshold)
    has_nearby_edge = torch.any(edge_nearby, dim=1)  # (N,)

    # Check if feet are lifted (not in contact)
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, num_feet, 3)
    feet_in_contact = torch.norm(contact_forces, dim=-1) > 1.0  # (N, num_feet)
    any_foot_lifted = ~torch.all(feet_in_contact, dim=1)  # (N,)

    # Reward: lift feet when edge is nearby
    reward = (has_nearby_edge & any_foot_lifted).float() * reward_scale

    # Only apply during jump approach phase if available
    if hasattr(command, "is_in_jump_approach_phase"):
        reward = torch.where(command.is_in_jump_approach_phase, reward, torch.zeros_like(reward))

    return reward
