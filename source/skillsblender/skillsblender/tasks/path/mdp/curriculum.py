"""Curriculum learning functions for path tracking tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING,  Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _mean_episode_reward(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> float | None:
    """Return the mean episode reward over the selected environments."""
    if not hasattr(env, "mean_episode_reward"):
        return None

    reward = env.mean_episode_reward
    if isinstance(reward, torch.Tensor):
        if env_ids is None:
            return float(reward.mean().item())
        return float(reward[env_ids].mean().item())

    if env_ids is None:
        return float(reward)

    return float(torch.as_tensor(reward)[env_ids].mean().item())


def _episode_reward_vector(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Return per-environment episode reward tensor if available."""
    if not hasattr(env, "mean_episode_reward"):
        return None
    reward = env.mean_episode_reward
    if isinstance(reward, torch.Tensor):
        return reward
    return torch.as_tensor(reward, device=env.device).repeat(env.num_envs)


def curriculum_path_length(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    command_name: str,
    initial_range: tuple[float, float],
    final_range: tuple[float, float],
    reward_threshold: float,
) -> None:
    """
    根据训练进度逐步增加路径长度。

    注意: 课程触发条件可按 env_ids 评估，但配置更新会作用于全部环境。

    参数:
        env: 环境实例
        env_ids: 参与课程评估的环境索引（None 表示全部环境）
        command_name: 命令管理器中路径命令的名称
        initial_range: 初始路径长度范围 (min, max)
        final_range: 最终路径长度范围 (min, max)
        reward_threshold: 触发课程进阶的平均奖励阈值
    """
    # 获取路径命令
    command = env.command_manager.get_term(command_name)

    # 获取当前的平均奖励（需要从环境中获取）

    mean_reward = _mean_episode_reward(env, env_ids)
    if mean_reward is None:
        # 如果环境没有平均奖励，不进行课程调整
        return

  

    # 如果平均奖励超过阈值，增加路径长度
    if mean_reward > reward_threshold:
        # 获取当前路径长度范围
        current_min = command.cfg.inpoints.end_to_start_pos[0]
        current_max = command.cfg.inpoints.end_to_start_pos[1]

        # 计算新的路径长度范围（逐步接近最终范围）
        step_size = 0.5  # 每次增加 0.5m
        new_min = min(current_min, final_range[0])
        new_max = min(current_max + step_size, final_range[1])

        # 更新路径长度范围
        command.cfg.inpoints.end_to_start_pos = (new_min, new_max, 0)

        print(f"[Curriculum] Path length updated: ({new_min:.1f}, {new_max:.1f})")


def curriculum_velocity_requirement(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    initial_velocity: float,
    final_velocity: float,
    reward_threshold: float,
) -> None:
    """
    根据训练进度逐步提高速度要求。

    注意: 课程触发条件可按 env_ids 评估，但配置更新会作用于全部环境。

    参数:
        env: 环境实例
        env_ids: 参与课程评估的环境索引（None 表示全部环境）
        initial_velocity: 初始目标速度
        final_velocity: 最终目标速度
        reward_threshold: 触发课程进阶的平均奖励阈值
    """
    # 获取当前的平均奖励
    mean_reward = _mean_episode_reward(env, env_ids)
    if mean_reward is None:
        return

    # 如果平均奖励超过阈值，提高速度要求
    if mean_reward > reward_threshold:
        # 这里需要根据具体的奖励函数来调整目标速度
        # 例如，可以修改 track_velocity_along_path_exp 函数中的目标速度
        # 注意: 这需要在奖励函数中支持动态参数调整
        print(f"[Curriculum] Velocity requirement could be increased")
        # 实际实现取决于奖励函数的设计


def curriculum_terrain_diffiƒculty(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    reward_threshold: float,
    max_difficulty: int = 5,
) -> None:
    """
    根据训练进度逐步增加地形难度。

    注意: 课程触发条件可按 env_ids 评估，但配置更新会作用于全部环境。

    参数:
        env: 环境实例
        env_ids: 参与课程评估的环境索引（None 表示全部环境）
        reward_threshold: 触发课程进阶的平均奖励阈值
        max_difficulty: 最大地形难度级别
    """
    # 获取当前的平均奖励
    mean_reward = _mean_episode_reward(env, env_ids)
    if mean_reward is None:
        return

    # 如果平均奖励超过阈值，增加地形难度
    if mean_reward > reward_threshold:
        # 获取地形配置
        if hasattr(env.scene, "terrain"):
            terrain = env.scene.terrain
            # 增加地形难度级别
            if hasattr(terrain.cfg, "difficulty_range"):
                current_max = terrain.cfg.difficulty_range[1]
                new_max = min(current_max + 0.1, 1.0)
                terrain.cfg.difficulty_range = (0.0, new_max)
                print(f"[Curriculum] Terrain difficulty updated: {new_max:.2f}")




# ==============================================================================
# Jump-Specific Curriculum (跳跃专用课程学习)
# ==============================================================================

def curriculum_jump_gap_width(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    reward_threshold: float,
    initial_gap_range: tuple[float, float] = (0.2, 0.3),  # Updated: gentler start
    intermediate_gap_range: tuple[float, float] = (0.3, 0.5),  # New: intermediate level
    final_gap_range: tuple[float, float] = (0.5, 0.8),  # Updated: reduced max
    step_size: float = 0.1,
) -> None:
    """
    根据训练进度逐步增加跳跃沟壑宽度。

    注意: 课程触发条件可按 env_ids 评估，但地形配置更新会作用于全部环境。

    参数:
        env: 环境实例
        env_ids: 参与课程评估的环境索引（None 表示全部环境）
        reward_threshold: 触发课程进阶的平均奖励阈值
        initial_gap_range: 初始沟壑宽度范围 (min, max) - level 0-1
        intermediate_gap_range: 中级沟壑宽度范围 (min, max) - level 2-3
        final_gap_range: 最终沟壑宽度范围 (min, max) - level 4+
        step_size: 每次增加的步长 (米)
    """
    # 获取当前的平均奖励
    mean_reward = _mean_episode_reward(env, env_ids)
    if mean_reward is None:
        return

    # 如果平均奖励超过阈值，增加沟壑宽度
    if mean_reward > reward_threshold:
        # 获取地形配置
        if hasattr(env.scene, "terrain"):
            terrain = env.scene.terrain
            if hasattr(terrain.cfg, "terrain_generator"):
                gen_cfg = terrain.cfg.terrain_generator

                # 更新窄沟壑配置
                if "narrow_gaps" in gen_cfg.sub_terrains:
                    narrow_cfg = gen_cfg.sub_terrains["narrow_gaps"]
                    current_max = narrow_cfg.gap_width_range[1]
                    new_max = min(current_max + step_size, intermediate_gap_range[1])
                    narrow_cfg.gap_width_range = (initial_gap_range[0], new_max)
                    print(f"[Jump Curriculum] Narrow gap width updated: {narrow_cfg.gap_width_range}")

                # 更新宽沟壑配置
                if "wide_gaps" in gen_cfg.sub_terrains:
                    wide_cfg = gen_cfg.sub_terrains["wide_gaps"]
                    current_max = wide_cfg.gap_width_range[1]
                    new_max = min(current_max + step_size, final_gap_range[1])
                    wide_cfg.gap_width_range = (intermediate_gap_range[0], new_max)
                    print(f"[Jump Curriculum] Wide gap width updated: {wide_cfg.gap_width_range}")


def curriculum_jump_height_requirement(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    command_name: str,
    reward_threshold: float,
    initial_height: float = 0.20,  # Updated: reduced from 0.25
    final_height: float = 0.40,  # Updated: reduced from 0.45
    step_size: float = 0.03,  # Updated: reduced from 0.05
) -> None:
    """
    根据训练进度逐步提高跳跃高度要求。

    注意: 课程触发条件可按 env_ids 评估，但命令配置更新会作用于全部环境。

    参数:
        env: 环境实例
        env_ids: 参与课程评估的环境索引（None 表示全部环境）
        command_name: 命令管理器中跳跃命令的名称
        reward_threshold: 触发课程进阶的平均奖励阈值
        initial_height: 初始跳跃高度 (米)
        final_height: 最终跳跃高度 (米)
        step_size: 每次增加的步长 (米)
    """
    # 获取当前的平均奖励
    mean_reward = _mean_episode_reward(env, env_ids)
    if mean_reward is None:
        return


    # 如果平均奖励超过阈值，增加跳跃高度
    if mean_reward > reward_threshold:
        command = env.command_manager.get_term(command_name)
        if hasattr(command.cfg, "jump_params"):
            current_height = command.cfg.jump_params.jump_height
            new_height = min(current_height + step_size, final_height)
            command.cfg.jump_params.jump_height = new_height
            print(f"[Jump Curriculum] Jump height requirement updated: {new_height:.2f}m")


def curriculum_jump_terrain_mix(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    reward_threshold: float,
    max_jump_proportion: float = 0.9,
) -> None:
    """
    根据训练进度逐步增加跳跃地形的比例，减少平地比例。

    注意: 课程触发条件可按 env_ids 评估，但地形配置更新会作用于全部环境。

    参数:
        env: 环境实例
        env_ids: 参与课程评估的环境索引（None 表示全部环境）
        reward_threshold: 触发课程进阶的平均奖励阈值
        max_jump_proportion: 跳跃地形的最大比例
    """
    # 获取当前的平均奖励
    mean_reward = _mean_episode_reward(env, env_ids)
    if mean_reward is None:
        return


    # 如果平均奖励超过阈值，增加跳跃地形比例
    if mean_reward > reward_threshold:
        if hasattr(env.scene, "terrain"):
            terrain = env.scene.terrain
            if hasattr(terrain.cfg, "terrain_generator"):
                gen_cfg = terrain.cfg.terrain_generator

                # 减少平地比例，增加跳跃地形比例
                if "flat" in gen_cfg.sub_terrains:
                    flat_cfg = gen_cfg.sub_terrains["flat"]
                    current_flat_prop = flat_cfg.proportion
                    new_flat_prop = max(current_flat_prop - 0.1, 1.0 - max_jump_proportion)
                    flat_cfg.proportion = new_flat_prop

                    # 重新分配比例给跳跃地形
                    jump_proportion = 1.0 - new_flat_prop
                    if "narrow_gaps" in gen_cfg.sub_terrains and "wide_gaps" in gen_cfg.sub_terrains:
                        gen_cfg.sub_terrains["narrow_gaps"].proportion = jump_proportion * 0.6
                        gen_cfg.sub_terrains["wide_gaps"].proportion = jump_proportion * 0.4

                    print(f"[Jump Curriculum] Terrain mix updated - Flat: {new_flat_prop:.2f}, Jumps: {jump_proportion:.2f}")


def curriculum_jump_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    reward_threshold: float,
    max_level: int = 6,
    down_threshold: float | None = None,
) -> None:
    """Per-env discrete terrain level curriculum for jump tasks."""
    rewards = _episode_reward_vector(env)
    if rewards is None:
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device)

    if down_threshold is None:
        down_threshold = reward_threshold * 0.5

    move_up = rewards[env_ids] > reward_threshold
    move_down = rewards[env_ids] < down_threshold

    if not hasattr(env, "_jump_terrain_levels"):
        env._jump_terrain_levels = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    levels = env._jump_terrain_levels
    levels[env_ids] = torch.clamp(levels[env_ids] + move_up.long() - move_down.long(), 0, max_level)
    env._jump_terrain_levels = levels

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "update_env_origins"):
        return

    terrain.update_env_origins(env_ids, move_up, move_down)


def curriculum_jump_terrain_levels_by_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    command_name: str,
    threshold: float = 0.5,
    down_ratio: float = 0.5,
) -> None:
    """Per-env discrete terrain levels based on path progress."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device)

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "update_env_origins"):
        return

    command = env.command_manager.get_term(command_name)
    robot_pos = command.robot_pos_w[env_ids]
    target_pos = command.target_pos_w[env_ids]
    distance = torch.norm(robot_pos[:, :2] - target_pos[:, :2], dim=1)

    if hasattr(command, "pos_path_w"):
        start_pos = command.pos_path_w[env_ids, 0, :2]
        initial_distance = torch.norm(start_pos - target_pos[:, :2], dim=1)
    else:
        initial_distance = torch.full_like(distance, threshold / down_ratio)

    move_up = distance <= threshold
    move_down = distance > initial_distance * down_ratio
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)


def curriculum_jump_speed_requirement(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    reward_threshold: float,
    initial_speed: float = 0.8,  # Updated: reduced from 1.0
    final_speed: float = 2.0,
    step_size: float = 0.08,  # Updated: reduced from 0.1
) -> None:
    """
    根据训练进度逐步提高跳跃时的速度要求。

    注意: 课程触发条件可按 env_ids 评估，但更新会作用于全部环境。

    参数:
        env: 环境实例
        env_ids: 参与课程评估的环境索引（None 表示全部环境）
        reward_threshold: 触发课程进阶的平均奖励阈值
        initial_speed: 初始速度要求 (m/s)
        final_speed: 最终速度要求 (m/s)
        step_size: 每次增加的步长 (m/s)
    """
    # 获取当前的平均奖励
    mean_reward = _mean_episode_reward(env, env_ids)
    if mean_reward is None:
        return


    # 如果平均奖励超过阈值，增加速度要求

    if mean_reward > reward_threshold:
        # 这里需要更新奖励函数中的target_velocity参数
        # 由于奖励函数参数在配置时固定，这里可以通过环境变量传递
        if not hasattr(env, "_jump_target_velocity"):
            env._jump_target_velocity = initial_speed

        current_speed = env._jump_target_velocity
        new_speed = min(current_speed + step_size, final_speed)
        env._jump_target_velocity = new_speed
        print(f"[Jump Curriculum] Jump speed requirement updated: {new_speed:.2f}m/s")

# """Curriculum learning functions for path tracking tasks."""

# from __future__ import annotations

# import torch
# from typing import TYPE_CHECKING, Sequence

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv


# def _mean_episode_reward(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | torch.Tensor | None,
# ) -> float | None:
#     """Return the mean episode reward over the selected environments."""
#     if not hasattr(env, "mean_episode_reward"):
#         return None

#     reward = env.mean_episode_reward
#     if isinstance(reward, torch.Tensor):
#         if env_ids is None:
#             return float(reward.mean().item())
#         return float(reward[env_ids].mean().item())

#     if env_ids is None:
#         return float(reward)

#     return float(torch.as_tensor(reward)[env_ids].mean().item())

# def curriculum_path_length(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | None, 
#     command_name: str,
#     initial_range: tuple[float, float],
#     final_range: tuple[float, float],
#     reward_threshold: float,
# ) -> None:
#     """
#     根据训练进度逐步增加路径长度。
#     """
#     # 获取路径命令
#     command = env.command_manager.get_term(command_name)

#     if not hasattr(env, "mean_episode_reward"):
#         return

#     current_reward = env.mean_episode_reward

#     # 如果平均奖励超过阈值，增加路径长度
#     if current_reward > reward_threshold:
#         # 获取当前路径长度范围
#         current_min = command.cfg.inpoints.end_to_start_pos[0]
#         current_max = command.cfg.inpoints.end_to_start_pos[1]

#         # 计算新的路径长度范围（逐步接近最终范围）
#         step_size = 0.5  # 每次增加 0.5m
#         new_min = min(current_min, final_range[0])
#         new_max = min(current_max + step_size, final_range[1])

#         # 更新路径长度范围
#         command.cfg.inpoints.end_to_start_pos = (new_min, new_max, 0)

#         print(f"[Curriculum] Path length updated: ({new_min:.1f}, {new_max:.1f})")


# def curriculum_velocity_requirement(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | None,  # <--- 新增
#     initial_velocity: float,
#     final_velocity: float,
#     reward_threshold: float,
# ) -> None:
#     """
#     根据训练进度逐步提高速度要求。
#     """
#     if not hasattr(env, "mean_episode_reward"):
#         return

#     current_reward = env.mean_episode_reward

#     if current_reward > reward_threshold:
#         print(f"[Curriculum] Velocity requirement could be increased")
#         # 实际实现取决于奖励函数的设计


# def curriculum_terrain_difficulty(  # <--- 修复了原来的拼写错误 (diffiƒculty)
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | None,  # <--- 新增
#     reward_threshold: float,
#     max_difficulty: int = 5,
# ) -> None:
#     """
#     根据训练进度逐步增加地形难度。
#     """
#     if not hasattr(env, "mean_episode_reward"):
#         return

#     current_reward = env.mean_episode_reward

#     if current_reward > reward_threshold:
#         if hasattr(env.scene, "terrain"):
#             terrain = env.scene.terrain
#             if hasattr(terrain.cfg, "difficulty_range"):
#                 current_max = terrain.cfg.difficulty_range[1]
#                 new_max = min(current_max + 0.1, 1.0)
#                 terrain.cfg.difficulty_range = (0.0, new_max)
#                 print(f"[Curriculum] Terrain difficulty updated: {new_max:.2f}")


# # ==============================================================================
# # Jump-Specific Curriculum (跳跃专用课程学习)
# # ==============================================================================

# def curriculum_jump_gap_width(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | None,  # <--- 关键修改：新增 env_ids 参数！
#     reward_threshold: float,
#     initial_gap_range: tuple[float, float] = (0.3, 0.5),
#     final_gap_range: tuple[float, float] = (0.6, 1.0),
#     step_size: float = 0.1,
# ) -> None:
#     """
#     根据训练进度逐步增加跳跃沟壑宽度。
#     """
#     if not hasattr(env, "mean_episode_reward"):
#         return

#     current_reward = env.mean_episode_reward

#     if current_reward > reward_threshold:
#         if hasattr(env.scene, "terrain"):
#             terrain = env.scene.terrain
#             if hasattr(terrain.cfg, "terrain_generator"):
#                 gen_cfg = terrain.cfg.terrain_generator

#                 # 更新窄沟壑配置
#                 if "narrow_gaps" in gen_cfg.sub_terrains:
#                     narrow_cfg = gen_cfg.sub_terrains["narrow_gaps"]
#                     current_max = narrow_cfg.gap_width_range[1]
#                     new_max = min(current_max + step_size, initial_gap_range[1])
#                     narrow_cfg.gap_width_range = (initial_gap_range[0], new_max)
#                     print(f"[Jump Curriculum] Narrow gap width updated: {narrow_cfg.gap_width_range}")

#                 # 更新宽沟壑配置
#                 if "wide_gaps" in gen_cfg.sub_terrains:
#                     wide_cfg = gen_cfg.sub_terrains["wide_gaps"]
#                     current_max = wide_cfg.gap_width_range[1]
#                     new_max = min(current_max + step_size, final_gap_range[1])
#                     wide_cfg.gap_width_range = (final_gap_range[0], new_max)
#                     print(f"[Jump Curriculum] Wide gap width updated: {wide_cfg.gap_width_range}")


# def curriculum_jump_height_requirement(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | None,  # <--- 新增
#     command_name: str,
#     reward_threshold: float,
#     initial_height: float = 0.25,
#     final_height: float = 0.45,
#     step_size: float = 0.05,
# ) -> None:
#     """
#     根据训练进度逐步提高跳跃高度要求。
#     """
#     if not hasattr(env, "mean_episode_reward"):
#         return

#     current_reward = env.mean_episode_reward

#     if current_reward > reward_threshold:
#         command = env.command_manager.get_term(command_name)
#         if hasattr(command.cfg, "jump_params"):
#             current_height = command.cfg.jump_params.jump_height
#             new_height = min(current_height + step_size, final_height)
#             command.cfg.jump_params.jump_height = new_height
#             print(f"[Jump Curriculum] Jump height requirement updated: {new_height:.2f}m")


# def curriculum_jump_terrain_mix(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | None,  # <--- 新增
#     reward_threshold: float,
#     max_jump_proportion: float = 0.9,
# ) -> None:
#     """
#     根据训练进度逐步增加跳跃地形的比例，减少平地比例。
#     """
#     if not hasattr(env, "mean_episode_reward"):
#         return

#     current_reward = env.mean_episode_reward

#     if current_reward > reward_threshold:
#         if hasattr(env.scene, "terrain"):
#             terrain = env.scene.terrain
#             if hasattr(terrain.cfg, "terrain_generator"):
#                 gen_cfg = terrain.cfg.terrain_generator

#                 # 减少平地比例，增加跳跃地形比例
#                 if "flat" in gen_cfg.sub_terrains:
#                     flat_cfg = gen_cfg.sub_terrains["flat"]
#                     current_flat_prop = flat_cfg.proportion
#                     new_flat_prop = max(current_flat_prop - 0.1, 1.0 - max_jump_proportion)
#                     flat_cfg.proportion = new_flat_prop

#                     # 重新分配比例给跳跃地形
#                     jump_proportion = 1.0 - new_flat_prop
#                     if "narrow_gaps" in gen_cfg.sub_terrains and "wide_gaps" in gen_cfg.sub_terrains:
#                         gen_cfg.sub_terrains["narrow_gaps"].proportion = jump_proportion * 0.6
#                         gen_cfg.sub_terrains["wide_gaps"].proportion = jump_proportion * 0.4

#                     print(f"[Jump Curriculum] Terrain mix updated - Flat: {new_flat_prop:.2f}, Jumps: {jump_proportion:.2f}")


# def curriculum_jump_speed_requirement(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int] | None,  # <--- 新增
#     reward_threshold: float,
#     initial_speed: float = 1.0,
#     final_speed: float = 2.0,
#     step_size: float = 0.1,
# ) -> None:
#     """
#     根据训练进度逐步提高跳跃时的速度要求。
#     """
#     if not hasattr(env, "mean_episode_reward"):
#         return

#     current_reward = env.mean_episode_reward

#     if current_reward > reward_threshold:
#         if not hasattr(env, "_jump_target_velocity"):
#             env._jump_target_velocity = initial_speed

#         current_speed = env._jump_target_velocity
#         new_speed = min(current_speed + step_size, final_speed)
#         env._jump_target_velocity = new_speed
#         print(f"[Jump Curriculum] Jump speed requirement updated: {new_speed:.2f}m/s")

