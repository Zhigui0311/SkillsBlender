"""多技能路径生成系统使用示例

该脚本演示如何使用新的路径生成系统：
1. 配置路径生成器
2. 生成单技能路径（walk, jump）
3. 生成多技能序列路径
4. 验证和可视化路径
"""

import torch
from skillsblender.tasks.path.mdp.commands.path_command_cfg import (
    PathGeneratorCfg,
    EnvironmentBounds,
    ValidationParams,
    WalkParams,
    JumpParams,
)
from skillsblender.tasks.path.utils.path_generator import TerrainAwarePathGenerator


def example_1_generate_walk_path():
    """示例 1：生成行走路径"""
    print("\n" + "="*60)
    print("示例 1：生成行走路径")
    print("="*60)

    # 1. 配置路径生成器
    cfg = PathGeneratorCfg(
        env_bounds=EnvironmentBounds(
            x_range=(-5.0, 5.0),
            y_range=(-5.0, 5.0),
            boundary_margin=0.5,
        ),
        num_waypoints=64,
        walk_params=WalkParams(v_ref=1.0, base_height_ref=0.34),
    )

    # 2. 创建路径生成器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_generator = TerrainAwarePathGenerator(cfg, device)

    # 3. 准备输入数据
    env_ids = torch.tensor([0, 1, 2], device=device)
    start_pos = torch.tensor([
        [0.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [2.0, 2.0, 0.5],
    ], device=device)
    start_yaw = torch.tensor([0.0, 0.5, 1.0], device=device)
    env_bounds = torch.tensor([
        [-5.0, 5.0, -5.0, 5.0],
        [-5.0, 5.0, -5.0, 5.0],
        [-5.0, 5.0, -5.0, 5.0],
    ], device=device)

    # 4. 生成路径
    path_result = path_generator.generate_path(
        env_ids=env_ids,
        skill_name="walk",
        start_pos=start_pos,
        start_yaw=start_yaw,
        env_bounds=env_bounds,
    )

    # 5. 打印结果
    print(f"✓ 生成了 {len(env_ids)} 条行走路径")
    print(f"  - Waypoints shape: {path_result.waypoints.shape}")
    print(f"  - Headings shape: {path_result.headings.shape}")
    print(f"  - Path lengths: {path_result.path_length.cpu().numpy()}")
    print(f"  - Valid paths: {path_result.valid.cpu().numpy()}")
    print(f"  - Number of segments: {path_result.num_segments.cpu().numpy()}")


def example_2_generate_jump_path():
    """示例 2：生成跳跃路径（需要模拟 height_scanner 数据）"""
    print("\n" + "="*60)
    print("示例 2：生成跳跃路径")
    print("="*60)

    # 1. 配置路径生成器
    cfg = PathGeneratorCfg(
        env_bounds=EnvironmentBounds(
            x_range=(-8.0, 8.0),
            y_range=(-8.0, 8.0),
            boundary_margin=0.8,
        ),
        num_waypoints=64,
        jump_params=JumpParams(
            scan_dist=6.0,
            gap_threshold=-0.15,
            takeoff_margin_min=0.3,
            takeoff_margin_max=0.5,
            jump_height_min=0.35,
            jump_height_max=0.6,
        ),
    )

    # 2. 创建路径生成器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_generator = TerrainAwarePathGenerator(cfg, device)

    # 3. 准备输入数据
    env_ids = torch.tensor([0], device=device)
    start_pos = torch.tensor([[0.0, 0.0, 0.5]], device=device)
    start_yaw = torch.tensor([0.0], device=device)
    env_bounds = torch.tensor([[-8.0, 8.0, -8.0, 8.0]], device=device)

    # 4. 模拟 height_scanner 数据（有 gap）
    num_rays = 60
    ray_hits_w = torch.zeros(1, num_rays, 3, device=device)

    # 前半段：正常高度
    ray_hits_w[0, :20, 0] = torch.linspace(0, 2.0, 20, device=device)
    ray_hits_w[0, :20, 2] = 0.5

    # 中间段：gap（低于阈值）
    ray_hits_w[0, 20:35, 0] = torch.linspace(2.0, 3.5, 15, device=device)
    ray_hits_w[0, 20:35, 2] = -0.3  # 低于 gap_threshold

    # 后半段：正常高度
    ray_hits_w[0, 35:, 0] = torch.linspace(3.5, 6.0, 25, device=device)
    ray_hits_w[0, 35:, 2] = 0.5

    terrain_data = {"height_scanner": ray_hits_w}

    # 5. 生成路径
    path_result = path_generator.generate_path(
        env_ids=env_ids,
        skill_name="jump",
        start_pos=start_pos,
        start_yaw=start_yaw,
        env_bounds=env_bounds,
        terrain_data=terrain_data,
    )

    # 6. 打印结果
    print(f"✓ 生成了跳跃路径")
    print(f"  - Waypoints shape: {path_result.waypoints.shape}")
    print(f"  - Path length: {path_result.path_length.item():.2f}m")
    print(f"  - Valid path: {path_result.valid.item()}")
    print(f"  - Detected gap: {path_result.valid.item()}")

    # 检查跳跃高度
    max_height = path_result.waypoints[0, :, 2].max().item()
    print(f"  - Max jump height: {max_height:.2f}m")


def example_3_generate_multi_skill_path():
    """示例 3：生成多技能序列路径"""
    print("\n" + "="*60)
    print("示例 3：生成多技能序列路径 (walk -> jump -> walk)")
    print("="*60)

    # 1. 配置路径生成器
    cfg = PathGeneratorCfg(
        env_bounds=EnvironmentBounds(
            x_range=(-10.0, 10.0),
            y_range=(-10.0, 10.0),
            boundary_margin=1.0,
        ),
        num_waypoints=64,
        enable_multi_skill=True,
        skill_sequence=["walk", "jump", "walk"],
    )

    # 2. 创建路径生成器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_generator = TerrainAwarePathGenerator(cfg, device)

    # 3. 准备输入数据
    env_ids = torch.tensor([0], device=device)
    start_pos = torch.tensor([[0.0, 0.0, 0.5]], device=device)
    start_yaw = torch.tensor([0.0], device=device)
    env_bounds = torch.tensor([[-10.0, 10.0, -10.0, 10.0]], device=device)

    # 4. 模拟 height_scanner 数据
    num_rays = 60
    ray_hits_w = torch.zeros(1, num_rays, 3, device=device)
    ray_hits_w[0, :20, 0] = torch.linspace(0, 2.0, 20, device=device)
    ray_hits_w[0, :20, 2] = 0.5
    ray_hits_w[0, 20:35, 0] = torch.linspace(2.0, 3.5, 15, device=device)
    ray_hits_w[0, 20:35, 2] = -0.3
    ray_hits_w[0, 35:, 0] = torch.linspace(3.5, 6.0, 25, device=device)
    ray_hits_w[0, 35:, 2] = 0.5

    terrain_data = {"height_scanner": ray_hits_w}

    # 5. 生成多技能路径
    path_result = path_generator.generate_multi_skill_path(
        env_ids=env_ids,
        skill_sequence=["walk", "jump", "walk"],
        start_pos=start_pos,
        start_yaw=start_yaw,
        env_bounds=env_bounds,
        terrain_data=terrain_data,
    )

    # 6. 打印结果
    print(f"✓ 生成了多技能序列路径")
    print(f"  - Waypoints shape: {path_result.waypoints.shape}")
    print(f"  - Total path length: {path_result.path_length.item():.2f}m")
    print(f"  - Number of segments: {path_result.num_segments.item()}")
    print(f"  - Segment skills: {path_result.segment_skills[0].cpu().numpy()}")
    print(f"  - Segment s0: {path_result.segment_s0[0].cpu().numpy()}")
    print(f"  - Segment s1: {path_result.segment_s1[0].cpu().numpy()}")
    print(f"  - Valid path: {path_result.valid.item()}")


def example_4_path_validation():
    """示例 4：路径验证"""
    print("\n" + "="*60)
    print("示例 4：路径验证")
    print("="*60)

    # 1. 配置路径生成器（使用严格的验证参数）
    cfg = PathGeneratorCfg(
        env_bounds=EnvironmentBounds(
            x_range=(-5.0, 5.0),
            y_range=(-5.0, 5.0),
        ),
        num_waypoints=64,
        validation_params=ValidationParams(
            max_waypoint_distance=0.2,
            max_heading_change=0.5,
            max_climb_angle=0.6,
            min_segment_length=0.5,
        ),
    )

    # 2. 创建路径生成器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_generator = TerrainAwarePathGenerator(cfg, device)

    # 3. 生成路径
    env_ids = torch.tensor([0], device=device)
    start_pos = torch.tensor([[0.0, 0.0, 0.5]], device=device)
    start_yaw = torch.tensor([0.0], device=device)
    env_bounds = torch.tensor([[-5.0, 5.0, -5.0, 5.0]], device=device)

    path_result = path_generator.generate_path(
        env_ids=env_ids,
        skill_name="walk",
        start_pos=start_pos,
        start_yaw=start_yaw,
        env_bounds=env_bounds,
    )

    # 4. 手动验证路径
    validation_result = path_generator.validator.validate(path_result)

    # 5. 打印验证结果
    print(f"✓ 路径验证完成")
    print(f"  - Valid: {validation_result.valid.item()}")
    print(f"  - Errors:")
    for error_type, error_mask in validation_result.errors.items():
        if error_mask.any():
            print(f"    - {error_type}: {error_mask.sum().item()} environments")
        else:
            print(f"    - {error_type}: ✓ Pass")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("多技能路径生成系统 - 使用示例")
    print("="*60)

    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA 不可用，使用 CPU")

    # 运行示例
    example_1_generate_walk_path()
    example_2_generate_jump_path()
    example_3_generate_multi_skill_path()
    example_4_path_validation()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
