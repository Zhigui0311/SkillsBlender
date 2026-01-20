from skillsblender.tasks.path.config.jump_env_cfg import JumpPathEnvCfg
from isaaclab.utils import configclass
import skillsblender.tasks.path.mdp as mdp

from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Go2JumpEnvCfg(JumpPathEnvCfg):
    """Unitree Go2 in gap terrain path following task configuration."""

    def __post_init__(self):

        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 观测缩放
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # 动作缩放
        self.actions.joint_pos_actoion.scale = 0.2

        # 基础奖励权重调整
        self.rewards.torques.weight = -0.0005
        self.rewards.action_rate.weight = -0.05
        self.rewards.joint_dev.weight = -0.2

        # 命令参数调整
        self.commands.path_tracking.ranges.num_waypoints = 100
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 5

        # ========================================================================
        # 跳跃专用奖励权重调整（根据训练效果微调）
        # ========================================================================
        # 增加跳跃高度追踪的重要性
        self.rewards.jump_height_tracking.weight = 4.0  # 默认3.0 → 4.0

        # 增加落地稳定性的重要性
        self.rewards.jump_landing_stability.weight = 2.5  # 默认2.0 → 2.5

        # 适度降低前向速度要求（避免过快导致失控）
        self.rewards.jump_forward_velocity.weight = 1.5  # 默认2.0 → 1.5


@configclass
class Go2JumpEnvCfg_PLAY(Go2JumpEnvCfg):
    """Unitree Go2 in gap terrain path following PLAY configuration for visualization."""

    def __post_init__(self):
        super().__post_init__()

        # 仿真参数（这些会在父类__post_init__中设置，但这里再设置一次确保正确）
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz control
        self.episode_length_s = 15.0  # 跳跃任务需要更长时间

        # 场景配置（少环境，便于观察）
        self.scene.num_envs = 8
        self.scene.env_spacing = 10.0  # 跳跃需要更大空间

        self.sim.render_interval = 2

        # 禁用地形难度递增（方便测试特定难度）
        self.scene.terrain.max_init_terrain_level = None

        # 禁用干扰事件
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 禁用课程学习
        self.curriculum = None

        # 物理材质
        self.sim.physics_material = self.scene.terrain.physics_material

        # 视角设置
        self.viewer.origin_type = "env"

        # 启用路径可视化（观察跳跃轨迹）
        self.commands.path_tracking.debug_vis = True

        # 禁用观测噪声（便于调试）
        self.observations.policy.enable_corruption = False
