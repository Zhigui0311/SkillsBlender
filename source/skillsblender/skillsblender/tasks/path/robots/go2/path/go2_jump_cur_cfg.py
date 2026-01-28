from skillsblender.tasks.path.config.jump_env_cfg import JumpPathEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import skillsblender.tasks.path.mdp as mdp
from .go2_jump_cfg import Go2JumpEnvCfg
import isaaclab.terrains as terrain_gen
from skillsblender.assets.robots.unitree import UNITREE_GO2_CFG


CUR_JUMP_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=16,
    num_cols=24,
    curriculum= False,
    difficulty_range=(0.0,1.0),
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_width = 0.5,
    sub_terrains={
        # 1. 基础平地 - 增加初始比例以便更好地学习基础跳跃
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.50),  # Updated: 增加到50%

        # 2. 窄沟壑 - 调整为更温和的初始范围
        "narrow_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.40,  # Updated: 调整比例
            gap_width_range=(0.2, 0.3),  # Updated: 更窄的初始范围
            platform_width=2.0,
        ),

        # 3. 宽沟壑：用于进阶跳跃训练
        "wide_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.10,  # Updated: 减少初始比例
            gap_width_range=(0.3, 0.5),  # Updated: 调整为中等宽度
            platform_width=2.5,
        ),
    },
)

@configclass
class JumpCurriculumCfg:
    """Jump-specific curriculum learning configuration."""

    # 沟壑宽度课程: 从窄沟壑逐步增加到宽沟壑
    jump_gap_width = CurrTerm(
        func=mdp.curriculum_jump_gap_width,
        params={
            "reward_threshold": 70.0,  # Updated: 提高阈值从50.0到70.0
            "initial_gap_range": (0.2, 0.3),  # Updated: 更温和的起始范围
            "intermediate_gap_range": (0.3, 0.5),  # New: 中级范围
            "final_gap_range": (0.5, 0.8),  # Updated: 降低最大宽度
            "step_size": 0.1
        }
    )

    # 跳跃高度要求课程: 逐步提高跳跃高度
    jump_height_requirement = CurrTerm(
        func=mdp.curriculum_jump_height_requirement,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 80.0,  # Updated: 提高阈值从60.0到80.0
            "initial_height": 0.20,  # Updated: 降低初始高度
            "final_height": 0.40,  # Updated: 降低最终高度
            "step_size": 0.03  # Updated: 减小步长
        }
    )

    # 地形混合课程: 逐步增加跳跃地形比例
    jump_terrain_mix = CurrTerm(
        func=mdp.curriculum_jump_terrain_mix,
        params={
            "reward_threshold": 60.0,  # Updated: 提高阈值从40.0到60.0
            "max_jump_proportion": 0.95
        }
    )

    # 速度要求课程: 逐步提高跳跃时的速度要求
    jump_speed_requirement = CurrTerm(
        func=mdp.curriculum_jump_speed_requirement,
        params={
            "reward_threshold": 75.0,  # Updated: 提高阈值从65.0到75.0
            "initial_speed": 0.8,  # Updated: 降低初始速度
            "final_speed": 2.0,
            "step_size": 0.08  # Updated: 减小步长
        }
    )

@configclass
class Go2JumpCurEnvCfg(Go2JumpEnvCfg):
    """Unitree Go2 in gap terrain path following task configuration with curriculum."""
    
    curriculum: JumpCurriculumCfg = JumpCurriculumCfg()
    
    def __post_init__(self):

        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator= CUR_JUMP_TERRAIN_CFG 
        
        # Command parameters adjustment
        self.commands.path_tracking.ranges.num_waypoints = 100
        self.commands.path_tracking.ranges.num_lookahead_waypoints = 5
        self.commands.path_tracking.jump_params.jump_height = 0.45

        # Configurable jump trajectory parameters
        self.commands.path_tracking.jump_params.gap_width_threshold = 0.55
        self.commands.path_tracking.jump_params.narrow_gap_endpoint_extension = 1.5
        self.commands.path_tracking.jump_params.wide_gap_endpoint_extension = 2.0
        self.commands.path_tracking.jump_params.narrow_gap_takeoff_margin = 0.4
        self.commands.path_tracking.jump_params.wide_gap_takeoff_margin = 0.5
        self.commands.path_tracking.jump_params.landing_margin = 0.5
        self.commands.path_tracking.jump_params.post_jump_distance = 0.0  # 0 = stop at landing
        
        self.rewards.flat_orientation.weight = -0.1
        self.rewards.base_lin_vel_z.weight = -0.1
        self.rewards.base_ang_vel_xy.weight = -0.02
        self.rewards.stalling_penalty.weight = -5.0  # Updated: 增强停滞惩罚

        # Jump-specific rewards with updated weights
        self.rewards.jump_forward_velocity.weight = 2.5  # 基础权重，将在接近gap时动态调整
        self.rewards.jump_clearance.weight = 3.0  # 强化间隙清除奖励
        self.rewards.jump_air_time.weight = 2.0
        self.rewards.jump_height_tracking.weight = 4.0  # 将根据距离动态调整

        # New rewards for fixing spinning behavior
        self.rewards.spinning_penalty.weight = -3.0  # 强力惩罚打转行为
        self.rewards.approach_momentum_reward.weight = 2.0  # 鼓励接近时的动量
        self.rewards.consistency_reward.weight = 1.5  # 鼓励持续前进 

        # Others
        # self.rewards.air_time_variance.weight = -4.0
        # self.rewards.feet_acc.weight = -2e-6
        
        if self.__class__.__name__ == "Go2JumpCurEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class Go2JumpCurEnvCfg_PLAY(Go2JumpCurEnvCfg):
    """Unitree Go2 in gap terrain path following PLAY configuration for visualization."""

    def __post_init__(self):
        super().__post_init__()

        # 仿真参数（这些会在父类__post_init__中设置，但这里再设置一次确保正确）
        self.sim.dt = 0.005 
        self.decimation = 4  
        self.episode_length_s = 15.0  
        self.sim.render_interval = 2

        # 场景配置（
        self.scene.num_envs = 32
        self.scene.env_spacing = 5.0  
        
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # 禁用地形难度递增（方便测试特定难度）
        self.scene.terrain.max_init_terrain_level = None

        # 禁用干扰事件
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum = None

        # # 物理材质
        # self.sim.physics_material = self.scene.terrain.physics_material

        # # 视角设置
        # self.viewer.origin_type = "env"

        # 启用路径可视化（观察跳跃轨迹）
        self.commands.path_tracking.debug_vis = True
        # 禁用观测噪声（便于调试）
        self.observations.policy.enable_corruption = False
        
