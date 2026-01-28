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
        # 1. 基础平地
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20), #但是我的训练的逻辑是规划路径然后训练
        
        # 2. 窄沟壑
        "narrow_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.5,
            gap_width_range=(0.3, 0.5),               
            platform_width=2.0,           
        ),
        
        # 3. 宽沟壑：用于进阶跳跃训练 (占比 40%)
        "wide_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.3,
            gap_width_range=(0.6, 1.0),  # 沟壑宽度 0.6m - 1.0m (挑战 Go2 极限)
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
            "reward_threshold": 50.0,           # 平均奖励超过此值时增加难度
            "initial_gap_range": (0.3, 0.5),   # 初始沟壑宽度范围
            "final_gap_range": (0.6, 1.0),     # 最终沟壑宽度范围
            "step_size": 0.1                    # 每次增加步长
        }
    )

    # 跳跃高度要求课程: 逐步提高跳跃高度
    jump_height_requirement = CurrTerm(
        func=mdp.curriculum_jump_height_requirement,
        params={
            "command_name": "path_tracking",
            "reward_threshold": 60.0,
            "initial_height": 0.25,            
            "final_height": 0.45,              
            "step_size": 0.05
        }
    )

    # 地形混合课程: 逐步增加跳跃地形比例
    jump_terrain_mix = CurrTerm(
        func=mdp.curriculum_jump_terrain_mix,
        params={
            "reward_threshold": 40.0,
            "max_jump_proportion": 0.95         # 跳跃地形的最大比例
        }
    )

    # 速度要求课程: 逐步提高跳跃时的速度要求 (可选)
    # jump_speed_requirement = CurrTerm(
    #     func=mdp.curriculum_jump_speed_requirement,
    #     params={
    #         "reward_threshold": 65.0,
    #         "initial_speed": 1.0,
    #         "final_speed": 2.0,
    #         "step_size": 0.1
    #     }
    # )

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
        
        self.rewards.flat_orientation.weight = -0.1
        self.rewards.base_lin_vel_z.weight = -0.1
        self.rewards.base_ang_vel_xy.weight = -0.02
        self.rewards.stalling_penalty.weight = -4.0

        self.rewards.jump_forward_velocity.weight = 2.5  # encourage commit to jump
        self.rewards.jump_clearance.weight = 3.0  # stronger reward to clear gaps
        self.rewards.jump_air_time.weight = 2.0 

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
        
