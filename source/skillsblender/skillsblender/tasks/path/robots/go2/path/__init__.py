# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##


gym.register(
    id="go2-path-jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_jump_cfg:Go2JumpEnvCfg", 
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

# ----------------------jump---------------------------------------

gym.register(
    id="go2-path-jump-curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_jump_cur_cfg:Go2JumpCurEnvCfg", 
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-jump-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_jump_cfg:Go2JumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-jump-curriculum-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_jump_cur_cfg:Go2JumpCurEnvCfg_PLAY", # Ensure this uses JumpPathCommandCfg
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

# curriculum-enabled (v1)
gym.register(
    id="go2-path-jump-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_jump_cur_cfg:Go2JumpCurEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-jump-v1-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_jump_cur_cfg:Go2JumpCurEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)


# ==============================================================================
# New Skill Environments Registration
# ==============================================================================

# ---------------------- Walk ----------------------
gym.register(
    id="go2-path-walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_walk_cfg:Go2WalkEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2WalkPPOCfg",
    },
)

gym.register(
    id="go2-path-walk-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_walk_cfg:Go2WalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2WalkPPOCfg",
    },
)

gym.register(
    id="go2-path-walk-slope-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_walk_slope_cfg:Go2WalkSlopeEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2WalkSlopePPOCfg",
    },
)

gym.register(
    id="go2-path-walk-slope-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_walk_slope_cfg:Go2WalkSlopeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2WalkSlopePPOCfg",
    },
)

# ---------------------- Stairs ----------------------
gym.register(
    id="go2-path-stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_stairs_cfg:Go2StairsEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2StairsPPOCfg",
    },
)

gym.register(
    id="go2-path-stairs-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_stairs_cfg:Go2StairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2StairsPPOCfg",
    },
)

gym.register(
    id="go2-path-stairs-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_stairs_cfg:Go2StairsCurEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2StairsPPOCfg",
    },
)

gym.register(
    id="go2-path-stairs-v1-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_stairs_cfg:Go2StairsCurEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2StairsPPOCfg",
    },
)

# ---------------------- Platform ----------------------
gym.register(
    id="go2-path-platform-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_platform_cfg:Go2PlatformEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PlatformPPOCfg",
    },
)

gym.register(
    id="go2-path-platform-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_platform_cfg:Go2PlatformEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PlatformPPOCfg",
    },
)

gym.register(
    id="go2-path-platform-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_platform_cfg:Go2PlatformCurEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PlatformPPOCfg",
    },
)

gym.register(
    id="go2-path-platform-v1-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_platform_cfg:Go2PlatformCurEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PlatformPPOCfg",
    },
)

# ---------------------- Climb ----------------------
gym.register(
    id="go2-path-climb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_climb_cfg:Go2ClimbEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ClimbPPOCfg",
    },
)

gym.register(
    id="go2-path-climb-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_climb_cfg:Go2ClimbEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ClimbPPOCfg",
    },
)

# ---------------------- Crouch ----------------------
gym.register(
    id="go2-path-crouch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_crouch_cfg:Go2CrouchEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2CrouchPPOCfg",
    },
)

gym.register(
    id="go2-path-crouch-curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_crouch_cfg:Go2CrouchCurEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2CrouchPPOCfg",
    },
)

gym.register(
    id="go2-path-crouch-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_crouch_cfg:Go2CrouchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2CrouchPPOCfg",
    },
)

gym.register(
    id="go2-path-crouch-curriculum-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_crouch_cfg:Go2CrouchCurEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2CrouchPPOCfg",
    },
)

gym.register(
    id="go2-path-crouch-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_crouch_cfg:Go2CrouchCurEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2CrouchPPOCfg",
    },
)

gym.register(
    id="go2-path-crouch-v1-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_crouch_cfg:Go2CrouchCurEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2CrouchPPOCfg",
    },
)

# ---------------------- Blender (Multi-Skill) ----------------------
gym.register(
    id="go2-path-blener-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_blener_cfg:Go2BlenerEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2BlenerPPOCfg",
    },
)

gym.register(
    id="go2-path-blener-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_blener_cfg:Go2BlenerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2BlenerPPOCfg",
    },
)

# ---------------------- Virtual Skills (Instruction Hallucination) ----------------------
gym.register(
    id="go2-path-virtual-jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.virtual.go2_vjump_cfg:Go2VJumpEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2VirtualJumpPPOCfg",
    },
)

gym.register(
    id="go2-path-virtual-jump-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.virtual.go2_vjump_cfg:Go2VJumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2VirtualJumpPPOCfg",
    },
)

gym.register(
    id="go2-path-virtual-crouch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.virtual.go2_vcrouch_cfg:Go2VCrouchEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2VirtualCrouchPPOCfg",
    },
)

gym.register(
    id="go2-path-virtual-crouch-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.virtual.go2_vcrouch_cfg:Go2VCrouchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2VirtualCrouchPPOCfg",
    },
)

# ==============================================================================
# Parkour (Multi-Skill Fusion) Environments
# ==============================================================================

# ---------------------- Direct Training (From Scratch) ----------------------
gym.register(
    id="go2-parkour-direct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_parkour_cfg:Go2ParkourDirectEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ParkourDirectPPOCfg",
    },
)

gym.register(
    id="go2-parkour-direct-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_parkour_cfg:Go2ParkourDirectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ParkourDirectPPOCfg",
    },
)

# ---------------------- Distillation Training ----------------------
gym.register(
    id="go2-parkour-distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_parkour_cfg:Go2ParkourDistillEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ParkourDistillPPOCfg",
    },
)

gym.register(
    id="go2-parkour-distill-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_parkour_cfg:Go2ParkourDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ParkourDistillPPOCfg",
    },
)

# ---------------------- CNN Path Encoder ----------------------
gym.register(
    id="go2-parkour-cnn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_parkour_cfg:Go2ParkourCNNEnvCfg",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ParkourCNNPPOCfg",
    },
)

gym.register(
    id="go2-parkour-cnn-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain.go2_parkour_cfg:Go2ParkourCNNEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2ParkourCNNPPOCfg",
    },
)
