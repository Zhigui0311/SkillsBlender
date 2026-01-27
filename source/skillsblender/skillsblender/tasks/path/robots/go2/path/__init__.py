# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##


gym.register(
    id="go2-path-flat-symmetry-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOWithSymmetryCfg",
        # "rsl_rl_with_symmetry_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOWithSymmetryCfg",
    },
)


# gym.register(
#     id="go2-path-flat-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg",
#         "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOCfg",
#     },
# )

gym.register(
    id="go2-path-flat-vel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathFlatCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOCfg",
    },
)

gym.register(
    id="go2-path-flat-vel-sym-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathFlatCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatVelPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-flat-vel-sym-test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathFlatCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2testPathFlatVelPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-flat-vel-sym-test-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2testPathFlatVelPPOWithSymmetryCfg",
    },
)

# -------flat play---------


gym.register(
    id="go2-path-flat-symmetry-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-flat-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOCfg",
    },
)
gym.register(
    id="go2-path-flat-vel-sym-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatVelPPOWithSymmetryCfg",
    },
)

# ----------------------jump---------------------------------------

gym.register(
    id="go2-path-jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_jump_cfg:Go2JumpEnvCfg", 
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-jump-curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_jump_cur_cfg:Go2JumpCurEnvCfg", 
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-jump-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_jump_cfg:Go2JumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-jump-curriculum-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_jump_cur_cfg:Go2JumpCurEnvCfg_PLAY", # Ensure this uses JumpPathCommandCfg
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)