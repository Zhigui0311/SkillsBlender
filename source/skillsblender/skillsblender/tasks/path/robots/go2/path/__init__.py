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
    id="go2-path-flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOCfg",
    },
)


gym.register(
    id="go2-path-flat-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOWithSymmetryCfg",
    },
)

gym.register(
    id="go2-path-flat-test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg",
        "rsl_rl_cfg_entry_point": f"skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2PathFlatPPOCfg",
    },
)




gym.register(
    id="go2-path-jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_flat_cfg:Go2PathEnvCfg", # Ensure this uses JumpPathCommandCfg
        "rsl_rl_cfg_entry_point": "skillsblender.tasks.path.agents.rsl_rl_ppo_cfg:GO2JumpPPOWithSymmetryCfg",
    },
)