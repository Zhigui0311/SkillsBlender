# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry for GO2 path tracking task (policy/critic obs + actions).

This version:
- NEVER touches `env` at module import time.
- Dynamically reads num_lookahead_waypoints and slice dims from the command cfg.
- Applies left-right symmetry to:
  * path_slice: y -> -y, yaw_rel -> -yaw_rel (x,z unchanged)
  * base_ang_vel / projected_gravity: same as IsaacLab convention
  * joints + actions: swap left/right legs + sign flips on hip joints
"""

# from __future__ import annotations

# import torch
# from tensordict import TensorDict
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv

# __all__ = ["compute_symmetric_states"]


# @torch.no_grad()
# def compute_symmetric_states(
#     env: ManagerBasedRLEnv,
#     obs: TensorDict | None = None,
#     actions: torch.Tensor | None = None,
#     command_name: str = "path_tracking",
# ):
#     """Augments observations/actions using left-right symmetry (x forward, y left).

#     Args:
#         env: ManagerBasedRLEnv (runtime object injected by runner).
#         obs: TensorDict with keys "policy" and optionally "critic".
#         actions: (B, act_dim)
#         command_name: command term name used to fetch PathCommand cfg.

#     Returns:
#         (obs_aug, actions_aug) where batch is doubled if input is not None.
#     """
    
#     # from pathlib import Path
#     # p = Path("/tmp/go2_sym_calls.txt")
#     # n = int(p.read_text()) if p.exists() else 0
#     # p.write_text(str(n + 1))
 



#     # ---- get dynamic path dims from command cfg (runtime, safe) ----
#     cmd = env.command_manager.get_term(command_name)  # PathCommand
#     W = int(cmd.cfg.ranges.num_lookahead_waypoints)
#     PATH_SLICE = int(cmd.cfg.slice_nums)  # = 4 * W

#     # ---------------- observations ----------------
#     if obs is not None:
#         B = obs.batch_size[0]
#         obs_aug = obs.repeat(2)

#         # policy
#         policy = obs["policy"]
#         obs_aug["policy"][:B] = policy
#         obs_aug["policy"][B : 2 * B] = _transform_policy_obs_left_right(
#             policy,
#             PATH_SLICE=PATH_SLICE,
#             W=W,
#         )

#         # critic (optional)
#         if "critic" in obs.keys():
#             critic = obs["critic"]
#             obs_aug["critic"][:B] = critic
#             obs_aug["critic"][B : 2 * B] = _transform_critic_obs_left_right(
#                 critic,
#                 PATH_SLICE=PATH_SLICE,
#                 W=W,
#             )
#     else:
#         obs_aug = None

#     # ---------------- actions ----------------
#     if actions is not None:
#         B = actions.shape[0]
#         actions_aug = torch.zeros(B * 2, actions.shape[1], device=actions.device)
#         actions_aug[:B] = actions
#         actions_aug[B : 2 * B] = _transform_actions_left_right(actions)
#     else:
#         actions_aug = None

#     return obs_aug, actions_aug


import torch
from tensordict import TensorDict

@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None, command_name: str = "path_tracking"):
    # ---- infer W from obs dims (no env access needed) ----
    PATH_SLICE = None
    W = None

    if obs is not None and "policy" in obs.keys():
        policy_dim = obs["policy"].shape[-1]
        PATH_SLICE = policy_dim - 42  # 3+3+12+12+12
        if PATH_SLICE % 4 != 0 or PATH_SLICE <= 0:
            raise RuntimeError(f"Bad PATH_SLICE inferred from policy_dim={policy_dim}: PATH_SLICE={PATH_SLICE}")
        W = PATH_SLICE // 4
    elif obs is not None and "critic" in obs.keys():
        critic_dim = obs["critic"].shape[-1]
        PATH_SLICE = critic_dim - 45  # 3+3+3+12+12+12
        if PATH_SLICE % 4 != 0 or PATH_SLICE <= 0:
            raise RuntimeError(f"Bad PATH_SLICE inferred from critic_dim={critic_dim}: PATH_SLICE={PATH_SLICE}")
        W = PATH_SLICE // 4
    else:
        # If only actions are passed, you can hardcode W or skip path transform.
        # Here we just skip path transforms (still do joint swap for actions).
        PATH_SLICE, W = 0, 0

    # ---------------- observations ----------------
    if obs is not None:
        B = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        # policy
        if "policy" in obs.keys():
            policy = obs["policy"]
            obs_aug["policy"][:B] = policy
            obs_aug["policy"][B:2*B] = _transform_policy_obs_left_right(policy, PATH_SLICE=PATH_SLICE, W=W)

        # critic
        if "critic" in obs.keys():
            critic = obs["critic"]
            obs_aug["critic"][:B] = critic
            obs_aug["critic"][B:2*B] = _transform_critic_obs_left_right(critic, PATH_SLICE=PATH_SLICE, W=W)
    else:
        obs_aug = None

    # ---------------- actions ----------------
    if actions is not None:
        B = actions.shape[0]
        actions_aug = torch.zeros(B * 2, actions.shape[1], device=actions.device)
        actions_aug[:B] = actions
        actions_aug[B:2*B] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug

# =============================================================================
# Obs transforms (LEFT-RIGHT)
# =============================================================================

def _transform_policy_obs_left_right(obs: torch.Tensor, *, PATH_SLICE: int, W: int) -> torch.Tensor:
    """Policy obs layout (from your ObservationsCfg.PolicyCfg order):
        0:PATH_SLICE          -> path_slice (flattened)
        PATH_SLICE: +3        -> base_ang_vel
        +3: +3                -> projected_gravity
        +3: +12               -> joint_pos
        +12:+12               -> joint_vel
        +12:+12               -> actions
    """
    obs = obs.clone()
    device = obs.device

    # Build slices dynamically
    s_path = slice(0, PATH_SLICE)
    s_ang = slice(s_path.stop, s_path.stop + 3)
    s_grav = slice(s_ang.stop, s_ang.stop + 3)
    s_jpos = slice(s_grav.stop, s_grav.stop + 12)
    s_jvel = slice(s_jpos.stop, s_jpos.stop + 12)
    s_act = slice(s_jvel.stop, s_jvel.stop + 12)

    # 1) path_slice left-right: (x,y,z,yaw_rel) for each waypoint
    obs[:, s_path] = _transform_path_slice_left_right(obs[:, s_path], W=W)

    # 2) base_ang_vel, projected_gravity: follow IsaacLab convention for LR mirror
    obs[:, s_ang] *= torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[:, s_grav] *= torch.tensor([1.0, -1.0, 1.0], device=device)

    # 3) joints + last action: swap legs + sign flips
    obs[:, s_jpos] = _switch_go2_joints_left_right(obs[:, s_jpos])
    obs[:, s_jvel] = _switch_go2_joints_left_right(obs[:, s_jvel])
    obs[:, s_act] = _switch_go2_joints_left_right(obs[:, s_act])

    return obs


def _transform_critic_obs_left_right(obs: torch.Tensor, *, PATH_SLICE: int, W: int) -> torch.Tensor:
    """Critic obs layout (from your ObservationsCfg.CriticCfg order):
        0:PATH_SLICE          -> path_slice
        PATH_SLICE:+3         -> base_lin_vel
        +3:+3                 -> base_ang_vel
        +3:+3                 -> projected_gravity
        +3:+12                -> joint_pos
        +12:+12               -> joint_vel
        +12:+12               -> actions
    """
    obs = obs.clone()
    device = obs.device

    s_path = slice(0, PATH_SLICE)
    s_lin = slice(s_path.stop, s_path.stop + 3)
    s_ang = slice(s_lin.stop, s_lin.stop + 3)
    s_grav = slice(s_ang.stop, s_ang.stop + 3)
    s_jpos = slice(s_grav.stop, s_grav.stop + 12)
    s_jvel = slice(s_jpos.stop, s_jpos.stop + 12)
    s_act = slice(s_jvel.stop, s_jvel.stop + 12)

    # 1) path_slice LR
    obs[:, s_path] = _transform_path_slice_left_right(obs[:, s_path], W=W)

    # 2) base_lin_vel LR: y flips, x/z same (common convention)
    obs[:, s_lin] *= torch.tensor([1.0, -1.0, 1.0], device=device)

    # 3) base_ang_vel, projected_gravity
    obs[:, s_ang] *= torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[:, s_grav] *= torch.tensor([1.0, -1.0, 1.0], device=device)

    # 4) joints + last action
    obs[:, s_jpos] = _switch_go2_joints_left_right(obs[:, s_jpos])
    obs[:, s_jvel] = _switch_go2_joints_left_right(obs[:, s_jvel])
    obs[:, s_act] = _switch_go2_joints_left_right(obs[:, s_act])

    return obs


def _transform_path_slice_left_right(path_flat: torch.Tensor, *, W: int) -> torch.Tensor:
    """path_flat: (B, 4*W), each waypoint [x, y, z, yaw_rel] in BODY frame.

    Left-right mirror (x forward, y left):
      y -> -y
      yaw_rel -> -yaw_rel
      x,z unchanged
    """
    B = path_flat.shape[0]
    # (B, W, 4)
    path = path_flat.view(B, W, 4).clone()

    # y flip
    path[..., 1] *= -1.0
    # yaw_rel flip
    path[..., 3] *= -1.0

    return path.view(B, 4 * W)



# Action transforms (LEFT-RIGHT)


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_go2_joints_left_right(actions[:])
    return actions


# =============================================================================
# Joint mapping helpers
# =============================================================================


def _switch_go2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right legs and flip hip signs.

    Works for shape (..., 12).
    """    
    #针对 unitree.py 定义的 [FR, FL, RR, RL] 顺序进行交换，之前的顺序不对
    joint_data_switched = torch.zeros_like(joint_data)

    # 交换 FR (0,1,2) 和 FL (3,4,5)
    joint_data_switched[..., [0, 1, 2]] = joint_data[..., [3, 4, 5]]
    joint_data_switched[..., [3, 4, 5]] = joint_data[..., [0, 1, 2]]
    
    # 交换 RR (6,7,8) 和 RL (9,10,11)
    joint_data_switched[..., [6, 7, 8]] = joint_data[..., [9, 10, 11]]
    joint_data_switched[..., [9, 10, 11]] = joint_data[..., [6, 7, 8]]

    # 翻转胯部关节（Hip）的符号：索引是 0, 3, 6, 9
    joint_data_switched[..., [0, 3, 6, 9]] *= -1.0
    return joint_data_switched


# def _switch_go2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
#     """Swap left/right legs and flip hip signs.

#     Works for shape (..., 12).
#     """
#     joint_data_switched = torch.zeros_like(joint_data)

#     # left <-- right
#     joint_data_switched[..., [0, 4, 8, 2, 6, 10]] = joint_data[..., [1, 5, 9, 3, 7, 11]]
#     # right <-- left
#     joint_data_switched[..., [1, 5, 9, 3, 7, 11]] = joint_data[..., [0, 4, 8, 2, 6, 10]]

#     # flip sign of hip joints
#     joint_data_switched[..., [0, 1, 2, 3]] *= -1.0
#     return joint_data_switched
