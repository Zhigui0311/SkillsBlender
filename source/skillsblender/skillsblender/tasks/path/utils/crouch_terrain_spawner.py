"""
内容：额外 spawn “crouch roof(顶棚)” 和 “climb ramp(坡道/台阶)” 的工具函数。
放置位置：建议放到 env 层（scene setup / reset hooks）附近，例如：
  <your_repo>/tasks/locomotion/envs/skill_props_spawner.py
用法（推荐）：
  - 在 env reset 时按 env_ids 选择是否 spawn 某种 props，并把 skill_tag_id 写到 env.extras["skill_tag_ids"]。
  - 这样 PathCommand 不需要读 terrain 元数据，直接读 extras 即可（teacher/student 都可用，但部署可关闭）。
注意：
  - 下面用的是“伪接口”，因为你项目里具体 spawn API（Usd/Scene/AssetCfg）不一定一致。
  - 你把 spawn_prim() / set_pose() 两个函数替换成你工程实际的实现即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import torch

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


@dataclass
class RoofSpec:
    length: float = 2.0
    width: float = 1.2
    thickness: float = 0.08
    height: float = 0.35  # 离地高度（决定 crouch 目标 base_height_ref）
    offset_x: float = 1.4  # 相对 start 沿前向的距离
    offset_y: float = 0.0


@dataclass
class RampSpec:
    length: float = 2.5
    width: float = 1.2
    height: float = 0.4
    offset_x: float = 1.2
    offset_y: float = 0.0


def spawn_prim(scene, prim_path: str, shape: str, size: tuple[float, float, float]):
    """
    TODO: 用你项目的实际 spawn API 替换这里：
    - 可能是 usd_utils.create_prim
    - 可能是 isaaclab.assets 里的 rigid object cfg
    """
    raise NotImplementedError("replace spawn_prim with your project spawn api")


def set_pose(scene, prim_path: str, pos_w: torch.Tensor, quat_w: torch.Tensor):
    """
    TODO: 用你项目的实际 set pose API 替换这里
    """
    raise NotImplementedError("replace set_pose with your project pose api")


def spawn_crouch_roof_for_envs(env, env_ids: torch.Tensor, start_pos_w: torch.Tensor, fwd_dir_xy: torch.Tensor, spec: RoofSpec):
    """
    为每个 env spawn 一个 roof（顶棚），用于 crouch 训练。
    - start_pos_w: (B,3)
    - fwd_dir_xy: (B,2) 单位向量
    """
    scene = env.scene
    device = start_pos_w.device

    # roof 盒子尺寸：x=length, y=width, z=thickness
    size = (spec.length, spec.width, spec.thickness)

    for bi, eid in enumerate(env_ids.tolist()):
        prim_path = f"{env.cfg.scene.env_ns}/env_{eid}/Props/CrouchRoof"
        spawn_prim(scene, prim_path, shape="box", size=size)

        center_xy = start_pos_w[bi, :2] + fwd_dir_xy[bi] * spec.offset_x
        pos_w = torch.tensor([center_xy[0], center_xy[1] + spec.offset_y, spec.height], device=device)
        quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # 无旋转
        set_pose(scene, prim_path, pos_w, quat_w)


def spawn_climb_ramp_for_envs(env, env_ids: torch.Tensor, start_pos_w: torch.Tensor, fwd_dir_xy: torch.Tensor, spec: RampSpec):
    """
    为每个 env spawn 一个 ramp（坡道），用于 climb 训练。
    简化：用一个斜盒子或楔形体；如果你没有楔形体，就用 mesh。
    """
    scene = env.scene
    device = start_pos_w.device

    # 先用 box 占位，实际最好用 wedge mesh
    size = (spec.length, spec.width, spec.height)

    for bi, eid in enumerate(env_ids.tolist()):
        prim_path = f"{env.cfg.scene.env_ns}/env_{eid}/Props/ClimbRamp"
        spawn_prim(scene, prim_path, shape="ramp_or_box", size=size)

        center_xy = start_pos_w[bi, :2] + fwd_dir_xy[bi] * spec.offset_x
        pos_w = torch.tensor([center_xy[0], center_xy[1] + spec.offset_y, spec.height * 0.5], device=device)
        quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        set_pose(scene, prim_path, pos_w, quat_w)


@clone
def spawn_xform(
    prim_path: str,
    cfg: "XformCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an empty Xform prim (for parenting)."""
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class XformCfg(SpawnerCfg):
    """Spawner config for an empty Xform prim."""

    func: Callable = spawn_xform


def spawn_crouch_obstacles(*args, **kwargs):
    """
    Placeholder entrypoint for crouch obstacle spawning.
    Implement this with your project-specific spawn API when you want obstacles.
    Currently a no-op so importing crouch_env_cfg won't fail.
    """
    return None
