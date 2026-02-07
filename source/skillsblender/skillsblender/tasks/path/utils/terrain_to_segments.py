"""
内容：从仿真 terrain 元数据推断“当前 env 的地形类型”，并生成对应的默认 segment 计划（主技能段的 s0/s1）。
放置位置：建议放到 commands/ 或 planners/ 目录，例如：
  <your_repo>/tasks/locomotion/commands/terrain_to_segments.py
用法：
  tag_int = get_skill_tag_from_terrain(env, env_ids)
  plans = default_segment_plan_for_tag(tag_int)
  然后 PathCommand 里按 plans 调 _append_segment(...)。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .terrain import (
    SKILL_TAG_WALK,
    SKILL_TAG_JUMP,
    SKILL_TAG_STAIRS,
    SKILL_TAG_PLATFORM,
    SKILL_TAG_CLIMB,
    SKILL_TAG_CROUCH,
    subterrain_name_to_skill_tag,
)


@dataclass
class SkillSegmentPlan:
    skill_tag: str
    s0: float
    s1: float


def _try_get_subterrain_names(env) -> list[str] | None:
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    if terrain is None:
        return None

    for attr in ("subterrain_names", "sub_terrain_names", "terrain_names", "terrain_types"):
        names = getattr(terrain, attr, None)
        if isinstance(names, (list, tuple)) and len(names) and isinstance(names[0], str):
            return list(names)

    mapping = getattr(terrain, "terrain_type_to_name", None)
    if isinstance(mapping, dict) and mapping:
        max_k = max(int(k) for k in mapping.keys())
        out = [""] * (max_k + 1)
        for k, v in mapping.items():
            out[int(k)] = str(v)
        return out

    return None


def _try_get_subterrain_type_ids(env, env_ids: torch.Tensor) -> torch.Tensor | None:
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    if terrain is None:
        return None

    for attr in ("terrain_type_ids", "terrain_ids", "subterrain_ids", "sub_terrain_ids"):
        ids = getattr(terrain, attr, None)
        if isinstance(ids, torch.Tensor) and ids.numel() >= int(env_ids.max()) + 1:
            return ids[env_ids].to(device=env_ids.device)

    return None


def get_skill_tag_from_terrain(env, env_ids: torch.Tensor) -> torch.Tensor:
    """
    返回 per-env 的 skill_tag_id（int），用于固定宽度 buffer。
    词表：
      0 walk, 1 jump, 2 stairs, 3 platform, 4 crouch, 5 climb
    """
    device = env_ids.device
    tag_int = torch.zeros(len(env_ids), dtype=torch.long, device=device)  # 默认 walk

    # 若你在 reset 时已经写入 env.extras["skill_tag_ids"]，这里优先读它
    extras = getattr(env, "extras", None)
    if isinstance(extras, dict) and "skill_tag_ids" in extras:
        ids = extras["skill_tag_ids"]
        if isinstance(ids, torch.Tensor) and ids.numel() >= int(env_ids.max()) + 1:
            return ids[env_ids].to(device=device, dtype=torch.long)

    names = _try_get_subterrain_names(env)
    type_ids = _try_get_subterrain_type_ids(env, env_ids)
    if names is None or type_ids is None:
        return tag_int

    for i in range(len(env_ids)):
        tid = int(type_ids[i].item())
        if tid < 0 or tid >= len(names):
            continue
        tag = subterrain_name_to_skill_tag(names[tid])
        if tag == SKILL_TAG_JUMP:
            tag_int[i] = 1
        elif tag == SKILL_TAG_STAIRS:
            tag_int[i] = 2
        elif tag == SKILL_TAG_PLATFORM:
            tag_int[i] = 3
        elif tag == SKILL_TAG_CLIMB:
            tag_int[i] = 5
        elif tag == SKILL_TAG_CROUCH:
            tag_int[i] = 4
        else:
            tag_int[i] = 0

    return tag_int


def default_segment_plan_for_tag(tag_int: torch.Tensor) -> list[list[SkillSegmentPlan]]:
    """
    兜底计划：每个 env 生成一个“主技能段”窗口（s0/s1，单位米，沿 planned forward 的距离）。
    walk: 空列表（表示 walk-only）
    """
    plans: list[list[SkillSegmentPlan]] = []
    for t in tag_int.tolist():
        if t == 2:  # stairs
            plans.append([SkillSegmentPlan(SKILL_TAG_STAIRS, 1.0, 3.0)])
        elif t == 3:  # platform
            plans.append([SkillSegmentPlan(SKILL_TAG_PLATFORM, 1.0, 3.0)])
        elif t == 4:  # crouch（通常需要额外 props；这里只给一个窗口）
            plans.append([SkillSegmentPlan(SKILL_TAG_CROUCH, 1.0, 3.0)])
        elif t == 1:  # jump（真实 jump 更建议用 height_scanner 检 gap）
            plans.append([SkillSegmentPlan(SKILL_TAG_JUMP, 1.2, 2.2)])
        elif t == 5:  # climb
            plans.append([SkillSegmentPlan(SKILL_TAG_CLIMB, 1.0, 3.0)])
        else:
            plans.append([])
    return plans
