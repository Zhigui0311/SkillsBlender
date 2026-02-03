"""
内容：把 ParkourPathCommand 改成“可扩展插件式”的多技能段生成器：
- 不再写死 if/elif skill_tag == ...
- 你新增技能时，只需要：
  1) 在技能注册表里加 skill_name（你 base 里 SKILL_NAMES）
  2) 在下面注册一个 handler：register_skill_handler("vault", ...)
  3) （可选）在 terrain/extras 里产出对应的 tag

放置位置：替换你现有的 parkour_path_command.py（同路径）。
关键接口：
- SKILL_HANDLERS: dict[str, Callable]    # 技能名 -> 生成该段 params + skill_id 的函数
- register_skill_handler(name, fn)       # 新技能只用加一行注册
- default_segment_plan_for_tag()         # 仍可用，或者你换成 planner 输出 segments

注意：
- 这里仍保留“pre-walk/post-walk 可缺省”的逻辑
- crouch/climb 的障碍建议在 reset 时 spawn，并写 env.extras["skill_tag_str"] 或 ["skill_tag_ids"] 让这里选中
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from .base_path_command import SegmentPathCommand
from skillsblender.tasks.path.utils.terrain_to_segments import get_skill_tag_from_terrain, default_segment_plan_for_tag


# ----------------------------
# 可扩展：技能段 handler 注册表
# ----------------------------

# handler 签名：输入 (cmd, env_id_tensor(1,), s0, s1) -> None（内部调用 _append_segment）
SkillHandler = Callable[["ParkourPathCommand", torch.Tensor, float, float], None]
SKILL_HANDLERS: Dict[str, SkillHandler] = {}


def register_skill_handler(skill_name: str, handler: SkillHandler):
    SKILL_HANDLERS[skill_name] = handler


# ----------------------------
# 可扩展：tag_int -> skill_name 映射
# 你新增技能时，可以扩展这个映射或直接从 extras 读 skill_name
# ----------------------------
TAG_INT_TO_SKILL_NAME = {
    0: "walk",
    1: "jump",
    2: "stairs_up",
    3: "climb",
    4: "crouch",
}


# ----------------------------
# 默认 handler：walk（可复用）
# ----------------------------
def _append_walk(cmd: "ParkourPathCommand", env_id_1: torch.Tensor, s0: float, s1: float):
    p = torch.zeros(cmd.num_seg_params, device=cmd.device)
    p[cmd.SEG_PARAM["v_ref"]] = 1.0
    cmd._append_segment(env_id_1, cmd.SKILL_ID["walk"], s0, s1, p.unsqueeze(0))


# ----------------------------
# 默认 handler：jump/stairs/climb/crouch（你后续可替换参数逻辑）
# ----------------------------
def _append_jump(cmd: "ParkourPathCommand", env_id_1: torch.Tensor, s0: float, s1: float):
    p = torch.zeros(cmd.num_seg_params, device=cmd.device)
    p[cmd.SEG_PARAM["v_ref"]] = 1.0
    p[cmd.SEG_PARAM["jump_height_ref"]] = 0.6
    cmd._append_segment(env_id_1, cmd.SKILL_ID["jump"], s0, s1, p.unsqueeze(0))


def _append_stairs_up(cmd: "ParkourPathCommand", env_id_1: torch.Tensor, s0: float, s1: float):
    p = torch.zeros(cmd.num_seg_params, device=cmd.device)
    p[cmd.SEG_PARAM["v_ref"]] = 0.9
    p[cmd.SEG_PARAM["clearance_ref"]] = 0.12
    cmd._append_segment(env_id_1, cmd.SKILL_ID["stairs_up"], s0, s1, p.unsqueeze(0))


def _append_climb(cmd: "ParkourPathCommand", env_id_1: torch.Tensor, s0: float, s1: float):
    p = torch.zeros(cmd.num_seg_params, device=cmd.device)
    p[cmd.SEG_PARAM["v_ref"]] = 0.8
    p[cmd.SEG_PARAM["misc"]] = 0.25
    cmd._append_segment(env_id_1, cmd.SKILL_ID["climb"], s0, s1, p.unsqueeze(0))


def _append_crouch(cmd: "ParkourPathCommand", env_id_1: torch.Tensor, s0: float, s1: float):
    p = torch.zeros(cmd.num_seg_params, device=cmd.device)
    p[cmd.SEG_PARAM["v_ref"]] = 0.8
    p[cmd.SEG_PARAM["base_height_ref"]] = 0.25
    cmd._append_segment(env_id_1, cmd.SKILL_ID["crouch"], s0, s1, p.unsqueeze(0))


# 预注册默认技能（以后你新增技能，只要 register_skill_handler("vault", fn)）
register_skill_handler("walk", _append_walk)
register_skill_handler("jump", _append_jump)
register_skill_handler("stairs_up", _append_stairs_up)
register_skill_handler("climb", _append_climb)
register_skill_handler("crouch", _append_crouch)


# ----------------------------
# 主类
# ----------------------------
class ParkourPathCommand(SegmentPathCommand):
    """
    多技能 PathCommand（可扩展）：

    - 轨迹几何：_generate_trajectory()（你已有）
    - 技能选择：
        1) 优先 env.extras["skill_name"]（字符串），最适合你“crouch/climb 由 spawn props 决定”的场景
        2) 其次 env.extras["skill_tag_ids"]（int）
        3) 最后 terrain 元数据推断（get_skill_tag_from_terrain）
    - 段生成：根据 skill_name 查 SKILL_HANDLERS 生成主技能段（前后可选 walk）
    """

    def _get_skill_name_for_envs(self, env_ids: torch.Tensor) -> list[str]:
        extras = getattr(self.env, "extras", None)
        out: list[str] = []

        # 1) 字符串 skill_name（推荐你 spawn props 时直接写这个）
        if isinstance(extras, dict) and "skill_name" in extras:
            sn = extras["skill_name"]
            if isinstance(sn, (list, tuple)) and len(sn) >= int(env_ids.max()) + 1:
                for eid in env_ids.tolist():
                    out.append(str(sn[eid]))
                return out
            if isinstance(sn, torch.Tensor):
                # 不支持 tensor->string，这里跳过
                pass

        # 2) int tag ids
        if isinstance(extras, dict) and "skill_tag_ids" in extras:
            ids = extras["skill_tag_ids"]
            if isinstance(ids, torch.Tensor) and ids.numel() >= int(env_ids.max()) + 1:
                for eid in env_ids.tolist():
                    out.append(TAG_INT_TO_SKILL_NAME.get(int(ids[eid].item()), "walk"))
                return out

        # 3) terrain 元数据
        tag_int = get_skill_tag_from_terrain(self.env, env_ids)
        for t in tag_int.tolist():
            out.append(TAG_INT_TO_SKILL_NAME.get(int(t), "walk"))
        return out

    def _resample_command(self, env_ids: torch.Tensor):
        # 1) 生成轨迹几何（xyz+yaw）
        pos_traj, yaw_traj = self._generate_trajectory(env_ids)
        self.pos_path_w[env_ids] = pos_traj
        self.heading_path_w[env_ids] = yaw_traj
        self.pos_path_w_cur[env_ids] = self.pos_path_w[env_ids, 0, :]
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False

        # 2) 生成主技能段窗口（s0/s1）
        # 你也可以换成：planner 输出 segments（而不是 default_segment_plan_for_tag）
        tag_int = get_skill_tag_from_terrain(self.env, env_ids)
        per_env_plans = default_segment_plan_for_tag(tag_int)

        # 3) 清段表
        self._clear_segments(env_ids)

        # 4) 估算路径长度（沿 XY 的近似）
        start_xy = self.pos_path_w[env_ids, 0, :2]
        end_xy = self.pos_path_w[env_ids, -1, :2]
        total_len = torch.norm(end_xy - start_xy, dim=-1)  # (B,)

        # 5) 获取每个 env 的 skill_name（优先 extras，可用于 crouch/climb props）
        skill_names = self._get_skill_name_for_envs(env_ids)

        # 6) 写段表：walk/skill/walk（可缺省），且主技能段由 handler 控制（可扩展）
        for bi, eid in enumerate(env_ids.tolist()):
            env_id_1 = torch.tensor([eid], device=self.device)
            L = float(total_len[bi].item())

            plans = per_env_plans[bi]
            if len(plans) == 0:
                # walk-only
                _append_walk(self, env_id_1, 0.0, L)
                continue

            seg = plans[0]
            s0 = float(max(0.0, seg.s0))
            s1 = float(min(L, seg.s1))

            # pre-walk（可缺省）
            if s0 > 1e-3:
                _append_walk(self, env_id_1, 0.0, s0)

            # main skill：用可扩展 handler
            skill_name = skill_names[bi]
            handler = SKILL_HANDLERS.get(skill_name, _append_walk)
            handler(self, env_id_1, s0, s1)

            # post-walk（可缺省）
            if s1 < L - 1e-3:
                _append_walk(self, env_id_1, s1, L)