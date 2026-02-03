# PathCommand 代码检查报告

## ✅ 现有 PathCommand 文件检查

### 1. **FlatPathCommand** (`flat_path_command.py`)

#### 问题：
1. ❌ **重复的 import** - 第 1-20 行和第 22-31 行重复导入
2. ⚠️ **代码冗余** - 有未使用的导入（第 9-18 行）

#### 正确的部分：
- ✅ 继承自 `SegmentPathCommand`
- ✅ 实现了 `_resample_command()` 方法
- ✅ 正确使用了 segment table
- ✅ 路径生成逻辑正确（直线路径）

#### 建议修复：
```python
# 删除第 1-20 行的重复导入
# 只保留第 22-31 行的导入即可
```

---

### 2. **StairsPathCommand** (`stairs_path_command.py`)

#### 检查结果：
- ✅ **完全正确**
- ✅ 继承自 `SegmentPathCommand`
- ✅ 实现了楼梯路径生成（台阶式高度变化）
- ✅ 支持上楼和下楼（`stairs_up` 参数）
- ✅ 正确使用了 segment table（walk + stairs + walk）
- ✅ 参数使用正确（`v_ref`, `clearance_ref`, `misc`）

**无需修改**

---

### 3. **ClimbPathCommand** (`climb_path_command.py`)

#### 检查结果：
- ✅ **完全正确**
- ✅ 继承自 `SegmentPathCommand`
- ✅ 实现了斜坡攀爬路径（线性高度增加）
- ✅ 正确使用了 segment table（walk + climb + walk）
- ✅ 参数使用正确

**无需修改**

---

### 4. **CrouchPathCommand** (`crouch_path_command.py`)

#### 问题：
- ❌ **文件几乎为空** - 只有 1 行代码
- ❌ **未实现** - 需要完整实现

#### 需要实现的内容：
```python
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_path_command import SegmentPathCommand
from .path_command_cfg import PathCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CrouchPathCommand(SegmentPathCommand):
    """Path with a CROUCH segment (low stance)."""

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: torch.Tensor):
        self._set_planned_frame_from_robot(env_ids)

        cp = self.cfg.crouch_params
        total_len = torch.full((len(env_ids),), float(self.cfg.ranges.default_path_len), device=self.device)

        s_min, s_max = cp.start_dist_range
        crouch_s0 = torch.empty(len(env_ids), device=self.device).uniform_(s_min, s_max)
        crouch_s1 = (crouch_s0 + float(cp.crouch_len)).clamp(max=total_len - 1e-3)

        self._path_len[env_ids] = total_len

        start = self._planned_start_pos[env_ids].clone()
        fwd = self._planned_forward_dir[env_ids]
        end = start.clone()
        end[:, :2] = end[:, :2] + fwd * total_len[:, None]

        alpha = self.t_alpha.view(1, -1, 1)
        pos = start[:, None, :] + (end[:, None, :] - start[:, None, :]) * alpha

        yaw_wp = self._planned_yaw[env_ids][:, None].repeat(1, self.num_waypoints)
        self.pos_path_w[env_ids] = pos
        self.heading_path_w[env_ids, :, 0] = yaw_wp

        self._clear_segments(env_ids)
        zero = torch.zeros(len(env_ids), device=self.device)

        # Leading walk segment
        lead_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        lead_params[:, self.SEG_PARAM["v_ref"]] = 1.0
        self._append_segment(env_ids, self.SKILL_ID["walk"], zero, crouch_s0, params=lead_params)

        # Crouch segment
        crouch_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        crouch_params[:, self.SEG_PARAM["v_ref"]] = 0.7  # Slower speed
        crouch_params[:, self.SEG_PARAM["base_height_ref"]] = float(cp.base_height_ref)  # Lower height
        self._append_segment(env_ids, self.SKILL_ID["crouch"], crouch_s0, crouch_s1, params=crouch_params)

        # Trailing walk segment
        trail_params = torch.zeros(len(env_ids), self.num_seg_params, device=self.device)
        trail_params[:, self.SEG_PARAM["v_ref"]] = 1.0
        self._append_segment(env_ids, self.SKILL_ID["walk"], crouch_s1, total_len, params=trail_params)

        self._num_segs[env_ids] = torch.clamp(self._num_segs[env_ids], min=1)
        self.current_waypoints_index[env_ids] = 0
        self.goal_reached[env_ids] = False
```

---

### 5. **JumpPathCommand** (`jump_path_command.py`)

#### 检查结果：
- ✅ **已经存在且正确**
- ✅ 实现了 gap 检测
- ✅ 实现了抛物线轨迹生成
- ✅ 正确使用了 segment table

**无需修改**（已经在之前的代码中）

---

## 🔌 新技能接口检查

### ✅ 接口设计完善

我已经为添加新技能留好了完整的接口：

#### 1. **SkillPathPlanner 抽象基类**

```python
class SkillPathPlanner(ABC):
    """技能路径规划器接口"""

    @abstractmethod
    def plan_segment(self, ...) -> SegmentPlan:
        """规划一个技能段的路径"""
        pass

    @abstractmethod
    def validate_terrain(self, ...) -> torch.Tensor:
        """验证地形是否适合该技能"""
        pass
```

#### 2. **在 TerrainAwarePathGenerator 中注册**

```python
class TerrainAwarePathGenerator:
    def __init__(self, cfg, device):
        # 注册技能路径规划器
        self.skill_planners: Dict[str, SkillPathPlanner] = {
            "walk": WalkPathPlanner(cfg.walk_params, device),
            "jump": JumpPathPlanner(cfg.jump_params, device),
            "stairs_up": StairsPathPlanner(cfg.stairs_params, device, direction="up"),
            "stairs_down": StairsPathPlanner(cfg.stairs_params, device, direction="down"),
            "climb": ClimbPathPlanner(cfg.climb_params, device),
            "crouch": CrouchPathPlanner(cfg.crouch_params, device),
            # 👇 在这里添加新技能
        }
```

#### 3. **在 base_path_command.py 中注册技能 ID**

```python
class SegmentPathCommand(CommandTerm):
    # 技能注册表
    SKILL_NAMES = [
        "walk",
        "jump",
        "stairs_up",
        "stairs_down",
        "crouch",
        "sidestep",
        "climb",
        # 👇 在这里添加新技能名称
    ]
    SKILL_ID = {name: i for i, name in enumerate(SKILL_NAMES)}
```

---

## 📝 添加新技能的完整步骤

### 示例：添加 "sidestep"（侧步）技能

#### Step 1: 创建配置类

在 `path_command_cfg.py` 中：

```python
@configclass
class SidestepParams:
    """侧步路径参数"""
    sidestep_len: float = 2.0  # 侧步段长度
    sidestep_distance: float = 0.5  # 侧向移动距离
    start_dist_range: Tuple[float, float] = (1.0, 2.0)
```

#### Step 2: 创建路径规划器

在 `skill_path_planners.py` 中：

```python
class SidestepPathPlanner(SkillPathPlanner):
    """侧步路径规划器"""

    def __init__(self, cfg: SidestepParams, device: torch.device):
        super().__init__(cfg, device)

    def plan_segment(
        self,
        env_ids: torch.Tensor,
        start_pos: torch.Tensor,
        start_yaw: torch.Tensor,
        terrain_data: Dict[str, torch.Tensor],
        segment_length: torch.Tensor,
    ) -> SegmentPlan:
        """生成侧步路径"""
        N = len(env_ids)
        device = start_pos.device

        # 计算侧向方向
        left_dir = torch.stack([
            -torch.sin(start_yaw),
            torch.cos(start_yaw),
            torch.zeros_like(start_yaw),
        ], dim=-1)

        # 生成侧步路径
        sidestep_len = min(self.cfg.sidestep_len, segment_length.min().item())
        num_waypoints = 32
        alpha = torch.linspace(0, 1, num_waypoints, device=device).view(1, -1, 1)

        # 前向移动 + 侧向移动
        forward_dir = torch.stack([
            torch.cos(start_yaw),
            torch.sin(start_yaw),
            torch.zeros_like(start_yaw),
        ], dim=-1)

        end_pos = start_pos + forward_dir * sidestep_len + left_dir * self.cfg.sidestep_distance
        waypoints = start_pos.unsqueeze(1) + (end_pos - start_pos).unsqueeze(1) * alpha

        # 保持航向
        headings = start_yaw.unsqueeze(-1).repeat(1, num_waypoints)

        # 构建参数
        segment_params = torch.zeros(N, 6, device=device)
        segment_params[:, 0] = 0.8  # v_ref（侧步较慢）

        return SegmentPlan(
            waypoints=waypoints,
            headings=headings,
            skill_id=5,  # SKILL_ID["sidestep"]
            segment_length=torch.full((N,), sidestep_len, device=device),
            segment_params=segment_params,
            valid=torch.ones(N, dtype=torch.bool, device=device),
        )

    def validate_terrain(self, terrain_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """验证地形"""
        return torch.tensor([True], device=self.device)
```

#### Step 3: 注册到 PathGeneratorCfg

在 `path_command_cfg.py` 中：

```python
@configclass
class PathGeneratorCfg:
    # ... 现有配置 ...
    sidestep_params: SidestepParams = field(default_factory=SidestepParams)  # 👈 添加这行
```

#### Step 4: 注册到 TerrainAwarePathGenerator

在 `path_generator.py` 中：

```python
def __init__(self, cfg: PathGeneratorCfg, device: torch.device):
    # ... 现有代码 ...
    self.skill_planners: Dict[str, SkillPathPlanner] = {
        # ... 现有技能 ...
        "sidestep": SidestepPathPlanner(cfg.sidestep_params, device),  # 👈 添加这行
    }
```

#### Step 5: 使用新技能

```python
# 生成侧步路径
path_result = path_generator.generate_path(
    env_ids=env_ids,
    skill_name="sidestep",  # 👈 使用新技能
    start_pos=start_pos,
    start_yaw=start_yaw,
    env_bounds=env_bounds,
)
```

---

## 📊 总结

### 现有代码问题：

| 文件 | 状态 | 问题 | 优先级 |
|------|------|------|--------|
| `flat_path_command.py` | ⚠️ 需要修复 | 重复导入 | 低 |
| `stairs_path_command.py` | ✅ 正确 | 无 | - |
| `climb_path_command.py` | ✅ 正确 | 无 | - |
| `crouch_path_command.py` | ❌ 未实现 | 文件为空 | **高** |
| `jump_path_command.py` | ✅ 正确 | 无 | - |

### 新技能接口：

✅ **接口完善** - 已经留好了完整的扩展接口
✅ **易于扩展** - 只需 5 个步骤即可添加新技能
✅ **统一规范** - 所有技能遵循相同的接口
✅ **类型安全** - 使用抽象基类确保实现完整

### 建议：

1. **立即修复** `crouch_path_command.py`（我已经提供了完整代码）
2. **可选修复** `flat_path_command.py` 的重复导入
3. **保持现有** 其他文件都正确，无需修改

需要我帮你修复这些问题吗？
