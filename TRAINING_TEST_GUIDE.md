# 跳跃参数配置化 - 训练测试指南

## 测试目标

验证新的可配置跳跃参数在实际训练中是否正常工作，并测试不同参数配置对跳跃行为的影响。

## 测试环境

- **配置文件**: `go2_jump_cur_cfg.py`
- **机器人**: Unitree GO2
- **地形**: 带有窄/宽间隙的课程学习地形

## 测试场景

### 场景 1: 默认配置测试（基线）

**目的**: 验证新参数的默认值与之前的硬编码行为一致

**配置**:
```python
gap_width_threshold = 0.55
narrow_gap_endpoint_extension = 1.5
wide_gap_endpoint_extension = 2.0
narrow_gap_takeoff_margin = 0.4
wide_gap_takeoff_margin = 0.5
landing_margin = 0.5
post_jump_distance = 0.0
```

**运行命令**:
```bash
python source/standalone/workflows/rsl_rl/train.py \
    --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
    --num_envs 4096 \
    --headless
```

**预期结果**:
- 训练应该正常启动
- 机器人应该能够跳过间隙
- 行为应该与之前的版本一致

---

### 场景 2: 保守配置测试

**目的**: 测试更安全的跳跃配置（更长的安全距离）

**修改配置** (`go2_jump_cur_cfg.py`):
```python
self.commands.path_tracking.jump_params.narrow_gap_endpoint_extension = 2.0  # 增加
self.commands.path_tracking.jump_params.wide_gap_endpoint_extension = 2.5    # 增加
self.commands.path_tracking.jump_params.narrow_gap_takeoff_margin = 0.3      # 减少（更早起跳）
self.commands.path_tracking.jump_params.wide_gap_takeoff_margin = 0.4        # 减少
self.commands.path_tracking.jump_params.landing_margin = 0.6                 # 增加
self.commands.path_tracking.jump_params.post_jump_distance = 0.5             # 启用跳跃后前进
```

**预期结果**:
- 机器人应该在间隙前更早起跳
- 着陆点应该距离间隙边缘更远
- 着陆后应该继续前进 0.5m

---

### 场景 3: 激进配置测试

**目的**: 测试更高效的跳跃配置（更短的距离）

**修改配置**:
```python
self.commands.path_tracking.jump_params.narrow_gap_endpoint_extension = 1.2  # 减少
self.commands.path_tracking.jump_params.wide_gap_endpoint_extension = 1.5    # 减少
self.commands.path_tracking.jump_params.narrow_gap_takeoff_margin = 0.5      # 增加（更晚起跳）
self.commands.path_tracking.jump_params.wide_gap_takeoff_margin = 0.6        # 增加
self.commands.path_tracking.jump_params.landing_margin = 0.4                 # 减少
self.commands.path_tracking.jump_params.post_jump_distance = 0.0             # 禁用
```

**预期结果**:
- 机器人应该在更接近间隙时起跳
- 着陆点应该更接近间隙边缘
- 整体轨迹应该更短、更高效

---

### 场景 4: Gap 分类阈值测试

**目的**: 测试修改 gap 分类阈值的效果

**修改配置**:
```python
self.commands.path_tracking.jump_params.gap_width_threshold = 0.4  # 从 0.55 改为 0.4
```

**预期结果**:
- 更多的 gap 会被分类为"宽 gap"
- 这些 gap 会使用 wide_gap 的参数（更大的延伸和边距）

---

## 快速测试命令

### 1. 可视化测试（推荐先运行）

使用 PLAY 配置进行可视化测试，快速验证参数效果：

```bash
python source/standalone/workflows/rsl_rl/play.py \
    --task SkillsBlender-Path-Go2-Jump-Cur-v0-PLAY \
    --num_envs 4 \
    --checkpoint /path/to/your/checkpoint.pt
```

**观察要点**:
- 机器人起跳位置（是否在配置的边距处）
- 着陆位置（是否在配置的延伸距离处）
- 跳跃后是否继续前进（如果启用了 post_jump_distance）

### 2. 小规模训练测试

在修改参数后，先用少量环境测试：

```bash
python source/standalone/workflows/rsl_rl/train.py \
    --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
    --num_envs 128 \
    --max_iterations 1000 \
    --headless
```

**检查要点**:
- 训练是否正常启动（无错误）
- 奖励是否收敛
- Tensorboard 中的指标是否正常

### 3. 完整训练

确认参数正常后，运行完整训练：

```bash
python source/standalone/workflows/rsl_rl/train.py \
    --task SkillsBlender-Path-Go2-Jump-Cur-v0 \
    --num_envs 4096 \
    --headless
```

---

## 监控指标

在训练过程中，通过 Tensorboard 监控以下指标：

### 关键指标
1. **Episode Length**: 应该根据参数调整而变化
   - 更长的延伸 → 更长的 episode
   - post_jump_distance > 0 → 更长的 episode

2. **Jump Success Rate**: 自定义指标（如果有）
   - 应该保持稳定或提高

3. **Reward Components**:
   - `jump_forward_velocity`: 应该正常
   - `jump_clearance`: 应该正常
   - `jump_landing_stability`: 应该正常

### 诊断指标
1. **Termination Reasons**:
   - `jump_gap_fall`: 应该很少（表示掉入间隙）
   - `jump_landing_failure`: 应该减少

2. **Path Tracking**:
   - `track_xy`: 应该保持高值
   - `track_velocity`: 应该正常

---

## 参数调优建议

### 如果机器人经常掉入间隙：
```python
# 增加安全距离
narrow_gap_endpoint_extension = 2.0  # 增加
wide_gap_endpoint_extension = 2.5    # 增加
landing_margin = 0.6                 # 增加
```

### 如果机器人起跳太晚：
```python
# 增加起飞边距（更早起跳）
narrow_gap_takeoff_margin = 0.5  # 增加
wide_gap_takeoff_margin = 0.6    # 增加
```

### 如果机器人起跳太早：
```python
# 减少起飞边距（更晚起跳）
narrow_gap_takeoff_margin = 0.3  # 减少
wide_gap_takeoff_margin = 0.4    # 减少
```

### 如果想让机器人跳跃后继续前进：
```python
post_jump_distance = 0.5  # 或更大的值
```

### 如果想调整 gap 分类：
```python
# 更多 gap 被视为"窄"
gap_width_threshold = 0.6  # 增加

# 更多 gap 被视为"宽"
gap_width_threshold = 0.4  # 减少
```

---

## 故障排查

### 问题 1: 训练启动失败

**可能原因**: 参数值不合理

**解决方案**:
- 检查所有参数 >= 0
- 检查 gap_width_threshold > 0
- 恢复默认值测试

### 问题 2: 机器人行为异常

**可能原因**: 参数组合不合理

**解决方案**:
- 检查 endpoint_extension >= landing_margin
- 检查 takeoff_margin 不要太大（< 1.0m）
- 逐个参数调整，观察效果

### 问题 3: 性能下降

**可能原因**: 参数不适合当前地形

**解决方案**:
- 检查地形配置的 gap 宽度范围
- 调整 gap_width_threshold 以匹配地形
- 参考默认配置

---

## 测试检查清单

训练前检查：
- [ ] 已备份当前配置文件
- [ ] 已设置新的跳跃参数
- [ ] 已检查参数值合理性
- [ ] 已准备好监控 Tensorboard

训练中检查：
- [ ] 训练正常启动（无错误）
- [ ] Episode length 符合预期
- [ ] 奖励正常收敛
- [ ] 无异常终止

训练后检查：
- [ ] 在 PLAY 模式下验证行为
- [ ] 检查跳跃成功率
- [ ] 对比不同配置的效果
- [ ] 记录最佳参数组合

---

## 实验记录模板

```
实验日期: ____________________
配置名称: ____________________

参数设置:
- gap_width_threshold: ______
- narrow_gap_endpoint_extension: ______
- wide_gap_endpoint_extension: ______
- narrow_gap_takeoff_margin: ______
- wide_gap_takeoff_margin: ______
- landing_margin: ______
- post_jump_distance: ______

训练结果:
- 最终奖励: ______
- Episode length: ______
- 跳跃成功率: ______

观察:
_________________________________
_________________________________

结论:
_________________________________
_________________________________
```

---

## 下一步

1. **运行场景 1（默认配置）** 作为基线
2. **对比场景 2 和 3** 观察参数影响
3. **根据结果调优参数**
4. **记录最佳配置** 用于生产

祝训练顺利！🚀
