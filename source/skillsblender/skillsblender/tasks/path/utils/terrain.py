"""
 SceneCfg/TerrainImporterCfg 里把 terrain_generator=... 指到这里的某个 *_TERRAIN_CFG。
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen

# 训练阶段用的稳定字符串标签（后续可以直接写进 extras 或 command）
SKILL_TAG_WALK = "walk"
SKILL_TAG_JUMP = "jump"
SKILL_TAG_STAIRS = "stairs"
SKILL_TAG_CLIMB = "climb"
SKILL_TAG_CROUCH = "crouch"  # crouch terrain uses extra spawned objects, not heightfield


# ----------------------------
# WALK: flat + mild rough
# ----------------------------
WALK_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.5,
    num_rows=12,
    num_cols=18,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.70),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)

# ----------------------------
# JUMP: gap
# ----------------------------
JUMP_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.5,
    num_rows=16,
    num_cols=24,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "narrow_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.65,
            gap_width_range=(0.30, 0.50),
            platform_width=2.0,
        ),
        "wide_gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.35,
            gap_width_range=(0.60, 1.00),
            platform_width=2.5,
        ),
    },
)

# ----------------------------
# STAIRS: stairs
# ----------------------------
STAIRS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.5,
    num_rows=12,
    num_cols=18,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.60,
            step_height_range=(0.05, 0.22),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.40,
            step_height_range=(0.05, 0.22),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
            holes=False,
        ),
    },
)

# ----------------------------
# CLIMB: slope/ramp（ climb 额外 spawn 物体更好；这个 cfg 仍可用于早期训练）
# ----------------------------
CLIMB_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.5,
    num_rows=12,
    num_cols=18,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.10, 0.45),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


def subterrain_name_to_skill_tag(name: str) -> str:
    """terrain generator 的 subterrain 名字 → skill tag"""
    n = name.lower()
    if "gap" in n:
        return SKILL_TAG_JUMP
    if "stair" in n:
        return SKILL_TAG_STAIRS
    if "slope" in n or "ramp" in n:
        return SKILL_TAG_CLIMB
    return SKILL_TAG_WALK