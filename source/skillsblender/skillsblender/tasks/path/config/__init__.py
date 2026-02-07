
from isaaclab.markers.config import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os


def _heading_marker_cfg(color: tuple[float, float, float], usd_path: str | None = None):
    """Create a heading marker (prefer arrow, fall back to slim box if unavailable)."""
    if usd_path:
        # Use a USD arrow asset when provided.
        return sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(0.25, 0.25, 0.25),
        )
    arrow_cls = getattr(sim_utils, "ArrowCfg", None)
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    if arrow_cls is not None:
        return arrow_cls(scale=(0.25, 0.06, 0.06), visual_material=material)
    # Fallback: elongated box that still conveys orientation.
    return sim_utils.CuboidCfg(size=(0.25, 0.05, 0.05), visual_material=material)

WAYPOINTS_MARKER_CFG = VisualizationMarkersCfg(
    # prim_path="/Visuals/PathMarkers",
    markers={
        "waypoints_slice": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  #color green
        ),
    }
)

START_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)), 
        ),
    }
)
GOAL_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), #color red
        ),
    }
)

# Heading arrows (desired vs actual yaw).
# Optional USD arrow (set env var to override).
_PATH_HEADING_USD = os.getenv("SKILLSBLENDER_ARROW_USD", "")
_ROBOT_HEADING_USD = os.getenv("SKILLSBLENDER_ROBOT_ARROW_USD", _PATH_HEADING_USD)

PATH_HEADING_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "path_heading": _heading_marker_cfg((0.0, 0.4, 1.0), usd_path=_PATH_HEADING_USD),  # blue
    }
)

ROBOT_HEADING_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "robot_heading": _heading_marker_cfg((0.0, 1.0, 0.0), usd_path=_ROBOT_HEADING_USD),  # green
    }
)
