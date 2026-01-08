
from isaaclab.markers.config import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

WAYPOINTS_MARKER_CFG = VisualizationMarkersCfg(
    # prim_path="/Visuals/PathMarkers",
    markers={
        "waypoints_slice": sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), 
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)