"""Configuration for the Locked Door 2D/3D demonstration."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import torch

@configclass
class LockedDoorSceneCfg(InteractiveSceneCfg):
    """Configuration for the locked door scene."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Top-Down Camera for Video Recording
    top_camera = CameraCfg(
        prim_path="/World/TopCamera",
        update_period=0.01,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.0, 5.5, 13.0),
            rot=(1.0, 0.0, 0.0, 0.0), # looking straight down, depends on convention though.
        )
    )

    # The Agent (a sphere that we control via velocities)
    agent = RigidObjectCfg(
        prim_path="/World/Agent",
        spawn=sim_utils.SphereCfg(
            radius=0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=5.0,
                angular_damping=5.0,
                max_linear_velocity=5.0,
                max_angular_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )

    # Goal Marker (a green cylinder)
    goal = RigidObjectCfg(
        prim_path="/World/Goal",
        spawn=sim_utils.CylinderCfg(
            radius=0.4,
            height=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 0.0, 0.02)),
    )

    # Button Marker (an orange cylinder)
    button = RigidObjectCfg(
        prim_path="/World/Button",
        spawn=sim_utils.CylinderCfg(
            radius=0.4,
            height=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 5.0, 0.02)),
    )

    # The Gate (a physical barrier we can move down when opened)
    gate = RigidObjectCfg(
        prim_path="/World/Gate",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 4.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 0.5)),
    )
