# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Isaac Lab scene config for the Drawer Manipulation task.

Manager-based environment using:
  - Franka Panda arm (Articulation)
  - Drawer cabinet (Articulation, prismatic joint)
  - Wrist RGB camera (Camera)
  - Wrist FT sensor (ContactSensor at wrist)

Phase 3 implementation target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DisturbanceType(Enum):
    NONE = "none"
    LIGHT = "light"    # translate cabinet mid-task
    HEAVY = "heavy"    # remove drawer from hand post-grasp


class UncertaintyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class DrawerManipEnvCfg:
    """Configuration for one drawer manipulation trial.

    Attributes:
        uncertainty_level:  Initial prior uncertainty on drawer position.
        disturbance_type:   Whether and how to disturb mid-task.
        sensor_noise_std:   Std-dev of additive Gaussian noise on camera detections.
        goal_open_dist_m:   Target drawer opening distance in metres (paper: 0.20).
        camera_fov_deg:     Wrist camera horizontal field of view in degrees.
        device:             Torch device for AICON graph tensors.
    """

    uncertainty_level: UncertaintyLevel = UncertaintyLevel.LOW
    disturbance_type: DisturbanceType = DisturbanceType.NONE
    sensor_noise_std: float = 0.01
    goal_open_dist_m: float = 0.20
    camera_fov_deg: float = 80.0
    device: str = "cuda:0"

    # ---------------------------------------------------------------------------
    # TODO (Phase 3): Add full Isaac Lab InteractiveSceneCfg here.
    #
    # Required:
    #   from isaaclab.utils import configclass
    #   from isaaclab.assets import ArticulationCfg
    #   from isaaclab.sensors import CameraCfg, ContactSensorCfg
    #   from isaaclab.scene import InteractiveSceneCfg
    #
    # Scene assets:
    #   ground_plane: sim_utils.GroundPlaneCfg
    #   table:        sim_utils.UsdFileCfg / BoxCfg
    #   cabinet:      ArticulationCfg pointing to drawer USD
    #   robot:        ArticulationCfg (Franka Panda from Isaac Lab assets)
    #   wrist_camera: CameraCfg(data_types=["rgb"], offset=OffsetCfg to EE)
    #   wrist_ft:     ContactSensorCfg(filter_prim_paths_expr=["/World/robot/panda_hand"])
    #
    # Events:
    #   reset:        randomise drawer handle position within uncertainty_level range
    #   disturb:      apply disturbance_type event at mid-episode
    # ---------------------------------------------------------------------------
