# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Isaac Lab scene and environment configuration for Blocks World.

This is a manager-based environment (InteractiveSceneCfg).
The AICON logic runs entirely in Python/PyTorch outside the sim graph —
the sim is used only to render physics and provide ground-truth state.

Phase 2 implementation target.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass


# ---------------------------------------------------------------------------
# TODO (Phase 2): Implement full Isaac Lab scene config.
#
# Required imports (uncomment when implementing):
#   import isaaclab.sim as sim_utils
#   from isaaclab.assets import RigidObjectCfg, ArticulationCfg
#   from isaaclab.scene import InteractiveSceneCfg
#   from isaaclab.utils import configclass
#
# Scene layout:
#   - ground_plane
#   - table (box USDGeom or asset)
#   - blocks[N]: RigidObject, coloured, stacked at spawn
#   - (optional) camera for visualisation
#
# Events:
#   - reset: randomise block positions/stacking via EventTermCfg
#   - State readout: o(X,Y) from RigidObject.data.root_pos_w using
#     height comparison + XY proximity threshold
# ---------------------------------------------------------------------------


@dataclass
class BlocksWorldEnvCfg:
    """Configuration for a Blocks World episode.

    Attributes:
        num_blocks: Number of blocks in this instance (paper: 10–30).
        num_towers_goal: Number of towers in the goal config (paper: 0–15).
        seed: Random seed for instance generation.
        block_size: Side length of each cube block in metres.
        proximity_threshold: XY distance threshold to decide o(X,Y) = 1.
        height_threshold: Z-height threshold to decide block is on another.
    """

    num_blocks: int = 10
    num_towers_goal: int = 3
    seed: int = 0
    block_size: float = 0.05
    proximity_threshold: float = 0.04
    height_threshold: float = 0.04
