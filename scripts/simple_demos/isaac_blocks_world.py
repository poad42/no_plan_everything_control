import argparse
import torch
import torchvision.io as tvio
import numpy as np

# Isaac Lab / Sim setup
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab Blocks World AICON")
parser.add_argument("--headless", action="store_true", help="Run headlessly")
parser.add_argument("--num_blocks", type=int, default=5, help="Number of blocks")
# Append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.shapes import CuboidCfg

import math
from no_plan_everything_control.aicon.utils import random_bw_instance, stacks_to_on_matrix
from no_plan_everything_control.envs.blocks_world.aicon_policy import BlocksWorldAICON

@configclass
class BlocksSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(color=(0.5, 0.5, 0.5)),
    )
    
    # We will spawn blocks dynamically in code, but let's define a camera
    camera = CameraCfg(
        prim_path="/World/Camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None, # use existing camera if desired or spawn one later
    )

def main():
    sim_cfg = SimulationCfg(dt=0.02)
    
    # Just standard Isaac Lab logic without relying too much on complex Scene configs for dynamic number of blocks
    # We can just spawn them with core APIs and use RigidObjectCfg if we need managers, but since we are kinematic-teleporting,
    # we can literally use `sim.spawn_cuboid` or just prim paths. 
    pass

if __name__ == "__main__":
    main()
    simulation_app.close()
