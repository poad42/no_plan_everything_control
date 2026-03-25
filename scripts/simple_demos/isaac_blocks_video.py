import argparse
import torch
import torchvision.io as tvio
import numpy as np
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab Blocks World Video AICON")
parser.add_argument("--num_blocks", type=int, default=5, help="Number of blocks")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import CameraCfg, Camera
import omni.replicator.core as rep
from pxr import UsdGeom, Gf

from no_plan_everything_control.aicon.utils import random_bw_instance, stacks_to_on_matrix
from no_plan_everything_control.envs.blocks_world.aicon_policy import BlocksWorldAICON

def create_blocks_world(n_blocks):
    import isaacsim.core.utils.stage as stage_utils
    
    # create table
    prim_utils.create_prim("/World/Table", "Cube", translation=(0.0, 0.0, 0.4), scale=(1.2, 0.6, 0.8))
    
    # create blocks
    block_prims = []
    colors = [
        (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8),
        (0.8, 0.8, 0.1), (0.1, 0.8, 0.8), (0.8, 0.1, 0.8)
    ]
    size = 0.08
    for i in range(n_blocks):
        path = f"/World/Block_{i}"
        prim_utils.create_prim(path, "Cube", translation=(0.0, 0.0, 0.0), scale=(size, size, size))
        block_prims.append(path)
        
    return block_prims, size

def get_stack_positions(stacks, block_size):
    positions = {}
    n_towers_max = 10
    base_x = np.linspace(-0.4, 0.4, n_towers_max)
    base_y = 0.0
    table_z = 0.8
    for t_idx, tower in enumerate(stacks):
        for b_idx, block in enumerate(tower):
            x = base_x[t_idx % n_towers_max]
            y = base_y
            z = table_z + block_size / 2.0 + b_idx * block_size
            positions[block] = (x, y, z)
    return positions

def interpolate_arc(p1, p2, param):
    p = np.array(p1) * (1-param) + np.array(p2) * param
    h = 0.3 * math.sin(param * math.pi)
    p[2] += h
    return p

def main():
    sim = SimulationContext(SimulationCfg(dt=0.03))
    sim.set_camera_view(eye=[0.0, -1.2, 1.3], target=[0.0, 0.0, 0.8])
    num_blocks = 5
    block_prims, size = create_blocks_world(num_blocks)
    
    # Init AICON
    initial_stacks = [[0, 1], [2], [3], [4]]
    goal_stacks = [[4, 3, 2, 1, 0]]
    
    policy = BlocksWorldAICON(n_blocks=num_blocks, goal_on=stacks_to_on_matrix(goal_stacks, num_blocks), interconnected_goal=True)
    policy.reset(initial_stacks)
    
    sim.reset()
    
    print("Setting up replicator for camera capture...")
    render_product = rep.create.render_product("/OmniverseKit_Persp", (640, 480))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    
    frames = []
    curr_stacks = [list(t) for t in initial_stacks]
    
    for _ in range(10): sim.step()
        
    def snap_frame():
        rgb_data = rgb_annotator.get_data()
        if rgb_data is not None and rgb_data.shape[0] > 0:
            frame = torch.from_numpy(rgb_data[..., :3].copy()).permute(2, 0, 1).byte()
            frames.append(frame)

    prev_positions = get_stack_positions(curr_stacks, size)
    for b, pos in prev_positions.items():
        prim_utils.set_prim_attribute_value(block_prims[b], "xformOp:translate", Gf.Vec3d(*pos))

    print("Beginning symbolic AICON transitions...")
    for step in range(30):
        action_tup, best_norm = policy._select_action(stacks_to_on_matrix(curr_stacks, num_blocks), policy._state.clear, policy._goal_cost(stacks_to_on_matrix(curr_stacks, num_blocks)))
        action, X, Y = action_tup
        
        if action == "stack":
            for t in curr_stacks:
                if X in t: t.remove(X); break
            for t in curr_stacks:
                if Y in t: t.append(X); break
        else:
            for t in curr_stacks:
                if X in t: t.remove(X); break
            curr_stacks.append([X])
            
        curr_stacks = [t for t in curr_stacks if t]
        target_positions = get_stack_positions(curr_stacks, size)
        
        frames_per_move = 25
        p_start = prev_positions[X]
        p_end = target_positions[X]
        
        for t in range(frames_per_move):
            param = t / float(frames_per_move - 1)
            pos_x = interpolate_arc(p_start, p_end, param)
            
            prim_utils.set_prim_attribute_value(block_prims[X], "xformOp:translate", Gf.Vec3d(*pos_x))
            sim.step()
            snap_frame()
            
        prev_positions = target_positions.copy()
        
        on_curr = stacks_to_on_matrix(curr_stacks, num_blocks)
        on_target = stacks_to_on_matrix(goal_stacks, num_blocks)
        if (on_curr - on_target).abs().sum() < 0.5:
            print("Goal Reached!")
            break
            
    for _ in range(15):
        sim.step()
        snap_frame()

    if len(frames) > 0:
        video_tensor = torch.stack(frames)
        os.makedirs("scripts/simple_demos/outputs", exist_ok=True)
        tvio.write_video("scripts/simple_demos/outputs/isaac_blocks_world.mp4", video_tensor, fps=24, video_codec="libx264")
        print("Generated scripts/simple_demos/outputs/isaac_blocks_world.mp4 successfully.")
    else:
        print("Failed to capture frames.")

if __name__ == "__main__":
    main()
    simulation_app.close()
