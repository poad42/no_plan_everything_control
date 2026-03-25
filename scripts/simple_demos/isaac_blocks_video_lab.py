import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source")))
import argparse
import torch
import numpy as np
import math
import os
# import torchvision.io as tvio

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab Blocks World Video AICON")
parser.add_argument("--num_blocks", type=int, default=5, help="Number of blocks")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Removed forced headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass

from no_plan_everything_control.aicon.utils import stacks_to_on_matrix
from no_plan_everything_control.envs.blocks_world.aicon_policy import BlocksWorldAICON

NUM_BLOCKS = 5
BLOCK_SIZE = 0.08
COLORS = [
    (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8),
    (0.8, 0.8, 0.1), (0.1, 0.8, 0.8)
]

@configclass
class BlocksSceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
    )

    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.5, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.4))
    )

    camera = CameraCfg(
        prim_path="/World/Camera",
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0),
        width=640,
        height=480,
        data_types=["rgb"]
    )

    block_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_0",
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[0]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False), # kinematic so we can teleport
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9))
    )
    block_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_1",
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[1]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9))
    )
    block_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_2",
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[2]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9))
    )
    block_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_3",
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[3]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9))
    )
    block_4 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_4",
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[4]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9))
    )

def get_stack_positions(stacks, block_size):
    positions = {}
    n_towers_max = 5
    base_x = np.linspace(-0.25, 0.25, n_towers_max)
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
    h = 0.2 * math.sin(param * math.pi)
    p[2] += h
    return p

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = BlocksSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(0.0)

    camera = scene.sensors["camera"]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[0.0, -0.9, 1.2]], device=sim.device),
        targets=torch.tensor([[0.0, 0.0, 0.8]], device=sim.device)
    )
    # Step to apply camera position
    sim.step()
    scene.update(0.0)

    # Initial and Goal configs for AICON
    initial_stacks = [[0, 1], [2], [3], [4]]
    goal_stacks = [[4, 3, 2, 1, 0]]
    policy = BlocksWorldAICON(n_blocks=NUM_BLOCKS, goal_on=stacks_to_on_matrix(goal_stacks, NUM_BLOCKS), interconnected_goal=True)
    policy.reset(initial_stacks)

    curr_stacks = [list(t) for t in initial_stacks]
    prev_positions = get_stack_positions(curr_stacks, BLOCK_SIZE)
    
    blocks = [scene[f"block_{i}"] for i in range(NUM_BLOCKS)]
    
    # set initial poses
    for i in range(NUM_BLOCKS):
        pos = prev_positions[i]
        root_state = blocks[i].data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(pos, device=sim.device, dtype=torch.float32)
        blocks[i].write_root_state_to_sim(root_state)
        blocks[i].reset()

    frames = []

    def snap_frame():
        rgb = camera.data.output["rgb"][0].cpu() # shape [H, W, 3] or [H, W, 4]
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        # Torchvision io expects [T, H, W, C] where channel is last in 0.16+, but save_video expects T, H, W, C uint8
        frames.append(rgb.to(torch.uint8))
        
    for _ in range(30):
        sim.step()
        scene.update(0.0)

    print("Beginning symbolic AICON transitions...")
    for step in range(30):
        action_tup, best_norm = policy._select_action(
            stacks_to_on_matrix(curr_stacks, NUM_BLOCKS), 
            policy._state.clear, 
            policy._goal_cost(stacks_to_on_matrix(curr_stacks, NUM_BLOCKS))
        )
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
        target_positions = get_stack_positions(curr_stacks, BLOCK_SIZE)
        
        frames_per_move = 30
        p_start = prev_positions[X]
        p_end = target_positions[X]
        
        for t in range(frames_per_move):
            param = t / float(frames_per_move - 1)
            pos_x = interpolate_arc(p_start, p_end, param)
            
            root_state = blocks[X].data.default_root_state.clone()
            root_state[:, :3] = torch.tensor(pos_x, device=sim.device, dtype=torch.float32)
            blocks[X].write_root_state_to_sim(root_state)
            
            sim.step()
            scene.update(0.0)
            snap_frame()
            
        prev_positions = target_positions.copy()
        
        on_curr = stacks_to_on_matrix(curr_stacks, NUM_BLOCKS)
        on_target = stacks_to_on_matrix(goal_stacks, NUM_BLOCKS)
        if (on_curr - on_target).abs().sum() < 0.5:
            print("Goal Reached!")
            break
            
    for _ in range(20):
        sim.step()
        scene.update(0.0)
        snap_frame()

    if len(frames) > 0:
        video_tensor = torch.stack(frames) # [T, H, W, 3]
        os.makedirs("outputs/videos", exist_ok=True)
        out_path = "outputs/videos/isaac_blocks_world.mp4"
        
        # Save as individual frames using PIL instead of PyAV MP4 compression 
        # since PyAV is not installed in IsaacLab pip_prebundle.
        from PIL import Image
        print(f"Captured {len(frames)} frames. Saving sequence to {out_path}_frames/ ...")
        os.makedirs(f"{out_path}_frames", exist_ok=True)
        for idx, frame in enumerate(video_tensor):
            img = Image.fromarray(frame.numpy(), 'RGB')
            img.save(f"{out_path}_frames/frame_{idx:04d}.png")
            
        print(f"Generated {out_path}_frames/ successfully.")
    else:
        print("Failed to capture frames.")

if __name__ == "__main__":
    main()
    simulation_app.close()
