"""
Standalone Isaac Lab visualization of the AICON Locked Door Demo
Run via: ./isaaclab.sh -p scripts/simple_demos/isaac_locked_door.py
"""

import argparse
import sys
from pathlib import Path

# Add the project root source to sys.path so we can import internal configs
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "source"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AICON Locked Door Demo")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# Import the Configuration
from no_plan_everything_control.envs.locked_door.env_cfg import LockedDoorSceneCfg

class PointMassAICON:
    def __init__(self, start_pos, goal_pos, button_pos, gate_x=5.0):
        self.device = torch.device('cpu') # Running PyTorch math on CPU since it's tiny
        self.agent_pos = torch.tensor(start_pos, dtype=torch.float32, requires_grad=True, device=self.device)
        self.goal_pos = torch.tensor(goal_pos, dtype=torch.float32, device=self.device)
        self.button_pos = torch.tensor(button_pos, dtype=torch.float32, device=self.device)
        self.gate_x = gate_x
        
    def gate_open_prob(self, pos):
        dist_to_button = torch.norm(pos - self.button_pos)
        return torch.exp(-0.5 * dist_to_button)

    def compute_cost(self, pos):
        goal_cost = torch.norm(pos - self.goal_pos)
        c_gate = self.gate_open_prob(pos)
        gate_distance = pos[0] - self.gate_x
        barrier = 100.0 * torch.exp(-0.5 * (gate_distance / 2.0)**2) 
        total_cost = goal_cost + (1.0 - c_gate) * barrier
        return total_cost

def main():
    # Setup physics context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set the camera specifically for top-down view of the demo
    sim.set_camera_view(eye=[5.0, 2.5, 12.0], target=[5.0, 2.5, 0.0])

    # Construct the Scene
    scene_cfg = LockedDoorSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    print("[INFO] Simulation setup complete.")

    start_pos = [0.0, 0.0]
    goal_pos = [10.0, 0.0]
    button_pos = [2.0, 5.0]
    gate_x = 5.0

    # Initialize AICON
    aicon = PointMassAICON(start_pos=start_pos, goal_pos=goal_pos, button_pos=button_pos, gate_x=gate_x)
    
    agent = scene["agent"]
    gate = scene["gate"]

    import os
    import torchvision.io as tv_io

    output_dir = "scripts/simple_demos/outputs"
    os.makedirs(output_dir, exist_ok=True)
    video_frames = []

    step = 0
    while simulation_app.is_running():
        # Step the scene
        scene.write_data_to_sim()
        sim.step()
        scene.update(0.01)

        # 1. Read agent X, Y position from Isaac Lab
        # Shape is (num_envs, 3)
        pos_3d = agent.data.root_pos_w.clone() 
        # For simplicity, just grab the first env's (only env's) x and y
        pos_2d = torch.tensor([pos_3d[0, 0].item(), pos_3d[0, 1].item()], dtype=torch.float32, requires_grad=True, device='cpu')
        
        # Override AICON state
        aicon.agent_pos = pos_2d
        
        if aicon.agent_pos.grad is not None:
            aicon.agent_pos.grad.zero_()
            
        # 2. Forward/Backward
        cost = aicon.compute_cost(aicon.agent_pos)
        cost.backward()
        
        grad = aicon.agent_pos.grad.clone()
        
        # 3. Calculate velocity command from normalized gradient
        lr = 4.0 # velocity gain
        grad_norm = torch.norm(grad)
        if grad_norm > 1e-6:
            vel_cmd = -lr * (grad / grad_norm)
        else:
            vel_cmd = torch.zeros_like(grad)
            
        # 4. Apply velocity to the agent (using rigid object api)
        # shape expected: (num_envs, 6) [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        vel_3d = torch.zeros(1, 6, device=scene.device)
        vel_3d[0, 0] = vel_cmd[0].item()
        vel_3d[0, 1] = vel_cmd[1].item()
        agent.write_root_velocity_to_sim(vel_3d)

        # 5. Evaluate active interconnection: Gate Opening
        c_gate = aicon.gate_open_prob(aicon.agent_pos).item()
        
        # Animate Gate opening depending on the connection probability
        # if c_gate=1.0, gate drops to -2.0 Z
        # if c_gate=0.0, gate stays at +0.5 Z
        target_z = 0.5 - 2.5 * c_gate
        current_gate_pos = gate.data.root_pos_w.clone()
        current_gate_pos[0, 2] = target_z
        gate.write_root_pose_to_sim(torch.cat([current_gate_pos, gate.data.root_quat_w], dim=-1))
        
        # End simulation if reached
        dist_to_goal = torch.norm(pos_2d - torch.tensor(goal_pos, dtype=torch.float32))
        if step % 50 == 0:
            print(f"[{step}] Pos: {pos_2d.tolist()}, Dist to goal: {dist_to_goal.item():.2f}")
            
        if dist_to_goal < 0.5:
            print(f"Goal reached at step {step}!")
            break
            
        # Capture frame explicitly every N steps (e.g., render 30fps out of 100Hz phys step)
        if step % 3 == 0:
            scene["top_camera"].update(0.01)
            img_tensor = scene["top_camera"].data.output["rgb"][0]
            # Convert from RGBA generic output to RGB uint8 for video writer
            frame = img_tensor[..., :3].cpu().byte()
            # torchvision requires [C, H, W] for video writer, image_tensor is [H, W, C]
            frame = frame.permute(2, 0, 1)
            video_frames.append(frame)
        
        step += 1

    print("Encoding video...")
    video_tensor = torch.stack(video_frames) # Shape: [T, C, H, W]
    video_path = os.path.join(output_dir, "locked_door_demo.mp4")
    tv_io.write_video(video_path, video_tensor.permute(0, 2, 3, 1), fps=30)
    print(f"Video saved to {video_path}")

    simulation_app.close()

if __name__ == "__main__":
    main()
