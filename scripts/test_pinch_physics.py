import argparse
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True, "num_envs": 1})

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_box_minus

import isaaclab_tasks

def main():
    env_cfg = parse_env_cfg("Isaac-Open-Drawer-Franka-IK-Rel-v0", device="cuda:0", num_envs=1)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    
    # 1. Approach: Wait a bit to let it settle
    for _ in range(50):
        # Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # zero motion, gripper open (1.0)
        action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device="cuda:0")
        env.step(action)
    
    print("[TEST] Stage 1 done: Settled")

    # 2. Approach the handle
    for step in range(250):
        obs = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
        drawer_pos = env.unwrapped.scene["cabinet"].data.joint_pos[:, 0]
        handle_pos = env.unwrapped.scene["cabinet"].data.body_pos_w[:, env.unwrapped.scene["cabinet"].find_bodies("drawer_top")[0][0]]
        handle_pos[:, 1] -= 0.1 # Move slightly along -y to handle front
        
        ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
        ee_quat = env.unwrapped.scene["ee_frame"].data.target_quat_w[:, 0, :]
        
        target_q = torch.tensor([[0.707, -0.707, 0.0, 0.0]], device=env.unwrapped.device)
        err_rot = quat_box_minus(target_q, ee_quat).squeeze(0)
        
        err_pos = (handle_pos - ee_pos).squeeze(0)
        
        # Scale down for velocity command
        vel = err_pos * 5.0
        rot = err_rot * 2.0
        
        action = torch.zeros((1, 7), device="cuda:0")
        action[0, :3] = vel.clamp(-1.0, 1.0)
        action[0, 3:6] = rot.clamp(-1.0, 1.0)
        action[0, 6] = 1.0 # open gripper
        
        env.step(action)
        if step % 20 == 0:
            dist = torch.norm(err_pos).item()
            print(f"[TEST] Approach dist: {dist:.4f}")
            
    print("[TEST] Stage 2 done: Reached handle")

    # 3. Close the gripper tightly
    for step in range(50):
        action = torch.zeros((1, 7), device="cuda:0")
        action[0, 6] = -1.0 # CLOSE gripper
        env.step(action)
        if step == 49:
            print("[TEST] Stage 3 done: Gripper Closed")
            
    # 4. Pull the drawer
    for step in range(200):
        action = torch.zeros((1, 7), device="cuda:0")
        action[0, 1] = 1.0 # Pull in +Y or -Y? The open axis for Franka cabinet is -Y? Actually let's check open_axis in observation.
        # Wait, earlier we logged open_axis = (-0.00, -1.00, -0.00) so pulling is -Y direction.
        action[0, 1] = -1.0 # Pull backward
        action[0, 6] = -1.0 # KEEP Gripper Closed
        env.step(action)
        
        if step % 20 == 0:
            drawer_joint = env.unwrapped.scene["cabinet"].data.joint_pos[0, 0].item()
            print(f"[TEST] Pull step {step} - Drawer Joint: {drawer_joint:.4f}")

    simulation_app.close()

if __name__ == "__main__":
    main()
