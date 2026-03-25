import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

def main():
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    scene_cfg.proxy_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/proxy_block",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.12, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.5)),
    )
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # Create the scene
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    
    print("\n[TEST] Successfully loaded pure test env with proxy block asset\n")
    
    for _ in range(50):
        # Step simulation
        sim.step()
        scene.write_data_to_sim()
        scene.update(sim_cfg.dt)
        
    print("\n[TEST] Successfully completed 50 steps\n")
    
if __name__ == "__main__":
    main()
    simulation_app.close()
