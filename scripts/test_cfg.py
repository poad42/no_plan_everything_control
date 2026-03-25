import argparse
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
cfg = parse_env_cfg('Isaac-Open-Drawer-Franka-IK-Rel-v0')
print("Robot Spawn:")
print(cfg.scene.robot.spawn)
print("\nCabinet Spawn:")
print(cfg.scene.cabinet.spawn)
simulation_app.close()
