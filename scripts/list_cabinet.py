from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
from pxr import Usd
stage = Usd.Stage.Open('/run/media/adhitya/Steam1/IsaacLab/source/isaaclab_assets/isaaclab_assets/props/sektion_cabinet.usd') if False else None
import isaaclab.sim as sim_utils
print("Done")
simulation_app.close()
