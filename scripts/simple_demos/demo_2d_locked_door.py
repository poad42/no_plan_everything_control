import torch
import torch.nn.functional as F

class PointMassAICON:
    def __init__(self, start_pos, goal_pos, button_pos, gate_x=5.0):
        self.device = torch.device('cpu')
        self.agent_pos = torch.tensor(start_pos, dtype=torch.float32, requires_grad=True, device=self.device)
        self.goal_pos = torch.tensor(goal_pos, dtype=torch.float32, device=self.device)
        self.button_pos = torch.tensor(button_pos, dtype=torch.float32, device=self.device)
        self.gate_x = gate_x
        self.door_open_state = torch.tensor(0.0, dtype=torch.float32)

    def step(self, lr=0.15):
        if self.agent_pos.grad is not None:
            self.agent_pos.grad.zero_()
            
        pos = self.agent_pos
        goal_dist = torch.norm(pos - self.goal_pos)
        dist_to_button = torch.norm(pos - self.button_pos)
        
        # --- Strict Physical Trigger via STE (Straight-Through Estimator) ---
        # 1. Soft gradient field for AICON backprop 
        p_button_soft = torch.exp(-0.25 * (dist_to_button**2))
        
        # 2. Hard physical trigger: Agent MUST actually physically touch the button
        p_button_hard = (dist_to_button < 0.4).float()
        
        # 3. STE Trick: The forward pass value is strictly 0.0 or 1.0. 
        # The backward pass gradient uses the soft exponential. 
        p_button = p_button_hard + (p_button_soft - p_button_soft.detach())
        
        # Component 2: Recursive Door State
        door_prev = self.door_open_state.detach()
        door_current = door_prev + p_button * (1.0 - door_prev)
        
        # Factor representing "distance multiplier" if the path is closed
        factor = 1.0 + 10.0 * (1.0 - door_current)
        
        # Physical gate barrier (Agent hits an energy wall)
        barrier = F.relu(pos[0] - (self.gate_x - 1.0))**2 * 100.0
        
        # Dynamically Composed Objective Cost
        cost = goal_dist * factor + barrier * (1.0 - door_current)
        cost.backward()
        
        grad = self.agent_pos.grad.clone()
        grad_norm = torch.norm(grad)
        if grad_norm > 1e-6:
            grad_normalized = grad / grad_norm
        else:
            grad_normalized = grad
            
        with torch.no_grad():
            self.agent_pos -= lr * grad_normalized
            
        # Hard physical constraint simulating collision
        if self.door_open_state.item() < 0.5 and self.agent_pos[0].item() > self.gate_x:
            self.agent_pos.data[0] = self.gate_x
            
        self.door_open_state = door_current.detach()
        return self.agent_pos.clone().detach().numpy(), cost.item(), self.door_open_state.item()

def main():
    aicon = PointMassAICON(start_pos=[0.0, 0.0], goal_pos=[10.0, 0.0], button_pos=[2.0, 4.0])
    print("Starting Strict AICON 2D Simulation (STE)...")
    for step in range(200):
        pos, cost, door = aicon.step(lr=0.2)
        if step % 10 == 0:
            print(f"Step {step:03d} | Pos: {pos[0]:.2f}, {pos[1]:.2f} | Door Open: {door:.2f} | Cost: {cost:.2f}")

if __name__ == "__main__":
    main()
