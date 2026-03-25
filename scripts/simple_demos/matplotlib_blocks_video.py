import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source")))
from no_plan_everything_control.aicon.utils import stacks_to_on_matrix
from no_plan_everything_control.envs.blocks_world.aicon_policy import BlocksWorldAICON

# Config
NUM_BLOCKS = 5
BLOCK_SIZE = 1.0
COLORS = ['#e63946', '#2a9d8f', '#457b9d', '#f4a261', '#e9c46a']

# Initial & Goal setups
initial_stacks = [[0, 1, 2], [3], [4]]
goal_stacks = [[2, 1], [0], [4, 3]]

print(f"Goal: {goal_stacks}")
print(f"Start: {initial_stacks}")

# Setup AICON
policy = BlocksWorldAICON(n_blocks=NUM_BLOCKS, goal_on=stacks_to_on_matrix(goal_stacks, NUM_BLOCKS), interconnected_goal=True)
policy.reset(initial_stacks)
curr_stacks = [list(t) for t in initial_stacks]

# Track history of stacks and the action labels
history = []
action_labels = []
history.append([list(t) for t in curr_stacks])
action_labels.append(None)  # no action for the initial state

# Run symbolic planner
print("Beginning symbolic AICON transitions...")
for step in range(30):
    on_mat = stacks_to_on_matrix(curr_stacks, NUM_BLOCKS)

    # Check goal
    if torch.allclose(on_mat, policy._goal_on):
        print(f"Goal reached at step {step}!")
        break

    action_tup, best_norm = policy._select_action(
        on_mat,
        policy._state.clear,
        policy._goal_cost(on_mat)
    )
    action, X, Y = action_tup

    if action == "stack":
        for t in curr_stacks:
            if X in t: t.remove(X); break
        for t in curr_stacks:
            if t and t[-1] == Y: t.append(X); break
        label = f"Stack {X} on {Y}"
        print(f"[{step}] {label} -> {curr_stacks}")
    elif action == "unstack":
        for t in curr_stacks:
            if X in t: t.remove(X); break
        curr_stacks.append([X])
        label = f"Unstack {X} from {Y}"
        print(f"[{step}] {label} -> {curr_stacks}")

    # Clean up empty stacks
    curr_stacks = [t for t in curr_stacks if len(t) > 0]
    history.append([list(t) for t in curr_stacks])
    action_labels.append(label)

n_steps = len(history)
is_solved = torch.allclose(
    stacks_to_on_matrix(history[-1], NUM_BLOCKS),
    policy._goal_on,
)

# Build frame sequence: hold start and end longer, show every intermediate
#   - 3 repeats of frame 0 (start)
#   - 1 each for intermediates
#   - 4 repeats of final frame (solved)
frame_sequence = [0] * 3
for i in range(1, n_steps - 1):
    frame_sequence.append(i)
frame_sequence += [n_steps - 1] * 4

# Animation
SCENE_W = 10.0
SCENE_H = 6.0
fig, ax = plt.subplots(figsize=(SCENE_W, SCENE_H))

def draw_frame(seq_idx):
    frame_idx = frame_sequence[seq_idx]
    ax.clear()
    ax.set_xlim(-0.5, SCENE_W - 0.5)
    ax.set_ylim(-0.8, SCENE_H - 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    stacks = history[frame_idx]
    is_first = frame_idx == 0
    is_last = frame_idx == n_steps - 1

    # Title line
    if is_first:
        title = "Initial State"
    elif is_last and is_solved:
        title = f"Step {frame_idx} — SOLVED"
    else:
        title = f"Step {frame_idx}"
    ax.text(0, SCENE_H - 1.8, title, fontsize=14, fontweight='bold', color='#222')

    # Action annotation below title
    if action_labels[frame_idx] is not None:
        ax.text(0, SCENE_H - 2.3, action_labels[frame_idx],
                fontsize=11, fontstyle='italic', color='#555')

    # Table surface
    table_w = max(len(stacks), len(goal_stacks)) * 1.5 + 1
    ax.add_patch(patches.Rectangle((-0.3, -0.25), max(table_w, 8.5), 0.25,
                                   color='#4a4a4a'))

    # Draw current stacks
    for stack_idx, stack in enumerate(stacks):
        x = stack_idx * 1.5
        for height, block_id in enumerate(stack):
            y = height * BLOCK_SIZE
            rect = patches.FancyBboxPatch(
                (x, y), BLOCK_SIZE, BLOCK_SIZE,
                boxstyle="round,pad=0.05",
                linewidth=2, edgecolor='black',
                facecolor=COLORS[block_id % len(COLORS)],
                zorder=3)
            ax.add_patch(rect)
            ax.text(x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2, str(block_id),
                    color='white', weight='bold', fontsize=14,
                    ha='center', va='center', zorder=4)

    # Solved badge
    if is_last and is_solved:
        ax.text(len(stacks) * 1.5 + 0.3, 0.3, "Goal reached!",
                fontsize=13, fontweight='bold', color='#2a9d8f',
                bbox=dict(boxstyle='round,pad=0.3', fc='#d4f5e9', ec='#2a9d8f', lw=1.5))

    # Goal inset (top-right)
    goal_x_start = SCENE_W - len(goal_stacks) * 1.2 - 0.5
    ax.text(goal_x_start, NUM_BLOCKS - 0.3, "Goal:", fontsize=10, color='#666')
    for stack_idx, stack in enumerate(goal_stacks):
        x = goal_x_start + stack_idx * 1.2
        for height, block_id in enumerate(stack):
            y = height * 0.8
            rect = patches.FancyBboxPatch(
                (x, y), BLOCK_SIZE * 0.8, BLOCK_SIZE * 0.8,
                boxstyle="round,pad=0.03",
                linewidth=1, edgecolor='black',
                facecolor=COLORS[block_id % len(COLORS)],
                alpha=0.55)
            ax.add_patch(rect)
            ax.text(x + BLOCK_SIZE * 0.4, y + BLOCK_SIZE * 0.4, str(block_id),
                    color='white', weight='bold', fontsize=10,
                    ha='center', va='center')

    # Step counter ribbon along the bottom
    ribbon_y = -0.65
    for i in range(n_steps):
        cx = 0.3 + i * 0.6
        if cx > SCENE_W - 1:
            break
        color = '#2a9d8f' if i <= frame_idx else '#ddd'
        ax.plot(cx, ribbon_y, 'o', markersize=8, color=color, zorder=5)
        if i < n_steps - 1 and cx + 0.6 < SCENE_W - 1:
            ax.plot([cx, cx + 0.6], [ribbon_y, ribbon_y], '-', color='#ccc', lw=1, zorder=4)

anim = animation.FuncAnimation(fig, draw_frame, frames=len(frame_sequence),
                                interval=900, repeat_delay=2500)

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "videos"))
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "matplotlib_blocks.gif")
print(f"Saving animation to {out_file}...")
anim.save(out_file, writer='pillow', fps=1.2)

print("Done! You can view the GIF directly in VS Code.")
