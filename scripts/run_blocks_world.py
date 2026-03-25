#!/usr/bin/env python3
# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Blocks World AICON experiment — Isaac Lab entry-point.

Launch with:
    cd /run/media/adhitya/Steam1/IsaacLab
    ./isaaclab.sh -p /run/media/adhitya/Steam1/no_plan_everything_control/scripts/run_blocks_world.py

Phase 1 (pure logic, no sim): set --headless --no-sim to run the AICON BW logic
against BFS baseline without launching Isaac Lab.

Phase 2 (full sim): launches Isaac Lab, spawns blocks, and evaluates the AICON
policy against the optimal BFS planner across 130 random instances.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AICON Blocks World experiment")
    p.add_argument("--num-instances", type=int, default=130, help="Number of BW instances (paper: 130)")
    p.add_argument("--min-blocks", type=int, default=10)
    p.add_argument("--max-blocks", type=int, default=30)
    p.add_argument("--min-towers", type=int, default=0)
    p.add_argument("--max-towers", type=int, default=15)
    p.add_argument("--no-sim", action="store_true", help="Run pure logic without Isaac Lab")
    p.add_argument("--interconnected-goal", action="store_true", default=True)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs/blocks_world")
    return p.parse_args()


def run_pure_logic(args: argparse.Namespace) -> None:
    """Phase 1: run AICON BW policy against BFS without Isaac Lab."""
    import random
    from no_plan_everything_control.aicon.utils import (
        random_bw_instance,
        stacks_to_on_matrix,
        optimal_bw_plan,
    )
    from no_plan_everything_control.envs.blocks_world.aicon_policy import BlocksWorldAICON

    results = []
    rng = random.Random(args.seed_offset)

    for i in range(args.num_instances):
        n_blocks = rng.randint(args.min_blocks, args.max_blocks)
        n_towers = rng.randint(args.min_towers, args.max_towers)
        seed = args.seed_offset + i

        initial, goal = random_bw_instance(n_blocks, n_towers, seed=seed)

        # Ground-truth optimal plan
        optimal = optimal_bw_plan(initial, goal, n_blocks)
        opt_steps = len(optimal)

        # AICON policy (symbolic, no sim)
        goal_on = stacks_to_on_matrix(goal, n_blocks)
        policy = BlocksWorldAICON(
            n_blocks=n_blocks,
            goal_on=goal_on,
            interconnected_goal=args.interconnected_goal,
        )
        policy.reset(initial)

        # Simulate actions symbolically (teleport blocks)
        from no_plan_everything_control.aicon.utils import stacks_to_on_matrix

        current_stacks = [list(t) for t in initial]
        n_steps = 0
        max_steps = opt_steps * 5 + 50
        solved = False

        for _ in range(max_steps):
            on = stacks_to_on_matrix(current_stacks, n_blocks)
            # Check if goal reached
            on_goal = stacks_to_on_matrix(goal, n_blocks)
            if (on - on_goal).abs().sum() < 0.5:
                solved = True
                break

            action, X, Y = policy.step(on)
            # Apply symbolic action
            if action == "stack":
                # Move block X onto block Y (teleport)
                for t in current_stacks:
                    if X in t:
                        t.remove(X)
                        break
                for t in current_stacks:
                    if Y in t:
                        t.append(X)
                        break
            else:  # unstack
                # Remove X from Y (place X on table as new singleton tower)
                for t in current_stacks:
                    if X in t:
                        t.remove(X)
                        break
                current_stacks.append([X])
            current_stacks = [t for t in current_stacks if t]
            n_steps += 1

        ratio = n_steps / max(opt_steps, 1)
        results.append(
            {"instance": i, "n_blocks": n_blocks, "n_towers": n_towers,
             "solved": solved, "aicon_steps": n_steps, "optimal_steps": opt_steps, "ratio": ratio}
        )
        print(f"[{i+1}/{args.num_instances}] n={n_blocks} towers={n_towers} "
              f"solved={solved} steps={n_steps} optimal={opt_steps} ratio={ratio:.2f}")

    # Summary
    n_solved = sum(r["solved"] for r in results)
    mean_ratio = sum(r["ratio"] for r in results) / len(results)
    print(f"\n=== RESULTS ===")
    print(f"Solved: {n_solved}/{args.num_instances} ({100*n_solved/args.num_instances:.1f}%)")
    print(f"Mean steps/optimal ratio: {mean_ratio:.3f}")

    # Save results
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({"summary": {"solved": n_solved, "total": args.num_instances, "mean_ratio": mean_ratio},
                   "instances": results}, f, indent=2)
    print(f"Results saved to {out / 'results.json'}")


def run_with_sim(args: argparse.Namespace) -> None:
    """Phase 2: run AICON BW with Isaac Lab simulation.

    TODO (Phase 2): Implement full Isaac Lab scene integration.
    """
    raise NotImplementedError(
        "Isaac Lab scene integration is Phase 2. "
        "Run with --no-sim for the pure logic experiment."
    )


def main() -> None:
    args = parse_args()
    if args.no_sim:
        run_pure_logic(args)
    else:
        # TODO (Phase 2): launch Isaac Lab app before calling run_with_sim
        run_with_sim(args)


if __name__ == "__main__":
    main()
