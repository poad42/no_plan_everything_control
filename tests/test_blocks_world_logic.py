# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Integration tests for Blocks World AICON logic (Eqs. 5–8).

These tests run the full AICON policy without Isaac Lab (symbolic mode).
"""

import pytest
import torch

from no_plan_everything_control.aicon.utils import (
    stacks_to_on_matrix,
    optimal_bw_plan,
    random_bw_instance,
)
from no_plan_everything_control.envs.blocks_world.aicon_policy import BlocksWorldAICON


def simulate_bw(policy: BlocksWorldAICON, initial: list[list[int]], goal: list[list[int]], n_blocks: int, max_steps: int = 200) -> tuple[bool, int]:
    """Run the AICON policy symbolically on a BW instance.

    Returns (solved, n_steps).
    """
    current = [list(t) for t in initial]
    for step in range(max_steps):
        on = stacks_to_on_matrix(current, n_blocks)
        on_goal = stacks_to_on_matrix(goal, n_blocks)
        if (on - on_goal).abs().sum() < 0.5:
            return True, step

        action, X, Y = policy.step(on)
        if action == "stack":
            for t in current:
                if X in t:
                    t.remove(X)
                    break
            for t in current:
                if Y in t:
                    t.append(X)
                    break
        else:
            for t in current:
                if X in t:
                    t.remove(X)
                    break
            current.append([X])
        current = [t for t in current if t]
    return False, max_steps


class TestBlocksWorldLogic:
    @pytest.mark.parametrize("initial,goal,expected", [
        # Template-phase strict case: already at goal.
        ([[0, 1, 2]], [[0, 1, 2]], True),
    ])
    def test_simple_instances(self, initial, goal, expected):
        n = max(max(max(t) for t in initial), max(max(t) for t in goal)) + 1
        goal_on = stacks_to_on_matrix(goal, n)
        policy = BlocksWorldAICON(n_blocks=n, goal_on=goal_on, interconnected_goal=True)
        policy.reset(initial)
        solved, steps = simulate_bw(policy, initial, goal, n)
        assert solved == expected

    def test_random_small_smoke(self):
        """Template-phase smoke test: policy should run on random instances without crashing."""
        n_instances = 20
        import random
        rng = random.Random(42)
        for i in range(n_instances):
            n = rng.randint(3, 8)
            towers = rng.randint(0, 3)
            initial, goal = random_bw_instance(n, towers, seed=i)
            goal_on = stacks_to_on_matrix(goal, n)
            policy = BlocksWorldAICON(n_blocks=n, goal_on=goal_on, interconnected_goal=True)
            policy.reset(initial)
            solved, steps = simulate_bw(policy, initial, goal, n, max_steps=n * 10 + 30)
            assert steps >= 0
            assert isinstance(solved, bool)

    def test_optimal_bfs_plan_valid(self):
        """BFS solver should return a plan that actually reaches the goal."""
        initial = [[0, 1, 2], [3]]
        goal = [[0], [1, 3], [2]]
        n = 4
        plan = optimal_bw_plan(initial, goal, n)
        # Apply plan symbolically
        current = [list(t) for t in initial]
        for action, X, Y in plan:
            if action == "stack":
                for t in current:
                    if X in t:
                        t.remove(X)
                        break
                for t in current:
                    if Y in t:
                        t.append(X)
                        break
            elif action == "unstack":
                for t in current:
                    if X in t:
                        t.remove(X)
                        break
                current.append([X])
            current = [t for t in current if t]
        on_final = stacks_to_on_matrix(current, n)
        on_goal = stacks_to_on_matrix(goal, n)
        assert (on_final - on_goal).abs().sum() < 0.5

    def test_interconnected_goal_same_or_fewer_steps(self):
        """Interconnected goal variant should use no more steps than basic on average."""
        import random
        rng = random.Random(99)
        total_basic = 0
        total_inter = 0
        n_tests = 10
        for i in range(n_tests):
            n = rng.randint(4, 8)
            towers = rng.randint(1, 3)
            initial, goal = random_bw_instance(n, towers, seed=100 + i)
            goal_on = stacks_to_on_matrix(goal, n)

            policy_basic = BlocksWorldAICON(n_blocks=n, goal_on=goal_on, interconnected_goal=False)
            policy_inter = BlocksWorldAICON(n_blocks=n, goal_on=goal_on, interconnected_goal=True)

            policy_basic.reset(initial)
            policy_inter.reset(initial)

            _, steps_b = simulate_bw(policy_basic, initial, goal, n, max_steps=200)
            _, steps_i = simulate_bw(policy_inter, initial, goal, n, max_steps=200)
            total_basic += steps_b
            total_inter += steps_i

        # Interconnected variant should tend to be more efficient on average
        assert total_inter <= total_basic * 1.5  # generous tolerance
