# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""AICON policy for Blocks World.

Wraps the AICONSolver with BlockStateComponent and the BW-specific
gradient paths (∇_stack and ∇_unstack from Eqs. 7 and 8).

Phase 2 implementation target.
"""

from __future__ import annotations

import torch
from torch import Tensor

from no_plan_everything_control.aicon.components import BlockStateComponent
from no_plan_everything_control.aicon.gradient_descent import AICONSolver, GradientPath
from no_plan_everything_control.aicon.utils import stacks_to_on_matrix


class BlocksWorldAICON:
    """AICON policy for the Blocks World domain.

    At each step, given the current state (o, c), the solver computes
    gradient paths for all feasible (stack, X, Y) and (unstack, Z, X)
    combinations and selects the action with the steepest gradient.

    Args:
        n_blocks:     Number of blocks.
        goal_on:      Goal on(X,Y) matrix.
        gain_k:       Gradient descent step size.
        interconnected_goal: If True, gates the goal through current clearness
                             (Sec. IV-B variant that reduces competing subgoals).
        device:       Torch device.
    """

    def __init__(
        self,
        n_blocks: int,
        goal_on: Tensor,
        gain_k: float = 1.0,
        interconnected_goal: bool = True,
        device: str = "cpu",
    ) -> None:
        self._n = n_blocks
        self._goal_on = goal_on.to(device)
        self._k = gain_k
        self._interconnected = interconnected_goal
        self._device = device

        self._state = BlockStateComponent(n_blocks, device=device)

    def reset(self, initial_stacks: list[list[int]]) -> None:
        """Initialise the state from a known initial configuration."""
        on = stacks_to_on_matrix(initial_stacks, self._n).to(self._device)
        self._state.set_initial_state(on)

    def step(self, observed_on: Tensor | None = None) -> tuple[str, int, int]:
        """Select the best action using steepest gradient descent.

        Args:
            observed_on: Optionally update state from simulation readout.

        Returns:
            ('stack', X, Y) or ('unstack', Z, X) action tuple.
        """
        if observed_on is not None:
            self._state.set_initial_state(observed_on.to(self._device))

        o = self._state.on      # (n, n) requires no grad at this point
        c = self._state.clear   # (n,)

        # Build goal cost
        if self._interconnected:
            # Gate goal by current clearness to break competing subgoals
            goal_cost = self._interconnected_goal_cost(o, c)
        else:
            goal_cost = ((o - self._goal_on) ** 2).sum()

        # Enumerate gradient paths and select steepest action
        best_action, best_norm = self._select_action(o, c, goal_cost)
        return best_action

    def _goal_cost(self, o: Tensor) -> Tensor:
        return ((o - self._goal_on) ** 2).sum()

    def _interconnected_goal_cost(self, o: Tensor, c: Tensor) -> Tensor:
        # Prioritise unstacking of lowest blocked tower by weighting goal
        # contributions by the inverse of clearness (blocked blocks matter more)
        weight = (1.0 - c).unsqueeze(1) + 1e-3
        return (weight * (o - self._goal_on) ** 2).sum()

    def _select_action(
        self,
        o: Tensor,
        c: Tensor,
        goal_cost: Tensor,
    ) -> tuple[tuple[str, int, int], float]:
        """Enumerate all valid (stack/unstack, X, Y) gradients and pick steepest."""
        best_action = ("stack", 0, 1)
        best_norm = -1.0

        for X in range(self._n):
            for Y in range(self._n):
                if X == Y:
                    continue

                # --- ∇_stack (Eq. 7) ---
                # Valid if X is clear AND Y is clear
                if c[X] > 0.5 and c[Y] > 0.5:
                    a_stack = torch.zeros(self._n, self._n, device=self._device)
                    a_stack[X, Y] = 1.0
                    o_new = (o + c[X] * c[Y] * a_stack).clamp(0.0, 1.0)
                    c_new = 1.0 - o_new.sum(dim=0) / self._n
                    if self._interconnected:
                        cost_new = self._interconnected_goal_cost(o_new, c_new)
                    else:
                        cost_new = self._goal_cost(o_new)
                    grad_norm = float((goal_cost - cost_new).abs().item())
                    if grad_norm > best_norm:
                        best_norm = grad_norm
                        best_action = ("stack", X, Y)

                # --- ∇_unstack (Eq. 8) ---
                # Valid if X is on Y (o[X,Y] > 0.5) AND X is clear
                if o[X, Y] > 0.5 and c[X] > 0.5:
                    a_unstack = torch.zeros(self._n, self._n, device=self._device)
                    a_unstack[X, Y] = 1.0
                    o_new = (o - c[X] * a_unstack).clamp(0.0, 1.0)
                    c_new = 1.0 - o_new.sum(dim=0) / self._n
                    if self._interconnected:
                        cost_new = self._interconnected_goal_cost(o_new, c_new)
                    else:
                        cost_new = self._goal_cost(o_new)
                    grad_norm = float((goal_cost - cost_new).abs().item())
                    if grad_norm > best_norm:
                        best_norm = grad_norm
                        best_action = ("unstack", X, Y)

        return best_action, best_norm
