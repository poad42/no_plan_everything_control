# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""AICON Gradient Descent Solver — Eqs. (3) and (4) of Mengers & Brock, ICRA 2025.

At each step:
    1. Components update their estimates (Eq. 1).
    2. Active interconnections are evaluated (Eq. 2).
    3. Multiple gradient paths G(g, a) are enumerated via autograd.
    4. The STEEPEST gradient is selected (Eq. 4).
    5. The action is updated via gradient descent (Eq. 3).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .components import BaseComponent
from .interconnections import BaseInterconnection


class GradientPath:
    """Describes one path through the AICON graph from goal to action.

    A path is defined by a forward function that computes a scalar cost
    along one particular chain of components and interconnections.
    The solver calls torch.autograd.grad on this path.
    """

    def __init__(self, name: str, forward_fn: Callable[[Tensor], Tensor]) -> None:
        self.name = name
        self._forward = forward_fn

    def gradient(self, action: Tensor) -> Tensor:
        """Compute ∂(cost_along_path) / ∂action.

        Args:
            action: The current action tensor (must have requires_grad=True or be a leaf).

        Returns:
            Gradient tensor with the same shape as action.
        """
        a = action.detach().requires_grad_(True)
        cost = self._forward(a)
        if not isinstance(cost, Tensor):
            cost = torch.tensor(float(cost), device=a.device, dtype=a.dtype)
        if cost.ndim > 0:
            # Path costs are required to be scalar; reduce gracefully if needed.
            cost = cost.sum()
        if not cost.requires_grad:
            return torch.zeros_like(action)
        grad = torch.autograd.grad(cost, a, create_graph=False, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(action)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return grad


class AICONSolver:
    """Steepest-gradient action selector.

    Usage::

        solver = AICONSolver(
            components=[comp1, comp2],
            interconnections=[gate1],
            gradient_paths=[path1, path2, path3],
            gain_k=0.1,
        )
        action = torch.zeros(action_dim)
        for _ in range(max_steps):
            action = solver.step(action, obs)
    """

    def __init__(
        self,
        components: list[BaseComponent],
        interconnections: list[BaseInterconnection],
        gradient_paths: list[GradientPath],
        gain_k: float = 0.1,
    ) -> None:
        if len(gradient_paths) == 0:
            raise ValueError("AICONSolver requires at least one gradient path.")
        self._components = components
        self._interconnections = interconnections
        self._paths = gradient_paths
        self._k = gain_k
        self._last_selected_path_name: str | None = None
        self._last_gradient_norms: dict[str, float] = {}

    def _evaluate_interconnections(
        self,
        interconnection_inputs: dict[str, tuple[Tensor, ...]] | None = None,
    ) -> None:
        """Evaluate active interconnections for bookkeeping and future routing.

        Each input entry maps an interconnection index (as string) to a tuple of
        tensors passed positionally into that interconnection.
        """
        if interconnection_inputs is None:
            return
        for i, inter in enumerate(self._interconnections):
            inputs = interconnection_inputs.get(str(i))
            if inputs is None:
                continue
            # Evaluate to ensure dynamic gates are refreshed each step.
            inter(*inputs)

    def step(
        self,
        action: Tensor,
        component_priors: dict[str, dict[str, Tensor]] | None = None,
        interconnection_inputs: dict[str, tuple[Tensor, ...]] | None = None,
    ) -> Tensor:
        """Perform one AICON step.

        Args:
            action: Current action tensor.
            component_priors: mapping from component index (as str) to its priors dict.
                              If None, all components are updated with empty priors.

        Returns:
            Updated action a_{t+1}.
        """
        if component_priors is None:
            component_priors = {}

        # Step 1 & 2 — update components and interconnections
        for i, comp in enumerate(self._components):
            priors = component_priors.get(str(i), {})
            comp.update(priors)
        self._evaluate_interconnections(interconnection_inputs)

        # Step 3 — compute all gradient paths
        gradients = [path.gradient(action) for path in self._paths]

        # Step 4 — select the steepest gradient (Eq. 4)
        norms = torch.stack([g.norm() for g in gradients])
        best_idx = int(torch.argmax(norms).item())
        grad_star = gradients[best_idx]
        self._last_selected_path_name = self._paths[best_idx].name
        self._last_gradient_norms = {
            f"{path.name}[{idx}]": float(norm.item())
            for idx, (path, norm) in enumerate(zip(self._paths, norms))
        }

        # Step 5 — gradient descent step (Eq. 3)
        action_new = action - self._k * grad_star
        return action_new

    @property
    def gain(self) -> float:
        return self._k

    @gain.setter
    def gain(self, value: float) -> None:
        self._k = value

    @property
    def last_selected_path_name(self) -> str | None:
        """Name of the steepest path chosen during the most recent step."""
        return self._last_selected_path_name

    @property
    def last_gradient_norms(self) -> dict[str, float]:
        """Gradient magnitudes for all paths from the most recent step."""
        return self._last_gradient_norms
