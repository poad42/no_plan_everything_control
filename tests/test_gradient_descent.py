# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Unit tests for the AICON gradient descent solver (Eqs. 3 & 4)."""

import pytest
import torch
from torch import Tensor

from no_plan_everything_control.aicon.gradient_descent import AICONSolver, GradientPath


def _make_quadratic_path(target: float) -> GradientPath:
    """Returns a path whose cost is (a - target)^2."""
    def forward(a: Tensor) -> Tensor:
        return (a - target) ** 2
    return GradientPath("quadratic", forward)


class TestGradientPath:
    def test_gradient_at_target_is_zero(self):
        path = _make_quadratic_path(3.0)
        a = torch.tensor(3.0)
        grad = path.gradient(a)
        assert grad.abs().item() < 1e-5

    def test_gradient_direction_correct(self):
        path = _make_quadratic_path(1.0)
        a = torch.tensor(0.0)  # to the left of target
        grad = path.gradient(a)
        # d/da (a-1)^2 = 2(a-1) = -2 at a=0 → gradient is negative
        assert grad.item() < 0

    def test_gradient_unused_returns_zero(self):
        """If action is not in the path (allow_unused), return zero grad."""
        def forward_no_a(_a: Tensor) -> Tensor:
            return torch.tensor(1.0)
        path = GradientPath("constant", forward_no_a)
        a = torch.tensor(5.0)
        grad = path.gradient(a)
        assert grad.item() == 0.0


class TestAICONSolver:
    def test_requires_at_least_one_path(self):
        with pytest.raises(ValueError):
            AICONSolver(components=[], interconnections=[], gradient_paths=[])

    def test_steepest_path_selected(self):
        """Solver should select the path with the largest gradient norm."""
        path_steep = _make_quadratic_path(10.0)  # far from target → large gradient
        path_flat = _make_quadratic_path(0.01)   # near target → small gradient

        solver = AICONSolver(
            components=[],
            interconnections=[],
            gradient_paths=[path_flat, path_steep],
            gain_k=0.1,
        )
        a = torch.tensor(0.0)
        a_new = solver.step(a)
        # Steep path has gradient +2*(a-10)=-20 → step should move a toward 10
        assert a_new.item() > a.item(), "Action should move toward steep path target"
        assert solver.last_selected_path_name == "quadratic"
        assert len(solver.last_gradient_norms) == 2

    def test_gradient_descent_converges(self):
        """Solver should converge to target on a single-path quadratic."""
        target = 5.0
        path = _make_quadratic_path(target)
        solver = AICONSolver(
            components=[],
            interconnections=[],
            gradient_paths=[path],
            gain_k=0.5,
        )
        a = torch.tensor(0.0)
        for _ in range(100):
            a = solver.step(a)
        assert abs(a.item() - target) < 0.1

    def test_stuck_path_zero_gradient(self):
        """A path at its minimum returns zero gradient → other path dominates."""
        path_at_min = _make_quadratic_path(0.0)  # a=0 is already at minimum
        path_active = _make_quadratic_path(2.0)  # still has gradient at a=0

        solver = AICONSolver(
            components=[],
            interconnections=[],
            gradient_paths=[path_at_min, path_active],
            gain_k=1.0,
        )
        a = torch.tensor(0.0)
        a_new = solver.step(a)
        # path_at_min gradient = 0, path_active gradient = 2*(0-2)=-4
        # Solver should move toward 2
        assert a_new.item() > 0.0
