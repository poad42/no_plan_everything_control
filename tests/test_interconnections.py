# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Unit tests for AICON Active Interconnections (Eq. 2)."""

import math
import pytest
import torch

from no_plan_everything_control.aicon.interconnections import GraspGate, SoftGate, VisibilityGate


class TestSoftGate:
    def test_one_above_threshold(self):
        gate = SoftGate(weight=20.0, threshold=0.5)
        out = gate(torch.tensor(1.0))
        assert out.item() > 0.99

    def test_zero_below_threshold(self):
        gate = SoftGate(weight=20.0, threshold=0.5)
        out = gate(torch.tensor(0.0))
        assert out.item() < 0.01

    def test_differentiable(self):
        gate = SoftGate(weight=10.0, threshold=0.5)
        x = torch.tensor(0.5, requires_grad=True)
        out = gate(x)
        out.backward()
        assert x.grad is not None


class TestVisibilityGate:
    def test_visible_when_centered(self):
        """Handle directly in front of camera → p_visible ≈ 1."""
        gate = VisibilityGate(fov_rad=math.radians(40.0))
        handle = torch.tensor([0.0, 0.0, 1.0])  # along camera Z axis
        p = gate(handle)
        assert p.item() > 0.99

    def test_invisible_when_far_off_axis(self):
        """Handle 90° off axis → p_visible ≈ 0."""
        gate = VisibilityGate(fov_rad=math.radians(40.0))
        handle = torch.tensor([10.0, 0.0, 0.01])  # almost perpendicular
        p = gate(handle)
        assert p.item() < 0.01

    def test_differentiable(self):
        gate = VisibilityGate(fov_rad=math.radians(40.0))
        handle = torch.tensor([0.3, 0.0, 1.0], requires_grad=True)
        p = gate(handle)
        p.backward()
        assert handle.grad is not None


class TestGraspGate:
    def test_high_when_grasping(self):
        gate = GraspGate(dist_threshold=0.05, dist_sharpness=100.0, ft_threshold=1.0, ft_sharpness=5.0)
        ee = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([0.0, 0.0, 0.01])  # very close
        x_hand = torch.tensor(0.0)  # closed hand
        ft = torch.tensor([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])  # high force
        p = gate(ee, target, x_hand, ft)
        assert p.item() > 0.5

    def test_low_when_far(self):
        gate = GraspGate(dist_threshold=0.05)
        ee = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([1.0, 0.0, 0.0])  # far away
        x_hand = torch.tensor(0.0)
        ft = torch.zeros(6)
        p = gate(ee, target, x_hand, ft)
        assert p.item() < 0.1

    def test_differentiable(self):
        gate = GraspGate()
        ee = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        target = torch.tensor([0.02, 0.0, 0.0])
        x_hand = torch.tensor(0.1)
        ft = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        p = gate(ee, target, x_hand, ft)
        p.backward()
        assert ee.grad is not None
