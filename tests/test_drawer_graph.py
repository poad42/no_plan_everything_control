# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Tests for DrawerAICONGraph component updates and gate logic.

Phase 3 implementation: these are smoke tests that will expand once
the full gradient path wiring is complete.
"""

import math
import pytest
import torch

from no_plan_everything_control.aicon.interconnections import GraspGate, VisibilityGate
from no_plan_everything_control.envs.drawer_manipulation.aicon_graph import DrawerAICONGraph


class TestDrawerAICONGraphComponents:
    def _make_obs(self, ee_pos=None, hand=0.0, ft=None):
        return {
            "ee_pos": torch.tensor(ee_pos or [0.0, 0.0, 0.5]),
            "ee_vel": torch.zeros(6),
            "hand_state": torch.tensor(hand),
            "ft_wrench": ft if ft is not None else torch.zeros(6),
            "rgb_image": torch.zeros(64, 64, 3, dtype=torch.uint8),
        }

    def test_ee_component_updates(self):
        graph = DrawerAICONGraph(device="cpu")
        obs = self._make_obs(ee_pos=[1.0, 0.0, 0.5])
        graph.step(obs)
        # EE component should have moved toward measurement
        assert graph.ee_component.state.norm().item() > 0.0

    def test_step_returns_action_shape(self):
        graph = DrawerAICONGraph(device="cpu")
        obs = self._make_obs()
        action = graph.step(obs)
        assert action.shape == (7,), f"Expected (7,) action, got {action.shape}"

    def test_hand_state_updates(self):
        graph = DrawerAICONGraph(device="cpu")
        obs = self._make_obs(hand=0.8)
        graph.step(obs)
        # After one step, hand component should reflect measurement
        assert abs(graph.hand_component.state[0].item() - 0.8) < 0.7


class TestVisibilityGateIntegration:
    def test_frontward_visible(self):
        gate = VisibilityGate(fov_rad=math.radians(40.0))
        # Handle directly in front (camera looks along -Z in camera frame → positive Z = forward)
        handle = torch.tensor([0.0, 0.0, 0.5])
        p = gate(handle)
        assert p.item() > 0.9

    def test_backward_invisible(self):
        gate = VisibilityGate(fov_rad=math.radians(40.0))
        handle = torch.tensor([0.0, 0.0, -0.5])  # behind camera
        p = gate(handle)
        assert p.item() < 0.5  # behind camera = not visible


class TestGraspGateIntegration:
    def test_grasped_scenario(self):
        gate = GraspGate(dist_threshold=0.08, dist_sharpness=50.0, ft_threshold=1.5, ft_sharpness=3.0)
        ee = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([0.02, 0.0, 0.0])
        x_hand = torch.tensor(0.05)  # almost closed
        ft = torch.tensor([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
        p = gate(ee, target, x_hand, ft)
        assert p.item() > 0.3

    def test_not_grasped_open_hand(self):
        gate = GraspGate()
        ee = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([0.01, 0.0, 0.0])
        x_hand = torch.tensor(0.9)  # open hand
        ft = torch.zeros(6)
        p = gate(ee, target, x_hand, ft)
        assert p.item() < 0.3
