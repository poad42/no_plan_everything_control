# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Unit tests for AICON components (Eq. 1)."""

import pytest
import torch

from no_plan_everything_control.aicon.components import (
    BlockStateComponent,
    EKFComponent,
    MovingAverageComponent,
)


class TestMovingAverageComponent:
    def test_converges_to_measurement(self):
        comp = MovingAverageComponent(state_dim=3, alpha=0.5)
        target = torch.tensor([1.0, 2.0, 3.0])
        for _ in range(50):
            comp.update({"measurement": target})
        assert torch.allclose(comp.state, target, atol=1e-3)

    def test_state_dim(self):
        comp = MovingAverageComponent(state_dim=5, alpha=0.1)
        assert comp.state.shape == (5,)


class TestEKFComponent:
    def test_update_moves_toward_measurement(self):
        comp = EKFComponent(
            state_dim=3,
            process_noise_cov=torch.eye(3) * 1e-4,
            measurement_noise_cov=torch.eye(3) * 1e-2,
        )
        z = torch.tensor([5.0, 5.0, 5.0])
        for _ in range(20):
            comp.update({"measurement": z})
        assert torch.allclose(comp.state, z, atol=0.1)

    def test_no_measurement_predict_only(self):
        comp = EKFComponent(
            state_dim=2,
            process_noise_cov=torch.eye(2) * 1e-4,
            measurement_noise_cov=torch.eye(2) * 1e-2,
        )
        state_before = comp.state.clone()
        comp.update({})
        # Without measurement, state stays same (identity process model)
        assert torch.allclose(comp.state, state_before, atol=1e-6)

    def test_uncertainty_increases_without_measurement(self):
        comp = EKFComponent(
            state_dim=2,
            process_noise_cov=torch.eye(2) * 0.1,
            measurement_noise_cov=torch.eye(2) * 1e-2,
        )
        sigma_before = comp.uncertainty.clone()
        comp.update({})
        # Uncertainty should increase (predict step adds Q)
        assert comp.uncertainty.diag().sum() > sigma_before.diag().sum()


class TestBlockStateComponent:
    def test_initial_all_clear(self):
        """Freshly initialised component: nothing stacked, all blocks clear."""
        n = 4
        comp = BlockStateComponent(n_blocks=n)
        assert torch.all(comp.clear > 0.9)
        assert comp.on.sum() < 1e-6

    def test_stack_action_updates_on(self):
        """Stacking block 0 on block 1 should set o[0,1] > 0."""
        n = 3
        comp = BlockStateComponent(n_blocks=n)
        a_stack = torch.zeros(n, n)
        a_stack[0, 1] = 1.0
        comp.update({"a_stack": a_stack})
        assert comp.on[0, 1] > 0.9, "Block 0 should be on block 1 after stack"

    def test_unstack_clears_on(self):
        """Unstacking X from Y should set o[X,Y] back toward 0."""
        n = 3
        comp = BlockStateComponent(n_blocks=n)
        # First stack 0 on 1
        a_stack = torch.zeros(n, n)
        a_stack[0, 1] = 1.0
        comp.update({"a_stack": a_stack})
        assert comp.on[0, 1] > 0.5

        # Now unstack
        a_unstack = torch.zeros(n, n)
        a_unstack[0, 1] = 1.0
        comp.update({"a_unstack": a_unstack})
        assert comp.on[0, 1] < 0.5, "Block 0 should no longer be on block 1 after unstack"

    def test_clear_updates_from_on(self):
        """c(X) should decrease when something is stacked on X (Eq. 5)."""
        n = 3
        comp = BlockStateComponent(n_blocks=n)
        # Stack block 0 on block 1 → block 1 should become less clear
        a_stack = torch.zeros(n, n)
        a_stack[0, 1] = 1.0
        comp.update({"a_stack": a_stack})
        # c(1) = 1 - (1/3)*o(X,1 for all X); o[0,1]=~1 so c(1) < 1
        assert comp.clear[1] < 0.9, "Block 1 should not be fully clear after stack"

    def test_eq5_compliance(self):
        """Verify Eq. (5): c_t(X) = 1 - (1/|B|) * sum_{Y} o_t(Y,X)."""
        n = 4
        comp = BlockStateComponent(n_blocks=n)
        a_stack = torch.zeros(n, n)
        a_stack[0, 2] = 1.0  # block 0 on block 2
        comp.update({"a_stack": a_stack})
        expected_c2 = 1.0 - comp.on[:, 2].sum().item() / n
        assert abs(comp.clear[2].item() - expected_c2) < 1e-5

    def test_gradient_flows_through_update(self):
        """Gradients must flow through BlockStateComponent for AICON to work."""
        n = 3
        comp = BlockStateComponent(n_blocks=n)
        a_stack = torch.zeros(n, n, requires_grad=True)
        comp.update({"a_stack": a_stack})
        # Compute a scalar cost on o
        cost = comp.on.sum()
        cost.backward()
        assert a_stack.grad is not None
        assert a_stack.grad.sum().item() != 0.0

    def test_set_initial_state(self):
        """set_initial_state should wire o and c correctly."""
        n = 3
        comp = BlockStateComponent(n_blocks=n)
        on = torch.zeros(n, n)
        on[1, 0] = 1.0  # block 1 is on block 0
        comp.set_initial_state(on)
        assert comp.on[1, 0] > 0.9
        # c(0) should be less than 1 (block 1 is on top of it)
        assert comp.clear[0] < 0.9
