# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""AICON Recursive Estimator Components — Eq. (1) of Mengers & Brock, ICRA 2025.

Each component owns one world-quantity estimate and updates it differentiably.
All operations are pure PyTorch so gradients flow through torch.autograd.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseComponent(ABC):
    """Abstract base for all AICON recursive estimator components.

    Implements:  x_t = f(x_{t-1}; c_1, c_2, ..., c_N)   [Eq. 1]
    """

    def __init__(self, state_dim: int, device: str = "cpu") -> None:
        self._device = device
        self._state_dim = state_dim
        self._state: Tensor = torch.zeros(state_dim, device=device)

    @abstractmethod
    def update(self, priors: dict[str, Tensor]) -> Tensor:
        """Update state estimate given priors from active interconnections.

        Args:
            priors: mapping from prior name to scalar/vector tensor.

        Returns:
            Updated state estimate x_t.
        """

    @property
    def state(self) -> Tensor:
        return self._state

    @property
    def uncertainty(self) -> Tensor | None:
        return None

    @property
    def device(self) -> str:
        return self._device


class MovingAverageComponent(BaseComponent):
    """Exponential moving average component.

    x_t = alpha * z_t + (1 - alpha) * x_{t-1}
    """

    def __init__(self, state_dim: int, alpha: float = 0.1, device: str = "cpu") -> None:
        super().__init__(state_dim, device)
        self._alpha = alpha

    def update(self, priors: dict[str, Tensor]) -> Tensor:
        """Expects priors['measurement'] as the new observation z_t."""
        z = priors["measurement"].to(self._device)
        self._state = self._alpha * z + (1.0 - self._alpha) * self._state
        return self._state


class EKFComponent(BaseComponent):
    """Extended Kalman Filter component.

    State: x, Covariance: Sigma
    Each subclass defines the process model F and measurement model H.

    All matrix operations use torch.linalg so gradients flow through.
    """

    def __init__(
        self,
        state_dim: int,
        process_noise_cov: Tensor,
        measurement_noise_cov: Tensor,
        device: str = "cpu",
    ) -> None:
        super().__init__(state_dim, device)
        self._Q = process_noise_cov.to(device)
        self._R = measurement_noise_cov.to(device)
        self._Sigma = torch.eye(state_dim, device=device) * 0.1

    @property
    def uncertainty(self) -> Tensor:
        return self._Sigma

    def _process_model(self, x: Tensor, priors: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns (x_pred, F) — predicted state and Jacobian.

        Override in subclasses to implement non-linear dynamics.
        """
        F = torch.eye(self._state_dim, device=self._device)
        return x, F

    def _measurement_model(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (z_pred, H) — predicted measurement and Jacobian.

        Override in subclasses for non-linear observations.
        """
        H = torch.eye(self._state_dim, device=self._device)
        return x, H

    def update(self, priors: dict[str, Tensor]) -> Tensor:
        """EKF predict + update step.

        Expects priors to optionally contain:
            'measurement':  actual observation z (skips update step if absent)
            'measurement_noise': override R for this step
        """
        # --- Predict ---
        x_pred, F = self._process_model(self._state, priors)
        Sigma_pred = F @ self._Sigma @ F.T + self._Q

        # --- Update (only if measurement is available) ---
        if "measurement" in priors:
            z = priors["measurement"].to(self._device)
            R = priors.get("measurement_noise", self._R).to(self._device)
            z_pred, H = self._measurement_model(x_pred)
            S = H @ Sigma_pred @ H.T + R
            K = Sigma_pred @ H.T @ torch.linalg.solve(S, torch.eye(S.shape[0], device=self._device))
            self._state = x_pred + K @ (z - z_pred)
            self._Sigma = (torch.eye(self._state_dim, device=self._device) - K @ H) @ Sigma_pred
        else:
            self._state = x_pred
            self._Sigma = Sigma_pred

        return self._state


class BlockStateComponent(BaseComponent):
    """Differentiable Blocks World state component.

    Tracks:
        o[X, Y]  — likelihood that block X is on block Y   (n_blocks × n_blocks tensor)
        c[X]     — likelihood that block X is clear        (n_blocks tensor)

    State transitions per Eqs. (5) and (6) of the paper.
    """

    def __init__(self, n_blocks: int, device: str = "cpu") -> None:
        # State is the flattened o matrix stored as 1D for BaseComponent compatibility.
        super().__init__(state_dim=n_blocks * n_blocks, device=device)
        self._n = n_blocks
        # o[i, j] = likelihood block i is on block j
        self._o = torch.zeros(n_blocks, n_blocks, device=device)
        # c[i] = likelihood block i is clear
        self._c = torch.ones(n_blocks, device=device)

    @property
    def on(self) -> Tensor:
        """o(X,Y): n_blocks × n_blocks, requires_grad-compatible."""
        return self._o

    @property
    def clear(self) -> Tensor:
        """c(X): n_blocks, derived from o via Eq. (5)."""
        return self._c

    def set_initial_state(self, on_matrix: Tensor) -> None:
        """Initialise from a ground-truth on(X,Y) matrix (from sim or problem spec)."""
        self._o = on_matrix.float().to(self._device)
        self._c = self._compute_clear(self._o)

    def _compute_clear(self, o: Tensor) -> Tensor:
        """Eq. (5): c_t(X) = 1 - (1/|B|) * sum_{Y in B} o_t(Y, X)."""
        # o[:, X] = all blocks Y on top of X => sum over first dim
        return 1.0 - o.sum(dim=0) / self._n

    def update(self, priors: dict[str, Tensor]) -> Tensor:
        """Apply stack/unstack actions differentiably via Eq. (6).

        Args:
            priors:
                'a_stack':   (n_blocks, n_blocks) action tensor — a_stack(X,Y)
                'a_unstack': (n_blocks, n_blocks) action tensor — a_unstack(X,Y)

        Returns:
            Flattened o tensor (for BaseComponent API compat).
        """
        a_stack = priors.get("a_stack", torch.zeros_like(self._o))
        a_unstack = priors.get("a_unstack", torch.zeros_like(self._o))

        requires_grad = bool(a_stack.requires_grad or a_unstack.requires_grad)
        c = self._c if requires_grad else self._c.detach()

        # Eq. (6):
        # o_t(X,Y) = o_{t-1}(X,Y)
        #           + c_{t-1}(X) * c_{t-1}(Y) * a_stack(X,Y)
        #           - c_{t-1}(X) * a_unstack(X,Y)
        self._o = (
            self._o
            + c.unsqueeze(1) * c.unsqueeze(0) * a_stack
            - c.unsqueeze(1) * a_unstack
        ).clamp(0.0, 1.0)

        self._c = self._compute_clear(self._o)
        self._state = self._o.view(-1)
        return self._state
