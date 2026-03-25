# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""AICON Active Interconnections — Eq. (2) of Mengers & Brock, ICRA 2025.

Each interconnection is a differentiable implicit function c(x_1, ..., x_M)
whose output changes with the overall system state, enabling dynamic routing
of gradient information between components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch import Tensor


class BaseInterconnection(ABC):
    """Abstract base for all active interconnections.

    Implements:  c(x_1, x_2, ..., x_M)   [Eq. 2]

    The output is always differentiable with respect to all inputs so that
    gradient paths passing through this interconnection remain valid.
    """

    @abstractmethod
    def __call__(self, *quantities: Tensor) -> Tensor:
        """Compute the interconnection value given the relevant quantities."""


class SoftGate(BaseInterconnection):
    """Differentiable scalar gate: c = sigmoid(w * (x - threshold)).

    Used as a generic soft switch between 0 and 1 based on a scalar input.
    """

    def __init__(self, weight: float = 10.0, threshold: float = 0.0) -> None:
        self._weight = weight
        self._threshold = threshold

    def __call__(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return torch.sigmoid(self._weight * (x - self._threshold))


class VisibilityGate(BaseInterconnection):
    """Estimates p_visible — the likelihood that the drawer handle is in the camera FOV.

    p_visible is computed from the angle between the handle's relative position
    vector and the camera's look-at axis.

    Args:
        fov_rad: half-angle of the camera field of view in radians.
        sharpness: steepness of the sigmoid transition at the FOV boundary.
    """

    def __init__(self, fov_rad: float = math.radians(40.0), sharpness: float = 20.0) -> None:
        self._half_fov = fov_rad
        self._sharpness = sharpness

    def __call__(self, handle_pos_camera: Tensor) -> Tensor:  # type: ignore[override]
        """Compute p_visible.

        Args:
            handle_pos_camera: (3,) handle position in camera frame.

        Returns:
            Scalar tensor p_visible ∈ [0, 1].
        """
        # Angle from optical axis (Z in camera frame)
        z = handle_pos_camera[2]
        xy = torch.norm(handle_pos_camera[:2])
        angle = torch.atan2(xy, z)
        # Sigmoid: 1 when angle < fov, 0 when angle > fov
        return torch.sigmoid(self._sharpness * (self._half_fov - angle))


class GraspGate(BaseInterconnection):
    """Estimates p_grasped — the likelihood that the robot hand holds the drawer handle.

    Combines:
        - Distance from end-effector to ideal grasp pose (below half_dist_threshold → high)
        - Hand state x_hand (closed hand increases likelihood)
        - Force-torque norm (high contact force increases likelihood)

    All three terms are soft so gradient flows through.
    """

    def __init__(
        self,
        dist_threshold: float = 0.05,
        dist_sharpness: float = 50.0,
        ft_threshold: float = 2.0,
        ft_sharpness: float = 2.0,
        hand_weight: float = 2.0,
    ) -> None:
        self._dist_thr = dist_threshold
        self._dist_sharp = dist_sharpness
        self._ft_thr = ft_threshold
        self._ft_sharp = ft_sharpness
        self._hand_weight = hand_weight

    def __call__(  # type: ignore[override]
        self,
        ee_pos: Tensor,
        grasp_target_pos: Tensor,
        x_hand: Tensor,
        ft_wrench: Tensor,
    ) -> Tensor:
        """Compute p_grasped.

        Args:
            ee_pos:           (3,) end-effector position.
            grasp_target_pos: (3,) desired grasp approach position.
            x_hand:           scalar in [0,1], 0=closed, 1=open.
            ft_wrench:        (6,) force-torque measurement.

        Returns:
            Scalar tensor p_grasped ∈ [0, 1].
        """
        dist = torch.norm(ee_pos - grasp_target_pos)
        p_close = torch.sigmoid(self._dist_sharp * (self._dist_thr - dist))

        ft_norm = torch.norm(ft_wrench)
        p_force = torch.sigmoid(self._ft_sharp * (ft_norm - self._ft_thr))

        # x_hand = 0 means closed → high grasp likelihood
        p_hand = torch.sigmoid(self._hand_weight * (0.5 - x_hand))

        # Require geometric proximity as a necessity, optionally boosted by force.
        # Ensure that contact likelihood can exceed 0.5 even if ft_norm is 0,
        # so that pure geometric grasps can trigger the blue_open phase!
        p_contact = p_close * (0.8 + 0.2 * p_force)
        return p_contact * p_hand
