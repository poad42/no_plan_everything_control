# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""AICON graph for drawer manipulation — Experiment 2.

Implements the component network from paper Section V:
    EEPoseComponent      — x_ee ∈ SE(3)
    HandStateComponent   — x_hand ∈ [0,1]
    DrawerHandleComponent — x_drawer ∈ R³ + Σ_drawer ∈ R³ˣ³
    KinematicJointComponent — x_kin = (φ,θ,q,p) + Σ_kin ∈ R⁶ˣ⁶
    VisibilityGate       — p_visible
    GraspGate            — p_grasped

Four emergent gradient paths (Fig. 4):
    Blue:   open drawer directly     (active when p_grasped high)
    Green:  approach to grasp        (active when p_grasped low, Σ small)
    Orange: adjust viewpoint         (active when Σ_drawer large, p_visible high)
    Red:    reestablish visibility   (active when p_visible low)

Phase 3 implementation target.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from no_plan_everything_control.aicon.components import EKFComponent, MovingAverageComponent
from no_plan_everything_control.aicon.interconnections import GraspGate, VisibilityGate
from no_plan_everything_control.aicon.gradient_descent import AICONSolver, GradientPath

try:
    from isaaclab.utils.math import quat_box_minus
except ImportError:
    pass


class DrawerAICONGraph:
    """AICON network for the drawer manipulation task.

    All components are EKF-based with uncertainty tracking.
    The goal g(q) = (q - goal_dist)^2 is defined over the drawer joint state q.

    Args:
        goal_open_dist_m: Target opening distance in metres.
        camera_fov_deg:   Camera field of view in degrees.
        gain_k:           Gradient descent gain.
        device:           Torch device.
    """

    # Kinematic joint state dimensions: (φ, θ, q, p_x, p_y, p_z) → 6
    KIN_DIM = 6
    # EE pose in SE(3) represented as 6-vector twist
    EE_DIM = 6
    # Drawer handle in R³
    DRAWER_DIM = 3

    def __init__(
        self,
        goal_open_dist_m: float = 0.20,
        camera_fov_deg: float = 80.0,
        gain_k: float = 0.05,
        device: str = "cpu",
    ) -> None:
        self._device = device
        self._goal_q = goal_open_dist_m

        # --- Components ---
        self.ee_component = EKFComponent(
            state_dim=self.EE_DIM,
            process_noise_cov=torch.eye(self.EE_DIM, device=device) * 1e-4,
            measurement_noise_cov=torch.eye(self.EE_DIM, device=device) * 1e-3,
            device=device,
        )
        self.hand_component = MovingAverageComponent(state_dim=1, alpha=0.3, device=device)
        self.drawer_component = EKFComponent(
            state_dim=self.DRAWER_DIM,
            process_noise_cov=torch.eye(self.DRAWER_DIM, device=device) * 5e-4,
            measurement_noise_cov=torch.eye(self.DRAWER_DIM, device=device) * 1e-2,
            device=device,
        )
        self.kin_component = EKFComponent(
            state_dim=self.KIN_DIM,
            process_noise_cov=torch.eye(self.KIN_DIM, device=device) * 1e-5,
            measurement_noise_cov=torch.eye(self.KIN_DIM, device=device) * 5e-3,
            device=device,
        )

        # --- Active Interconnections ---
        self.visibility_gate = VisibilityGate(
            fov_rad=math.radians(camera_fov_deg / 2.0)
        )
        self.grasp_gate = GraspGate(
            dist_threshold=0.06,
            hand_weight=10.0,
        )

        # --- Solver (gradient paths wired in Phase 3) ---
        self._gain_k = gain_k
        self.last_selected_path_name: str | None = None
        self.last_gradient_norms: dict[str, float] = {}
        self.last_p_visible: float = 0.0
        self.last_p_grasped: float = 0.0

    def step(self, obs: dict[str, Tensor]) -> Tensor:
        """One AICON step given sensor observations.

        Args:
            obs: dictionary containing sensor readings:
                'ee_pos':       (3,) EE position in world frame
                'ee_vel':       (6,) EE velocity twist
                'hand_state':   scalar in [0,1]
                'rgb_image':    (H, W, 3) uint8 tensor from wrist camera
                'ft_wrench':    (6,) force-torque measurement
                'depth_image':  (H, W) float tensor (optional, for handle detection)

        Returns:
            (6,) EE velocity command + (1,) gripper command = (7,) action.

        TODO (Phase 3): Implement full forward pass and gradient path enumeration.
        """
        # --- Update EE component from proprioception ---
        ee_pos = obs["ee_pos"].to(self._device)
        ee_meas = torch.cat([ee_pos, torch.zeros(3, device=self._device, dtype=ee_pos.dtype)])
        self.ee_component.update({"measurement": ee_meas})

        # --- Update hand state ---
        self.hand_component.update({"measurement": obs["hand_state"].unsqueeze(0)})

        # Primary drawer measurement comes from camera/depth detection.
        if "detected_handle_pos_w" in obs:
            self.drawer_component.update({"measurement": obs["detected_handle_pos_w"].to(self._device)})
        # Optional privileged debug fallback.
        elif "handle_pos_w" in obs:
            self.drawer_component.update({"measurement": obs["handle_pos_w"].to(self._device)})

        # --- Compute likelihoods ---
        handle_in_cam = self._project_handle_to_camera(obs)
        p_visible = self.visibility_gate(handle_in_cam)

        x_hand = self.hand_component.state[0]
        p_grasped = self.grasp_gate(
            ee_pos=obs["ee_pos"],
            grasp_target_pos=self.drawer_component.state,
            x_hand=x_hand,
            ft_wrench=obs["ft_wrench"],
        )
        self.last_p_visible = float(p_visible.item())
        self.last_p_grasped = float(p_grasped.item())

        # --- Update drawer handle component ---
        drawer_priors: dict[str, Tensor] = {}
        if "detected_handle_pos_w" in obs:
            drawer_priors["measurement"] = obs["detected_handle_pos_w"].to(self._device)
        elif p_visible > 0.5 and "rgb_image" in obs:
            detected_pos = self._detect_handle_from_rgb(obs["rgb_image"], obs.get("depth_image"))
            if detected_pos is not None:
                drawer_priors["measurement"] = detected_pos
        elif p_grasped > 0.5:
            # Handle moves with EE when grasped
            drawer_priors["measurement"] = obs["ee_pos"]
        self.drawer_component.update(drawer_priors)

        # --- Update kinematic model ---
        # Use actual drawer joint extension (q) if available
        if "drawer_q" in obs:
            meas = self.kin_component.state.clone()
            meas[2] = obs["drawer_q"][0].to(self._device)
            self.kin_component.update({"measurement": meas})

        # --- Select action via steepest gradient ---
        action = self._select_action(p_visible, p_grasped, obs)
        return action

    def _project_handle_to_camera(self, obs: dict[str, Tensor]) -> Tensor:
        """Project known handle position into camera frame.

        TODO (Phase 3): use actual camera extrinsics from Isaac Lab.
        For now, returns handle_pos - ee_pos as a proxy.
        """
        if "handle_pos_camera" in obs:
            return obs["handle_pos_camera"].to(self._device)
        if "handle_pos_w" in obs:
            return obs["handle_pos_w"].to(self._device) - obs["ee_pos"].to(self._device)
        return self.drawer_component.state - obs["ee_pos"]

    def _detect_handle_from_rgb(
        self,
        rgb: Tensor,
        depth: Tensor | None = None,
    ) -> Tensor | None:
        """Detect drawer handle position from wrist RGB image.

        TODO (Phase 3): Implement lightweight detection (centroid on orange handle,
        colour threshold, or depth-based cluster centroid).

        Returns:
            (3,) handle position in world frame, or None if not detected.
        """
        return None

    def _select_action(
        self,
        p_visible: Tensor,
        p_grasped: Tensor,
        obs: dict[str, Tensor],
    ) -> Tensor:
        """Select action using four gradient paths.

        This implements a template of the paper's Fig. 4 path logic:
        - Blue: open drawer when grasped
        - Green: approach handle for grasp
        - Orange: move to informative viewpoint under high uncertainty
        - Red: reestablish visibility when handle is not visible
        """
        ee_pos = obs["ee_pos"].to(self._device)
        drawer_pos = self.drawer_component.state
        sigma_drawer = self.drawer_component.uncertainty

        # Goal term g(q) = (q - q_goal)^2; use absolute error as shared scaling.
        q_est = self.kin_component.state[2]
        goal_scale = (q_est - self._goal_q).abs() + 1e-3

        # Use drawer axis from observation when available (sim-time privileged signal).
        if "drawer_axis_w" in obs:
            open_axis = obs["drawer_axis_w"].to(self._device)
            open_axis = open_axis / (torch.norm(open_axis) + 1e-6)
        else:
            open_axis = torch.tensor([1.0, 0.0, 0.0], device=self._device)

        # Path targets in action space. Scale them similarly to prevent
        # gradient magnitude bias from dominating path selection.
        target_open = (open_axis * 0.3).clamp(-0.3, 0.3)
        
        # Approach from far in front of the drawer first, then slide in.
        dist_to_handle = torch.norm(drawer_pos - ee_pos).item()
        # If far away vertically, stay far back to avoid clipping cabinet face.
        z_dist = abs(drawer_pos[2] - ee_pos[2]).item()
        if z_dist > 0.1:
            swoop_offset_m = 0.30  # Stay 30cm away horizontally when moving vertically
        else:
            swoop_offset_m = 0.0  # Just go to the drawer
            
        pre_grasp_pos = drawer_pos + open_axis * swoop_offset_m
        target_approach = (pre_grasp_pos - ee_pos).clamp(-0.3, 0.3)
        target_view = (drawer_pos + torch.tensor([0.0, 0.12, 0.10], device=self._device) - ee_pos).clamp(-0.3, 0.3)
        target_reobserve = (drawer_pos + torch.tensor([0.0, -0.15, 0.12], device=self._device) - ee_pos).clamp(-0.3, 0.3)

        # Use uncertainty trace as a smooth scalar signal.
        uncertainty = torch.trace(sigma_drawer)
        uncertainty_norm = uncertainty / (uncertainty + 1.0)
        det_conf = obs.get("handle_detection_conf", torch.tensor(0.0, device=self._device))
        if not isinstance(det_conf, Tensor):
            det_conf = torch.tensor(float(det_conf), device=self._device)
        det_conf = det_conf.clamp(0.0, 1.0)

        w_open = p_grasped * goal_scale
        w_approach = (1.0 - p_grasped) * p_visible * (1.0 - uncertainty_norm) * goal_scale
        w_view = (1.0 - p_grasped) * p_visible * uncertainty_norm * goal_scale
        w_reobserve = (1.0 - p_visible) * goal_scale
        w_probe = (1.0 - p_grasped) * p_visible * (1.0 - det_conf) * goal_scale

        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self._device)
        side_axis = torch.cross(open_axis, up_axis, dim=0)
        if torch.norm(side_axis) < 1e-6:
            side_axis = torch.tensor([0.0, 1.0, 0.0], device=self._device)
        side_axis = side_axis / (torch.norm(side_axis) + 1e-6)
        approach_axis = -open_axis

        target_probe_left = (drawer_pos + 0.08 * side_axis - ee_pos).clamp(-0.3, 0.3)
        target_probe_right = (drawer_pos - 0.08 * side_axis - ee_pos).clamp(-0.3, 0.3)
        target_probe_up = (drawer_pos + 0.08 * up_axis - ee_pos).clamp(-0.3, 0.3)
        target_probe_front_up = (
            drawer_pos + 0.06 * approach_axis + 0.04 * up_axis - ee_pos
        ).clamp(-0.3, 0.3)
        target_probe_front_down = (
            drawer_pos + 0.06 * approach_axis - 0.04 * up_axis - ee_pos
        ).clamp(-0.3, 0.3)

        ee_quat = obs.get("ee_quat", None)
        handle_quat = obs.get("handle_quat", None)
        target_approach_rot = None
        
        # We need the hand to face +Y (the drawer pulls out in -Y).
        # We will match the handle's exact orientation to prevent IK singularities during the pinch.
        if ee_quat is not None and handle_quat is not None and 'quat_box_minus' in globals():
            try:
                # Match the handle's orientation directly. This ensures the fingers
                # are perfectly aligned with the handle geometry (horizontally/vertically)
                # and won't get stuck in a kinematic singularity trying to achieve a hardcoded quat.
                target_q = handle_quat
                err_rot = quat_box_minus(target_q.unsqueeze(0), ee_quat.unsqueeze(0)).squeeze(0)
                target_approach_rot = err_rot.clamp(-1.0, 1.0)
            except Exception as e:
                pass

        def _path_cost(target_vel: Tensor, target_gripper: float, weight: Tensor, target_rot: Tensor | None = None):
            def _forward(a: Tensor) -> Tensor:
                vel_cost = ((a[:3] - target_vel) ** 2).sum()
                if target_rot is not None:
                    rot_cost = ((a[3:6] - target_rot) ** 2).sum()
                else:
                    # Keep rotational command modest in this template stage.
                    rot_cost = 0.03 * (a[3:6] ** 2).sum()
                
                grip_cost = 0.2 * (a[6] - target_gripper) ** 2
                return weight * (vel_cost + rot_cost + grip_cost)

            return _forward

        paths = [
            # Gripper convention for BinaryJointPositionAction: negative=close, positive=open.
            GradientPath("blue_open", _path_cost(target_open, 0.00, w_open)),
            GradientPath("green_approach", _path_cost(target_approach, 1.0, w_approach, target_rot=target_approach_rot)),
            GradientPath("orange_view", _path_cost(target_view, 1.0, w_view)),
            GradientPath("red_reobserve", _path_cost(target_reobserve, 1.0, w_reobserve)),
            GradientPath("probe_left", _path_cost(target_probe_left, 0.5, 0.60 * w_probe)),
            GradientPath("probe_right", _path_cost(target_probe_right, 0.5, 0.60 * w_probe)),
            GradientPath("probe_up", _path_cost(target_probe_up, 0.5, 0.70 * w_probe)),
            GradientPath("probe_front_up", _path_cost(target_probe_front_up, 0.5, 0.75 * w_probe)),
            GradientPath("probe_front_down", _path_cost(target_probe_front_down, 0.5, 0.75 * w_probe)),
        ]

        solver = AICONSolver(
            components=[],
            interconnections=[],
            gradient_paths=paths,
            gain_k=self._gain_k,
        )

        a0 = torch.zeros(7, device=self._device)
        action = solver.step(a0)
        self.last_selected_path_name = solver.last_selected_path_name
        self.last_gradient_norms = solver.last_gradient_norms
        return action.clamp(-1.0, 1.0)
