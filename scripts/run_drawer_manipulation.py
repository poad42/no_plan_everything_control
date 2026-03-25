#!/usr/bin/env python3
# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Drawer Manipulation AICON experiment — Isaac Lab entry-point.

Launch with:
    cd /run/media/adhitya/Steam1/IsaacLab
    ./isaaclab.sh -p /run/media/adhitya/Steam1/no_plan_everything_control/scripts/run_drawer_manipulation.py --headless

This script runs an actual Isaac Lab cabinet-opening simulation and maps the
template AICON graph actions onto the task action space for smoke-level
validation. It logs metrics from the real simulation (EE-handle distance and
drawer joint position) and writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# Ensure local package import works when running through IsaacLab python.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "source"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

parser = argparse.ArgumentParser(description="Run AICON Drawer Manipulation simulation smoke test")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--log_every", type=int, default=25)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--goal-dist-m", type=float, default=0.20, help="Target drawer opening in metres")
parser.add_argument("--aicon-gain", type=float, default=1.5, help="Gradient descent gain k for AICON policy.")
parser.add_argument("--output-dir", type=str, default="outputs/drawer_manipulation")
parser.add_argument("--record-video", action="store_true", default=False, help="Record an MP4 rollout.")
parser.add_argument("--video-length", type=int, default=240, help="Length of recorded rollout in steps.")
parser.add_argument("--dump-ee-data", action="store_true", default=False, help="Dump wrist camera and EE sensor snapshots.")
parser.add_argument("--dump-interval", type=int, default=20, help="Dump interval in simulation steps.")
parser.add_argument(
    "--use-privileged-handle",
    action="store_true",
    default=False,
    help="Use simulator handle pose as fallback (debug only, not paper-faithful).",
)
parser.add_argument(
    "--fixed-joint-assist",
    action="store_true",
    default=False,
    help="Attach EE to drawer handle with a runtime fixed joint when grasp confidence is high (debug ablation).",
)
parser.add_argument(
    "--fixed-joint-attach-threshold",
    type=float,
    default=0.72,
    help="Attach fixed joint when p_grasped exceeds this threshold.",
)
parser.add_argument(
    "--fixed-joint-detach-threshold",
    type=float,
    default=0.25,
    help="Detach fixed joint when p_grasped falls below this threshold.",
)
parser.add_argument(
    "--fixed-joint-detach-steps",
    type=int,
    default=6,
    help="Require this many consecutive low-grasp steps before detaching the fixed joint.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

from no_plan_everything_control.envs.drawer_manipulation.aicon_graph import DrawerAICONGraph


class FixedJointAssist:
    """Optional runtime fixed-joint helper for grasp/open ablation experiments."""

    def __init__(self, sim: Any, env_idx: int = 0) -> None:
        self._enabled = False
        self._joint_path = f"/World/envs/env_{env_idx}/DrawerHandleAssistJoint"
        self._ee_prim_path = f"/World/envs/env_{env_idx}/Robot/panda_hand"
        self._handle_prim_path = f"/World/envs/env_{env_idx}/Cabinet/drawer_handle_top"
        self._joint_prim = None
        self._is_attached = False

        try:
            import omni.physx.scripts.physicsUtils as physics_utils
            from pxr import Gf

            self._physx_utils = physics_utils
            self._Gf = Gf
            self._stage = sim._stage
            self._enabled = True
            print(f"[Assist] fixed-joint helper ready at {self._joint_path}")
        except Exception as exc:  # pragma: no cover - runtime-specific guard
            self._physx_utils = None
            self._Gf = None
            self._stage = None
            print(f"[Assist] disabled: failed to initialize fixed-joint helper ({exc})")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_attached(self) -> bool:
        return self._is_attached

    def attach(self) -> bool:
        """Create a fixed joint between panda hand and drawer handle."""
        if not self._enabled:
            return False
        if self._is_attached:
            return True

        try:
            self._joint_prim = self._physx_utils.add_joint_fixed(
                stage=self._stage,
                jointPath=self._joint_path,
                actor0=self._ee_prim_path,
                actor1=self._handle_prim_path,
                localPos0=self._Gf.Vec3f(0.0, 0.0, 0.0),
                localRot0=self._Gf.Quatf(1.0, 0.0, 0.0, 0.0),
                localPos1=self._Gf.Vec3f(0.0, 0.0, 0.0),
                localRot1=self._Gf.Quatf(1.0, 0.0, 0.0, 0.0),
                breakForce=1.0e30,
                breakTorque=1.0e30,
            )
            self._is_attached = self._joint_prim is not None
            if self._is_attached:
                print(f"[Assist] attached fixed joint: {self._joint_path}")
            else:
                print(f"[Assist] failed to attach fixed joint: {self._joint_path}")
            return self._is_attached
        except Exception as exc:  # pragma: no cover - runtime-specific guard
            print(f"[Assist] attach exception: {exc}")
            return False

    def detach(self) -> bool:
        """Remove the runtime fixed joint if present."""
        if not self._enabled:
            return False
        if not self._is_attached:
            return True

        try:
            self._stage.RemovePrim(self._joint_path)
            self._joint_prim = None
            self._is_attached = False
            print(f"[Assist] detached fixed joint: {self._joint_path}")
            return True
        except Exception as exc:  # pragma: no cover - runtime-specific guard
            print(f"[Assist] detach exception: {exc}")
            return False


def _map_action(graph_action: torch.Tensor, action_shape: tuple[int, ...], device: str) -> torch.Tensor:
    """Map 7D graph action [vx,vy,vz,wx,wy,wz,gripper] onto env action tensor."""
    act = torch.zeros(action_shape, device=device)
    if len(action_shape) != 2:
        return act
    
    usable = min(7, action_shape[1])
    act[:, :usable] = graph_action[:usable].unsqueeze(0).expand(action_shape[0], usable)
    return act


def _attach_end_effector_sensors(env_cfg) -> None:
    """Augment the stock cabinet env with wrist camera and finger contact sensing."""
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.cabinet.spawn.activate_contact_sensors = True

    env_cfg.scene.wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=128,
        width=128,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
    )
    env_cfg.scene.contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
        update_period=0.0,
        history_length=6,
        track_friction_forces=True,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/drawer_handle_top"],
    )


def _read_wrist_camera(
    scene,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read wrist camera outputs and camera calibration/pose tensors."""
    wrist_cam = scene["wrist_cam"]
    rgb_image = wrist_cam.data.output["rgb"][0]
    if rgb_image.shape[-1] > 3:
        rgb_image = rgb_image[..., :3]

    depth_image = None
    if "distance_to_image_plane" in wrist_cam.data.output:
        depth_image = wrist_cam.data.output["distance_to_image_plane"][0]
        if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
            depth_image = depth_image[..., 0]

    cam_pos_w = wrist_cam.data.pos_w[0].to(device)
    cam_quat_w_ros = wrist_cam.data.quat_w_ros[0].to(device)
    intrinsics = wrist_cam.data.intrinsic_matrices[0].to(device)
    return rgb_image.to(device), None if depth_image is None else depth_image.to(device), cam_pos_w, cam_quat_w_ros, intrinsics


def _detect_handle_from_wrist_depth(
    depth_image: torch.Tensor | None,
    cam_pos_w: torch.Tensor,
    cam_quat_w_ros: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor | None, float]:
    """Estimate handle world position from wrist depth via robust nearest-surface clustering."""
    if depth_image is None:
        return None, 0.0

    depth = depth_image.float()
    h, w = depth.shape
    y0, y1 = int(0.2 * h), int(0.9 * h)
    x0, x1 = int(0.15 * w), int(0.85 * w)
    roi = depth[y0:y1, x0:x1]

    valid = torch.isfinite(roi) & (roi > 0.15) & (roi < 1.20)
    if int(valid.sum().item()) < 40:
        return None, 0.0

    dvals = roi[valid]
    d_ref = torch.quantile(dvals, 0.06)
    cluster = valid & (roi <= (d_ref + 0.03))
    if int(cluster.sum().item()) < 10:
        cluster = valid & (roi <= (d_ref + 0.08))
    if int(cluster.sum().item()) < 10:
        return None, 0.0

    ys, xs = torch.nonzero(cluster, as_tuple=True)
    z = roi[ys, xs].mean()

    u = xs.float().mean() + float(x0)
    v = ys.float().mean() + float(y0)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x_cam = (u - cx) * z / (fx + 1e-8)
    y_cam = (v - cy) * z / (fy + 1e-8)
    p_cam = torch.stack([x_cam, y_cam, z])

    p_w = quat_apply(cam_quat_w_ros.unsqueeze(0), p_cam.unsqueeze(0))[0] + cam_pos_w
    cluster_ratio = float(cluster.sum().item()) / float(valid.sum().item() + 1e-6)
    depth_std = float(roi[cluster].std().item()) if int(cluster.sum().item()) > 1 else 0.0
    conf = max(0.0, min(1.0, 2.5 * cluster_ratio - 4.0 * depth_std))
    return p_w, conf


def _read_contact_wrench(scene, device: torch.device) -> torch.Tensor:
    """Aggregate filtered finger contact forces into a 6D wrench-like tensor."""
    wrench = torch.zeros(6, device=device)
    if "contact_grasp" not in scene.keys() or scene["contact_grasp"] is None:
        return wrench

    contact_force = scene["contact_grasp"].data.net_forces_w[0]
    if contact_force.ndim == 2:
        wrench[:3] = contact_force.sum(dim=0)
    elif contact_force.ndim == 1:
        wrench[:3] = contact_force
    return wrench


def _read_hand_state(robot, gripper_joint_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize Franka finger opening to [0, 1], where 1 means open."""
    finger_pos = robot.data.joint_pos[0, gripper_joint_ids]
    return (finger_pos.mean() / 0.04).clamp(0.0, 1.0).to(device)


def _save_ppm(path: Path, rgb_image: torch.Tensor) -> None:
    """Write an RGB tensor (HxWx3 uint8) to a binary PPM image file."""
    rgb_cpu = rgb_image.detach().to("cpu")
    if rgb_cpu.dtype != torch.uint8:
        rgb_cpu = rgb_cpu.clamp(0, 255).to(torch.uint8)
    h, w, c = rgb_cpu.shape
    if c != 3:
        raise ValueError("PPM writer expects 3 channels")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb_cpu.contiguous().numpy().tobytes())


def _dump_ee_snapshot(
    dump_dir: Path,
    step: int,
    rgb_image: torch.Tensor,
    depth_image: torch.Tensor | None,
    obs: dict[str, torch.Tensor],
    graph: DrawerAICONGraph,
    ft_norm: float,
    ee_drawer_dist: float,
) -> None:
    """Persist wrist camera frames and policy-input snapshot for offline inspection."""
    step_tag = f"{step:04d}"
    _save_ppm(dump_dir / f"wrist_rgb_{step_tag}.ppm", rgb_image)
    if depth_image is not None:
        torch.save(depth_image.detach().to("cpu"), dump_dir / f"wrist_depth_{step_tag}.pt")

    snapshot = {
        "step": step,
        "ft_norm": ft_norm,
        "ee_handle_dist": ee_drawer_dist,
        "p_visible": graph.last_p_visible,
        "p_grasped": graph.last_p_grasped,
        "path": graph.last_selected_path_name,
        "ee_pos": obs["ee_pos"].detach().to("cpu").tolist(),
        "ee_vel": obs["ee_vel"].detach().to("cpu").tolist(),
        "hand_state": float(obs["hand_state"].detach().to("cpu").item()),
        "ft_wrench": obs["ft_wrench"].detach().to("cpu").tolist(),
        "handle_pos_camera": obs["handle_pos_camera"].detach().to("cpu").tolist()
        if "handle_pos_camera" in obs
        else None,
        "detected_handle_pos_w": obs["detected_handle_pos_w"].detach().to("cpu").tolist()
        if "detected_handle_pos_w" in obs
        else None,
        "rgb_shape": list(rgb_image.shape),
        "rgb_dtype": str(rgb_image.dtype),
        "depth_present": depth_image is not None,
        "depth_shape": list(depth_image.shape) if depth_image is not None else None,
    }
    (dump_dir / f"policy_input_{step_tag}.json").write_text(json.dumps(snapshot, indent=2))


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    _attach_end_effector_sensors(env_cfg)
    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    video_dir = PROJECT_ROOT / args_cli.output_dir / "videos"
    if args_cli.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=min(args_cli.video_length, args_cli.num_steps),
            disable_logger=True,
        )

    graph = DrawerAICONGraph(
        goal_open_dist_m=args_cli.goal_dist_m,
        gain_k=args_cli.aicon_gain,
        device=env.unwrapped.device,
    )

    print(f"[SIM] task={args_cli.task} num_envs={args_cli.num_envs} device={env.unwrapped.device}")
    print(f"[SIM] observation_space={env.observation_space}")
    print(f"[SIM] action_space={env.action_space}")

    env.reset()
    robot = env.unwrapped.scene["robot"]
    gripper_joint_ids = robot.find_joints(["panda_finger_joint.*"])[0]

    action_norm_acc = 0.0
    ee_drawer_dist_acc = 0.0
    drawer_pos_acc = 0.0
    ft_norm_acc = 0.0
    rgb_mean_acc = 0.0
    ft_norm_max = 0.0
    detected_handle_ema: torch.Tensor | None = None
    steps_done = 0
    close_hold_steps = 0
    attach_events = 0
    detach_events = 0
    assist_low_grasp_steps = 0
    dump_dir = PROJECT_ROOT / args_cli.output_dir / "ee_observations"
    assist = None
    if args_cli.fixed_joint_assist:
        assist = FixedJointAssist(env.unwrapped.sim, env_idx=0)

    while simulation_app.is_running() and steps_done < args_cli.num_steps:
        ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
        ee_quat = env.unwrapped.scene["ee_frame"].data.target_quat_w[:, 0, :]
        handle_pos = env.unwrapped.scene["cabinet_frame"].data.target_pos_w[:, 0, :]
        handle_quat = env.unwrapped.scene["cabinet_frame"].data.target_quat_w[:, 0, :]
        cabinet_joint_pos = env.unwrapped.scene["cabinet"].data.joint_pos[:, 0]
        handle_rot = matrix_from_quat(handle_quat[0].unsqueeze(0))[0]
        handle_x = handle_rot[:, 0]
        rgb_image, depth_image, cam_pos_w, cam_quat_w_ros, intrinsics = _read_wrist_camera(
            env.unwrapped.scene,
            env.unwrapped.device,
        )
        detected_handle_pos_w, handle_det_conf = _detect_handle_from_wrist_depth(
            depth_image=depth_image,
            cam_pos_w=cam_pos_w,
            cam_quat_w_ros=cam_quat_w_ros,
            intrinsics=intrinsics,
        )
        if detected_handle_pos_w is not None:
            # Cabinet workspace prior + temporal smoothing to reject outliers.
            valid_detection = (
                0.35 <= float(detected_handle_pos_w[0].item()) <= 0.95
                and -0.20 <= float(detected_handle_pos_w[1].item()) <= 0.20
                and 0.25 <= float(detected_handle_pos_w[2].item()) <= 0.80
            )
            if valid_detection:
                if detected_handle_ema is None:
                    detected_handle_ema = detected_handle_pos_w
                else:
                    detected_handle_ema = 0.8 * detected_handle_ema + 0.2 * detected_handle_pos_w
            elif detected_handle_ema is not None:
                detected_handle_pos_w = None

        if detected_handle_pos_w is None and detected_handle_ema is not None:
            detected_handle_pos_w = detected_handle_ema
        ft_wrench = _read_contact_wrench(env.unwrapped.scene, env.unwrapped.device)
        hand_state = _read_hand_state(robot, gripper_joint_ids, env.unwrapped.device)

        obs = {
            "ee_pos": ee_pos[0],
            "ee_quat": ee_quat[0],
            "ee_vel": torch.zeros(6, device=env.unwrapped.device),
            "hand_state": hand_state,
            "ft_wrench": ft_wrench,
            "drawer_q": cabinet_joint_pos[0:1],
            "handle_quat": handle_quat[0],
            # Approximate opening direction from cabinet frame for open-path target.
            "drawer_axis_w": -handle_x,
            "rgb_image": rgb_image,
        }
        if args_cli.use_privileged_handle:
            # Debug fallback; keep disabled for paper-faithful runs.
            obs["handle_pos_w"] = handle_pos[0]
            obs["handle_pos_camera"] = quat_apply_inverse(
                cam_quat_w_ros.unsqueeze(0),
                (handle_pos[0] - cam_pos_w).unsqueeze(0),
            )[0]
        elif detected_handle_pos_w is not None:
            obs["detected_handle_pos_w"] = detected_handle_pos_w
            obs["handle_pos_camera"] = quat_apply_inverse(
                cam_quat_w_ros.unsqueeze(0),
                (detected_handle_pos_w - cam_pos_w).unsqueeze(0),
            )[0]
        if depth_image is not None:
            obs["depth_image"] = depth_image

        # Paper-style policy: execute steepest-gradient action directly.
        graph_action = graph.step(obs)

        # IK-Rel actions in Isaac Lab's DifferentialIKController use world-frame
        # deltas by default (target_pos = current_pos + delta_pos in world coordinates).
        # We pass the world-frame graph_action directly to the environment.
        graph_action_local = graph_action.clone()

        # Gripper convention for BinaryJointPositionAction: negative => close.
        if graph_action_local[6].item() < 0.0:
            close_hold_steps += 1
        else:
            close_hold_steps = 0

        if assist is not None and assist.is_enabled:
            if (
                not assist.is_attached
                and graph.last_p_grasped >= args_cli.fixed_joint_attach_threshold
                and close_hold_steps >= 2
            ):
                if assist.attach():
                    attach_events += 1
                    assist_low_grasp_steps = 0
            elif assist.is_attached:
                if graph.last_p_grasped <= args_cli.fixed_joint_detach_threshold:
                    assist_low_grasp_steps += 1
                else:
                    assist_low_grasp_steps = 0

                should_detach = close_hold_steps == 0 or assist_low_grasp_steps >= args_cli.fixed_joint_detach_steps
                if should_detach and assist.detach():
                    detach_events += 1
                    assist_low_grasp_steps = 0

        action = _map_action(graph_action_local, env.action_space.shape, env.unwrapped.device)
        env.step(action)

        ee_drawer_dist = torch.norm(handle_pos[0] - ee_pos[0]).item()
        drawer_pos = float(cabinet_joint_pos[0].item())
        action_norm = float(graph_action_local.norm().item())
        ft_norm = float(torch.norm(ft_wrench).item())
        rgb_mean = float(rgb_image.float().mean().item())

        ee_drawer_dist_acc += ee_drawer_dist
        drawer_pos_acc += drawer_pos
        action_norm_acc += action_norm
        ft_norm_acc += ft_norm
        rgb_mean_acc += rgb_mean
        ft_norm_max = max(ft_norm_max, ft_norm)
        steps_done += 1

        if args_cli.dump_ee_data and (steps_done % args_cli.dump_interval == 0 or steps_done == 1):
            _dump_ee_snapshot(
                dump_dir=dump_dir,
                step=steps_done,
                rgb_image=rgb_image,
                depth_image=depth_image,
                obs=obs,
                graph=graph,
                ft_norm=ft_norm,
                ee_drawer_dist=ee_drawer_dist,
            )

        if steps_done % args_cli.log_every == 0 or steps_done == 1:
            print(
                f"[SIM] step={steps_done:04d} "
                f"ee_pos=({ee_pos[0][0].item():.2f},{ee_pos[0][1].item():.2f},{ee_pos[0][2].item():.2f}) "
                f"handle=({handle_pos[0][0].item():.2f},{handle_pos[0][1].item():.2f},{handle_pos[0][2].item():.2f}) "
                f"ee_quat=({ee_quat[0][0].item():.2f},{ee_quat[0][1].item():.2f},{ee_quat[0][2].item():.2f},{ee_quat[0][3].item():.2f}) "
                f"handle_quat=({handle_quat[0][0].item():.2f},{handle_quat[0][1].item():.2f},{handle_quat[0][2].item():.2f},{handle_quat[0][3].item():.2f}) "
                f"open_axis=({obs['drawer_axis_w'][0].item():.2f},{obs['drawer_axis_w'][1].item():.2f},{obs['drawer_axis_w'][2].item():.2f}) "
                f"ee_handle_dist={ee_drawer_dist:.4f} "
                f"drawer_joint={drawer_pos:.4f} "
                f"action_norm={action_norm:.4f} "
                f"ft_norm={ft_norm:.4f} "
                f"rgb_mean={rgb_mean:.2f} "
                f"close_hold={close_hold_steps} "
                f"assist={'on' if (assist is not None and assist.is_attached) else 'off'} "
                f"path={graph.last_selected_path_name} "
                f"p_vis={graph.last_p_visible:.3f} "
                f"p_grasp={graph.last_p_grasped:.3f}"
            )

    out_dir = PROJECT_ROOT / args_cli.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "num_steps": steps_done,
        "mean_ee_handle_distance": ee_drawer_dist_acc / max(steps_done, 1),
        "mean_drawer_joint_pos": drawer_pos_acc / max(steps_done, 1),
        "mean_action_norm": action_norm_acc / max(steps_done, 1),
        "mean_ft_norm": ft_norm_acc / max(steps_done, 1),
        "max_ft_norm": ft_norm_max,
        "mean_rgb_intensity": rgb_mean_acc / max(steps_done, 1),
        "fixed_joint_assist": bool(args_cli.fixed_joint_assist),
        "fixed_joint_attached_final": bool(assist.is_attached) if assist is not None else False,
        "fixed_joint_attach_events": attach_events,
        "fixed_joint_detach_events": detach_events,
        "device": str(env.unwrapped.device),
        "video_dir": str(video_dir) if args_cli.record_video else None,
        "ee_dump_dir": str(dump_dir) if args_cli.dump_ee_data else None,
    }
    summary_path = out_dir / "sim_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[SIM] wrote summary: {summary_path}")

    if assist is not None and assist.is_attached:
        assist.detach()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
