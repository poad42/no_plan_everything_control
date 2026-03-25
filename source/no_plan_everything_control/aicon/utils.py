# Copyright (c) 2025, no_plan_everything_control contributors.
# SPDX-License-Identifier: MIT

"""Utility functions for the AICON kernel."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# SE(3) / SO(3) helpers
# ---------------------------------------------------------------------------

def so3_log_map(R: Tensor) -> Tensor:
    """Logarithm map of SO(3), returns axis-angle vector (3,).

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) axis-angle representation.
    """
    # Clamp for numerical safety
    cos_angle = ((R.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    if angle.abs() < 1e-6:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (angle / (2.0 * torch.sin(angle))) * axis


def se3_log_map(T: Tensor) -> Tensor:
    """Logarithm map of SE(3).

    Args:
        T: (4, 4) homogeneous transformation matrix.

    Returns:
        (6,) twist vector [v, omega].
    """
    R = T[:3, :3]
    t = T[:3, 3]
    omega = so3_log_map(R)
    angle = omega.norm()
    if angle < 1e-6:
        v = t
    else:
        skew = _skew(omega / angle)
        A_inv = (
            torch.eye(3, device=T.device, dtype=T.dtype)
            - 0.5 * skew
            + ((1.0 - angle / (2.0 * torch.tan(angle / 2.0))) / (angle ** 2)) * (skew @ skew)
        )
        v = A_inv @ t
    return torch.cat([v, omega])


def _skew(v: Tensor) -> Tensor:
    """Skew-symmetric matrix from 3-vector."""
    return torch.tensor(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        device=v.device,
        dtype=v.dtype,
    )


# ---------------------------------------------------------------------------
# EKF covariance helpers
# ---------------------------------------------------------------------------

def cov_to_info(Sigma: Tensor) -> Tensor:
    """Convert covariance to information matrix via matrix inverse."""
    return torch.linalg.inv(Sigma)


def info_to_cov(Lambda: Tensor) -> Tensor:
    """Convert information matrix to covariance."""
    return torch.linalg.inv(Lambda)


# ---------------------------------------------------------------------------
# Blocks World helpers
# ---------------------------------------------------------------------------

def random_bw_instance(
    n_blocks: int,
    n_towers: int,
    seed: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate a random Blocks World problem instance.

    Args:
        n_blocks: total number of blocks (10–30 as in the paper).
        n_towers: number of towers in the goal state (0–15 as in the paper).
        seed: random seed for reproducibility.

    Returns:
        (initial_stacks, goal_stacks):
            Each is a list of towers, where each tower is a list of block indices
            ordered bottom to top.
    """
    import random
    rng = random.Random(seed)

    block_ids = list(range(n_blocks))
    rng.shuffle(block_ids)

    def split_into_towers(blocks: list[int], k: int) -> list[list[int]]:
        if k == 0:
            return [[b] for b in blocks]
        towers: list[list[int]] = [[] for _ in range(k)]
        for i, b in enumerate(blocks):
            towers[i % k].append(b)
        return [t for t in towers if t]

    initial = split_into_towers(block_ids[:], max(1, rng.randint(1, n_towers + 1)))

    rng.shuffle(block_ids)
    goal = split_into_towers(block_ids[:], n_towers if n_towers > 0 else 1)

    return initial, goal


def stacks_to_on_matrix(stacks: list[list[int]], n_blocks: int) -> Tensor:
    """Convert tower-of-stacks representation to o(X,Y) matrix.

    Args:
        stacks: list of towers, each tower is bottom-to-top list of block ids.
        n_blocks: total number of blocks.

    Returns:
        (n_blocks, n_blocks) float tensor, entry [i, j] = 1.0 if block i is on block j.
    """
    on = torch.zeros(n_blocks, n_blocks)
    for tower in stacks:
        for k in range(1, len(tower)):
            on[tower[k], tower[k - 1]] = 1.0  # block tower[k] is on tower[k-1]
    return on


def optimal_bw_plan(
    initial_stacks: list[list[int]],
    goal_stacks: list[list[int]],
    n_blocks: int,
) -> list[tuple[str, int, int]]:
    """Compute an optimal Blocks World plan via BFS.

    Args:
        initial_stacks: initial tower configuration (bottom-to-top).
        goal_stacks: goal tower configuration (bottom-to-top).
        n_blocks: total number of blocks.

    Returns:
        List of (action, X, Y) tuples:
            ('stack', X, Y)   = place X on top of Y
            ('unstack', X, Y) = remove X from Y (X must be on Y)
    """
    # Represent state as tuple of tuples (hashable)
    def to_state(stacks: list[list[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(t) for t in sorted(stacks, key=lambda t: t[0]) if t)

    def get_block_position(state: tuple, block: int) -> tuple[int, int]:
        """Returns (tower_idx, pos_in_tower) for block."""
        for ti, tower in enumerate(state):
            if block in tower:
                return ti, tower.index(block)
        raise ValueError(f"Block {block} not found in state.")

    def is_clear(state: tuple, block: int) -> bool:
        ti, pos = get_block_position(state, block)
        return pos == len(state[ti]) - 1

    def apply_stack(state: list[list[int]], X: int, Y: int) -> list[list[int]]:
        new_state = [list(t) for t in state]
        # Remove X from its current tower
        for t in new_state:
            if X in t:
                t.remove(X)
                break
        # Place X on top of Y's tower
        for t in new_state:
            if Y in t:
                t.append(X)
                break
        return [t for t in new_state if t]

    def state_tuple(stacks: list[list[int]]) -> tuple:
        return tuple(tuple(t) for t in sorted(stacks, key=lambda t: t[0]) if t)

    initial_state = [list(t) for t in initial_stacks]
    goal_state_t = state_tuple([list(t) for t in goal_stacks])

    queue: deque[tuple[list[list[int]], list]] = deque()
    queue.append((initial_state, []))
    visited = {state_tuple(initial_state)}

    while queue:
        state, plan = queue.popleft()
        st = state_tuple(state)

        if st == goal_state_t:
            return plan

        # Generate all valid moves
        for X in range(n_blocks):
            if not any(X in t for t in state):
                continue
            # X is clear?
            for t in state:
                if X in t and t[-1] == X:
                    # If X has a block below, allow unstacking X to table.
                    x_idx = t.index(X)
                    if x_idx > 0:
                        Y = t[x_idx - 1]
                        new_state = [list(tt) for tt in state]
                        for tt in new_state:
                            if X in tt:
                                tt.remove(X)
                                break
                        new_state.append([X])
                        new_state = [tt for tt in new_state if tt]
                        nst = state_tuple(new_state)
                        if nst not in visited:
                            visited.add(nst)
                            queue.append((new_state, plan + [("unstack", X, Y)]))

                    # X is clear — can stack on any clear Y != X
                    for Y in range(n_blocks):
                        if Y == X:
                            continue
                        for t2 in state:
                            if Y in t2 and t2[-1] == Y:
                                new_state = apply_stack(state, X, Y)
                                nst = state_tuple(new_state)
                                if nst not in visited:
                                    visited.add(nst)
                                    queue.append((new_state, plan + [("stack", X, Y)]))
                    break

    return []  # No solution found (should not happen for valid BW instances)
