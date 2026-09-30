"""Joint Distance Feature (JDF) and task reward utilities.

The feature is expressed in the object frame, making it invariant to a common
rigid transform of the hand and object.  For every hand joint it stores the
vector to the closest sampled object-surface point, plus its scalar distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TIP_INDICES = np.asarray([4, 8, 12, 16, 20])


def joint_distance_feature(joints: np.ndarray, surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened closest-point vectors and distances.

    Args:
        joints: ``(B, J, 3)`` joint positions in the object frame.
        surface: ``(B, P, 3)`` object surface samples in the object frame.
    """
    if joints.ndim != 3 or surface.ndim != 3 or joints.shape[0] != surface.shape[0]:
        raise ValueError("expected joints (B,J,3) and surface (B,P,3)")
    delta = surface[:, None, :, :] - joints[:, :, None, :]
    squared = np.einsum("bjpk,bjpk->bjp", delta, delta)
    nearest = np.argmin(squared, axis=2)
    batch = np.arange(joints.shape[0])[:, None]
    joint = np.arange(joints.shape[1])[None, :]
    vectors = delta[batch, joint, nearest]
    distances = np.sqrt(squared[batch, joint, nearest])
    return vectors.reshape(joints.shape[0], -1).astype(np.float32), distances.astype(np.float32)


@dataclass(frozen=True)
class RewardWeights:
    precision: float = 2.0
    multi_contact: float = 1.0
    lift_progress: float = 8.0
    lift_target: float = 4.0
    stability: float = 0.15
    action_rate: float = 0.01
    drop: float = 5.0
    success: float = 10.0


def grasp_lift_reward(
    distances: np.ndarray,
    contacts: np.ndarray,
    height: np.ndarray,
    initial_height: np.ndarray,
    object_velocity: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    lift_target: float,
    phase_is_lift: np.ndarray,
    success_height_tolerance: float,
    success_speed: float,
    weights: RewardWeights,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Dense reward for accurate enclosure followed by a stable 10 cm lift."""
    tip_distance = distances[:, TIP_INDICES].mean(axis=1)
    precision = np.exp(-tip_distance / 0.025)
    finger_contact = contacts[:, 1:].reshape(len(contacts), 5, 3).max(axis=2).sum(axis=1)
    multi_contact = np.minimum(finger_contact / 3.0, 1.0)
    lift = np.maximum(height - initial_height, 0.0)
    progress = np.minimum(lift / lift_target, 1.0) * phase_is_lift
    speed = np.linalg.norm(object_velocity, axis=1)
    stable = speed * phase_is_lift
    action_rate = np.square(action - previous_action).mean(axis=1)
    dropped = (height < initial_height - 0.025).astype(np.float32)
    success = (
        (lift >= lift_target - success_height_tolerance)
        & (speed <= success_speed)
        & (finger_contact >= 3)
    ).astype(np.float32)
    terms = {
        "precision": precision,
        "multi_contact": multi_contact,
        "lift_progress": progress,
        "lift_target": progress * progress,
        "stability": stable,
        "action_rate": action_rate,
        "drop": dropped,
        "success": success,
    }
    reward = (
        weights.precision * precision
        + weights.multi_contact * multi_contact
        + weights.lift_progress * progress
        + weights.lift_target * terms["lift_target"]
        - weights.stability * stable
        - weights.action_rate * action_rate
        - weights.drop * dropped
        + weights.success * success
    )
    return reward.astype(np.float32), terms

