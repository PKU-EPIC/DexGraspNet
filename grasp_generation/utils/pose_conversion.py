from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import torch
import numpy as np
from typing import Tuple


def pose_to_hand_config(
    hand_pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(hand_pose.shape) == 0:
        hand_pose = hand_pose.unsqueeze(0)  # Make sure hand pose at least 2d.

    batch_size = hand_pose.shape[0]

    joint_angles = hand_pose[:, 9:].cpu().numpy()
    rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[:, 3:9]).cpu().numpy()
    trans = hand_pose[:, :3].cpu().numpy()

    assert trans.shape == (batch_size, 3)
    assert rot.shape == (batch_size, 3, 3)
    assert joint_angles.shape == (batch_size, 16)

    return trans, rot, joint_angles


def hand_config_to_pose(
    trans: np.ndarray,
    rot: np.ndarray,
    joint_angles: np.ndarray,
) -> torch.Tensor:
    # Unsqueeze if no batch dim.

    if len(trans.shape) == 1:
        assert rot.shape == (3, 3)
        assert joint_angles.shape == (16,)
        trans = trans[None, :]
        rot = rot[None, :, :]
        joint_angles = joint_angles[None, :]

    # Shape checks.
    batch_size = trans.shape[0]
    assert trans.shape == (batch_size, 3)
    assert rot.shape == (batch_size, 3, 3)
    assert joint_angles.shape == (batch_size, 16)

    # Convert rotation matrix batch to rot6d tensors.
    rot6d_torch = (
        torch.from_numpy(rot[:, :, :2]).transpose(2, 1).reshape(batch_size, -1)
    )
    assert rot6d_torch.shape == (batch_size, 6)

    # Convert trans and joint angles to tensors.
    trans_torch = torch.from_numpy(trans)
    joint_angles_torch = torch.from_numpy(joint_angles)

    hand_pose = torch.cat([trans_torch, rot6d_torch, joint_angles_torch], dim=1).float()

    assert hand_pose.shape == (batch_size, 25)

    return hand_pose
