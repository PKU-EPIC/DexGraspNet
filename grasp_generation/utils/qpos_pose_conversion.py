from utils.hand_model_type import translation_names, rot_names
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import transforms3d
import torch
import numpy as np
from typing import List, Dict, Any, Tuple


def pose_to_qpos(
    hand_pose: torch.Tensor,
    joint_names: List[str],
):
    assert len(hand_pose.shape) == 1

    qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
    rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
    euler = transforms3d.euler.mat2euler(rot, axes="sxyz")
    qpos.update(dict(zip(rot_names, euler)))
    qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
    return qpos


def qpos_to_pose(
    qpos: Dict[str, Any], joint_names: List[str], unsqueeze_batch_dim: bool = True
):
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor(
        [qpos[name] for name in translation_names]
        + rot
        + [qpos[name] for name in joint_names],
        dtype=torch.float,
    )

    if unsqueeze_batch_dim:
        hand_pose = hand_pose.unsqueeze(0)
        assert len(hand_pose.shape) == 2
    else:
        assert len(hand_pose.shape) == 1
    return hand_pose

def qpos_to_translation_rot_jointangles(qpos: Dict[str, Any], joint_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation = np.array([qpos[name] for name in translation_names])
    rot = np.array([qpos[name] for name in rot_names])
    joint_angles = np.array([qpos[name] for name in joint_names])
    return translation, rot, joint_angles

