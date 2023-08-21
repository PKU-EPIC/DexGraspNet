from typing import Tuple, Dict, Any
import torch
import pypose as pp
import transforms3d
import numpy as np

NUM_FINGERS = 4
DEXGRASPNET_TRANS_NAMES = ["WRJTx", "WRJTy", "WRJTz"]
DEXGRASPNET_ROT_NAMES = ["WRJRx", "WRJRy", "WRJRz"]
ALLEGRO_JOINT_NAMES = [
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
]


def get_grasp_config_from_grasp_data(
    grasp_data: dict,
) -> Tuple[pp.LieTensor, torch.Tensor, pp.LieTensor]:
    """
    Given a dict of grasp data, return a tuple of the wrist pose, joint angles, and fingertip transforms.
    """
    qpos = grasp_data["qpos"]

    # Get wrist pose.
    wrist_translation = torch.tensor([qpos[tn] for tn in DEXGRASPNET_TRANS_NAMES])
    assert wrist_translation.shape == (3,)

    euler_angles = torch.tensor([qpos[rn] for rn in DEXGRASPNET_ROT_NAMES])
    wrist_quat = torch.tensor(transforms3d.euler.euler2quat(*euler_angles, axes="sxyz"))
    wrist_quat = wrist_quat[[1, 2, 3, 0]]  # Convert (x, y, z, w) -> (w, x, y, z)
    assert wrist_quat.shape == (4,)

    wrist_pose = pp.SE3(torch.cat([wrist_translation, wrist_quat], dim=0))

    # Get joint angles.
    joint_angles = torch.tensor([qpos[jn] for jn in ALLEGRO_JOINT_NAMES])
    assert joint_angles.shape == (16,)

    # Get grasp orientations.
    fingertip_transforms = get_fingertip_transforms_from_grasp_data(grasp_data)
    grasp_orientations = pp.from_matrix(fingertip_transforms.matrix(), pp.SO3_type)

    return wrist_pose, joint_angles, grasp_orientations


def get_fingertip_transforms_from_grasp_data(grasp_data: dict) -> pp.LieTensor:
    """
    Given a dict of grasp data, return a tensor of fingertip transforms.
    """
    (
        contact_candidates,
        target_contact_candidates,
    ) = get_contact_candidates_and_target_candidates(grasp_data)
    start_points, end_points, up_points = get_start_and_end_and_up_points(
        contact_candidates, target_contact_candidates, NUM_FINGERS
    )
    transforms = torch.stack(
        [
            get_transform(start, end, up)
            for start, end, up in zip(start_points, end_points, up_points)
        ],
        axis=0,
    )

    assert transforms.lshape == (NUM_FINGERS,)
    assert transforms.ltype == pp.SE3_type

    return transforms


def get_contact_candidates_and_target_candidates(
    grasp_data: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    link_name_to_contact_candidates = grasp_data["link_name_to_contact_candidates"]
    link_name_to_target_contact_candidates = grasp_data[
        "link_name_to_target_contact_candidates"
    ]
    contact_candidates = np.concatenate(
        [
            contact_candidate
            for _, contact_candidate in link_name_to_contact_candidates.items()
        ],
        axis=0,
    )
    target_contact_candidates = np.concatenate(
        [
            target_contact_candidate
            for _, target_contact_candidate in link_name_to_target_contact_candidates.items()
        ],
        axis=0,
    )
    return contact_candidates, target_contact_candidates


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def get_transform(start: np.ndarray, end: np.ndarray, up: np.ndarray) -> np.ndarray:
    # BRITTLE: Assumes new_z and new_y are pretty much perpendicular
    # If not, tries to find closest possible
    new_z = normalize(end - start)
    # new_y should be perpendicular to new_z
    up_dir = normalize(up - start)
    new_y = normalize(up_dir - np.dot(up_dir, new_z) * new_z)
    new_x = np.cross(new_y, new_z)

    transform = np.eye(4)
    transform[:3, :3] = np.stack([new_x, new_y, new_z], axis=1)
    transform[:3, 3] = start
    return pp.from_matrix(transform, pp.SE3_type)


def get_start_and_end_and_up_points(
    contact_candidates: np.ndarray,
    target_contact_candidates: np.ndarray,
    num_fingers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # BRITTLE: Assumes same number of contact points per finger
    # BRITTLE: Assumes UP_POINT_IDX is position of contact candidate up from center
    UP_POINT_IDX = 3
    contact_candidates_per_finger = contact_candidates.reshape(num_fingers, -1, 3)
    target_contact_candidates_per_finger = target_contact_candidates.reshape(
        num_fingers, -1, 3
    )
    start_points = contact_candidates_per_finger.mean(axis=1)
    end_points = target_contact_candidates_per_finger.mean(axis=1)
    up_points = contact_candidates_per_finger[:, UP_POINT_IDX, :]
    assert start_points.shape == end_points.shape == up_points.shape == (num_fingers, 3)
    return np.array(start_points), np.array(end_points), np.array(up_points)
