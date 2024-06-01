from utils.hand_model import HandModel
# from utils.object_model import ObjectModel
import torch
from typing import Dict, Tuple, Optional
from collections import defaultdict


FINGERTIP_KEYWORDS = ["link_3.0", "link_7.0", "link_11.0", "link_15.0"]
DEFAULT_DIST_MOVE_FINGER = 0.05  # NOTE: Important parameter to vary
DEFAULT_DIST_MOVE_FINGER_BACKWARDS = -0.015  # NOTE: Important parameter to vary


### HELPERS START ###
def _compute_link_name_to_contact_candidates(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
) -> Dict[str, torch.Tensor]:
    batch_size = joint_angles.shape[0]

    current_status = hand_model.chain.forward_kinematics(joint_angles)
    link_name_to_contact_candidates = {}
    for i, link_name in enumerate(hand_model.mesh):
        # Compute contact candidates
        untransformed_contact_candidates = hand_model.mesh[link_name][
            "contact_candidates"
        ]
        if len(untransformed_contact_candidates) == 0:
            continue

        contact_candidates = (
            current_status[link_name]
            .transform_points(untransformed_contact_candidates)
            .expand(batch_size, -1, 3)
        )
        contact_candidates = contact_candidates @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        assert contact_candidates.shape == (
            batch_size,
            len(untransformed_contact_candidates),
            3,
        )
        link_name_to_contact_candidates[link_name] = contact_candidates

    return link_name_to_contact_candidates


def _compute_fingertip_name_to_contact_candidates(
    link_name_to_contact_candidates: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # HACK: hardcoded
    # Merge links associated with same fingertip
    fingertip_keywords = FINGERTIP_KEYWORDS
    fingertip_name_to_contact_candidates = {}
    for fingertip_keyword in fingertip_keywords:
        merged_contact_candidates = []
        for link_name, contact_candidates in link_name_to_contact_candidates.items():
            batch_size, n_contact_candidates, _ = contact_candidates.shape
            assert contact_candidates.shape == (batch_size, n_contact_candidates, 3)

            if fingertip_keyword in link_name:
                merged_contact_candidates.append(contact_candidates)

        fingertip_name_to_contact_candidates[fingertip_keyword] = torch.cat(
            merged_contact_candidates, dim=1
        )
    return fingertip_name_to_contact_candidates


### HELPERS END ###


def compute_fingertip_mean_contact_positions(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
) -> torch.Tensor:
    """Get the mean position of the contact candidates for each fingertip"""
    batch_size = joint_angles.shape[0]
    num_fingers = hand_model.num_fingers

    # Each link has a set of contact candidates
    link_name_to_contact_candidates = _compute_link_name_to_contact_candidates(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )
    # Merge contact candidates for links associated with same fingertip
    fingertip_name_to_contact_candidates = (
        _compute_fingertip_name_to_contact_candidates(
            link_name_to_contact_candidates=link_name_to_contact_candidates,
        )
    )

    # Iterate in deterministic order
    fingertip_names = FINGERTIP_KEYWORDS
    fingertip_mean_positions = []
    for i, fingertip_name in enumerate(fingertip_names):
        contact_candidates = fingertip_name_to_contact_candidates[fingertip_name]
        num_contact_candidates_this_link = contact_candidates.shape[1]
        assert contact_candidates.shape == (
            batch_size,
            num_contact_candidates_this_link,
            3,
        )

        fingertip_mean_positions.append(contact_candidates.mean(dim=1))

    fingertip_mean_positions = torch.stack(fingertip_mean_positions, dim=1)
    assert fingertip_mean_positions.shape == (batch_size, num_fingers, 3)
    return fingertip_mean_positions


# def compute_closest_contact_point_info(
#     joint_angles: torch.Tensor,
#     hand_model: HandModel,
#     object_model: ObjectModel,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Big function to compute the closest contact point on the object for each fingertip and return relevant info"""
#     # For each link, find the closest point on the object
#     link_name_to_contact_candidates = _compute_link_name_to_contact_candidates(
#         joint_angles=joint_angles,
#         hand_model=hand_model,
#     )

#     # Merge links associated with same fingertip
#     fingertip_name_to_contact_candidates = (
#         _compute_fingertip_name_to_contact_candidates(
#             link_name_to_contact_candidates=link_name_to_contact_candidates,
#         )
#     )

#     # To be populated
#     num_fingers = len(fingertip_name_to_contact_candidates)
#     batch_size = joint_angles.shape[0]
#     all_hand_contact_nearest_points = torch.zeros((batch_size, num_fingers, 3)).to(
#         joint_angles.device
#     )
#     all_nearest_object_to_hand_directions = torch.zeros(
#         (batch_size, num_fingers, 3)
#     ).to(joint_angles.device)
#     all_nearest_distances = torch.zeros((batch_size, num_fingers)).to(
#         joint_angles.device
#     )
#     all_hand_contact_nearest_point_indices = (
#         torch.zeros((batch_size, num_fingers)).long().to(joint_angles.device)
#     )

#     fingertip_names = FINGERTIP_KEYWORDS
#     for i, fingertip_name in enumerate(fingertip_names):
#         contact_candidates = fingertip_name_to_contact_candidates[fingertip_name]
#         n_contact_candidates = contact_candidates.shape[1]
#         assert contact_candidates.shape == (batch_size, n_contact_candidates, 3)

#         # From cal_distance, interiors are positive dist, exteriors are negative dist
#         # Normals point from object to hand
#         (
#             distances_interior_positive,
#             object_to_hand_directions,
#         ) = object_model.cal_distance(contact_candidates)
#         distances_interior_negative = (
#             -distances_interior_positive
#         )  # Large positive distance => far away
#         nearest_point_index = distances_interior_negative.argmin(dim=1)
#         nearest_distances = torch.gather(
#             input=distances_interior_negative,
#             dim=1,
#             index=nearest_point_index.unsqueeze(1),
#         )
#         hand_contact_nearest_points = torch.gather(
#             input=contact_candidates,
#             dim=1,
#             index=nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
#         )
#         nearest_object_to_hand_directions = torch.gather(
#             input=object_to_hand_directions,
#             dim=1,
#             index=nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
#         )

#         assert hand_contact_nearest_points.shape == (batch_size, 1, 3)
#         assert nearest_object_to_hand_directions.shape == (batch_size, 1, 3)
#         assert nearest_distances.shape == (batch_size, 1)
#         assert nearest_point_index.shape == (batch_size,)

#         # Update tensors
#         all_hand_contact_nearest_points[:, i : i + 1, :] = hand_contact_nearest_points
#         all_nearest_object_to_hand_directions[:, i : i + 1, :] = (
#             nearest_object_to_hand_directions
#         )
#         all_nearest_distances[:, i : i + 1] = nearest_distances
#         all_hand_contact_nearest_point_indices[:, i] = nearest_point_index

#     return (
#         all_hand_contact_nearest_points,
#         all_nearest_object_to_hand_directions,
#         all_nearest_distances,
#         all_hand_contact_nearest_point_indices,
#     )


def compute_fingertip_dirs(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = joint_angles.shape[0]
    untransformed_center = torch.tensor([0, 0, 0]).to(joint_angles.device)
    untransformed_center_to_tip_dir = torch.tensor([0, 0, 1.0]).to(joint_angles.device)
    untransformed_center_to_right_dir = torch.tensor([0, 1.0, 0]).to(
        joint_angles.device
    )

    untransformed_points = torch.stack(
        [
            untransformed_center,
            untransformed_center + untransformed_center_to_tip_dir,
            untransformed_center + untransformed_center_to_right_dir,
        ],
        dim=0,
    )

    current_status = hand_model.chain.forward_kinematics(joint_angles)
    fingertip_link_names = set(
        ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]
    )
    center_to_right_dirs, center_to_tip_dirs = [], []
    for i, link_name in enumerate(hand_model.mesh):
        if link_name not in fingertip_link_names:
            continue

        transformed_points = current_status[link_name].transform_points(
            untransformed_points
        )
        transformed_points = transformed_points @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        assert transformed_points.shape == (
            batch_size,
            len(untransformed_points),
            3,
        )
        center = transformed_points[:, 0, :]
        center_to_tip_dir = transformed_points[:, 1, :] - center
        center_to_right_dir = transformed_points[:, 2, :] - center
        assert center_to_tip_dir.shape == (batch_size, 3)
        assert center_to_right_dir.shape == (batch_size, 3)
        assert torch.allclose(
            center_to_tip_dir.norm(dim=-1),
            torch.ones(
                batch_size,
                device=center_to_tip_dir.device,
                dtype=center_to_tip_dir.dtype,
            ),
        )
        assert torch.allclose(
            center_to_right_dir.norm(dim=-1),
            torch.ones(
                batch_size,
                device=center_to_right_dir.device,
                dtype=center_to_right_dir.dtype,
            ),
        )

        center_to_right_dirs.append(center_to_right_dir)
        center_to_tip_dirs.append(center_to_tip_dir)

    center_to_right_dirs = torch.stack(center_to_right_dirs, dim=1)
    center_to_tip_dirs = torch.stack(center_to_tip_dirs, dim=1)
    assert center_to_right_dirs.shape == (batch_size, hand_model.num_fingers, 3)
    assert center_to_tip_dirs.shape == (batch_size, hand_model.num_fingers, 3)
    return center_to_right_dirs, center_to_tip_dirs


# def compute_grasp_orientations(
#     joint_angles_start: torch.Tensor,
#     hand_model: HandModel,
#     object_model: ObjectModel,
# ) -> torch.Tensor:
#     # Can't just compute_grasp_dirs because we need to know the orientation of the fingers
#     # Each finger has a rotation matrix [x, y, z] where x y z are column vectors
#     #    * z is direction the fingertip moves
#     #    * y is direction "up" along finger (from finger center to fingertip), modified to be perpendicular to z
#     #    * x is direction "right" along finger (from finger center to fingertip), modified to be perpendicular to z and y
#     # if y.cross(z) == 0, then need backup
#     batch_size = joint_angles_start.shape[0]

#     (
#         hand_contact_nearest_points,
#         nearest_object_to_hand_directions,
#         _,
#         _,
#     ) = compute_closest_contact_point_info(
#         joint_angles=joint_angles_start,
#         hand_model=hand_model,
#         object_model=object_model,
#     )
#     nearest_hand_to_object_directions = -nearest_object_to_hand_directions
#     z_dirs = nearest_hand_to_object_directions
#     assert z_dirs.shape == (batch_size, hand_model.num_fingers, 3)

#     if torch.any(z_dirs.norm(dim=-1) == 0):
#         bad_inds = torch.where(z_dirs.norm(dim=-1) == 0)
#         z_dirs[bad_inds] = -hand_contact_nearest_points[bad_inds]
#         print(
#             f"WARNING: {len(bad_inds)} z_dirs have 0 norm, using hand_contact_nearest_points instead"
#         )

#     assert (z_dirs.norm(dim=-1).min() > 0).all()
#     z_dirs = z_dirs / z_dirs.norm(dim=-1, keepdim=True)

#     (center_to_right_dirs, center_to_tip_dirs) = compute_fingertip_dirs(
#         joint_angles=joint_angles_start,
#         hand_model=hand_model,
#     )
#     option_1_ok = torch.cross(center_to_tip_dirs, z_dirs).norm(dim=-1, keepdim=True) > 0

#     y_dirs = torch.where(
#         option_1_ok,
#         center_to_tip_dirs
#         - (center_to_tip_dirs * z_dirs).sum(dim=-1, keepdim=True) * z_dirs,
#         center_to_right_dirs
#         - (center_to_right_dirs * z_dirs).sum(dim=-1, keepdim=True) * z_dirs,
#     )
#     assert (y_dirs.norm(dim=-1).min() > 0).all()
#     y_dirs = y_dirs / y_dirs.norm(dim=-1, keepdim=True)

#     x_dirs = torch.cross(y_dirs, z_dirs)
#     assert (x_dirs.norm(dim=-1).min() > 0).all()
#     x_dirs = x_dirs / x_dirs.norm(dim=-1, keepdim=True)
#     grasp_orientations = torch.stack([x_dirs, y_dirs, z_dirs], dim=-1)

#     assert grasp_orientations.shape == (batch_size, hand_model.num_fingers, 3, 3)
#     return grasp_orientations


def compute_fingertip_init_targets(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    grasp_orientations: torch.Tensor,
    dist_move_finger_backwards: Optional[float] = None,
) -> torch.Tensor:
    if dist_move_finger_backwards is None:
        dist_move_finger_backwards = DEFAULT_DIST_MOVE_FINGER_BACKWARDS

    # Sanity check
    batch_size = joint_angles_start.shape[0]
    num_fingers = hand_model.num_fingers
    assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)

    # Get grasp directions
    grasp_directions = grasp_orientations[:, :, :, 2]
    assert grasp_directions.shape == (batch_size, num_fingers, 3)

    # Get current positions
    fingertip_mean_positions = compute_fingertip_mean_contact_positions(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
    )
    assert fingertip_mean_positions.shape == (batch_size, num_fingers, 3)

    fingertip_targets = (
        fingertip_mean_positions + grasp_directions * dist_move_finger_backwards
    )
    return fingertip_targets


def compute_fingertip_targets(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    grasp_orientations: torch.Tensor,
    dist_move_finger: Optional[float] = None,
) -> torch.Tensor:
    if dist_move_finger is None:
        dist_move_finger = DEFAULT_DIST_MOVE_FINGER

    # Sanity check
    batch_size = joint_angles_start.shape[0]
    num_fingers = hand_model.num_fingers
    assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)

    # Get grasp directions
    grasp_directions = grasp_orientations[:, :, :, 2]
    assert grasp_directions.shape == (batch_size, num_fingers, 3)

    # Get current positions
    fingertip_mean_positions = compute_fingertip_mean_contact_positions(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
    )
    assert fingertip_mean_positions.shape == (batch_size, num_fingers, 3)

    fingertip_targets = fingertip_mean_positions + grasp_directions * dist_move_finger
    return fingertip_targets


def compute_optimized_joint_angle_targets_given_fingertip_targets(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    fingertip_targets: torch.Tensor,
) -> Tuple[torch.Tensor, defaultdict]:
    # Sanity check
    batch_size = hand_model.batch_size
    num_fingers = hand_model.num_fingers
    num_xyz = 3
    assert fingertip_targets.shape == (batch_size, num_fingers, num_xyz)

    # Store original hand pose for later
    original_hand_pose = hand_model.hand_pose.detach().clone()

    # Optimize joint angles
    joint_angle_targets_to_optimize = (
        joint_angles_start.detach().clone().requires_grad_(True)
    )
    debug_info = defaultdict(list)
    N_ITERS = 100
    for i in range(N_ITERS):
        contact_points_hand_to_optimize = compute_fingertip_mean_contact_positions(
            joint_angles=joint_angle_targets_to_optimize, hand_model=hand_model
        )
        loss = (
            (fingertip_targets.detach().clone() - contact_points_hand_to_optimize)
            .square()
            .sum()
        )
        GRAD_STEP_SIZE = 5

        loss.backward(retain_graph=True)

        with torch.no_grad():
            joint_angle_targets_to_optimize -= (
                joint_angle_targets_to_optimize.grad * GRAD_STEP_SIZE
            )
            joint_angle_targets_to_optimize.grad.zero_()
        debug_info["loss"].append(loss.item())
        debug_info["fingertip_targets"].append(fingertip_targets.detach().clone())
        debug_info["contact_points_hand"].append(
            contact_points_hand_to_optimize.detach().clone()
        )

    # Update hand pose parameters
    new_hand_pose = original_hand_pose.detach().clone()
    new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
    hand_model.set_parameters(new_hand_pose)

    return joint_angle_targets_to_optimize, debug_info


def compute_optimized_joint_angle_targets_given_grasp_orientations(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    grasp_orientations: torch.Tensor,
    dist_move_finger: Optional[float] = None,
) -> Tuple[torch.Tensor, defaultdict]:
    # Get fingertip targets
    fingertip_targets = compute_fingertip_targets(
        joint_angles_start=joint_angles_start,
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
        dist_move_finger=dist_move_finger,
    )

    (
        joint_angle_targets,
        debug_info,
    ) = compute_optimized_joint_angle_targets_given_fingertip_targets(
        joint_angles_start=joint_angles_start,
        hand_model=hand_model,
        fingertip_targets=fingertip_targets,
    )
    return joint_angle_targets, debug_info


def compute_init_joint_angles_given_grasp_orientations(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    grasp_orientations: torch.Tensor,
    dist_move_finger_backwards: Optional[float] = None,
) -> Tuple[torch.Tensor, defaultdict]:
    # Get fingertip targets
    init_fingertip_targets = compute_fingertip_init_targets(
        joint_angles_start=joint_angles_start,
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
        dist_move_finger_backwards=dist_move_finger_backwards,
    )

    (
        init_joint_angles,
        debug_info,
    ) = compute_optimized_joint_angle_targets_given_fingertip_targets(
        joint_angles_start=joint_angles_start,
        hand_model=hand_model,
        fingertip_targets=init_fingertip_targets,
    )
    return init_joint_angles, debug_info
