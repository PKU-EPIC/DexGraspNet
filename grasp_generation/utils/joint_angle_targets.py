from utils.hand_model import HandModel
from utils.object_model import ObjectModel
import torch
from typing import Union, Dict, Any, Tuple, Optional, List
import numpy as np

from enum import Enum, auto


class AutoName(Enum):
    # https://docs.python.org/3.9/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


class OptimizationMethod(AutoName):
    DESIRED_PENETRATION_DEPTH = auto()
    DESIRED_DIST_TOWARDS_OBJECT_SURFACE_ONE_STEP = auto()
    DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS = auto()
    DESIRED_DIST_TOWARDS_FINGERS_CENTER_ONE_STEP = auto()
    DESIRED_DIST_TOWARDS_FINGERS_CENTER_MULTIPLE_STEP = auto()


def compute_fingertip_positions(
    hand_model: HandModel,
    device: torch.device,
) -> torch.Tensor:
    current_status = hand_model.chain.forward_kinematics(hand_model.hand_pose[:, 9:])
    batch_size = hand_model.hand_pose.shape[0]

    link_name_to_positions = {}
    for i, link_name in enumerate(hand_model.mesh):
        contact_candidates = hand_model.mesh[link_name]["contact_candidates"]
        if len(contact_candidates) == 0:
            continue

        contact_candidates = (
            current_status[link_name]
            .transform_points(contact_candidates)
            .expand(batch_size, -1, 3)
        )
        contact_candidates = contact_candidates @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        fingertip_position = contact_candidates.mean(dim=1)

        assert fingertip_position.shape == (batch_size, 3)
        link_name_to_positions[link_name] = fingertip_position

    fingertip_positions = []

    return torch.stack(fingertip_positions, dim=1).to(device)


def compute_fingers_center(
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
    dist_thresh_close: float = 0.01,
) -> torch.Tensor:
    # (batch_size, num_fingers, 3)
    fingertip_positions = compute_fingertip_positions(
        hand_model=hand_model, device=device
    )

    # Interiors are positive dist, exteriors are negative dist
    # (batch_size, num_fingers)
    distances, _ = object_model.cal_distance(fingertip_positions)
    admitted = -distances < dist_thresh_close

    # (batch_size, num_fingers, 3)
    admitted_fingertip_positions = torch.where(
        admitted[..., None],
        fingertip_positions,
        torch.zeros_like(fingertip_positions, device=device),
    )

    # (batch_size, 3)
    fingers_center = admitted_fingertip_positions.sum(dim=1) / admitted.sum(
        dim=1, keepdim=True
    )
    return fingers_center


def compute_loss_desired_penetration_depth(
    joint_angle_targets_to_optimize: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
    cached_target_points: Optional[torch.Tensor] = None,
    cached_contact_nearest_point_indexes: Optional[torch.Tensor] = None,
    dist_thresh_to_move_finger: float = 0.01,
    desired_penetration_depth: float = 0.003,
    return_debug_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    num_links = len(hand_model.mesh)
    batch_size = joint_angle_targets_to_optimize.shape[0]

    contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_distances = torch.zeros((batch_size, num_links)).to(device)
    contact_nearest_point_indexes = (
        torch.zeros((batch_size, num_links)).long().to(device)
    )

    current_status = hand_model.chain.forward_kinematics(
        joint_angle_targets_to_optimize
    )
    for i, link_name in enumerate(hand_model.mesh):
        contact_candidates = hand_model.mesh[link_name]["contact_candidates"]
        if len(contact_candidates) == 0:
            continue

        contact_candidates = (
            current_status[link_name]
            .transform_points(contact_candidates)
            .expand(batch_size, -1, 3)
        )
        contact_candidates = contact_candidates @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        # Interiors are positive dist, exteriors are negative dist
        # Normals point from object to hand
        distances, normals = object_model.cal_distance(contact_candidates)
        if cached_contact_nearest_point_indexes is None:
            nearest_point_index = distances.argmax(dim=1)
        else:
            nearest_point_index = cached_contact_nearest_point_indexes[:, i]
        nearest_distances = torch.gather(distances, 1, nearest_point_index.unsqueeze(1))
        nearest_points_hand = torch.gather(
            contact_candidates,
            1,
            nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
        )
        nearest_normals = torch.gather(
            normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        admitted = -nearest_distances < dist_thresh_to_move_finger
        contact_distances[:, i : i + 1] = torch.where(
            admitted, -nearest_distances, contact_distances[:, i : i + 1]
        )
        admitted = admitted.reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points_hand[:, i : i + 1, :] = torch.where(
            admitted, nearest_points_hand, contact_points_hand[:, i : i + 1, :]
        )
        contact_normals[:, i : i + 1, :] = torch.where(
            admitted, nearest_normals, contact_normals[:, i : i + 1, :]
        )
        contact_nearest_point_indexes[:, i] = nearest_point_index

    if cached_target_points is None:
        target_points = contact_points_hand - contact_normals * (
            contact_distances[..., None] + desired_penetration_depth
        )
    else:
        target_points = cached_target_points
    loss = (target_points.detach().clone() - contact_points_hand).square().sum()

    if not return_debug_info:
        return loss

    return loss, {
        "target_points": target_points,
        "contact_points_hand": contact_points_hand,
        "contact_normals": contact_normals,
        "contact_distances": contact_distances,
        "contact_nearest_point_indexes": contact_nearest_point_indexes,
    }


def compute_loss_desired_dist_move(
    joint_angle_targets_to_optimize: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
    cached_target_points: Optional[torch.Tensor] = None,
    cached_contact_nearest_point_indexes: Optional[torch.Tensor] = None,
    dist_thresh_to_move_finger: float = 0.001,
    dist_move_link: float = 0.001,
    return_debug_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    num_links = len(hand_model.mesh)
    batch_size = joint_angle_targets_to_optimize.shape[0]

    contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_nearest_point_indexes = (
        torch.zeros((batch_size, num_links)).long().to(device)
    )

    current_status = hand_model.chain.forward_kinematics(
        joint_angle_targets_to_optimize
    )
    for i, link_name in enumerate(hand_model.mesh):
        contact_candidates = hand_model.mesh[link_name]["contact_candidates"]
        if len(contact_candidates) == 0:
            continue

        contact_candidates = (
            current_status[link_name]
            .transform_points(contact_candidates)
            .expand(batch_size, -1, 3)
        )
        contact_candidates = contact_candidates @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        # Interiors are positive dist, exteriors are negative dist
        # Normals point from object to hand
        distances, normals = object_model.cal_distance(contact_candidates)
        if cached_contact_nearest_point_indexes is None:
            nearest_point_index = distances.argmax(dim=1)
        else:
            nearest_point_index = cached_contact_nearest_point_indexes[:, i]
        nearest_distances = torch.gather(distances, 1, nearest_point_index.unsqueeze(1))
        nearest_points_hand = torch.gather(
            contact_candidates,
            1,
            nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
        )
        nearest_normals = torch.gather(
            normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        admitted = -nearest_distances < dist_thresh_to_move_finger
        admitted = admitted.reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points_hand[:, i : i + 1, :] = torch.where(
            admitted, nearest_points_hand, contact_points_hand[:, i : i + 1, :]
        )
        contact_normals[:, i : i + 1, :] = torch.where(
            admitted, nearest_normals, contact_normals[:, i : i + 1, :]
        )
        contact_nearest_point_indexes[:, i] = nearest_point_index

    if cached_target_points is None:
        target_points = contact_points_hand - contact_normals * dist_move_link
    else:
        target_points = cached_target_points
    loss = (target_points.detach().clone() - contact_points_hand).square().sum()
    if not return_debug_info:
        return loss

    return loss, {
        "target_points": target_points,
        "contact_points_hand": contact_points_hand,
        "contact_normals": contact_normals,
        "contact_nearest_point_indexes": contact_nearest_point_indexes,
    }


def compute_optimized_joint_angle_targets(
    optimization_method: OptimizationMethod,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> Tuple[torch.Tensor, List[float], List[Dict[str, Any]]]:
    # TODO: Many of the parameters here are hardcoded
    original_hand_pose = hand_model.hand_pose.detach().clone()
    joint_angle_targets_to_optimize = (
        original_hand_pose.detach().clone()[:, 9:].requires_grad_(True)
    )

    losses = []
    debug_infos = []
    if optimization_method == OptimizationMethod.DESIRED_PENETRATION_DEPTH:
        N_ITERS = 100
        cached_target_points = None
        cached_contact_nearest_point_indexes = None
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_penetration_depth(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=cached_target_points,
                cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
                dist_thresh_to_move_finger=0.01,
                desired_penetration_depth=0.005,
                return_debug_info=True,
            )
            if cached_target_points is None:
                cached_target_points = debug_info["target_points"]
            if cached_contact_nearest_point_indexes is None:
                cached_contact_nearest_point_indexes = debug_info[
                    "contact_nearest_point_indexes"
                ]
            grad_step_size = 5
            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_ONE_STEP
    ):
        N_ITERS = 1
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                dist_thresh_to_move_finger=0.001,
                dist_move_link=0.001,
                return_debug_info=True,
            )
            grad_step_size = 500

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    ):
        N_ITERS = 100
        # Use cached target and indices to continue moving the same points toward the same targets for each iter
        # Otherwise, would be moving different points to different targets each iter
        cached_target_points = None
        cached_contact_nearest_point_indexes = None
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=cached_target_points,
                cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
                dist_thresh_to_move_finger=0.01,
                dist_move_link=0.01,
                return_debug_info=True,
            )
            if cached_target_points is None:
                cached_target_points = debug_info["target_points"]
            if cached_contact_nearest_point_indexes is None:
                cached_contact_nearest_point_indexes = debug_info[
                    "contact_nearest_point_indexes"
                ]
            grad_step_size = 5

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_FINGERS_CENTER_ONE_STEP
    ):
        N_ITERS = 1
        num_links = len(hand_model.mesh)
        target_points = compute_fingers_center(
            hand_model=hand_model, object_model=object_model, device=device
        )
        target_points = target_points.reshape(-1, 1, 3).repeat(1, num_links, 1)
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=target_points,
                dist_thresh_to_move_finger=0.001,
                dist_move_link=0.001,
                return_debug_info=True,
            )
            grad_step_size = 5

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_FINGERS_CENTER_MULTIPLE_STEP
    ):
        N_ITERS = 100
        # Use cached target and indices to continue moving the same points toward the same targets for each iter
        # Otherwise, would be moving different points to different targets each iter
        num_links = len(hand_model.mesh)
        cached_target_points = compute_fingers_center(
            hand_model=hand_model, object_model=object_model, device=device
        )
        cached_target_points = cached_target_points.reshape(-1, 1, 3).repeat(
            1, num_links, 1
        )

        cached_contact_nearest_point_indexes = None
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=cached_target_points,
                cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
                dist_thresh_to_move_finger=0.01,
                dist_move_link=0.01,
                return_debug_info=True,
            )
            if cached_target_points is None:
                cached_target_points = debug_info["target_points"]
            if cached_contact_nearest_point_indexes is None:
                cached_contact_nearest_point_indexes = debug_info[
                    "contact_nearest_point_indexes"
                ]
            grad_step_size = 1

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    else:
        raise NotImplementedError

    # Update hand pose parameters
    new_hand_pose = original_hand_pose.detach().clone()
    new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
    hand_model.set_parameters(new_hand_pose)

    return joint_angle_targets_to_optimize, losses, debug_infos


def compute_optimized_canonicalized_hand_pose(
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
    dist_thresh_to_move_finger: float = 0.01,
    desired_dist_from_object: float = 0.005,
) -> Tuple[torch.Tensor, List[float], List[Dict[str, Any]]]:
    # Canonicalized hand pose = hand pose modified so that the fingers are desired_dist_from_object away from the object
    # TODO: Consider optimization T and R as well (not just joint angles)
    desired_penetration_depth = -desired_dist_from_object
    original_hand_pose = hand_model.hand_pose.detach().clone()
    joint_angle_targets_to_optimize = (
        original_hand_pose.detach().clone()[:, 9:].requires_grad_(True)
    )

    losses = []
    debug_infos = []
    N_ITERS = 100
    cached_target_points = None
    cached_contact_nearest_point_indexes = None
    for i in range(N_ITERS):
        desired_penetration_depth = -desired_dist_from_object
        loss, debug_info = compute_loss_desired_penetration_depth(
            joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
            hand_model=hand_model,
            object_model=object_model,
            device=device,
            cached_target_points=cached_target_points,
            cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
            dist_thresh_to_move_finger=dist_thresh_to_move_finger,
            desired_penetration_depth=desired_penetration_depth,
            return_debug_info=True,
        )
        if cached_target_points is None:
            cached_target_points = debug_info["target_points"]
        if cached_contact_nearest_point_indexes is None:
            cached_contact_nearest_point_indexes = debug_info[
                "contact_nearest_point_indexes"
            ]
        grad_step_size = 5
        loss.backward(retain_graph=True)

        with torch.no_grad():
            joint_angle_targets_to_optimize -= (
                joint_angle_targets_to_optimize.grad * grad_step_size
            )
            joint_angle_targets_to_optimize.grad.zero_()
        losses.append(loss.item())
        debug_infos.append(debug_info)

    # Update hand pose parameters
    new_hand_pose = original_hand_pose.detach().clone()
    new_hand_pose[:, 9:] = joint_angle_targets_to_optimize.detach().clone()
    hand_model.set_parameters(new_hand_pose)

    return new_hand_pose, losses, debug_infos


def compute_optimized_joint_angle_targets_given_directions(
    hand_model: HandModel,
    grasp_dirs_array: torch.Tensor,
    dist_move_link: float = 0.01,
) -> Tuple[torch.Tensor, List[float], List[Dict[str, Any]]]:
    # Sanity check
    batch_size = hand_model.batch_size
    num_fingers = hand_model.num_fingers
    num_xyz = 3
    assert grasp_dirs_array.shape == (batch_size, num_fingers, num_xyz)

    # Compute target positions
    original_hand_pose = hand_model.hand_pose.detach().clone()
    original_joint_angle_targets = original_hand_pose[:, 9:].detach().clone()
    original_contact_points_hand = get_contact_points_hand(
        hand_model, original_joint_angle_targets
    )
    DIST_MOVE_LINK = dist_move_link
    target_points = original_contact_points_hand + grasp_dirs_array * DIST_MOVE_LINK

    joint_angle_targets_to_optimize = (
        original_joint_angle_targets.detach().clone().requires_grad_(True)
    )
    losses = []
    debug_infos = []
    N_ITERS = 100
    for i in range(N_ITERS):
        contact_points_hand_to_optimize = get_contact_points_hand(
            hand_model, joint_angle_targets_to_optimize
        )
        loss = (
            (target_points.detach().clone() - contact_points_hand_to_optimize)
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
        losses.append(loss.item())
        debug_infos.append(
            {
                "target_points": target_points.detach().clone(),
                "contact_points_hand": contact_points_hand_to_optimize.detach().clone(),
            }
        )

    # Update hand pose parameters
    new_hand_pose = original_hand_pose.detach().clone()
    new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
    hand_model.set_parameters(new_hand_pose)

    return joint_angle_targets_to_optimize, losses, debug_infos


def get_contact_points_hand(
    hand_model: HandModel, joint_angles: torch.Tensor
) -> torch.Tensor:
    batch_size = joint_angles.shape[0]
    num_fingers = hand_model.num_fingers
    num_xyz = 3

    # Forward kinematics
    current_status = hand_model.chain.forward_kinematics(joint_angles)
    all_contact_candidates = []
    for i, link_name in enumerate(hand_model.mesh):
        contact_candidates = hand_model.mesh[link_name]["contact_candidates"]
        if len(contact_candidates) == 0:
            continue

        contact_candidates = (
            current_status[link_name]
            .transform_points(contact_candidates)
            .expand(batch_size, -1, 3)
        )
        contact_candidates = contact_candidates @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)
        all_contact_candidates.append(contact_candidates)
    all_contact_candidates = torch.cat(all_contact_candidates, dim=1).to(
        hand_model.device
    )

    # TODO: Get finger centers
    # BRITTLE: Assumes ordering of links matches fingers nicely
    all_contact_candidates = all_contact_candidates.reshape(
        batch_size, num_fingers, -1, num_xyz
    )
    num_links_per_finger = all_contact_candidates.shape[2]
    VERBOSE = False
    if VERBOSE:
        print(f"Assuming {num_links_per_finger} links per finger")
    contact_points_hand = all_contact_candidates.mean(dim=2)
    assert contact_points_hand.shape == (batch_size, num_fingers, num_xyz)
    return contact_points_hand


def compute_link_name_to_contact_candidates(
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


def compute_fingertip_name_to_contact_candidates(
    link_name_to_contact_candidates: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # HACK: hardcoded
    # Merge links associated with same fingertip
    fingertip_keywords = ["link_3.0", "link_7.0", "link_11.0", "link_15.0"]
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


def compute_closest_contact_point_info(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # For each link, find the closest point on the object
    link_name_to_contact_candidates = compute_link_name_to_contact_candidates(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )

    # Merge links associated with same fingertip
    fingertip_name_to_contact_candidates = compute_fingertip_name_to_contact_candidates(
        link_name_to_contact_candidates=link_name_to_contact_candidates,
    )

    # To be populated
    num_fingers = len(fingertip_name_to_contact_candidates)
    batch_size = joint_angles.shape[0]
    all_hand_contact_nearest_points = torch.zeros((batch_size, num_fingers, 3)).to(
        device
    )
    all_nearest_object_to_hand_directions = torch.zeros(
        (batch_size, num_fingers, 3)
    ).to(device)
    all_nearest_distances = torch.zeros((batch_size, num_fingers)).to(device)
    all_hand_contact_nearest_point_indices = (
        torch.zeros((batch_size, num_fingers)).long().to(device)
    )

    for i, (fingertip_name, contact_candidates) in enumerate(
        fingertip_name_to_contact_candidates.items()
    ):
        n_contact_candidates = contact_candidates.shape[1]
        assert contact_candidates.shape == (batch_size, n_contact_candidates, 3)

        # From cal_distance, interiors are positive dist, exteriors are negative dist
        # Normals point from object to hand
        (
            distances_interior_positive,
            object_to_hand_directions,
        ) = object_model.cal_distance(contact_candidates)
        distances_interior_negative = (
            -distances_interior_positive
        )  # Negative, so large distance => far away
        nearest_point_index = distances_interior_negative.argmin(dim=1)
        nearest_distances = torch.gather(
            input=distances_interior_positive,
            dim=1,
            index=nearest_point_index.unsqueeze(1),
        )
        hand_contact_nearest_points = torch.gather(
            inputs=contact_candidates,
            dim=1,
            index=nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
        )
        nearest_object_to_hand_directions = torch.gather(
            inputs=object_to_hand_directions,
            dim=1,
            index=nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
        )

        assert hand_contact_nearest_points.shape == (batch_size, 1, 3)
        assert nearest_object_to_hand_directions.shape == (batch_size, 1, 3)
        assert nearest_distances.shape == (batch_size, 1)
        assert nearest_point_index.shape == (batch_size,)

        # Update tensors
        all_hand_contact_nearest_points[:, i : i + 1, :] = hand_contact_nearest_points
        all_nearest_object_to_hand_directions[
            :, i : i + 1, :
        ] = nearest_object_to_hand_directions
        all_nearest_distances[:, i : i + 1] = nearest_distances
        all_hand_contact_nearest_point_indices[:, i] = nearest_point_index

    return (
        all_hand_contact_nearest_points,
        all_nearest_object_to_hand_directions,
        all_nearest_distances,
        all_hand_contact_nearest_point_indices,
    )


def compute_fingertip_targets_and_hand_contact_nearest_point_indices(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Define both the fingertip targets and the indices of the contact points on the hand that should move towards those targets

    (
        hand_contact_nearest_points,
        nearest_object_to_hand_directions,
        nearest_distances,
        hand_contact_nearest_point_indices,
    ) = compute_closest_contact_point_info(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
        object_model=object_model,
        device=device,
    )

    optimization_method = (
        OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    )
    if (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    ):
        fingertip_targets = (
            hand_contact_nearest_points - nearest_object_to_hand_directions * 0.01
        )
    elif optimization_method == OptimizationMethod.DESIRED_PENETRATION_DEPTH:
        fingertip_targets = (
            hand_contact_nearest_points
            - nearest_object_to_hand_directions * (nearest_distances[..., None] + 0.05)
        )
    else:
        raise NotImplementedError

    return fingertip_targets, hand_contact_nearest_point_indices


def compute_grasp_orientations(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> torch.Tensor:
    # Can't just compute_grasp_dirs because we need to know the orientation of the fingers
    # Each finger has a rotation matrix [x, y, z] where x y z are column vectors
    #    * z is direction the fingertip moves
    #    * y is direction "up" along finger (from finger center to fingertip), modified to be perpendicular to z
    #    * x is direction "right" along finger (from finger center to fingertip), modified to be perpendicular to z and y
    # if y.cross(z) == 0, then need backup

    fingertip_targets, hand_
    (
        hand_contact_nearest_points,
        nearest_object_to_hand_directions,
        nearest_distances,
        hand_contact_nearest_point_indices,
    ) = compute_closest_contact_point_info(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
        object_model=object_model,
        device=device,
    )

    assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)
    return grasp_orientations

def compute_joint_angle_targets(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    grasp_orientations: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = joint_angles_start.shape[0]
    num_fingers = hand_model.num_fingers
    assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)

    grasp_directions = grasp_orientations[:, :, :, 2]
    assert grasp_directions.shape == (batch_size, num_fingers, 3, 3)
