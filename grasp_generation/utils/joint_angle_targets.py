from utils.hand_model import HandModel
from utils.object_model import ObjectModel
import torch
from typing import Union, Dict, Any, Tuple, Optional, List

from enum import Enum, auto


class AutoName(Enum):
    # https://docs.python.org/3.9/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


class OptimizationMethod(AutoName):
    DESIRED_PENETRATION_DIST = auto()
    DESIRED_DIST_MOVE_ONE_STEP = auto()
    DESIRED_DIST_MOVE_MULTIPLE_STEPS = auto()
    DESIRED_DIST_MOVE_TOWARDS_CENTER_ONE_STEP = auto()
    DESIRED_DIST_MOVE_TOWARDS_CENTER_MULTIPLE_STEP = auto()


def compute_fingertip_positions(
    hand_model: HandModel,
    device: torch.device,
) -> torch.Tensor:
    current_status = hand_model.chain.forward_kinematics(hand_model.hand_pose[:, 9:])
    batch_size = hand_model.hand_pose.shape[0]

    fingertip_positions = []
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
        fingertip_positions.append(fingertip_position)

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


def compute_loss_desired_penetration_dist(
    joint_angle_targets_to_optimize: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
    dist_thresh_to_move_finger: float = 0.01,
    desired_penetration_dist: float = 0.003,
    return_debug_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    num_links = len(hand_model.mesh)
    batch_size = joint_angle_targets_to_optimize.shape[0]

    contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_distances = torch.zeros((batch_size, num_links)).to(device)

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
        nearest_point_index = distances.argmax(dim=1)
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

    target_points = contact_points_hand - contact_normals * (
        contact_distances[..., None] + desired_penetration_dist
    )

    loss = (target_points.detach().clone() - contact_points_hand).square().sum()

    if not return_debug_info:
        return loss

    return loss, {
        "target_points": target_points,
        "contact_points_hand": contact_points_hand,
        "contact_normals": contact_normals,
        "contact_distances": contact_distances,
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
    if optimization_method == OptimizationMethod.DESIRED_PENETRATION_DIST:
        N_ITERS = 100
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_penetration_dist(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                dist_thresh_to_move_finger=0.01,
                desired_penetration_dist=0.005,
                return_debug_info=True,
            )
            grad_step_size = 50
            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif optimization_method == OptimizationMethod.DESIRED_DIST_MOVE_ONE_STEP:
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

    elif optimization_method == OptimizationMethod.DESIRED_DIST_MOVE_MULTIPLE_STEPS:
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
        == OptimizationMethod.DESIRED_DIST_MOVE_TOWARDS_CENTER_ONE_STEP
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
            grad_step_size = 10

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
        == OptimizationMethod.DESIRED_DIST_MOVE_TOWARDS_CENTER_MULTIPLE_STEP
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
) -> Tuple[torch.Tensor, List[float], List[Dict[str, Any]]]:
    # TODO: Many of the parameters here are hardcoded
    # TODO: Consider optimization T and R as well (not just joint angles)
    original_hand_pose = hand_model.hand_pose.detach().clone()
    joint_angle_targets_to_optimize = (
        original_hand_pose.detach().clone()[:, 9:].requires_grad_(True)
    )

    losses = []
    debug_infos = []
    N_ITERS = 100
    for i in range(N_ITERS):
        loss, debug_info = compute_loss_desired_penetration_dist(
            joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
            hand_model=hand_model,
            object_model=object_model,
            device=device,
            dist_thresh_to_move_finger=0.01,
            desired_penetration_dist=-0.005,
            return_debug_info=True,
        )
        grad_step_size = 50
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
    new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
    hand_model.set_parameters(new_hand_pose)

    return new_hand_pose, losses, debug_infos
