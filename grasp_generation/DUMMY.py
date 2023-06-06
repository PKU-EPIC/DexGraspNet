from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import handmodeltype_to_expectedcontactlinknames
import torch
from typing import Union, Dict, Any, Tuple, Optional


def compute_loss_desired_penetration_dist(
    joint_angle_targets_to_optimize: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    batch_size: int,
    device: torch.device,
    dist_thresh_to_move_finger: float = 0.01,
    desired_penetration_dist: float = 0.003,
    return_debug_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    num_links = len(hand_model.mesh)
    contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_distances = torch.zeros((batch_size, num_links)).to(device)

    expected_contact_link_names = handmodeltype_to_expectedcontactlinknames[
        hand_model.hand_model_type
    ]

    current_status = hand_model.chain.forward_kinematics(
        joint_angle_targets_to_optimize
    )
    for i, link_name in enumerate(hand_model.mesh):
        surface_points = hand_model.mesh[link_name]["contact_candidates"]
        if len(surface_points) == 0:
            continue
        if link_name not in expected_contact_link_names:
            continue

        surface_points = (
            current_status[link_name]
            .transform_points(surface_points)
            .expand(batch_size, -1, 3)
        )
        surface_points = surface_points @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        # Interiors are positive dist, exteriors are negative dist
        # Normals point from object to hand
        distances, normals = object_model.cal_distance(surface_points)
        nearest_point_index = distances.argmax(dim=1)
        nearest_distances = torch.gather(distances, 1, nearest_point_index.unsqueeze(1))
        nearest_points_hand = torch.gather(
            surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        nearest_normals = torch.gather(
            normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        admited = -nearest_distances < dist_thresh_to_move_finger
        contact_distances[:, i : i + 1] = torch.where(
            admited, -nearest_distances, contact_distances[:, i : i + 1]
        )
        admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points_hand[:, i : i + 1, :] = torch.where(
            admited, nearest_points_hand, contact_points_hand[:, i : i + 1, :]
        )
        contact_normals[:, i : i + 1, :] = torch.where(
            admited, nearest_normals, contact_normals[:, i : i + 1, :]
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
    batch_size: int,
    device: torch.device,
    cached_target_points: Optional[torch.Tensor] = None,
    cached_contact_nearest_point_indexes: Optional[torch.Tensor] = None,
    dist_thresh_to_move_finger: float = 0.001,
    dist_move_link: float = 0.001,
    return_debug_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    num_links = len(hand_model.mesh)
    contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_nearest_point_indexes = torch.zeros((batch_size, num_links)).long().to(device)

    expected_contact_link_names = handmodeltype_to_expectedcontactlinknames[
        hand_model.hand_model_type
    ]

    current_status = hand_model.chain.forward_kinematics(
        joint_angle_targets_to_optimize
    )
    for i, link_name in enumerate(hand_model.mesh):
        surface_points = hand_model.mesh[link_name]["contact_candidates"]
        if len(surface_points) == 0:
            continue
        if link_name not in expected_contact_link_names:
            continue

        surface_points = (
            current_status[link_name]
            .transform_points(surface_points)
            .expand(batch_size, -1, 3)
        )
        surface_points = surface_points @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        # Interiors are positive dist, exteriors are negative dist
        # Normals point from object to hand
        distances, normals = object_model.cal_distance(surface_points)
        if cached_contact_nearest_point_indexes is None:
            nearest_point_index = distances.argmax(dim=1)
        else:
            nearest_point_index = cached_contact_nearest_point_indexes[:, i]
        nearest_distances = torch.gather(distances, 1, nearest_point_index.unsqueeze(1))
        nearest_points_hand = torch.gather(
            surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        nearest_normals = torch.gather(
            normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        admited = -nearest_distances < dist_thresh_to_move_finger
        admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points_hand[:, i : i + 1, :] = torch.where(
            admited, nearest_points_hand, contact_points_hand[:, i : i + 1, :]
        )
        contact_normals[:, i : i + 1, :] = torch.where(
            admited, nearest_normals, contact_normals[:, i : i + 1, :]
        )
        contact_nearest_point_indexes[:, i : i + 1] = nearest_point_index

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