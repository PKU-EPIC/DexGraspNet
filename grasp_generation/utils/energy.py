"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: energy functions
"""

import torch
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from typing import Dict, Tuple

ENERGY_NAMES = [
    "Force Closure",
    "Hand Contact Point to Object Distance",
    "Hand Object Penetration",
    "Hand Self Penetration",
    "Joint Limits Violation",
    "Finger Finger Distance",
    "Finger Palm Distance",
]

ENERGY_NAME_TO_SHORTHAND_DICT = {
    "Force Closure": "E_fc",
    "Hand Contact Point to Object Distance": "E_dis",
    "Hand Object Penetration": "E_pen",
    "Hand Self Penetration": "E_spen",
    "Joint Limits Violation": "E_joints",
    "Finger Finger Distance": "E_ff",
    "Finger Palm Distance": "E_fp",
}

assert set(ENERGY_NAMES) == set(ENERGY_NAME_TO_SHORTHAND_DICT.keys())


def _cal_force_closure(
    object_to_hand_contact_point_normal: torch.Tensor,
    contact_points: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size, n_contact, _ = object_to_hand_contact_point_normal.shape

    transformation_matrix = torch.tensor(
        [
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float,
        device=device,
    )
    g = (
        torch.cat(
            [
                torch.eye(3, dtype=torch.float, device=device)
                .expand(batch_size, n_contact, 3, 3)
                .reshape(batch_size, 3 * n_contact, 3),
                (contact_points @ transformation_matrix).view(
                    batch_size, 3 * n_contact, 3
                ),
            ],
            dim=2,
        )
        .float()
        .to(device)
    )
    norm = torch.norm(
        object_to_hand_contact_point_normal.reshape(batch_size, 1, 3 * n_contact) @ g,
        dim=[1, 2],
    )
    E_fc = norm * norm
    return E_fc


def cal_energy(
    hand_model: HandModel,
    object_model: ObjectModel,
    energy_name_to_weight_dict: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert set(energy_name_to_weight_dict.keys()) == set(ENERGY_NAMES)

    # Prepare useful variables
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device
    (
        object_to_hand_contact_point_distances,
        object_to_hand_contact_point_normal,
    ) = object_model.cal_distance(hand_model.contact_points)

    # Compute energies
    unweighted_energy_matrix = torch.zeros(
        batch_size, len(ENERGY_NAMES), dtype=torch.float, device=device
    )
    for i, energy_name in enumerate(ENERGY_NAMES):
        if energy_name == "Force Closure":
            unweighted_energy_matrix[:, i] = _cal_force_closure(
                object_to_hand_contact_point_normal=object_to_hand_contact_point_normal,
                contact_points=hand_model.contact_points,
                device=device,
            )
        elif energy_name == "Hand Contact Point to Object Distance":
            unweighted_energy_matrix[:, i] = torch.sum(
                object_to_hand_contact_point_distances.abs(), dim=-1, dtype=torch.float
            ).to(device)
        elif energy_name == "Hand Object Penetration":
            object_scale = (
                object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
            )
            object_surface_points = (
                object_model.surface_points_tensor * object_scale
            )  # (n_objects * batch_size_each, num_samples, 3)
            hand_to_object_surface_point_distances = hand_model.cal_distance(
                object_surface_points
            )
            hand_to_object_surface_point_distances[
                hand_to_object_surface_point_distances <= 0
            ] = 0
            unweighted_energy_matrix[:, i] = hand_to_object_surface_point_distances.sum(
                -1
            )
        elif energy_name == "Hand Self Penetration":
            unweighted_energy_matrix[:, i] = hand_model.self_penetration()
        elif energy_name == "Joint Limits Violation":
            unweighted_energy_matrix[:, i] = torch.sum(
                (hand_model.hand_pose[:, 9:] > hand_model.joints_upper)
                * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper),
                dim=-1,
            ) + torch.sum(
                (hand_model.hand_pose[:, 9:] < hand_model.joints_lower)
                * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]),
                dim=-1,
            )
        elif energy_name == "Finger Finger Distance":
            unweighted_energy_matrix[:, i] = torch.cdist(hand_model.contact_points, hand_model.contact_points, p=2).reshape(batch_size, -1).sum(dim=-1)
        elif energy_name == "Finger Palm Distance":
            palm_position = hand_model.global_translation[:, None, :]
            unweighted_energy_matrix[:, i] = (palm_position - hand_model.contact_points).norm(dim=-1).sum(dim=-1)
        else:
            raise ValueError(f"Unknown energy name: {energy_name}")

    energy_weights = torch.tensor([energy_name_to_weight_dict[energy_name] for energy_name in ENERGY_NAMES],, device=device)
    weighted_energy_matrix = unweighted_energy_matrix * energy_weights[None, ...]
    energy = weighted_energy_matrix.sum(dim=-1)
    return energy, unweighted_energy_matrix, weighted_energy_matrix
