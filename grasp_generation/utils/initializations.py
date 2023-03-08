"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: initializations
"""

import torch
import transforms3d
import math
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import numpy as np
import torch.nn.functional


def initialize_convex_hull(hand_model, object_model, args):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices
    
    Parameters
    ----------
    hand_model: hand_model.HandModel
    object_model: object_model.ObjectModel
    args: Namespace
    """

    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    # initialize translation and rotation

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        
        # get inflated convex hull

        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]
        vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # sample points

        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_size_each)
        p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=batch_size_each)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        # sample parameters

        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        deviate_theta = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        process_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)
        rotate_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)

        # solve transformation
        # rotation_hand: rotate the hand to align its grasping direction with the +z axis
        # rotation_local: jitter the hand's orientation in a cone
        # rotation_global and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull

        rotation_local = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'), dtype=torch.float, device=device)
            rotation_global[j] = torch.tensor(transforms3d.euler.euler2mat(math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'), dtype=torch.float, device=device)
        translation[i * batch_size_each: (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
        rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)
        rotation[i * batch_size_each: (i + 1) * batch_size_each] = rotation_global @ rotation_local @ rotation_hand
    
    # initialize joint angles
    # joint_angles_mu: hand-crafted canonicalized hand articulation
    # use truncated normal distribution to jitter the joint angles

    joint_angles_mu = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i], hand_model.joints_lower[i] - 1e-6, hand_model.joints_upper[i] + 1e-6)

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    # initialize contact point indices

    contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)

    hand_model.set_parameters(hand_pose, contact_point_indices)
