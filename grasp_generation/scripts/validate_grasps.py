"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import translation_names, rot_names, HandModelType, handmodeltype_to_joint_names
from utils.qpos_pose_conversion import qpos_to_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand_model_type', default=HandModelType.SHADOW_HAND, type=HandModelType.from_string, choices=list(HandModelType))
    parser.add_argument('--gpu', default=3, type=int)
    parser.add_argument('--val_batch', default=500, type=int)
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    parser.add_argument('--grasp_path', default="../data/graspdata", type=str)
    parser.add_argument('--result_path', default="../data/dataset", type=str)
    parser.add_argument('--object_code',
                        default="sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2",
                        type=str)
    # if index is received, then the debug mode is on
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)

    args = parser.parse_args()

    joint_names = handmodeltype_to_joint_names[args.hand_model_type]

    os.environ.pop("CUDA_VISIBLE_DEVICES")
    os.makedirs(args.result_path, exist_ok=True)

    if not args.no_force:
        # Read in hand state and scale tensor
        device = torch.device(
            f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        data_dict = np.load(os.path.join(
            args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
        batch_size = data_dict.shape[0]
        hand_state = []
        scale_tensor = []
        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            hand_pose = qpos_to_pose(qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False).to(device)
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)

        # hand model
        hand_model = HandModel(
            hand_model_type=args.hand_model_type,
            n_surface_points=2000,
            device=device
        )
        hand_model.set_parameters(hand_state)

        # object model
        object_model = ObjectModel(
            data_root_path=args.mesh_path,
            batch_size_each=batch_size,
            num_samples=0,
            device=device
        )
        object_model.initialize(args.object_code)
        object_model.object_scale_tensor = scale_tensor

        # calculate contact points and contact normals
        num_links = len(hand_model.mesh)
        contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
        contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)

        for i, link_name in enumerate(hand_model.mesh):
            if len(hand_model.mesh[link_name]['surface_points']) == 0:
                continue
            surface_points = hand_model.current_status[link_name].transform_points(
                hand_model.mesh[link_name]['surface_points']).expand(batch_size, -1, 3)
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2) + hand_model.global_translation.unsqueeze(1)
            distances, normals = object_model.cal_distance(
                surface_points)
            nearest_point_index = distances.argmax(dim=1)
            nearest_distances = torch.gather(
                distances, 1, nearest_point_index.unsqueeze(1))
            nearest_points_hand = torch.gather(
                surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            nearest_normals = torch.gather(
                normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            admited = -nearest_distances < args.thres_cont
            admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
            contact_points_hand[:, i:i+1, :] = torch.where(
                admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])
            contact_normals[:, i:i+1, :] = torch.where(
                admited, nearest_normals, contact_normals[:, i:i+1, :])

        target_points = contact_points_hand + contact_normals * args.dis_move
        loss = (target_points.detach().clone() -
                contact_points_hand).square().sum()
        loss.backward()
        with torch.no_grad():
            hand_state[:, 9:] += hand_state.grad[:, 9:] * args.grad_move
            hand_state.grad.zero_()

    if (args.index is not None):
        sim = IsaacValidator(hand_model_type=args.hand_model_type, gpu=args.gpu, mode="gui")
    else:
        sim = IsaacValidator(hand_model_type=args.hand_model_type, gpu=args.gpu)

    data_dict = np.load(os.path.join(
        args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']
        rot = [qpos[name] for name in rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        translations.append(np.array([qpos[name]
                            for name in translation_names]))
        hand_poses.append(np.array([qpos[name] for name in joint_names]))
        scale_array.append(scale)

        if "E_pen" in data_dict[i]:
            E_pen_array.append(data_dict[i]["E_pen"])
        elif args.index is not None:
            print(f"Warning: E_pen not found in data_dict[{i}]")
            print("This is expected behavior if you are validating already validated grasps")
            E_pen_array.append(0)
        else:
            raise ValueError(f"E_pen not found in data_dict[{i}]")
    E_pen_array = np.array(E_pen_array)
    if not args.no_force:
        hand_poses = hand_state[:, 9:]

    if (args.index is not None):
        sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                       os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
        index = args.index
        sim.add_env_single_test_rotation(rotations[index], translations[index], hand_poses[index],
                           scale_array[index], 0)
        result = sim.run_sim()
        print(f"result = {result}")
        print(f"E_pen < args.penetration_threshold = {E_pen_array[index]:.7f} < {args.penetration_threshold:.7f} = {E_pen_array[index] < args.penetration_threshold}")
        estimated = E_pen_array < args.penetration_threshold
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                           os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
            for index in range(offset, offset_):
                sim.add_env_all_test_rotations(rotations[index], translations[index], hand_poses[index],
                            scale_array[index])
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            offset = offset_
        for i in range(batch_size):
            simulated[i] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)

        estimated = E_pen_array < args.penetration_threshold
        valid = simulated * estimated
        print(
            f'estimated: {estimated.sum().item()}/{batch_size}, '
            f'simulated: {simulated.sum().item()}/{batch_size}, '
            f'valid: {valid.sum().item()}/{batch_size}')
        result_list = []
        for i in range(batch_size):
            if (valid[i]):
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                result_list.append(new_data_dict)
        np.save(os.path.join(args.result_path, args.object_code +
                '.npy'), result_list, allow_pickle=True)
    sim.destroy()
