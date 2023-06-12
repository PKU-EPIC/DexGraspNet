"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: generate grasps in large-scale, use multiple graphics cards, no logging
"""

import os
import sys

sys.path.append(os.path.realpath("."))

import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy, ENERGY_NAMES, ENERGY_NAME_TO_SHORTHAND_DICT
from utils.optimizer import Annealing
from utils.hand_model_type import handmodeltype_to_joint_names
from utils.qpos_pose_conversion import pose_to_qpos
from utils.seed import set_seed
from utils.generate_grasps_argument_parser import GenerateGraspsArgumentParser

from torch.multiprocessing import set_start_method
from typing import Tuple, List
import trimesh
import plotly.graph_objects as go
import wandb
from datetime import datetime

try:
    set_start_method("spawn")
except RuntimeError:
    pass


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
np.seterr(all="raise")


def get_meshes(
    hand_model: HandModel,
    object_model: ObjectModel,
    object_idx: int,
    batch_idx: int,
    batch_size_each: int,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    idx = object_idx * batch_size_each + batch_idx

    # Get hand mesh
    hand_mesh = hand_model.get_trimesh_data(i=idx)

    # Get object mesh
    scale = object_model.object_scale_tensor[object_idx][batch_idx].item()
    object_mesh = object_model.object_mesh_list[object_idx].copy().apply_scale(scale)
    return hand_mesh, object_mesh


def generate(args_list):
    args, object_code_list, id, gpu_list = args_list
    args: GenerateGraspsArgumentParser = args

    # Log to wandb
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{args.wandb_name}_{time_str}" if len(args.wandb_name) > 0 else time_str
    wandb.init(
        entity="tylerlum",
        project="DexGraspNet_v1",
        name=name,
        config=args,
    )

    set_seed(args.seed)

    # prepare models

    worker = multiprocessing.current_process()._identity[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[worker - 1]
    device = torch.device("cuda")

    hand_model = HandModel(
        hand_model_type=args.hand_model_type,
        device=device,
    )

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=2000,
        device=device,
    )
    object_model.initialize(object_code_list)

    initialize_convex_hull(hand_model, object_model, args)

    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        "switch_possibility": args.switch_possibility,
        "starting_temperature": args.starting_temperature,
        "temperature_decay": args.temperature_decay,
        "annealing_period": args.annealing_period,
        "step_size": args.step_size,
        "stepsize_period": args.stepsize_period,
        "n_contacts_per_finger": args.n_contacts_per_finger,
        "mu": args.mu,
        "device": device,
    }
    optimizer = Annealing(hand_model, **optim_config)

    # optimize
    energy_name_to_weight_dict = {
        "Force Closure": args.w_fc,
        "Hand Contact Point to Object Distance": args.w_dis,
        "Hand Object Penetration": args.w_pen,
        "Hand Self Penetration": args.w_spen,
        "Joint Limits Violation": args.w_joints,
        "Finger Finger Distance": args.w_ff,
        "Finger Palm Distance": args.w_fp,
    }

    energy, unweighted_energy_matrix, weighted_energy_matrix = cal_energy(
        hand_model, object_model, energy_name_to_weight_dict=energy_name_to_weight_dict
    )

    energy.sum().backward(retain_graph=True)

    idx_to_visualize = 0
    for step in tqdm(range(args.n_iter), desc=f"optimizing {id}"):
        wandb_log_dict = {}
        wandb_log_dict["optimization_step"] = step

        s = optimizer.try_step()

        optimizer.zero_grad()

        (
            new_energy,
            new_unweighted_energy_matrix,
            new_weighted_energy_matrix,
        ) = cal_energy(
            hand_model,
            object_model,
            energy_name_to_weight_dict=energy_name_to_weight_dict,
        )
        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, temperature = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            unweighted_energy_matrix[accept] = new_unweighted_energy_matrix[accept]
            weighted_energy_matrix[accept] = new_weighted_energy_matrix[accept]

        # Log
        wandb_log_dict.update(
            {
                "accept": accept.sum().item(),
                "temperature": temperature.item(),
                "energy": energy.mean().item(),
                f"accept_{idx_to_visualize}": accept[idx_to_visualize].item(),
                f"energy_{idx_to_visualize}": energy[idx_to_visualize].item(),
            }
        )
        for i, energy_name in enumerate(ENERGY_NAMES):
            shorthand = ENERGY_NAME_TO_SHORTHAND_DICT[energy_name]
            uw_shorthand = f"unweighted_{shorthand}"
            wandb_log_dict.update(
                {
                    uw_shorthand: unweighted_energy_matrix[:, i].mean().item(),
                    shorthand: weighted_energy_matrix[:, i].mean().item(),
                    f"{uw_shorthand}_{idx_to_visualize}": unweighted_energy_matrix[
                        idx_to_visualize, i
                    ].item(),
                    f"{shorthand}_{idx_to_visualize}": weighted_energy_matrix[
                        idx_to_visualize, i
                    ].item(),
                }
            )

        # Visualize
        if step % args.visualization_freq == 0:
            fig_title = f"hand_object_visualization_{idx_to_visualize}"
            fig = go.Figure(
                layout=go.Layout(
                    scene=dict(
                        xaxis=dict(title="X"),
                        yaxis=dict(title="Y"),
                        zaxis=dict(title="Z"),
                        aspectmode="data",
                    ),
                    showlegend=True,
                    title=fig_title,
                )
            )
            plots = [
                *hand_model.get_plotly_data(
                    i=idx_to_visualize,
                    opacity=1.0,
                    with_contact_points=True,
                    with_contact_candidates=True,
                ),
                *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
            ]
            for plot in plots:
                fig.add_trace(plot)
            wandb_log_dict[fig_title] = fig
        wandb.log(wandb_log_dict)

    # save results
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(args.batch_size_each):
            idx = i * args.batch_size_each + j
            scale = object_model.object_scale_tensor[i][j].item()
            qpos = pose_to_qpos(
                hand_pose=hand_model.hand_pose[idx].detach().cpu(),
                joint_names=joint_names,
            )
            qpos_st = pose_to_qpos(
                hand_pose=hand_pose_st[idx].detach().cpu(),
                joint_names=joint_names,
            )
            data = dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                energy=energy[idx].item(),
            )

            data.update(
                {
                    ENERGY_NAME_TO_SHORTHAND_DICT[
                        energy_name
                    ]: unweighted_energy_matrix[idx, k].item()
                    for k, energy_name in enumerate(ENERGY_NAMES)
                }
            )

            data_list.append(data)

        np.save(
            os.path.join(args.result_path, object_code + ".npy"),
            data_list,
            allow_pickle=True,
        )


if __name__ == "__main__":
    args = GenerateGraspsArgumentParser().parse_args()

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"gpu_list: {gpu_list}")

    # check whether arguments are valid and process arguments
    set_seed(args.seed)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.exists(args.data_root_path):
        raise ValueError(f"data_root_path {args.data_root_path} doesn't exist")

    if (args.object_code_list is not None) + args.all != 1:
        raise ValueError(
            "exactly one among 'object_code_list' 'all' should be specified"
        )

    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = os.listdir(args.data_root_path)
        print(f"First 10: {object_code_list_all[:10]}")

    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError(
                "object_code_list isn't a subset of dirs in data_root_path"
            )
    else:
        object_code_list = object_code_list_all

    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, object_code + ".npy")):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(
            f"batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}"
        )

    print(f"n_objects: {len(object_code_list)}")

    # generate

    set_seed(args.seed)
    random.shuffle(object_code_list)
    objects_each = args.max_total_batch_size // args.batch_size_each
    object_code_groups = [
        object_code_list[i : i + objects_each]
        for i in range(0, len(object_code_list), objects_each)
    ]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    with multiprocessing.Pool(len(gpu_list)) as p:
        it = tqdm(
            p.imap(generate, process_args),
            total=len(process_args),
            desc="generating",
            maxinterval=1000,
        )
        list(it)
