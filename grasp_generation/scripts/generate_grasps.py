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
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType
from utils.qpos_pose_conversion import pose_to_qpos
from utils.seed import set_seed

from torch.multiprocessing import set_start_method
from typing import Tuple, List, Optional, Dict, Any
import trimesh
import plotly.graph_objects as go
import wandb
from datetime import datetime
from tap import Tap

try:
    set_start_method("spawn")
except RuntimeError:
    pass


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
np.seterr(all="raise")


class GenerateGraspsArgumentParser(Tap):
    # experiment settings
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    wandb_name: str = ""
    wandb_entity: str = "tylerlum"
    wandb_project: str = "DexGraspNet_v1"
    visualization_freq: int = 2000
    result_path: str = "../data/graspdata"
    data_root_path: str = "../data/meshdata"
    object_code_list: Optional[List[str]] = None
    all: bool = False
    overwrite: bool = False
    todo: bool = False
    seed: int = 1
    batch_size_each: int = 500
    max_total_batch_size: int = 1000
    n_iter: int = 6000

    # hyper parameters
    switch_possibility: float = 0.5
    mu: float = 0.98
    step_size: float = 0.005
    stepsize_period: int = 50
    starting_temperature: float = 18
    annealing_period: int = 30
    temperature_decay: float = 0.95
    n_contacts_per_finger: int = 1
    w_fc: float = 1.0
    w_dis: float = 300.0
    w_pen: float = 100.0
    w_spen: float = 100.0
    w_joints: float = 1.0
    w_ff: float = 3.0
    w_fp: float = 0.0

    # initialization settings
    jitter_strength: float = 0.1
    distance_lower: float = 0.2
    distance_upper: float = 0.3
    theta_lower: float = -math.pi / 6
    theta_upper: float = math.pi / 6

    # energy thresholds
    thres_fc: float = 0.3
    thres_dis: float = 0.005
    thres_pen: float = 0.001

    # verbose (grasps throughout)
    store_grasps_mid_optimization_path: Optional[str] = None
    store_grasps_mid_optimization_freq: Optional[int] = None


def create_visualization_figure(
    hand_model: HandModel,
    object_model: ObjectModel,
    idx_to_visualize: int,
) -> Tuple[go.Figure, str]:
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
    return fig, fig_title


def get_energy_term_log_dict(
    unweighted_energy_matrix: torch.Tensor,
    weighted_energy_matrix: torch.Tensor,
    idx_to_visualize: int,
) -> Dict[str, Any]:
    log_dict = {}
    for i, energy_name in enumerate(ENERGY_NAMES):
        shorthand = ENERGY_NAME_TO_SHORTHAND_DICT[energy_name]
        uw_shorthand = f"unweighted_{shorthand}"
        log_dict.update(
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
    return log_dict


def save_results(
    hand_model: HandModel,
    object_model: ObjectModel,
    object_code_list: List[str],
    hand_pose_st: torch.Tensor,
    energy: torch.Tensor,
    unweighted_energy_matrix: torch.Tensor,
    output_folder: str,
) -> None:
    num_objects, num_grasps_per_object = object_model.object_scale_tensor.shape
    assert len(object_code_list) == num_objects
    assert hand_pose_st.shape[0] == num_objects * num_grasps_per_object

    joint_names = handmodeltype_to_joint_names[hand_model.hand_model_type]
    for object_i, object_code in enumerate(object_code_list):
        object_grasp_data_list = []
        for object_grasp_j in range(num_grasps_per_object):
            grasp_idx = object_i * num_grasps_per_object + object_grasp_j

            scale = object_model.object_scale_tensor[object_i, object_grasp_j].item()
            qpos = pose_to_qpos(
                hand_pose=hand_model.hand_pose[grasp_idx].detach().cpu(),
                joint_names=joint_names,
            )
            qpos_st = pose_to_qpos(
                hand_pose=hand_pose_st[grasp_idx].detach().cpu(),
                joint_names=joint_names,
            )
            object_grasp_data = dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                energy=energy[grasp_idx].item(),
            )

            object_grasp_data.update(
                {
                    ENERGY_NAME_TO_SHORTHAND_DICT[
                        energy_name
                    ]: unweighted_energy_matrix[grasp_idx, k].item()
                    for k, energy_name in enumerate(ENERGY_NAMES)
                }
            )

            object_grasp_data_list.append(object_grasp_data)

        np.save(
            os.path.join(output_folder, object_code + ".npy"),
            object_grasp_data_list,
            allow_pickle=True,
        )


def generate(
    args_tuple: Tuple[GenerateGraspsArgumentParser, List[str], int, List[str]]
) -> None:
    args, object_code_list, id, gpu_list = args_tuple

    # Log to wandb
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{args.wandb_name}_{time_str}" if len(args.wandb_name) > 0 else time_str
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
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

    initialize_convex_hull(
        hand_model=hand_model,
        object_model=object_model,
        distance_lower=args.distance_lower,
        distance_upper=args.distance_upper,
        theta_lower=args.theta_lower,
        theta_upper=args.theta_upper,
        hand_model_type=args.hand_model_type,
        jitter_strength=args.jitter_strength,
        n_contacts_per_finger=args.n_contacts_per_finger,
    )

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

        # Store grasps mid optimization
        if (
            args.store_grasps_mid_optimization_freq is not None
            and (step % args.store_grasps_mid_optimization_freq == 0)
            and args.store_grasps_mid_optimization_path is not None
        ):
            new_output_folder = os.path.join(
                args.store_grasps_mid_optimization_path, str(step)
            )
            os.makedirs(new_output_folder, exist_ok=True)
            save_results(
                hand_model=hand_model,
                object_model=object_model,
                object_code_list=object_code_list,
                hand_pose_st=hand_pose_st,
                energy=energy,
                unweighted_energy_matrix=unweighted_energy_matrix,
                output_folder=new_output_folder,
            )

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
        wandb_log_dict.update(
            get_energy_term_log_dict(
                unweighted_energy_matrix=unweighted_energy_matrix,
                weighted_energy_matrix=weighted_energy_matrix,
                idx_to_visualize=idx_to_visualize,
            )
        )

        # Visualize
        if step % args.visualization_freq == 0:
            fig, fig_title = create_visualization_figure(
                hand_model=hand_model,
                object_model=object_model,
                idx_to_visualize=idx_to_visualize,
            )
            wandb_log_dict[fig_title] = fig

        wandb.log(wandb_log_dict)

    # save results
    save_results(
        hand_model=hand_model,
        object_model=object_model,
        object_code_list=object_code_list,
        hand_pose_st=hand_pose_st,
        energy=energy,
        unweighted_energy_matrix=unweighted_energy_matrix,
        output_folder=args.result_path,
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
        print(f"First 10 in object_code_list_all: {object_code_list_all[:10]}")

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
