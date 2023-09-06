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
from utils.pose_conversion import pose_to_hand_config
from utils.seed import set_seed
from utils.parse_object_code_and_scale import object_code_and_scale_to_str

from torch.multiprocessing import set_start_method
from typing import Tuple, List, Optional, Dict, Any
import trimesh
import plotly.graph_objects as go
import wandb
from datetime import datetime
from tap import Tap
import pathlib

try:
    set_start_method("spawn")
except RuntimeError:
    pass


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
np.seterr(all="raise")


class GenerateHandConfigDictsArgumentParser(Tap):
    # experiment settings
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata_trial")
    output_hand_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/hand_config_dicts"
    )
    object_scale: float = 0.1
    seed: int = 1
    batch_size_each_object: int = 500
    n_objects_per_batch: int = (
        2  # Runs batch_size_each_object * n_objects_per_batch grasps per GPU
    )
    n_iter: int = 4000
    use_multiprocess: bool = False

    # Logging
    use_wandb: bool = False
    wandb_name: str = ""
    wandb_entity: str = "tylerlum"
    wandb_project: str = "DexGraspNet_v1"
    wandb_visualization_freq: Optional[int] = 50

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
    w_dis: float = 100.0
    w_pen: float = 300.0
    w_spen: float = 100.0
    w_joints: float = 1.0
    w_ff: float = 3.0
    w_fp: float = 0.0
    use_penetration_energy: bool = False
    penetration_iters_frac: float = (
        0.7  # Fraction of iterations to perform penetration energy calculation
    )
    object_num_samples_for_penetration_energy: int = 2000

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
    store_grasps_mid_optimization_freq: Optional[int] = 50
    store_grasps_mid_optimization_iters: Optional[List[int]] = None


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


def save_hand_config_dicts(
    hand_model: HandModel,
    object_model: ObjectModel,
    object_code_list: List[str],
    object_scale: float,
    hand_pose_start: torch.Tensor,
    energy: torch.Tensor,
    unweighted_energy_matrix: torch.Tensor,
    output_folder_path: pathlib.Path,
) -> None:
    """
    Save results to output_folder_path
        * <output_folder_path>/<object_code>_<object_scale>.npy

    TODO: update docstring.
    """
    num_objects, num_grasps_per_object = object_model.object_scale_tensor.shape
    assert len(object_code_list) == num_objects
    assert hand_pose_start.shape[0] == num_objects * num_grasps_per_object
    assert (object_model.object_scale_tensor == object_scale).all()

    for _, object_code in enumerate(object_code_list):
        energy_dict = {}

        trans, rot, joint_angles = pose_to_hand_config(
            hand_pose=hand_model.hand_pose.detach().cpu(),
        )
        trans_start, rot_start, joint_angles_start = pose_to_hand_config(
            hand_pose=hand_pose_start.detach().cpu(),
        )

        for k, energy_name in enumerate(ENERGY_NAMES):
            energy_dict[energy_name] = (
                unweighted_energy_matrix[:, k].detach().cpu().numpy()
            )

        object_code_and_scale_str = object_code_and_scale_to_str(
            object_code, object_scale
        )

        hand_config_dict = {
            "trans": trans,
            "rot": rot,
            "joint_angles": joint_angles,
            "trans_start": trans_start,
            "rot_start": rot_start,
            "joint_angles_start": joint_angles_start,
            **energy_dict,
        }

        np.save(
            output_folder_path / f"{object_code_and_scale_str}.npy",
            hand_config_dict,
            allow_pickle=True,
        )


def generate(
    args_tuple: Tuple[GenerateHandConfigDictsArgumentParser, List[str], int, List[str]]
) -> None:
    args, object_code_list, id, gpu_list = args_tuple

    # Log to wandb
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{args.wandb_name}_{time_str}" if len(args.wandb_name) > 0 else time_str
    if args.use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=name,
            config=args,
        )

    set_seed(args.seed)

    # prepare models
    if args.use_multiprocess:
        worker = multiprocessing.current_process()._identity[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[worker - 1]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    hand_model = HandModel(
        hand_model_type=args.hand_model_type,
        device=device,
    )

    object_model = ObjectModel(
        meshdata_root_path=str(args.meshdata_root_path),
        batch_size_each=args.batch_size_each_object,
        scale=args.object_scale,
        num_samples=args.object_num_samples_for_penetration_energy,
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

    hand_pose_start = hand_model.hand_pose.detach()

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
        hand_model,
        object_model,
        energy_name_to_weight_dict=energy_name_to_weight_dict,
        use_penetration_energy=args.use_penetration_energy,
    )

    energy.sum().backward(retain_graph=True)

    idx_to_visualize = 0
    step_first_compute_penetration_energy = int(
        args.n_iter * args.penetration_iters_frac
    )
    pbar = tqdm(range(args.n_iter), desc="optimizing", dynamic_ncols=True)
    for step in pbar:
        wandb_log_dict = {}
        wandb_log_dict["optimization_step"] = step

        use_penetration_energy = (
            args.use_penetration_energy
            and step >= step_first_compute_penetration_energy
        )

        # When we start using penetration energy, we must recompute the current energy with penetration energy
        # Else the current energy will appear artificially better than all new energies
        # So optimizer will stop accepting new energies
        if step == step_first_compute_penetration_energy:
            if args.use_penetration_energy:
                assert (
                    use_penetration_energy
                ), f"On step {step}, use_penetration_energy is {use_penetration_energy} but should be True"
            (
                updated_energy,
                updated_unweighted_energy_matrix,
                updated_weighted_energy_matrix,
            ) = cal_energy(
                hand_model,
                object_model,
                energy_name_to_weight_dict=energy_name_to_weight_dict,
                use_penetration_energy=use_penetration_energy,
            )
            energy[:] = updated_energy
            unweighted_energy_matrix[:] = updated_unweighted_energy_matrix
            weighted_energy_matrix[:] = updated_weighted_energy_matrix

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
            use_penetration_energy=use_penetration_energy,
        )
        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, temperature = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            unweighted_energy_matrix[accept] = new_unweighted_energy_matrix[accept]
            weighted_energy_matrix[accept] = new_weighted_energy_matrix[accept]

        # Store grasps mid optimization
        if (
            args.store_grasps_mid_optimization_iters is not None
            and args.store_grasps_mid_optimization_freq is not None
            and step % args.store_grasps_mid_optimization_freq == 0
        ) or (
            args.store_grasps_mid_optimization_iters is not None
            and step in args.store_grasps_mid_optimization_iters
        ):
            new_output_folder = (
                pathlib.Path(f"{args.output_hand_config_dicts_path}")
                / "mid_optimization"
                / str(step)
            )
            new_output_folder.mkdir(parents=True, exist_ok=True)
            save_hand_config_dicts(
                hand_model=hand_model,
                object_model=object_model,
                object_code_list=object_code_list,
                object_scale=args.object_scale,
                hand_pose_start=hand_pose_start,
                energy=energy,
                unweighted_energy_matrix=unweighted_energy_matrix,
                output_folder_path=new_output_folder,
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
        if (
            args.wandb_visualization_freq is not None
            and step % args.wandb_visualization_freq == 0
        ):
            fig, fig_title = create_visualization_figure(
                hand_model=hand_model,
                object_model=object_model,
                idx_to_visualize=idx_to_visualize,
            )
            wandb_log_dict[fig_title] = fig

        if args.use_wandb:
            wandb.log(wandb_log_dict)

        pbar.set_description(f"optimizing, mean energy: {energy.mean().item():.4f}")

    save_hand_config_dicts(
        hand_model=hand_model,
        object_model=object_model,
        object_code_list=object_code_list,
        object_scale=args.object_scale,
        hand_pose_start=hand_pose_start,
        energy=energy,
        unweighted_energy_matrix=unweighted_energy_matrix,
        output_folder_path=args.output_hand_config_dicts_path,
    )


def main(args: GenerateHandConfigDictsArgumentParser) -> None:
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"gpu_list: {gpu_list}")

    # check whether arguments are valid and process arguments
    set_seed(args.seed)
    args.output_hand_config_dicts_path.mkdir(parents=True, exist_ok=True)

    if not args.meshdata_root_path.exists():
        raise ValueError(f"meshdata_root_path {args.meshdata_root_path} doesn't exist")

    object_code_list = [path.name for path in args.meshdata_root_path.iterdir()]
    print(f"First 10 in object_code_list_all: {object_code_list[:10]}")
    print(f"len(object_code_list): {len(object_code_list)}")

    # generate
    set_seed(args.seed)
    random.shuffle(object_code_list)
    object_code_groups = [
        object_code_list[i : i + args.n_objects_per_batch]
        for i in range(0, len(object_code_list), args.n_objects_per_batch)
    ]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    if args.use_multiprocess:
        with multiprocessing.Pool(len(gpu_list)) as p:
            it = tqdm(
                p.imap(generate, process_args),
                total=len(process_args),
                desc="generating",
                maxinterval=1000,
            )
            list(it)
    else:
        for process_arg in tqdm(process_args, desc="generating", maxinterval=1000):
            generate(process_arg)


if __name__ == "__main__":
    args = GenerateHandConfigDictsArgumentParser().parse_args()
    main(args)
