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
from utils.hand_model_type import HandModelType
from utils.pose_conversion import pose_to_hand_config
from utils.seed import set_seed
from utils.parse_object_code_and_scale import object_code_and_scale_to_str, parse_object_code_and_scale
from utils.timers import LoopTimer

from torch.multiprocessing import set_start_method
from typing import Tuple, List, Optional, Dict, Any
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
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_v2_trial")
    output_hand_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/hand_config_dicts"
    )
    rand_object_scale: bool = False
    object_scale: Optional[float] = 0.075
    min_object_scale: float = 0.05
    max_object_scale: float = 0.125
    seed: Optional[int] = None
    batch_size_each_object: int = 250
    n_objects_per_batch: int = (
        20  # Runs batch_size_each_object * n_objects_per_batch grasps per GPU
    )
    n_iter: int = 2000
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
    # n_contacts_per_finger: int = 15
    # w_fc: float = 0.75
    # w_dis: float = 200.0
    # w_pen: float = 1500.0
    # w_spen: float = 100.0
    # w_joints: float = 1.0
    # w_ff: float = 3.0
    # w_fp: float = 0.0
    n_contacts_per_finger: int = 5
    w_fc: float = 0.5
    w_dis: float = 500
    w_pen: float = 300.0
    w_spen: float = 100.0
    w_joints: float = 1.0
    w_ff: float = 3.0
    w_fp: float = 0.0
    w_tpen: float = 100.0  # TODO: Tune
    use_penetration_energy: bool = False
    penetration_iters_frac: float = (
        0.0  # Fraction of iterations to perform penetration energy calculation
    )
    object_num_surface_samples: int = 5000
    object_num_samples_calc_penetration_energy: int = 500

    # initialization settings
    jitter_strength: float = 0.1
    distance_lower: float = 0.2
    distance_upper: float = 0.3
    theta_lower: float = -math.pi / 6
    theta_upper: float = math.pi / 6

    # energy function params
    thres_dis: float = 0.015
    thres_pen: float = 0.015

    # verbose (grasps throughout)
    store_grasps_mid_optimization_freq: Optional[int] = None
    store_grasps_mid_optimization_iters: Optional[List[int]] = None
    # store_grasps_mid_optimization_iters: Optional[List[int]] = [25] + [
        # int(ff * 2500) for ff in [0.2, 0.5, 0.95]  # TODO: May add this back
    # ]

    # Continue from previous run
    no_continue: bool = False


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
    object_codes: List[str],
    object_scales: List[float],
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
    assert len(object_codes) == num_objects
    assert hand_pose_start.shape[0] == num_objects * num_grasps_per_object
    correct_object_scales = (
        torch.Tensor(object_scales)
        .unsqueeze(-1)
        .expand(-1, object_model.batch_size_each)
    ).to(device=object_model.object_scale_tensor.device)
    assert (object_model.object_scale_tensor == correct_object_scales).all()

    # Reshape hand poses and energy terms to be (num_objects, num_grasps_per_object, ...)
    # an aside: it's absolutely ridiculous that we have to do this 🙃

    hand_pose = (
        hand_model.hand_pose.detach()
        .cpu()
        .reshape(num_objects, num_grasps_per_object, -1)
    )
    hand_pose_start = (
        hand_pose_start.detach().cpu().reshape(num_objects, num_grasps_per_object, -1)
    )

    unweighted_energy_matrix = unweighted_energy_matrix.reshape(
        num_objects, num_grasps_per_object, -1
    )
    energy = energy.reshape(num_objects, num_grasps_per_object)

    for ii, object_code in enumerate(object_codes):
        trans, rot, joint_angles = pose_to_hand_config(hand_pose=hand_pose[ii])

        trans_start, rot_start, joint_angles_start = pose_to_hand_config(
            hand_pose=hand_pose_start[ii]
        )

        energy_dict = {}
        for k, energy_name in enumerate(ENERGY_NAMES):
            energy_dict[energy_name] = (
                unweighted_energy_matrix[ii, :, k].detach().cpu().numpy()
            )
        energy_dict["Total Energy"] = energy[ii].detach().cpu().numpy()

        object_code_and_scale_str = object_code_and_scale_to_str(
            object_code, object_scales[ii]
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
    args_tuple: Tuple[
        GenerateHandConfigDictsArgumentParser,
        List[str],
        int,
        List[str],
        List[float],
    ]
) -> None:
    args, object_codes, id, gpu_list, object_scales = args_tuple
    try:
        loop_timer = LoopTimer()

        # Log to wandb
        with loop_timer.add_section_timer("wandb and setup"):
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = f"{args.wandb_name}_{time_str}" if len(args.wandb_name) > 0 else time_str
            if args.use_wandb:
                wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=name,
                    config=args,
                )

            # prepare models
            if args.use_multiprocess:
                worker = multiprocessing.current_process()._identity[0]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[worker - 1]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            device = torch.device("cuda")

        with loop_timer.add_section_timer("create hand model"):
            hand_model = HandModel(
                hand_model_type=args.hand_model_type,
                device=device,
                n_surface_points=1000,  # Need this for table penetration
            )

        with loop_timer.add_section_timer("create object model"):
            object_model = ObjectModel(
                meshdata_root_path=str(args.meshdata_root_path),
                batch_size_each=args.batch_size_each_object,
                num_samples=args.object_num_surface_samples,
                num_calc_samples=args.object_num_samples_calc_penetration_energy,
                device=device,
            )
            object_model.initialize(object_codes, object_scales)

        with loop_timer.add_section_timer("init convex hull"):
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

        with loop_timer.add_section_timer("create optimizer"):
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
        with loop_timer.add_section_timer("energy forward"):
            energy_name_to_weight_dict = {
                "Force Closure": args.w_fc,
                "Hand Contact Point to Object Distance": args.w_dis,
                "Hand Object Penetration": args.w_pen,
                "Hand Self Penetration": args.w_spen,
                "Joint Limits Violation": args.w_joints,
                "Finger Finger Distance": args.w_ff,
                "Finger Palm Distance": args.w_fp,
                "Hand Table Penetration": args.w_tpen,
            }
            energy, unweighted_energy_matrix, weighted_energy_matrix = cal_energy(
                hand_model,
                object_model,
                energy_name_to_weight_dict=energy_name_to_weight_dict,
                use_penetration_energy=args.use_penetration_energy,
                thres_dis=args.thres_dis,
                thres_pen=args.thres_pen,
            )

        with loop_timer.add_section_timer("energy backward"):
            energy.sum().backward(retain_graph=True)

        idx_to_visualize = 0
        step_first_compute_penetration_energy = int(
            args.n_iter * args.penetration_iters_frac
        )
        pbar = tqdm(range(args.n_iter), desc="optimizing", dynamic_ncols=True)
        for step in pbar:
            with loop_timer.add_section_timer("wandb and setup"):
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
                    thres_dis=args.thres_dis,
                    thres_pen=args.thres_pen,
                )
                energy[:] = updated_energy
                unweighted_energy_matrix[:] = updated_unweighted_energy_matrix
                weighted_energy_matrix[:] = updated_weighted_energy_matrix

            with loop_timer.add_section_timer("optimizer try step zero grad"):
                _ = optimizer.try_step()
                optimizer.zero_grad()

            with loop_timer.add_section_timer("energy forward"):
                (
                    new_energy,
                    new_unweighted_energy_matrix,
                    new_weighted_energy_matrix,
                ) = cal_energy(
                    hand_model,
                    object_model,
                    energy_name_to_weight_dict=energy_name_to_weight_dict,
                    use_penetration_energy=use_penetration_energy,
                    thres_dis=args.thres_dis,
                    thres_pen=args.thres_pen,
                )
            with loop_timer.add_section_timer("energy backward"):
                new_energy.sum().backward(retain_graph=True)

            with loop_timer.add_section_timer("update energy"):

                with torch.no_grad():
                    accept, temperature = optimizer.accept_step(energy, new_energy)

                    energy[accept] = new_energy[accept]
                    unweighted_energy_matrix[accept] = new_unweighted_energy_matrix[accept]
                    weighted_energy_matrix[accept] = new_weighted_energy_matrix[accept]

            # Store grasps mid optimization
            with loop_timer.add_section_timer("save mid optimization grasps"):
                if (
                    args.store_grasps_mid_optimization_freq is not None
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
                        object_codes=object_codes,
                        object_scales=object_scales,
                        hand_pose_start=hand_pose_start,
                        energy=energy,
                        unweighted_energy_matrix=unweighted_energy_matrix,
                        output_folder_path=new_output_folder,
                    )

            # Log
            with loop_timer.add_section_timer("wandb and setup"):
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

            PRINT_LOOP_TIMER_EVERY_LOOP = False
            if PRINT_LOOP_TIMER_EVERY_LOOP:
                loop_timer.pretty_print_section_times()

        with loop_timer.add_section_timer("save final grasps"):
            save_hand_config_dicts(
                hand_model=hand_model,
                object_model=object_model,
                object_codes=object_codes,
                object_scales=object_scales,
                hand_pose_start=hand_pose_start,
                energy=energy,
                unweighted_energy_matrix=unweighted_energy_matrix,
                output_folder_path=args.output_hand_config_dicts_path,
            )
        loop_timer.pretty_print_section_times()

    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_codes} and continuing")


def main(args: GenerateHandConfigDictsArgumentParser) -> None:
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"gpu_list: {gpu_list}")

    # check whether arguments are valid and process arguments
    args.output_hand_config_dicts_path.mkdir(parents=True, exist_ok=True)

    if not args.meshdata_root_path.exists():
        raise ValueError(f"meshdata_root_path {args.meshdata_root_path} doesn't exist")

    # generate
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(datetime.now().microsecond)

    object_codes = [path.name for path in args.meshdata_root_path.iterdir()]
    random.shuffle(object_codes)
    print(f"First 10 in object_codes: {object_codes[:10]}")
    print(f"len(object_codes): {len(object_codes)}")

    if args.rand_object_scale:
        object_scales = np.random.uniform(
            low=args.min_object_scale,
            high=args.max_object_scale,
            size=(len(object_codes),),
        )
    else:
        assert args.object_scale is not None
        object_scales = np.ones(len(object_codes)) * args.object_scale
    print(f"First 10 in object_scales: {object_scales[:10]}")
    print(f"len(object_scales): {len(object_scales)}")

    existing_object_code_and_scale_strs = (
        [path.stem for path in list(args.output_hand_config_dicts_path.glob("*.npy"))]
        if args.output_hand_config_dicts_path.exists()
        else []
    )
    new_object_code_and_scale_strs = [
        object_code_and_scale_to_str(object_code, object_scale)
        for object_code, object_scale in zip(object_codes, object_scales)
    ]

    if args.no_continue:
        # Compare input and output directories
        print(
            f"Found {len(existing_object_code_and_scale_strs)} objects in {args.output_hand_config_dicts_path}"
        )
        raise ValueError(
            f"Output folder {args.output_hand_config_dicts_path} already exists. Please delete it or set --no_continue to False."
        )
    elif len(existing_object_code_and_scale_strs) > 0:
        print(
            f"Found {len(existing_object_code_and_scale_strs)} objects in {args.output_hand_config_dicts_path}"
        )

        new_object_code_and_scale_strs = list(
            set(new_object_code_and_scale_strs)
            - set(existing_object_code_and_scale_strs)
        )
        print(f"Generating remaining {len(object_codes)} hand_config_dicts")

        object_codes, object_scales = [], []
        for object_code_and_scale_str in new_object_code_and_scale_strs:
            object_code, object_scale = parse_object_code_and_scale(
                object_code_and_scale_str
            )
            object_codes.append(object_code)
            object_scales.append(object_scale)

    object_code_groups = [
        object_codes[i : i + args.n_objects_per_batch]
        for i in range(0, len(object_codes), args.n_objects_per_batch)
    ]

    object_scale_groups = [
        object_scales[i : i + args.n_objects_per_batch]
        for i in range(0, len(object_scales), args.n_objects_per_batch)
    ]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append(
            (args, object_code_group, id + 1, gpu_list, object_scale_groups[id])
        )

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
