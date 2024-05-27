from __future__ import annotations
import subprocess
import random
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.realpath("."))
from utils.isaac_validator import ValidationType
from utils.hand_model_type import HandModelType
import multiprocessing

from functools import partial

from tap import Tap
from typing import Optional, List
import pathlib


class EvalAllGraspConfigDictsArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.GRAVITY_AND_TABLE
    gpu: int = 0
    max_grasps_per_batch: int = 5000
    debug_index: Optional[int] = None
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    use_gui: bool = False
    use_cpu: bool = False  # NOTE: Tyler has had big discrepancy between using GPU vs CPU, hypothesize that CPU is safer
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_v2")
    output_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )
    num_random_pose_noise_samples_per_grasp: Optional[int] = None
    move_fingers_back_at_init: bool = False
    randomize_order_seed: Optional[int] = None
    mid_optimization_steps: List[int] = []
    use_multiprocess: bool = True
    num_workers: int = 3


def get_object_code_and_scale_strs_to_process(
    input_grasp_config_dicts_path: pathlib.Path,
    output_evaled_grasp_config_dicts_path: pathlib.Path,
) -> List[str]:
    input_object_code_and_scale_strs = [
        path.stem for path in list(input_grasp_config_dicts_path.glob("*.npy"))
    ]

    print(
        f"Found {len(input_object_code_and_scale_strs)} object codes in input_grasp_config_dicts_path ({input_grasp_config_dicts_path})"
    )

    # Compare input and output directories
    existing_object_code_and_scale_strs = (
        [
            path.stem
            for path in list(output_evaled_grasp_config_dicts_path.glob("*.npy"))
        ]
        if output_evaled_grasp_config_dicts_path.exists()
        else []
    )
    print(
        f"Found {len(existing_object_code_and_scale_strs)} object codes in {output_evaled_grasp_config_dicts_path}"
    )

    # Sanity check that we are going into the right folder
    only_in_output = set(existing_object_code_and_scale_strs) - set(
        input_object_code_and_scale_strs
    )
    print(f"Num only in output: {len(only_in_output)}")
    assert len(only_in_output) == 0, f"Object codes only in output: {only_in_output}"

    # Don't redo old work
    only_in_input = set(input_object_code_and_scale_strs) - set(
        existing_object_code_and_scale_strs
    )
    print(f"Num codes only in input: {len(only_in_input)}")

    return list(only_in_input)


def print_and_run_command_safe(
    object_code_and_scale_str: str,
    args: EvalAllGraspConfigDictsArgumentParser,
    script_to_run: pathlib.Path,
    input_grasp_config_dicts_path: pathlib.Path,
    output_evaled_grasp_config_dicts_path: pathlib.Path,
):
    command = " ".join(
        [
            f"CUDA_VISIBLE_DEVICES={args.gpu}",
            f"python {script_to_run}",
            f"--hand_model_type {args.hand_model_type.name}",
            f"--validation_type {args.validation_type.name}",
            f"--gpu {args.gpu}",
            f"--meshdata_root_path {args.meshdata_root_path}",
            f"--input_grasp_config_dicts_path {input_grasp_config_dicts_path}",
            f"--output_evaled_grasp_config_dicts_path {output_evaled_grasp_config_dicts_path}",
            f"--object_code_and_scale_str {object_code_and_scale_str}",
            f"--max_grasps_per_batch {args.max_grasps_per_batch}",
            f"--num_random_pose_noise_samples_per_grasp {args.num_random_pose_noise_samples_per_grasp}" if args.num_random_pose_noise_samples_per_grasp is not None else "",
            "--move_fingers_back_at_init" if args.move_fingers_back_at_init else "",
        ]
    )

    if args.debug_index is not None:
        command += f" --debug_index {args.debug_index}"

    if args.use_gui:
        command += " --use_gui"

    if args.use_cpu:
        command += " --use_cpu"

    print(f"Running command: {command}")

    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_code_and_scale_str} and continuing")


def eval_all_grasp_config_dicts(
    args: EvalAllGraspConfigDictsArgumentParser,
    input_grasp_config_dicts_path: pathlib.Path,
    output_evaled_grasp_config_dicts_path: pathlib.Path,
) -> None:
    # Check if script exists
    script_to_run = pathlib.Path("scripts/eval_grasp_config_dict.py")
    assert script_to_run.exists(), f"Script {script_to_run} does not exist"

    input_object_code_and_scale_strs = get_object_code_and_scale_strs_to_process(
        input_grasp_config_dicts_path=input_grasp_config_dicts_path,
        output_evaled_grasp_config_dicts_path=output_evaled_grasp_config_dicts_path,
    )
    if args.randomize_order_seed is not None:
        random.Random(args.randomize_order_seed).shuffle(
            input_object_code_and_scale_strs
        )

    print(f"Processing {len(input_object_code_and_scale_strs)} object codes")
    print(f"First 10 object codes: {input_object_code_and_scale_strs[:10]}")

    map_fn = partial(
        print_and_run_command_safe,
        args=args,
        script_to_run=script_to_run,
        input_grasp_config_dicts_path=input_grasp_config_dicts_path,
        output_evaled_grasp_config_dicts_path=output_evaled_grasp_config_dicts_path,
    )

    if args.use_multiprocess:
        with multiprocessing.Pool(args.num_workers) as p:
            p.map(
                map_fn,
                input_object_code_and_scale_strs,
            )
    else:
        pbar = tqdm(input_object_code_and_scale_strs, dynamic_ncols=True)
        for object_code_and_scale_str in pbar:
            pbar.set_description(f"Processing {object_code_and_scale_str}")

            print_and_run_command_safe(
                object_code_and_scale_str=object_code_and_scale_str,
                args=args,
                script_to_run=script_to_run,
                input_grasp_config_dicts_path=input_grasp_config_dicts_path,
                output_evaled_grasp_config_dicts_path=output_evaled_grasp_config_dicts_path,
            )


def main(args: EvalAllGraspConfigDictsArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    eval_all_grasp_config_dicts(
        args=args,
        input_grasp_config_dicts_path=args.input_grasp_config_dicts_path,
        output_evaled_grasp_config_dicts_path=args.output_evaled_grasp_config_dicts_path,
    )

    for mid_optimization_step in args.mid_optimization_steps:
        print("!" * 80)
        print(f"Running mid_optimization_step: {mid_optimization_step}")
        print("!" * 80 + "\n")
        mid_optimization_input_grasp_config_dicts_path = (
            args.input_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        mid_optimization_output_grasp_config_dicts_path = (
            args.output_evaled_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        eval_all_grasp_config_dicts(
            args=args,
            input_grasp_config_dicts_path=mid_optimization_input_grasp_config_dicts_path,
            output_evaled_grasp_config_dicts_path=mid_optimization_output_grasp_config_dicts_path,
        )


if __name__ == "__main__":
    args = EvalAllGraspConfigDictsArgumentParser().parse_args()
    main(args)
