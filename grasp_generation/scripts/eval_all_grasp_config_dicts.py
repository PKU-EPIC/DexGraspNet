from __future__ import annotations
import subprocess
import random
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.realpath("."))
from utils.isaac_validator import ValidationType
from utils.hand_model_type import HandModelType
from utils.joint_angle_targets import OptimizationMethod
from utils.parse_object_code_and_scale import parse_object_code_and_scale

from tap import Tap
from typing import Optional
import pathlib


class EvalAllGraspConfigDictsArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    max_grasps_per_batch: int = 500
    debug_index: Optional[int] = None
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    use_gui: bool = False
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    output_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )
    randomize_order_seed: Optional[int] = None


def get_object_code_and_scale_strs_to_process(
    args: EvalAllGraspConfigDictsArgumentParser,
) -> list:
    input_object_code_and_scale_strs = [
        path.stem for path in args.input_grasp_config_dicts_path.iterdir()
    ]

    print(
        f"Found {len(input_object_code_and_scale_strs)} object codes in args.input_grasp_config_dicts_path ({args.input_grasp_config_dicts_path})"
    )

    # Compare input and output directories
    existing_object_code_and_scale_strs = (
        [path.stem for path in args.output_evaled_grasp_config_dicts_path.iterdir()]
        if args.output_evaled_grasp_config_dicts_path.exists()
        else []
    )
    print(
        f"Found {len(existing_object_code_and_scale_strs)} object codes in {args.output_evaled_grasp_config_dicts_path}"
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


def main(args: EvalAllGraspConfigDictsArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    # Check if script exists
    script_to_run = pathlib.Path("scripts/eval_grasp_config_dict.py")
    assert script_to_run.exists(), f"Script {script_to_run} does not exist"

    input_object_code_and_scale_strs = get_object_code_and_scale_strs_to_process(args)
    if args.randomize_order_seed is not None:
        random.Random(args.randomize_order_seed).shuffle(
            input_object_code_and_scale_strs
        )
    print(f"Processing {len(input_object_code_and_scale_strs)} object codes")
    print(f"First 10 object codes: {input_object_code_and_scale_strs[:10]}")

    pbar = tqdm(input_object_code_and_scale_strs, dynamic_ncols=True)
    for object_code_and_scale_str in pbar:
        pbar.set_description(f"Processing {object_code_and_scale_str}")

        command = " ".join(
            [
                f"CUDA_VISIBLE_DEVICES={args.gpu}",
                f"python {script_to_run}",
                f"--hand_model_type {args.hand_model_type.name}",
                f"--validation_type {args.validation_type.name}",
                f"--gpu {args.gpu}",
                f"--meshdata_root_path {args.meshdata_root_path}",
                f"--input_grasp_config_dicts_path {args.input_grasp_config_dicts_path}",
                f"--output_evaled_grasp_config_dicts_path {args.output_evaled_grasp_config_dicts_path}",
                f"--object_code_and_scale_str {object_code_and_scale_str}",
                f"--max_grasps_per_batch {args.max_grasps_per_batch}",
            ]
        )

        if args.debug_index is not None:
            command += f" --debug_index {args.debug_index}"

        if args.use_gui:
            command += " --use_gui"

        print(f"Running command: {command}")

        try:
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Skipping {object_code_and_scale_str} and continuing")
            continue


if __name__ == "__main__":
    args = EvalAllGraspConfigDictsArgumentParser().parse_args()
    main(args)
