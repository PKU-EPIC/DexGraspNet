import subprocess
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


class EvalAllGraspsArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    randomize_order_seed: Optional[int] = None
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    output_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )


def get_object_code_and_scale_strs_to_process(
    args: EvalAllGraspsArgumentParser,
) -> list:
    input_object_code_and_scale_strs = [
        path.stem for path in args.input_grasp_config_dicts_path.iterdir()
    ]

    print(
        f"Found {len(input_object_code_and_scale_strs)} object codes in args.input_grasp_config_dicts_path ({args.input_grasp_config_dicts_path})"
    )

    # Compare input and output directories
    existing_object_code_and_scale_strs = [
        path.stem for path in args.output_evaled_grasp_config_dicts_path.iterdir()
    ]
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


def main(args: EvalAllGraspsArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    # Check if script exists
    script_to_run = pathlib.Path("scripts/eval_grasp_config_dicts.py")
    assert script_to_run.exists(), f"Script {script_to_run} does not exist"

    input_object_code_and_scale_strs = get_object_code_and_scale_strs_to_process(args)

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
                f"--input_grasp_config_dicts_path {args.input_grasp_config_dicts_path}",
                f"--output_evaled_grasp_config_dicts_path {args.output_evaled_grasp_config_dicts_path}",
                f"--object_code_and_scale_str {object_code_and_scale_str}",
            ]
        )
        print(f"Running command: {command}")

        try:
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Skipping {object_code} and continuing")
            continue


if __name__ == "__main__":
    args = EvalAllGraspsArgumentParser().parse_args()
    main(args)
