"""
Last modified date: 2023.06.13
Author: Tyler Lum
Description: Create NeRF Data in Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from tap import Tap
from tqdm import tqdm
import subprocess
from typing import Optional


class GenerateNerfDataArgumentParser(Tap):
    gpu: int = 0
    mesh_path: str = "../data/meshdata"
    output_nerf_path: str = "../data/nerfdata"
    randomize_order_seed: Optional[int] = None
    only_objects_in_this_graspdata_path: Optional[str] = None


def get_object_codes_to_process(args: GenerateNerfDataArgumentParser):
    # Get input object codes
    if args.only_objects_in_this_graspdata_path is not None:
        input_object_codes = [
            os.path.splitext(object_code_dot_npy)[0]
            for object_code_dot_npy in os.listdir(
                args.only_objects_in_this_graspdata_path
            )
        ]
        print(
            f"Found {len(input_object_codes)} object codes in args.only_objects_in_this_graspdata_path ({args.only_objects_in_this_graspdata_path})"
        )
    else:
        input_object_codes = [object_code for object_code in os.listdir(args.mesh_path)]
        print(
            f"Found {len(input_object_codes)} object codes in args.mesh_path ({args.mesh_path})"
        )

    # Check for existing object codes
    existing_object_code_files = (
        [object_code for object_code in os.listdir(args.output_nerf_path)]
        if os.path.exists(args.output_nerf_path)
        else []
    )
    print(f"Found {len(existing_object_code_files)} object codes in {args.output_nerf_path}")

    # Sanity check that we are going into the right folder
    object_codes_only_in_output = set(existing_object_code_files) - set(input_object_codes)
    print(f"Num object codes only in output: {len(object_codes_only_in_output)}")
    assert (
        len(object_codes_only_in_output) == 0
    ), f"Object codes only in output: {object_codes_only_in_output}"

    # Don't redo old work
    object_codes_only_in_input = set(input_object_codes) - set(existing_object_code_files)
    print(f"Num objects only in input: {len(object_codes_only_in_input)}")
    object_codes_only_in_input = list(object_codes_only_in_input)
    print("Processing these only...")
    return object_codes_only_in_input


def main(args: GenerateNerfDataArgumentParser):
    # Check if script exists
    script_to_run = "scripts/generate_nerf_data_one_object.py"
    assert os.path.exists(script_to_run)

    input_object_codes = get_object_codes_to_process(args)

    # Randomize order
    if args.randomize_order_seed is not None:
        import random

        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(input_object_codes)

    for i, object_code in tqdm(
        enumerate(input_object_codes),
        desc="Generating NeRF data for all objects",
        dynamic_ncols=True,
        total=len(input_object_codes),
    ):
        command = " ".join(
            [
                f"python {script_to_run}",
                f"--gpu {args.gpu}",
                f"--mesh_path {args.mesh_path}",
                f"--output_nerf_path {args.output_nerf_path}",
                f"--object_code {object_code}",
            ]
        )
        print(f"Running command {i}: {command}")
        subprocess.run(command, shell=True, check=True)
        print(f"Finished command {i}")


if __name__ == "__main__":
    args = GenerateNerfDataArgumentParser().parse_args()
    main(args)
