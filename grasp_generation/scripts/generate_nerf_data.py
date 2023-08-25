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
from typing import Optional, Tuple
import pathlib
from utils.parse_object_code_and_str import parse_object_code_and_str


class GenerateNerfDataArgumentParser(Tap):
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    output_nerfdata_path: pathlib.Path = pathlib.Path("../data/nerfdata")
    randomize_order_seed: Optional[int] = None
    only_objects_in_this_path: Optional[pathlib.Path] = None


def get_object_codes_and_scales_to_process(
    args: GenerateNerfDataArgumentParser,
) -> Tuple[list, list]:
    # Get input object codes
    if args.only_objects_in_this_path is not None:
        input_object_codes, input_object_scales = [], []
        for path in args.only_objects_in_this_path.iterdir():
            object_code_and_scale_str = path.stem
            object_code, object_scale = parse_object_code_and_str(
                object_code_and_scale_str
            )
            input_object_codes.append(object_code)
            input_object_scales.append(object_scale)

        print(
            f"Found {len(input_object_codes)} object codes in args.only_objects_in_this_path ({args.only_objects_in_this_path})"
        )
    else:
        input_object_codes = [
            object_code for object_code in os.listdir(args.meshdata_root_path)
        ]
        HARDCODED_OBJECT_SCALE = 0.1
        input_object_scales = [HARDCODED_OBJECT_SCALE] * len(input_object_codes)
        print(
            f"Found {len(input_object_codes)} object codes in args.mesh_path ({args.meshdata_root_path})"
        )
        print(f"Using hardcoded scale {HARDCODED_OBJECT_SCALE} for all objects")

    return input_object_codes, input_object_scales


def main(args: GenerateNerfDataArgumentParser):
    # Check if script exists
    script_to_run = pathlib.Path("scripts/generate_nerf_data_one_object_one_scale.py")
    assert script_to_run.exists(), f"Script {script_to_run} does not exist"

    input_object_codes, input_object_scales = get_object_codes_and_scales_to_process(
        args
    )

    # Randomize order
    if args.randomize_order_seed is not None:
        import random

        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(input_object_codes)

    for i, (object_code, object_scale) in tqdm(
        enumerate(zip(input_object_codes, input_object_scales)),
        desc="Generating NeRF data for all objects",
        dynamic_ncols=True,
        total=len(input_object_codes),
    ):
        command = " ".join(
            [
                f"python {script_to_run}",
                f"--gpu {args.gpu}",
                f"--meshdata_root_path {args.meshdata_root_path}",
                f"--output_nerfdata_path {args.output_nerfdata_path}",
                f"--object_code {object_code}",
                f"--object_scale {object_scale}",
            ]
        )
        print(f"Running command {i}: {command}")
        subprocess.run(command, shell=True, check=True)
        print(f"Finished command {i}")


if __name__ == "__main__":
    args = GenerateNerfDataArgumentParser().parse_args()
    main(args)
