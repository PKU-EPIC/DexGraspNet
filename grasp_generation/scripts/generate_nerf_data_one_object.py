"""
Last modified date: 2023.06.13
Author: Tyler Lum
Description: Create NeRF Data in Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.object_model import ObjectModel
from tap import Tap
from tqdm import tqdm
import subprocess


class GenerateNerfDataOneObjectArgumentParser(Tap):
    gpu: int = 0
    mesh_path: str = "../data/meshdata"
    output_nerf_path: str = "../data/nerfdata"
    object_code: str = "sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2"


def main(args: GenerateNerfDataOneObjectArgumentParser):
    # Check if script exists
    script_to_run = "scripts/generate_nerf_data_one_object_one_scale.py"
    assert os.path.exists(script_to_run)

    object_model = ObjectModel(
        data_root_path=args.mesh_path,
        batch_size_each=1,
    )
    object_scales = object_model.scale_choice.tolist()

    for i, object_scale in tqdm(
        enumerate(object_scales),
        desc=f"Generating NeRF data for {args.object_code} at different scales",
        dynamic_ncols=True,
        total=len(object_scales),
    ):
        command = " ".join(
            [
                f"python {script_to_run}",
                f"--gpu {args.gpu}",
                f"--mesh_path {args.mesh_path}",
                f"--output_nerf_path {args.output_nerf_path}",
                f"--object_code {args.object_code}",
                f"--object_scale {object_scale}",
            ]
        )
        print(f"Running command {i}: {command}")
        subprocess.run(command, shell=True, check=True)
        print(f"Finished command {i}")


if __name__ == "__main__":
    args = GenerateNerfDataOneObjectArgumentParser().parse_args()
    main(args)
