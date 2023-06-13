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


class GenerateNerfDataArgumentParser(Tap):
    gpu: int = 0
    mesh_path: str = "../data/meshdata"
    output_nerf_path: str = "../data/nerfdata"


def main(args: GenerateNerfDataArgumentParser):
    # Check if script exists
    script_to_run = "scripts/generate_nerf_data_one_object.py"
    assert os.path.exists(script_to_run)

    object_codes = [
        object_code
        for object_code in os.listdir(args.mesh_path)
        if os.path.isdir(os.path.join(args.mesh_path, object_code))
    ]
    for i, object_code in tqdm(
        enumerate(object_codes),
        desc="Generating NeRF data for all objects",
        dynamic_ncols=True,
        total=len(object_codes),
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
