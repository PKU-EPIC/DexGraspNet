"""
Last modified date: 2023.06.13
Author: Tyler Lum
Description: Create NeRF Data in Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.isaac_validator import IsaacValidator
from utils.object_model import ObjectModel
from utils.seed import set_seed
from tap import Tap
from tqdm import tqdm
import pathlib


class GenerateNerfDataOneObjectOneScaleArgumentParser(Tap):
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    output_nerfdata_path: pathlib.Path = pathlib.Path("../data/nerfdata")
    object_code: str = "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5"
    object_scale: float = 0.1


def main(args: GenerateNerfDataOneObjectOneScaleArgumentParser):
    set_seed(42)
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    output_nerf_object_path = (
        args.output_nerfdata_path
        / f"{args.object_code}_{args.object_scale:.2f}".replace(".", "_")
    )
    if output_nerf_object_path.exists():
        print(f"Skipping {args.object_code} at scale {args.object_scale:.2f}")
        return

    # Create sim
    sim = IsaacValidator(
        gpu=args.gpu,
    )

    # For each scale, create NeRF dataset
    args.output_nerfdata_path.mkdir(parents=True, exist_ok=True)
    sim.set_obj_asset(
        obj_root=str(args.meshdata_root_path / args.object_code / "coacd"),
        obj_file="coacd.urdf",
    )
    sim.add_env_nerf_data_collection(
        obj_scale=args.object_scale,
    )
    sim.save_images(folder=str(output_nerf_object_path))
    sim.create_train_val_test_split(
        folder=str(output_nerf_object_path), train_frac=0.8, val_frac=0.1
    )
    sim.reset_simulator()
    sim.destroy()


if __name__ == "__main__":
    args = GenerateNerfDataOneObjectOneScaleArgumentParser().parse_args()
    main(args)
