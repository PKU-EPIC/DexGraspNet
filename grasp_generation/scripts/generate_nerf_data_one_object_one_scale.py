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


class GenerateNerfDataOneObjectOneScaleArgumentParser(Tap):
    gpu: int = 0
    mesh_path: str = "../data/meshdata"
    output_nerf_path: str = "../data/nerfdata"
    object_code: str = "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5"
    object_scale: float = 0.1


def main(args: GenerateNerfDataOneObjectOneScaleArgumentParser):
    # TODO: Trying to do multiple scales in one python script caused segfaults for some reason: 872597665ff528720b46b4f0c0a95a4513c38f7c
    # TODO: Currently assumes origin of urdf is ~same as center of mesh.bounds
    set_seed(42)
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    output_nerf_object_path = os.path.join(
        args.output_nerf_path,
        f"{args.object_code}_{args.object_scale:.2f}".replace(".", "_"),
    )
    if os.path.exists(output_nerf_object_path):
        print(f"Skipping {args.object_code} at scale {args.object_scale:.2f}")
        return

    # Create sim
    sim = IsaacValidator(
        gpu=args.gpu,
    )

    # For each scale, create NeRF dataset
    os.makedirs(args.output_nerf_path, exist_ok=True)
    sim.set_obj_asset(
        obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
        obj_file="coacd.urdf",
    )
    sim.add_env_nerf_data_collection(
        obj_scale=args.object_scale,
    )
    sim.save_images(folder=output_nerf_object_path)
    sim.create_train_val_test_split(
        folder=output_nerf_object_path, train_frac=0.8, val_frac=0.1
    )
    sim.reset_simulator()
    sim.destroy()


if __name__ == "__main__":
    args = GenerateNerfDataOneObjectOneScaleArgumentParser().parse_args()
    main(args)
