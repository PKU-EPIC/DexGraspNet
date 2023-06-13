"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.isaac_validator import IsaacValidator
from utils.object_model import ObjectModel
from utils.seed import set_seed
from tap import Tap
from tqdm import tqdm


class GenerateNerfDataArgumentParser(Tap):
    gpu: int = 0
    mesh_path: str = "../data/meshdata"
    output_nerf_path: str = "../data/nerfdata"
    object_code: str = "sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2"


def main(args: GenerateNerfDataArgumentParser):
    # TODO: Currently assumes origin of urdf is ~same as center of mesh.bounds
    set_seed(42)
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    # Create sim
    sim = IsaacValidator(
        gpu=args.gpu,
    )
    object_model = ObjectModel(
        data_root_path=args.mesh_path,
        batch_size_each=1,
    )
    obj_scales = object_model.scale_choice.tolist()

    # For each scale, create NeRF dataset
    os.makedirs(args.output_nerf_path, exist_ok=True)
    for obj_scale in tqdm(
        obj_scales,
        desc=f"Generating NeRF data for {args.object_code} at different scales",
        dynamic_ncols=True,
    ):
        sim.set_obj_asset(
            obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
            obj_file="coacd.urdf",
        )
        sim.add_env_nerf_data_collection(
            obj_scale=obj_scale,
        )

        output_nerf_object_path = os.path.join(
            args.output_nerf_path, f"{args.object_code}_{obj_scale:.2f}".replace(".", "_")
        )
        sim.save_images(folder=output_nerf_object_path)
        # sim.create_train_val_test_split(
        #     folder=output_nerf_object_path, train_frac=0.8, val_frac=0.1
        # )
        # sim.reset_simulator()
    sim.destroy()


if __name__ == "__main__":
    args = GenerateNerfDataArgumentParser().parse_args()
    main(args)
