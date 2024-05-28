"""
Last modified date: 2023.06.13
Author: Tyler Lum
Description: Create NeRF Data in Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.isaac_validator import IsaacValidator, ValidationType
from utils.seed import set_seed
from utils.parse_object_code_and_scale import object_code_and_scale_to_str
from tap import Tap
import pathlib


class GenerateNerfDataOneObjectOneScaleArgumentParser(Tap):
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_v2")
    output_nerfdata_path: pathlib.Path = pathlib.Path("../data/nerfdata")
    object_code: str = "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5"
    object_scale: float = 0.1
    generate_seg: bool = False
    generate_depth: bool = False
    num_cameras: int = 250
    debug_with_gui: bool = False


def main(args: GenerateNerfDataOneObjectOneScaleArgumentParser):
    set_seed(42)
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    object_code_and_scale_str = object_code_and_scale_to_str(
        args.object_code, args.object_scale
    )
    output_nerf_object_path = args.output_nerfdata_path / object_code_and_scale_str
    if output_nerf_object_path.exists():
        print(f"{output_nerf_object_path} exists, skipping {object_code_and_scale_str}")
        return
    from utils.timers import LoopTimer

    loop_timer = LoopTimer()

    # Create sim
    with loop_timer.add_section_timer("create sim"):
        sim = IsaacValidator(
            gpu=args.gpu,
            # validation_type=ValidationType.NO_GRAVITY_SHAKING,  # Floating object, no table
            validation_type=ValidationType.GRAVITY_AND_TABLE,  # Object on table
            mode="gui" if args.debug_with_gui else "headless",
        )

    with loop_timer.add_section_timer("set obj asset"):
        args.output_nerfdata_path.mkdir(parents=True, exist_ok=True)
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / args.object_code / "coacd"),
            obj_file="coacd.urdf",
            vhacd_enabled=False,  # Disable vhacd because it should be faster and not needed
        )

    with loop_timer.add_section_timer("add env"):
        sim.add_env_nerf_data_collection(
            obj_scale=args.object_scale,
        )

    with loop_timer.add_section_timer("run_sim_till_object_settles_upright"):
        is_valid, log_text = sim.run_sim_till_object_settles_upright()
        if not is_valid:
            log_failures_path = pathlib.Path(
                str(args.output_nerfdata_path) + "_failures.txt"
            )
            print(
                f"Skipping {object_code_and_scale_str} because {log_text}, writing to {log_failures_path}"
            )
            with open(log_failures_path, "a") as f:
                f.write(f"{object_code_and_scale_str}: {log_text}\n")

    with loop_timer.add_section_timer("save images light"):
        sim.save_images_lightweight(
            folder=str(output_nerf_object_path),
            generate_seg=args.generate_seg,
            generate_depth=args.generate_depth,
            num_cameras=args.num_cameras,
        )
    with loop_timer.add_section_timer("create no split data"):
        sim.create_no_split_data(folder=str(output_nerf_object_path))
    with loop_timer.add_section_timer("destroy"):
        sim.reset_simulator()
        sim.destroy()

    loop_timer.pretty_print_section_times()


if __name__ == "__main__":
    args = GenerateNerfDataOneObjectOneScaleArgumentParser().parse_args()
    main(args)
