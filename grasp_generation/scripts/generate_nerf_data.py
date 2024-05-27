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
from typing import Optional, Tuple, List
import pathlib
from utils.parse_object_code_and_scale import parse_object_code_and_scale
import multiprocessing


class GenerateNerfDataArgumentParser(Tap):
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_v2")
    output_nerfdata_path: pathlib.Path = pathlib.Path("../data/nerfdata")
    num_cameras: int = 250
    randomize_order_seed: Optional[int] = None
    only_objects_in_this_path: Optional[pathlib.Path] = None
    use_multiprocess: bool = True
    num_workers: int = 4
    no_continue: bool = False


def get_object_code_and_scale_strs_from_folder(
    folder_path: pathlib.Path,
) -> List[str]:
    if not folder_path.exists():
        return []

    object_code_and_scale_strs = []
    for file_path in folder_path.iterdir():
        object_code_and_scale_str = file_path.stem
        try:
            parse_object_code_and_scale(object_code_and_scale_str)
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Skipping {object_code_and_scale_str} and continuing")
            continue
        object_code_and_scale_strs.append(object_code_and_scale_str)
    return object_code_and_scale_strs


def get_object_codes_and_scales_to_process(
    args: GenerateNerfDataArgumentParser,
) -> Tuple[List[str], List[float]]:
    # Get input object codes
    if args.only_objects_in_this_path is None:
        input_object_codes = [
            object_code for object_code in os.listdir(args.meshdata_root_path)
        ]
        HARDCODED_OBJECT_SCALE = 0.06
        input_object_scales = [HARDCODED_OBJECT_SCALE] * len(input_object_codes)
        print(
            f"Found {len(input_object_codes)} object codes in args.mesh_path ({args.meshdata_root_path})"
        )
        print(f"Using hardcoded scale {HARDCODED_OBJECT_SCALE} for all objects")
        return input_object_codes, input_object_scales

    input_object_code_and_scale_strs = get_object_code_and_scale_strs_from_folder(
        args.only_objects_in_this_path
    )
    print(
        f"Found {len(input_object_code_and_scale_strs)} object codes in args.only_objects_in_this_path ({args.only_objects_in_this_path})"
    )

    existing_object_code_and_scale_strs = get_object_code_and_scale_strs_from_folder(
        args.output_nerfdata_path
    )

    if args.no_continue and len(existing_object_code_and_scale_strs) > 0:
        print(
            f"Found {len(existing_object_code_and_scale_strs)} existing object codes in args.output_nerfdata_path ({args.output_nerfdata_path})."
        )
        print("Exiting because --no_continue was specified.")
        exit()
    elif len(existing_object_code_and_scale_strs) > 0:
        print(
            f"Found {len(existing_object_code_and_scale_strs)} existing object codes in args.output_nerfdata_path ({args.output_nerfdata_path})."
        )
        print("Continuing because --no_continue was not specified.")

        input_object_code_and_scale_strs = list(
            set(input_object_code_and_scale_strs)
            - set(existing_object_code_and_scale_strs)
        )
        print(
            f"Continuing with {len(input_object_code_and_scale_strs)} object codes after filtering."
        )

    input_object_codes, input_object_scales = [], []
    for object_code_and_scale_str in input_object_code_and_scale_strs:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )
        input_object_codes.append(object_code)
        input_object_scales.append(object_scale)

    return input_object_codes, input_object_scales


def run_command(
    object_code: str,
    object_scale: float,
    args: GenerateNerfDataArgumentParser,
    script_to_run: pathlib.Path,
):
    command = " ".join(
        [
            f"python {str(script_to_run)}",
            f"--gpu {args.gpu}",
            f"--meshdata_root_path {args.meshdata_root_path}",
            f"--output_nerfdata_path {args.output_nerfdata_path}",
            f"--object_code {object_code}",
            f"--object_scale {object_scale}",
            f"--num_cameras {args.num_cameras}",
        ]
    )
    print(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_code} and continuing")
    print(f"Finished object {object_code}.")


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
        random.Random(args.randomize_order_seed).shuffle(input_object_scales)

    if args.use_multiprocess:
        print(f"Using multiprocessing with {args.num_workers} workers.")
        with multiprocessing.Pool(args.num_workers) as p:
            p.starmap(
                run_command,
                zip(
                    input_object_codes,
                    input_object_scales,
                    [args] * len(input_object_codes),
                    [script_to_run] * len(input_object_codes),
                ),
            )
    else:
        for i, (object_code, object_scale) in tqdm(
            enumerate(zip(input_object_codes, input_object_scales)),
            desc="Generating NeRF data for all objects",
            dynamic_ncols=True,
            total=len(input_object_codes),
        ):
            run_command(
                object_code=object_code,
                object_scale=object_scale,
                args=args,
                script_to_run=script_to_run,
            )


if __name__ == "__main__":
    args = GenerateNerfDataArgumentParser().parse_args()
    main(args)
