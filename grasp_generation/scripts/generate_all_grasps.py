from tap import Tap
import os
import sys
import pathlib
from typing import List
from datetime import datetime

sys.path.append(os.path.realpath("."))
DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class ArgParser(Tap):
    input_meshdata_path: pathlib.Path = pathlib.Path("../data/meshdata")
    base_data_path: pathlib.Path = pathlib.Path("../data")
    experiment_name: str = DATETIME_STR
    use_multiprocess: bool = True
    generate_nerf_data: bool = False


def process_data(args: ArgParser):
    # Generate hand configs.
    hand_gen_command = (
        f"python scripts/generate_hand_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --output_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + " --rand_object_scale"
        + " --use_penetration_energy"
    )

    if args.use_multiprocess:
        hand_gen_command += " --use_multiprocess"

    print(f"Running: {hand_gen_command}")
    os.system(hand_gen_command)

    # Get resulting mid-opt steps
    mid_opt_path = (
        args.base_data_path
        / args.experiment_name
        / "hand_config_dicts"
        / "mid_optimization"
    )
    mid_opt_steps = [int(str(x.stem)) for x in mid_opt_path.iterdir()]

    # Generate grasp configs.
    grasp_gen_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
    )

    print(f"Running: {grasp_gen_command}")
    os.system(grasp_gen_command)

    # Eval final grasp configs.
    eval_final_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --output_evaled_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_evaled_grasp_config_dicts'}"
        + f" --meshdata_root_path {args.input_meshdata_path}"
    )

    print(f"Running: {eval_final_grasp_command}")
    os.system(eval_final_grasp_command)

    # Augment grasp configs.
    augment_grasp_command = (
        "python scripts/augment_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_evaled_grasp_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --augment_only_successes"
    )

    print(f"Running: {augment_grasp_command}")
    os.system(augment_grasp_command)

    # Generate grasps for "folded in" mid opt ones.
    grasp_gen_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --mid_opt_steps {','.join([str(x) for x in mid_opt_steps])}"
    )

    # Relabel open hand grasps.
    relabel_command = (
        "python scripts/generate_grasp_config_dicts.py"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts' / 'opened_hand'}"
        + f" --output_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts' / 'opened_hand'}"
    )

    print(f"Running: {relabel_command}")
    os.system(relabel_command)

    # Merge grasp configs.
    merge_grasp_command = (
        "python scripts/merge_config_dicts.py"
        + f" --input_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --output_config_dicts_path {args.base_data_path / args.experiment_name / 'grasp_config_dicts'}"
    )

    print(f"Running: {merge_grasp_command}")
    os.system(merge_grasp_command)

    # Eval grasp configs.
    eval_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'grasp_config_dicts'}"
        + f" --output_evaled_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'evaled_grasp_config_dicts'}"
        + f" --meshdata_root_path {args.input_meshdata_path}"
    )

    print(f"Running: {eval_grasp_command}")
    os.system(eval_grasp_command)

    # Generate NeRF data.
    if args.generate_nerf_data:
        nerf_data_command = (
            "python scripts/generate_nerf_data.py"
            + f" --meshdata_root_path {args.input_meshdata_path}"
            + f" --output_nerfdata_path {args.base_data_path / args.experiment_name / 'nerfdata'}"
            + f" --only_objects_in_this_path {args.base_data_path / args.experiment_name / 'evaled_grasp_config_dicts'}"
        )

        print(f"Running: {nerf_data_command}")
        os.system(nerf_data_command)

    print("Done!")


if __name__ == "__main__":
    args = ArgParser().parse_args()
    process_data(args)
