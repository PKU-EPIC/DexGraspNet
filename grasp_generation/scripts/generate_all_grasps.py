from tap import Tap
import os
import subprocess
import sys
import pathlib
from typing import Optional
from datetime import datetime

sys.path.append(os.path.realpath("."))
DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class ArgParser(Tap):
    input_meshdata_path: pathlib.Path = pathlib.Path("../data/meshdata")
    base_data_path: pathlib.Path = pathlib.Path("../data")
    experiment_name: str = DATETIME_STR
    use_multiprocess: bool = True
    generate_nerf_data: bool = False
    results_path: Optional[pathlib.Path] = None
    gcloud_results_path: Optional[pathlib.Path] = None


def print_and_run(command: str) -> None:
    print(f"Running {command}")
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def process_data(args: ArgParser):
    if args.results_path is not None:
        assert args.gcloud_results_path is not None

        # Generate sync command.
        sync_command = (
            f"gsutil -m rsync -r {args.results_path} gs:\/\/{args.gcloud_results_path}"
        )

    # Generate hand configs.
    hand_gen_command = (
        f"python scripts/generate_hand_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --output_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + " --rand_object_scale" # Turning off so we don't have to regen nerfs every time.
        + " --use_penetration_energy"
    )

    if args.use_multiprocess:
        hand_gen_command += " --use_multiprocess"

    print_and_run(hand_gen_command)

    if args.results_path is not None:
        print_and_run(sync_command)

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

    # Eval final grasp configs.
    eval_final_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --output_evaled_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_evaled_grasp_config_dicts'}"
        + f" --meshdata_root_path {args.input_meshdata_path}"
    )

    print_and_run(eval_final_grasp_command)
    if args.results_path is not None:
        print_and_run(sync_command)

    # Augment grasp configs.
    augment_grasp_command = (
        "python scripts/augment_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_evaled_grasp_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --augment_only_successes"
    )

    print_and_run(augment_grasp_command)

    # Generate grasps for "folded in" mid opt ones.
    grasp_gen_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --mid_optimization_steps {' '.join([str(x) for x in mid_opt_steps])}"
    )

    print_and_run(grasp_gen_command)

    # Relabel open hand grasps.
    relabel_command = (
        "python scripts/generate_grasp_config_dicts.py"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts' / 'opened_hand'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts' / 'opened_hand'}"
    )

    print_and_run(relabel_command)

    # Merge grasp configs.
    merge_grasp_command = (
        "python scripts/merge_config_dicts.py"
        + f" --input_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --output_config_dicts_path {args.base_data_path / args.experiment_name / 'grasp_config_dicts'}"
    )

    print_and_run(merge_grasp_command)

    # Eval grasp configs.
    eval_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'grasp_config_dicts'}"
        + f" --output_evaled_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'evaled_grasp_config_dicts'}"
        + f" --meshdata_root_path {args.input_meshdata_path}"
    )

    print_and_run(eval_grasp_command)
    if args.results_path is not None:
        print_and_run(sync_command)

    # Generate NeRF data.
    if args.generate_nerf_data:
        nerf_data_command = (
            "python scripts/generate_nerf_data.py"
            + f" --meshdata_root_path {args.input_meshdata_path}"
            + f" --output_nerfdata_path {args.base_data_path / args.experiment_name / 'nerfdata'}"
            + f" --only_objects_in_this_path {args.base_data_path / args.experiment_name / 'evaled_grasp_config_dicts'}"
        )

        print_and_run(nerf_data_command)
        if args.results_path is not None:
            print_and_run(sync_command)

    print("Done!")


if __name__ == "__main__":
    args = ArgParser().parse_args()
    process_data(args)
