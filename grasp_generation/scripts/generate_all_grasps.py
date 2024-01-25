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
    num_random_pose_noise_samples_per_grasp: Optional[int] = None


def print_and_run(command: str) -> None:
    print("= " * 80)
    print(f"Running {command}")
    print("= " * 80 + "\n")
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
        + " --use_penetration_energy"
        # + " --rand_object_scale" # Turning off so we don't have to regen nerfs every time.
        # + " --object_scale 0.03" # For cube only to get 200cm => 6cm
        # + " --batch_size_each_object 1000 --n_objects_per_batch 5"  # For more grasps per object
        # + " --store_grasps_mid_optimization_freq 200"  # For more low-quality grasps
    )
    if args.use_multiprocess:
        hand_gen_command += " --use_multiprocess"
    print_and_run(hand_gen_command)

    if args.results_path is not None:
        print_and_run(sync_command)

    # Get resulting mid-opt steps
    hand_configs_mid_opt_path = (
        args.base_data_path
        / args.experiment_name
        / "hand_config_dicts"
        / "mid_optimization"
    )
    hand_configs_mid_opt_steps = [
        int(str(x.stem)) for x in hand_configs_mid_opt_path.iterdir()
    ]

    # Generate raw grasp configs.
    init_grasp_gen_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --mid_optimization_steps {' '.join([str(x) for x in hand_configs_mid_opt_steps])}"
    )
    print_and_run(init_grasp_gen_command)

    # Eval raw grasp configs.
    init_eval_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_grasp_config_dicts'}"
        + f" --output_evaled_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_evaled_grasp_config_dicts'}"
        + f" --meshdata_root_path {args.input_meshdata_path}"
        + (
            f" --num_random_pose_noise_samples_per_grasp {args.num_random_pose_noise_samples_per_grasp}"
            if args.num_random_pose_noise_samples_per_grasp is not None
            else ""
        )
        + f" --mid_optimization_steps {' '.join([str(x) for x in hand_configs_mid_opt_steps])}"
    )
    print_and_run(init_eval_grasp_command)

    if args.results_path is not None:
        print_and_run(sync_command)

    # Augment successful grasp configs.
    augment_grasp_command = (
        "python scripts/augment_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'raw_evaled_grasp_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_hand_config_dicts'}"
        + f" --augment_only_successes"
        + f" --mid_optimization_steps {' '.join([str(x) for x in hand_configs_mid_opt_steps])}"
    )
    print_and_run(augment_grasp_command)

    # Relabel open hand grasps.
    opened_update_augmented_grasp_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_hand_config_dicts_opened_hand'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_grasp_config_dicts_opened_hand'}"
        + f" --mid_optimization_steps {' '.join([str(x) for x in hand_configs_mid_opt_steps])}"
    )
    print_and_run(opened_update_augmented_grasp_command)

    # Relabel closed hand grasps.
    closed_update_augmented_grasp_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_hand_config_dicts_closed_hand'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_grasp_config_dicts_closed_hand'}"
        + f" --mid_optimization_steps {' '.join([str(x) for x in hand_configs_mid_opt_steps])}"
    )
    print_and_run(closed_update_augmented_grasp_command)

    # Eval grasp configs.
    opened_eval_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_grasp_config_dicts_opened_hand'}"
        + f" --output_evaled_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_evaled_grasp_config_dicts_opened_hand'}"
        + f" --meshdata_root_path {args.input_meshdata_path}"
        + (
            f" --num_random_pose_noise_samples_per_grasp {args.num_random_pose_noise_samples_per_grasp}"
            if args.num_random_pose_noise_samples_per_grasp is not None
            else ""
        )
        + f" --mid_optimization_steps {' '.join([str(x) for x in hand_configs_mid_opt_steps])}"
    )
    print_and_run(opened_eval_grasp_command)
    closed_eval_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_grasp_config_dicts_closed_hand'}"
        + f" --output_evaled_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'augmented_raw_evaled_grasp_config_dicts_closed_hand'}"
        + f" --meshdata_root_path {args.input_meshdata_path}"
        + (
            f" --num_random_pose_noise_samples_per_grasp {args.num_random_pose_noise_samples_per_grasp}"
            if args.num_random_pose_noise_samples_per_grasp is not None
            else ""
        )
        + f" --mid_optimization_steps {' '.join([str(x) for x in hand_configs_mid_opt_steps])}"
    )
    print_and_run(closed_eval_grasp_command)

    # Merge grasp configs.
    merge_grasp_command = (
        "python scripts/merge_config_dicts.py"
        + f" --input_config_dicts_paths {args.base_data_path / args.experiment_name / 'raw_evaled_grasp_config_dicts'} {args.base_data_path / args.experiment_name / 'augmented_raw_evaled_grasp_config_dicts_opened_hand'} {args.base_data_path / args.experiment_name / 'augmented_raw_evaled_grasp_config_dicts_closed_hand'}"
        + f" --output_config_dicts_path {args.base_data_path / args.experiment_name / 'evaled_grasp_config_dicts'}"
    )
    print_and_run(merge_grasp_command)

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
