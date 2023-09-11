import subprocess
from tap import Tap
import pathlib
from typing import List, Dict
import math
import random
import pickle
from datetime import datetime

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR_PATH_ON_BUCKET = "experiments"
EXPERIMENT_DIR_PATH_LOCAL = pathlib.Path("../data/experiments")

ALL_MESHDATA_PATH_ON_BUCKET = "meshdata"


class ArgParser(Tap):
    gcp_instance_names: List[str]
    input_meshdata_path: pathlib.Path = pathlib.Path("../data/meshdata")
    experiment_name: str = DATETIME_STR
    dexgraspnet_root_path_on_instance: pathlib.Path = pathlib.Path("github_repos/dexgraspnet")
    seed: int = 42


def print_and_run(command: str) -> None:
    print(f"Running: {command}")
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def create_instance_name_to_object_codes_dict(
    input_meshdata_path: pathlib.Path, gcp_instance_names: List[str], seed: int
) -> Dict[str, List[str]]:
    assert input_meshdata_path.exists(), f"{input_meshdata_path} does not exist"
    assert (
        len(gcp_instance_names) > 0
    ), f"len(gcp_instance_names) = {len(gcp_instance_names)}"

    all_object_codes = [path.name for path in input_meshdata_path.iterdir()]
    n_object_codes = len(all_object_codes)
    print(f"Found {n_object_codes} object_codes")
    print(f"First 10: {all_object_codes[:10]}")

    n_instances = len(gcp_instance_names)
    n_object_codes_per_instance = math.ceil(n_object_codes / n_instances)

    random.Random(seed).shuffle(all_object_codes)
    instance_name_to_object_codes_dict = {}
    for instance_i, instance_name in enumerate(gcp_instance_names):
        start_idx = instance_i * n_object_codes_per_instance
        end_idx = min(start_idx + n_object_codes_per_instance, n_object_codes)

        instance_name_to_object_codes_dict[instance_name] = all_object_codes[
            start_idx:end_idx
        ]
    return instance_name_to_object_codes_dict


def main() -> None:
    args = ArgParser().parse_args()

    # Create instance_name_to_object_codes_dict for this experiment
    instance_name_to_object_codes_dict = create_instance_name_to_object_codes_dict(
        input_meshdata_path=args.input_meshdata_path,
        gcp_instance_names=args.gcp_instance_names,
        seed=args.seed,
    )
    EXPERIMENT_DIR_PATH_LOCAL.mkdir(parents=True, exist_ok=True)
    experiment_file = EXPERIMENT_DIR_PATH_LOCAL / f"{args.experiment_name}.pkl"
    with open(experiment_file, "wb") as handle:
        pickle.dump(
            instance_name_to_object_codes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    # Upload experiment_dir (instance_name_to_object_codes_dicts) to GCP
    # Both source and destination paths must be directories
    # Don't upload meshdata, should already be there and takes ~5min to check if synced
    # Also, don't sync with input_meshdata_path, this could be subset, ALL_MESHDATA_PATH_ON_BUCKET should have all meshes
    # If need to upload, run "gsutil -m rsync -r ../data/meshdata gs://learned-nerf-grasping/meshdata"
    print_and_run(f"gsutil -m rsync -r {str(EXPERIMENT_DIR_PATH_LOCAL)} gs://learned-nerf-grasping/{EXPERIMENT_DIR_PATH_ON_BUCKET}")

    # Run experiment on GCP
    for instance_name in args.gcp_instance_names:
        cd_command = f"cd {str(dexgraspnet_root_path_on_instance / 'grasp_generation')}"
        run_experiment_command = " ".join(
            [
                "python scripts/run_gcp_experiment.py",
                f"--experiment_name {args.experiment_name}",
            ]
        )
        print_and_run(f"gcloud compute ssh {instance_name} --command='{cd_command} && {run_experiment_command}'")


if __name__ == "__main__":
    main()
