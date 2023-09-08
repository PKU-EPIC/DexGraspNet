import subprocess
from tap import Tap
import pathlib
from typing import List, Dict
import math
import random
import pickle
import datetime

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class ArgParser(Tap):
    gcp_instance_names: List[str]
    input_meshdata_path: pathlib.Path = pathlib.Path("../data/meshdata")
    experiment_name: str = DATETIME_STR
    seed: int = 42


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
    experiment_dict_filename = f"{args.experiment_name}.pkl"
    with open(experiment_dict_filename, "wb") as handle:
        pickle.dump(
            instance_name_to_object_codes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    # Upload meshdata and instance_name_to_object_codes_dict to GCP
    # TODO: Check that these paths work
    subprocess.run(
        f"gsutil -m rsync -r {args.input_meshdata_path} gs://learned-nerf-grasping",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"gsutil -m rsync -r {experiment_dict_filename} gs://learned-nerf-grasping",
        shell=True,
        check=True,
    )

    # Run experiment on GCP
    for instance_name in args.gcp_instance_names:
        cd_command = "cd DexGraspNet/grasp_generation"
        run_experiment_command = " ".join(
            [
                "python scripts/run_gcp_experiment.py",
                f"--experiment_name {args.experiment_name}",
            ]
        )
        subprocess.run(
            f"gcloud compute ssh {instance_name} --command='{cd_command} && {run_experiment_command}'",
            shell=True,
            check=True,
        )


if __name__ == "__main__":
    main()
