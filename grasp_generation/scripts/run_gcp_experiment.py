import subprocess
from tap import Tap
import pathlib
from datetime import datetime
import socket
import pickle

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

EXPERIMENT_DIR_PATH_ON_BUCKET = "experiments"
EXPERIMENT_DIR_PATH_LOCAL = pathlib.Path("../data/experiments")

ALL_MESHDATA_PATH_ON_BUCKET = "meshdata"
ALL_MESHDATA_PATH_LOCAL = pathlib.Path("../data/meshdata")

DEXGRASPNET_GIVEN_DATA_PATH_ON_BUCKET = "dexgraspnet_given_data"
DEXGRASPNET_GIVEN_DATA_PATH_LOCAL = pathlib.Path("../data/dexgraspnet_given_data")


class ArgParser(Tap):
    experiment_name: str = DATETIME_STR
    no_continue: bool = False


def print_and_run(command: str) -> None:
    print(f"Running {command}")
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def main() -> None:
    args = ArgParser().parse_args()

    instance_name = socket.gethostname()
    # Download experiment files (instance_name_to_object_codes_dicts) from GCP
    # Both source and destination paths must be directories
    # Don't run command to download meshdata unless needed, takes ~5min to check if synced
    if not ALL_MESHDATA_PATH_LOCAL.exists():
        print(
            f"ALL_MESHDATA_PATH_LOCAL = {ALL_MESHDATA_PATH_LOCAL} does not exist, downloading..."
        )
        FASTER_DOWNLOAD_COMPRESSED = True
        if FASTER_DOWNLOAD_COMPRESSED:
            DEXGRASPNET_GIVEN_DATA_PATH_LOCAL.mkdir(
                parents=True, exist_ok=True
            )  # Must make dir before populating the dir with rsync
            print_and_run(
                f"gsutil -m rsync -r gs://learned-nerf-grasping/{DEXGRASPNET_GIVEN_DATA_PATH_ON_BUCKET} {str(DEXGRASPNET_GIVEN_DATA_PATH_LOCAL)}",
            )
            meshdata_tar_gz_path = DEXGRASPNET_GIVEN_DATA_PATH_LOCAL / "meshdata.tar.gz"
            assert (
                meshdata_tar_gz_path.exists()
            ), f"Strange, {meshdata_tar_gz_path} missing"
            print_and_run(
                f"tar -xf str(meshdata_tar_gz_path) --directory {str(ALL_MESHDATA_PATH_LOCAL.parent)}",
            )
            assert (
                ALL_MESHDATA_PATH_LOCAL.exists()
            ), f"Strange, failed to create {ALL_MESHDATA_PATH_LOCAL}"
        else:
            ALL_MESHDATA_PATH_LOCAL.mkdir(
                parents=True, exist_ok=True
            )  # Must make dir before populating the dir with rsync
            print_and_run(
                f"gsutil -m rsync -r gs://learned-nerf-grasping/{ALL_MESHDATA_PATH_ON_BUCKET} {str(ALL_MESHDATA_PATH_LOCAL)}",
            )
    EXPERIMENT_DIR_PATH_LOCAL.mkdir(
        parents=True, exist_ok=True
    )  # Must make dir before populating the dir with rsync
    print_and_run(
        f"gsutil -m rsync -r gs://learned-nerf-grasping/{EXPERIMENT_DIR_PATH_ON_BUCKET} {str(EXPERIMENT_DIR_PATH_LOCAL)}"
    )

    # Get object_codes
    experiment_file = EXPERIMENT_DIR_PATH_LOCAL / f"{args.experiment_name}.pkl"
    with open(experiment_file, "rb") as handle:
        instance_name_to_object_codes_dict = pickle.load(handle)
    object_codes = instance_name_to_object_codes_dict[instance_name]
    print(f"Found {len(object_codes)} object_codes for {instance_name}")

    # Make new input_meshdata_path
    new_input_meshdata_path = (
        pathlib.Path("../data") / args.experiment_name / instance_name / "meshdata"
    )
    if new_input_meshdata_path.exists() and args.no_continue:
        raise ValueError(
            f"Found {new_input_meshdata_path}. Either delete it or run without --no_continue"
        )
    elif not new_input_meshdata_path.exists():
        new_input_meshdata_path.mkdir(parents=True, exist_ok=True)
        CREATE_SYMLINKS = True
        for object_code in object_codes:
            if CREATE_SYMLINKS:
                print_and_run(
                    " ".join(
                        [
                            "ln -s",
                            str((ALL_MESHDATA_PATH_LOCAL / object_code).resolve()),
                            str((new_input_meshdata_path / object_code).resolve()),
                        ]
                    ),
                )
            else:
                print_and_run(
                    " ".join(
                        [
                            "cp -r",
                            str(ALL_MESHDATA_PATH_LOCAL / object_code),
                            str(new_input_meshdata_path / object_code),
                        ]
                    ),
                )
        print(f"Done copying meshdata")
    else:  # Check all correct objects are in new_input_meshdata_path
        print(f"Found {new_input_meshdata_path}. Continuing experiment.")
        existing_object_codes = [
            object_code.name
            for object_code in new_input_meshdata_path.iterdir()
            if object_code.is_dir()
        ]
        if set(existing_object_codes) != set(object_codes):
            print(
                f"existing object codes not in current list: {set(existing_object_codes) - set(object_codes)}"
            )
            print(
                f"object codes in list not currently existing: {set(object_codes)-set(existing_object_codes) }"
            )
            print("Strange, existing_object_codes != object_codes; continuing...")

    results_path = pathlib.Path("../data") / args.experiment_name
    results_path.mkdir(parents=True, exist_ok=True)

    # Run
    print_and_run(
        " ".join(
            [
                "python scripts/generate_all_grasps.py",
                f"--input_meshdata_path {new_input_meshdata_path}",
                # f"--base_data_path {}",
                f"--experiment_name {args.experiment_name}",
                f"--results_path {results_path}",
                f"--gcloud_results_path learned-nerf-grasping/{args.experiment_name}",
            ]
        ),
    )
    print(f"Done generating grasps.")

    # Upload results back to bucket.
    print_and_run(
        f"gsutil -m rsync -r {results_path} gs://learned-nerf-grasping/{args.experiment_name}"
    )

    print("Done!")


if __name__ == "__main__":
    main()
