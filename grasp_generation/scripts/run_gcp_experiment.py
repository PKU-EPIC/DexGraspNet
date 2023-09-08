import subprocess
from tap import Tap
import pathlib
import datetime
import socket
import pickle

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

EXPERIMENT_DIR_PATH_ON_BUCKET = "experiments"
EXPERIMENT_DIR_PATH_LOCAL = pathlib.Path("../data/experiments")

ALL_MESHDATA_PATH_ON_BUCKET = "all_meshdata"
ALL_MESHDATA_PATH_LOCAL = pathlib.Path("../data/all_meshdata")

class ArgParser(Tap):
    experiment_name: str = DATETIME_STR


def main() -> None:
    args = ArgParser().parse_args()

    instance_name = socket.gethostname()

    # Download experiment files and all_meshdata if needed (will do nothing if up to date)
    # Both source and destination paths must be directories
    subprocess.run(
        f"gsutil -m rsync -r gs://learned-nerf-grasping/{EXPERIMENT_DIR_PATH_ON_BUCKET} {str(EXPERIMENT_DIR_PATH_LOCAL)}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"gsutil -m rsync -r gs://learned-nerf-grasping/{ALL_MESHDATA_PATH_ON_BUCKET} {str(ALL_MESHDATA_PATH_LOCAL)}",
        shell=True,
        check=True,
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
    new_input_meshdata_path.mkdir(parents=True, exist_ok=True)
    for object_code in object_codes:
        subprocess.run(
            " ".join(
                [
                    "cp -r",
                    str(ALL_MESHDATA_PATH_LOCAL / object_code),
                    str(new_input_meshdata_path / object_code),
                ]
            ),
            shell=True,
            check=True,
        )

    # Run
    subprocess.run(
        " ".join(
            [
                "python scripts/generate_all_grasps.py",
                f"--input_meshdata_path {new_input_meshdata_path}",
                f"--experiment_name {args.experiment_name}",
            ]
        ),
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    main()
