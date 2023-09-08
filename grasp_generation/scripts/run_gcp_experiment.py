import subprocess
from tap import Tap
import pathlib
import datetime
import socket
import pickle

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class ArgParser(Tap):
    experiment_name: str = DATETIME_STR


def main() -> None:
    args = ArgParser().parse_args()

    instance_name = socket.gethostname()

    # Get object_codes
    # TODO: Check that this path works
    experiment_dict_filename = f"{args.experiment_name}.pkl"
    with open(experiment_dict_filename, "rb") as handle:
        instance_name_to_object_codes_dict = pickle.load(handle)
    object_codes = instance_name_to_object_codes_dict[instance_name]
    print(f"Found {len(object_codes)} object_codes for {instance_name}")

    # Make new input_meshdata_path
    # TODO: Check that this path works
    original_input_meshdata_path = pathlib.Path("../data/meshdata")
    new_input_meshdata_path = (
        pathlib.Path("../data") / args.experiment_name / instance_name / "meshdata"
    )
    new_input_meshdata_path.mkdir(parents=True, exist_ok=True)
    for object_code in object_codes:
        subprocess.run(
            " ".join(
                [
                    "cp -r",
                    str(original_input_meshdata_path / object_code),
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
