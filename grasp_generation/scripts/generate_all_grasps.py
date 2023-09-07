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
    scales: List[float] = [0.075, 0.1, 0.125, 0.15]
    experiment_name: str = DATETIME_STR
    use_multiprocess: bool = True


def process_data(args: ArgParser):
    for scale in args.scales:
        # Generate hand configs.
        hand_gen_command = (
            f"python scripts/generate_hand_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
            + f" --object_scale {scale}"
            + f" --output_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        )

        if args.use_multiprocess:
            hand_gen_command += " --use_multiprocess"

        print(f"Running: {hand_gen_command}")
        os.system(hand_gen_command)


if __name__ == "__main__":
    args = ArgParser().parse_args()
    process_data(args)
