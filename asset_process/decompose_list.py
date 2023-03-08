import argparse
from utils.extract_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--coacd_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    with open("run.sh", "w") as f:
        for mesh in os.listdir(args.src):
            f.write(
                f"python decompose.py --data_root_path {args.src} --result_path {args.dst} --object_code {mesh[:-4]} --coacd_path {args.coacd_path}\n")
