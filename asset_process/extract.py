import argparse
from utils.extract_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument('--set', type=str,
                        choices=["core", "sem", "mujoco", "ddg"], required=True)
    parser.add_argument('--meta', type=str)
    args = parser.parse_args()

    if(args.set == "core"):
        extract_core(args.src, args.dst)
    elif(args.set == "sem"):
        extract_sem(args.src, args.dst, args.meta)
    elif(args.set == "mujoco"):
        extract_mujoco(args.src, args.dst)
    elif(args.set == "ddg"):
        extract_ddg(args.src, args.dst)
