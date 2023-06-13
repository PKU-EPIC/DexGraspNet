"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: generate shell script for grasp validation
"""

from tap import Tap
import os


class ValidateGraspsBatchArgumentParser(Tap):
    val_batch: int = 500
    mesh_path: str = "../data/meshdata"
    src: str = "../data/graspdata"
    dst: str = "../data/dataset"


if __name__ == "__main__":
    args = ValidateGraspsBatchArgumentParser().parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        print("Please set CUDA_VISIBLE_DEVICES")
        exit()

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"gpu_list: {gpu_list}")

    gpu_cnt = 0

    with open("run.sh", "w") as f:
        for code in os.listdir(args.src):
            f.write(
                f"python scripts/validate_grasps.py --gpu {gpu_list[gpu_cnt%len(gpu_list)]} --val_batch {args.val_batch} --mesh_path {args.mesh_path} --grasp_path {args.src} --result_path {args.dst} --object_code {code[:-4]}\n"
            )
            gpu_cnt += 1
