import subprocess
import os
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', default="../data/graspdata", type=str)
    parser.add_argument('--result_path', default="../data/dataset_2023-05-22_distalonly", type=str)
    args = parser.parse_args()
    print(f"args = {args}")
 
    object_code_files = os.listdir(args.input_data_path)
    pbar = tqdm(object_code_files)
    for object_code_file in pbar:
        object_code = object_code_file.split(".")[0]
        pbar.set_description(f"Processing {object_code}")
        command = " ".join([
            "CUDA_VISIBLE_DEVICES=0",
            "python scripts/validate_grasps.py",
            f"--grasp_path {args.input_data_path}",
            "--gpu 0",
            f"--result_path {args.result_path}",
            f"--object_code {object_code}"
        ])
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
