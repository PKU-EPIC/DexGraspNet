import subprocess
import os
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', default="../data/graspdata_2023-05-22_distalonly", type=str)
    parser.add_argument('--result_path', default="../data/dataset_2023-05-22_distalonly", type=str)
    args = parser.parse_args()
    print(f"args = {args}")
 
    # Compare input and output directories
    input_object_code_files = os.listdir(args.input_data_path)
    print(f"Found {len(input_object_code_files)} object codes in {args.input_data_path}")
    existing_object_code_files = os.listdir(args.result_path)
    print(f"Found {len(existing_object_code_files)} object codes in {args.result_path}")

    # Sanity check that we are going into the right folder
    objects_only_in_output = set(existing_object_code_files) - set(input_object_code_files)
    print(f"Num objects only in output: {len(objects_only_in_output)}")
    assert len(objects_only_in_output) == 0, f"Objects only in output: {objects_only_in_output}"

    # Don't redo old work
    object_only_in_input = set(input_object_code_files) - set(existing_object_code_files)
    print(f"Num objects only in input: {len(object_only_in_input)}")
    object_only_in_input = list(object_only_in_input)
    print("Processing these only...")

    pbar = tqdm(object_only_in_input)
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
