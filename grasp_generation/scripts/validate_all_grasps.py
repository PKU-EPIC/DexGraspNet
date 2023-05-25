import subprocess
import os
from tqdm import tqdm
import argparse
from utils.hand_model_type import HandModelType

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand_model_type",
        default=HandModelType.SHADOW_HAND,
        type=HandModelType.from_string,
        choices=list(HandModelType),
    )
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument('--grasp_path', default="../data/graspdata_2023-05-22_distalonly", type=str)
    parser.add_argument('--result_path', default="../data/dataset_2023-05-22_distalonly", type=str)
    args = parser.parse_args()
    print(f"args = {args}")
 
    # Compare input and output directories
    input_object_code_files = os.listdir(args.grasp_path)
    print(f"Found {len(input_object_code_files)} object codes in {args.grasp_path}")
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
            f"CUDA_VISIBLE_DEVICES={args.gpu}",
            "python scripts/validate_grasps.py",
            f"--hand_model_type {args.hand_model_type}",
            f"--gpu {args.gpu}",
            f"--grasp_path {args.grasp_path}",
            f"--result_path {args.result_path}",
            f"--object_code {object_code}"
        ])
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
