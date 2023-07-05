import subprocess
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.realpath("."))
from utils.isaac_validator import ValidationType
from utils.hand_model_type import HandModelType
from utils.joint_angle_targets import OptimizationMethod

from tap import Tap


class ValidateAllGraspsArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    optimization_method: OptimizationMethod = (
        OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    )
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    grasp_path: str = "../data/graspdata_2023-05-24_allegro_distalonly"
    result_path: str = "../data/dataset_2023-05-24_allegro_distalonly"


if __name__ == "__main__":
    args = ValidateAllGraspsArgumentParser().parse_args()
    print(f"args = {args}")

    # Compare input and output directories
    input_object_code_files = os.listdir(args.grasp_path)
    print(f"Found {len(input_object_code_files)} object codes in {args.grasp_path}")
    existing_object_code_files = (
        os.listdir(args.result_path) if os.path.exists(args.result_path) else []
    )
    print(f"Found {len(existing_object_code_files)} object codes in {args.result_path}")

    # Sanity check that we are going into the right folder
    objects_only_in_output = set(existing_object_code_files) - set(
        input_object_code_files
    )
    print(f"Num objects only in output: {len(objects_only_in_output)}")
    assert (
        len(objects_only_in_output) == 0
    ), f"Objects only in output: {objects_only_in_output}"

    # Don't redo old work
    object_only_in_input = set(input_object_code_files) - set(
        existing_object_code_files
    )
    print(f"Num objects only in input: {len(object_only_in_input)}")
    object_only_in_input = list(object_only_in_input)
    print("Processing these only...")

    pbar = tqdm(object_only_in_input)
    for object_code_file in pbar:
        object_code = object_code_file.split(".")[0]
        pbar.set_description(f"Processing {object_code}")

        command = " ".join(
            [
                f"CUDA_VISIBLE_DEVICES={args.gpu}",
                "python scripts/validate_grasps.py",
                f"--hand_model_type {args.hand_model_type.name}",
                f"--optimization_method {args.optimization_method.name}",
                f"--validation_type {args.validation_type.name}",
                f"--gpu {args.gpu}",
                f"--grasp_path {args.grasp_path}",
                f"--result_path {args.result_path}",
                f"--object_code {object_code}",
            ]
        )
        print(f"Running command: {command}")

        try:
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Skipping {object_code} and continuing")
            continue
