import subprocess

command_parts = [
    "python scripts/validate_all_grasps.py",
    "--optimization_method DESIRED_DIST_TOWARDS_FINGERS_CENTER_MULTIPLE_STEP",
    "--validation_type NO_GRAVITY_SHAKING",
    "--grasp_path ../data/2023-07-01_graspdata",
    "--result_path ../data/2023-07-01_dataset_DESIRED_DIST_TOWARDS_FINGERS_CENTER_MULTIPLE_STEP_v2",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)
