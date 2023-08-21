import subprocess

command_parts = [
    "python scripts/validate_all_grasps.py",
    "--optimization_method DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS",
    "--validation_type NO_GRAVITY_SHAKING",
    "--grasp_path ../data/2023-07-01_graspdata_mid_optimization/0",
    "--result_path ../data/2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_mid_optimization/0",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)