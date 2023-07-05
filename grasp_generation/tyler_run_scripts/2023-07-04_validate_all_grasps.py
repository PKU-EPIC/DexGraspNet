import subprocess

command_parts = [
    "python scripts/validate_all_grasps.py",
    "--grasp_path ../data/2023-07-01_graspdata",
    "--result_path ../data/2023-07-01_dataset",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)