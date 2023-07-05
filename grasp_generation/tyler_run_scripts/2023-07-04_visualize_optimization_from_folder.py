import subprocess

command_parts = [
    "python tests/visualize_optimization_from_folder.py",
    "--input_folder ../data/2023-07-01_graspdata_mid_optimization",
    "--object_code ddg-gd_banana_poisson_001",
    "--save_to_html",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)