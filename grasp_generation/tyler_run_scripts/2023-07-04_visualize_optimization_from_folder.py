import subprocess

command_parts = [
    "python tests/visualize_optimization_from_folder.py",
    "--input_folder ../data/2023-07-01_dryrun_graspdata_mid_optimization",
    "--object_code mujoco-Olive_Kids_Butterfly_Garden_Pencil_Case",
    "--save_to_html",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)
