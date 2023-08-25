import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasp_config_dicts.py",
    "--input_hand_config_dicts_path ../data/2023-08-25_hand_config_dicts",
    "--output_grasp_config_dicts_path ../data/2023-08-25_grasp_config_dicts",
    "--seed 0",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)
