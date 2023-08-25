import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0 python scripts/eval_all_grasp_config_dicts.py",
    "--input_grasp_config_dicts_path ../data/2023-08-25_grasp_config_dicts",
    "--output_evaled_grasp_config_dicts_path ../data/2023-08-25_evaled_grasp_config_dicts",
    "--randomize_order_seed 0",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)
