import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0 python scripts/generate_nerf_data.py",
    "--output_nerfdata_path ../data/2023-08-25_nerfdata",
    "--randomize_order_seed 0",
    "--only_objects_in_this_path ../data/2023-08-25_evaled_grasp_config_dicts",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)
