import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0 python scripts/generate_hand_config_dicts.py",
    "--seed 0",
    "--output_hand_config_dicts_path ../data/2023-08-23_hand_config_dicts",
    "--wandb_name graspdata",
    "--use_wandb",
    "--object_scale 0.1",
    "--n_iter 400",
    "--wandb_visualization_freq 100",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)