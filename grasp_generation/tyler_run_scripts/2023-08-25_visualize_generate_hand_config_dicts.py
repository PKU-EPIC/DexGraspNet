import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0 python scripts/generate_hand_config_dicts.py",
    "--output_hand_config_dicts_path ../data/2023-08-25_visualize_hand_config_dicts",
    "--object_scale 0.1",
    "--seed 0",
    "--wandb_name hand_config_dicts",
    "--batch_size_each_object 500",
    "--n_objects_per_batch 2",
    "--n_iter 400",
    "--use_wandb",
    "--wandb_visualization_freq 100",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)