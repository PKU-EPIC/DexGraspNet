import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0",
    "python scripts/generate_nerf_data.py",
    "--gpu 0",
    "--output_nerf_path ../data/2023-08-16_nerfdataset",
    "--randomize_order_seed 100",
    "--only_objects_in_this_graspdata_path ../data/2023-07-01_dataset_DESIRED_DIST_TOWARDS_FINGERS_CENTER_MULTIPLE_STEP_v2",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)
