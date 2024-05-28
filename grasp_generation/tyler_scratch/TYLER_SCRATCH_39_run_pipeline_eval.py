# %%
import subprocess
import pathlib
import random
from tqdm import tqdm

# %%
EXPERIMENT_NAME = "2024-04-18_eval"
experiment_folder = pathlib.Path(f"/juno/u/tylerlum/github_repos/nerf_grasping/experiments/{EXPERIMENT_NAME}")
optimized_grasp_config_dicts_folder = experiment_folder / "optimized_grasp_config_dicts"
assert optimized_grasp_config_dicts_folder.exists()

# %%
object_grasp_config_dicts = sorted(list(optimized_grasp_config_dicts_folder.iterdir()))
random.Random(0).shuffle(object_grasp_config_dicts)

for object_grasp_config_dict in tqdm(object_grasp_config_dicts):
    command = f"CUDA_VISIBLE_DEVICES=0 python scripts/eval_grasp_config_dict.py --hand_model_type ALLEGRO_HAND --validation_type GRAVITY_AND_TABLE --gpu 0 --meshdata_root_path ../data/rotated_meshdata --input_grasp_config_dicts_path {experiment_folder / 'optimized_grasp_config_dicts'} --output_evaled_grasp_config_dicts_path {experiment_folder / 'evaled_optimized_grasp_config_dicts'} --object_code_and_scale_str {object_grasp_config_dict.stem} --max_grasps_per_batch 5000"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)