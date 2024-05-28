# %%
import subprocess
import pathlib
import random
from tqdm import tqdm

# %%
existing_meshdata_path = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata")
assert existing_meshdata_path.exists()

new_meshdata_path = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-05_rotated_meshdata_subset")
new_meshdata_path.mkdir(parents=True, exist_ok=False)

# %%
N_OBJECTS = 200
object_code_folders = list(existing_meshdata_path.iterdir())
subset_object_code_folders = random.sample(object_code_folders, N_OBJECTS)

for subset_object_code_folder in tqdm(subset_object_code_folders):
    new_folder = new_meshdata_path / subset_object_code_folder.name
    cp_command = f"cp -r {subset_object_code_folder} {new_folder}"
    subprocess.run(cp_command, shell=True, check=True)

# %%
