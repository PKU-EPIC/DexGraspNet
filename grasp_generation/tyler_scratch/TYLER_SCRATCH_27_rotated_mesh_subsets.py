# %%
import pathlib
import subprocess
import math

# %%
existing_meshdata_path = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata")
assert existing_meshdata_path.exists()

# %%
existing_meshdata_folders = sorted(list(existing_meshdata_path.iterdir()))
assert len(existing_meshdata_folders) > 2000

# %%
num_per_folder = 250
N_new_folders = math.ceil(len(existing_meshdata_folders) / num_per_folder)
new_meshdata_folders = [
    existing_meshdata_path.parent / f"2024-04-08_rotated_meshdata_subset_{i}"
    for i in range(N_new_folders)
]

# %%
for i, new_meshdata_folder in enumerate(new_meshdata_folders):
    new_meshdata_folder.mkdir(parents=True, exist_ok=False)

    start_idx = i * num_per_folder
    end_idx = min(start_idx + num_per_folder, len(existing_meshdata_folders))
    for j in range(start_idx, end_idx):
        existing_meshdata_folder = existing_meshdata_folders[j]
        ln_command = f"ln -sr {existing_meshdata_folder} {new_meshdata_folder}"
        subprocess.run(ln_command, shell=True, check=True)
# %%
