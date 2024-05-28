# %%
import pathlib
import subprocess
from tqdm import tqdm

# %%
rotated_meshdata = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata")
rotated_meshdata_v2 = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata_v2")
assert rotated_meshdata.exists()
assert rotated_meshdata_v2.exists()

rotated_meshdata_v2_only = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata_v2_only")
rotated_meshdata_v2_only.mkdir(parents=True, exist_ok=False)

# %%
num_rotated_meshes = len(list(rotated_meshdata.iterdir()))
num_rotated_meshes_v2 = len(list(rotated_meshdata_v2.iterdir()))
print(f"num_rotated_meshes: {num_rotated_meshes}")
print(f"num_rotated_meshes_v2: {num_rotated_meshes_v2}")

# %%
rotated_mesh_names_set = set([p.name for p in rotated_meshdata.iterdir()])
rotated_mesh_names_v2_set = set([p.name for p in rotated_meshdata_v2.iterdir()])

# %%
assert rotated_mesh_names_set.issubset(rotated_mesh_names_v2_set)
only_in_v2 = list(rotated_mesh_names_v2_set - rotated_mesh_names_set)
print(f"only_in_v2: {len(only_in_v2)}")

# %%
for mesh_name in tqdm(only_in_v2):
    orig_mesh_path = rotated_meshdata_v2 / mesh_name
    assert orig_mesh_path.exists()

    new_mesh_path = rotated_meshdata_v2_only / mesh_name
    assert not new_mesh_path.exists()

    subprocess.run(f"cp -r {orig_mesh_path} {new_mesh_path}", shell=True, check=True)
# %%
