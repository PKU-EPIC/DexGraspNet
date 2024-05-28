# %%
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from transforms3d.euler import quat2euler

# %%
folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-05_rotated_grasps/raw_evaled_grasp_config_dicts_DEBUG/")
assert folder.exists()

# %%
object_files = list(folder.glob("*.npy"))
assert len(object_files) > 0

# %%
bad_files = []
for file in tqdm(object_files):
    data_dict = np.load(file, allow_pickle=True).item()
    object_poses_before_grasp_array = data_dict["object_poses_before_grasp"]
    quat_xyzw = object_poses_before_grasp_array[..., -4:].reshape(-1, 4)
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    quat_w = quat_wxyz[..., 0]
    rpy_list = []
    for i in range(quat_wxyz.shape[0]):
        rpy = quat2euler(quat_wxyz[i])
        rpy_list.append(rpy)
    rpy_array = np.array(rpy_list)
    abs_rpy_array = np.abs(rpy_array)

    passed_eval = data_dict["passed_eval"]
    if np.min(quat_w) < 0.95:
    # if np.max(np.rad2deg(abs_rpy_array)) > 10:
        bad_files.append(file)
        print(f"{file.name}, min_w: {np.min(quat_w)}, max_w: {np.max(quat_w)}")
        print(f"max(passed_eval): {np.max(passed_eval)}")
        print(f"max_rpy (deg): {np.max(np.rad2deg(abs_rpy_array[..., 0]))}, {np.argmax(np.rad2deg(abs_rpy_array[..., 1]))}, {np.argmax(np.rad2deg(abs_rpy_array[..., 2]))}")
        print()
print(f"bad_files: {len(bad_files)}")

# %%
n_passed_evals_list = []
for file in tqdm(object_files):
    data_dict = np.load(file, allow_pickle=True).item()
    n_passed_evals = np.sum(data_dict["passed_eval"] > 0.9)
    if n_passed_evals == 0:
        n_passed_evals_list.append(n_passed_evals)
        print(file.name)

print(f"Num with 0 passed evals: {len(n_passed_evals_list)}")

# %%
y_pos_list = []
for file in tqdm(object_files):
    data_dict = np.load(file, allow_pickle=True).item()
    object_poses_before_grasp_array = data_dict["object_poses_before_grasp"]
    y_pos = object_poses_before_grasp_array[..., 1]
    y_pos_list.append(y_pos)

# %%
plt.hist(np.concatenate(y_pos_list).reshape(-1), bins=100)

# %%
# meshdata = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspN../data/rotated_meshdata_v2")
# assert meshdata.exists()
# decomposed_obj_files = list(meshdata.rglob("**/decomposed.obj"))
# assert len(decomposed_obj_files) > 0

# %%
# import trimesh
# MAX = 0
# pbar = tqdm(decomposed_obj_files)
# for decomposed_obj_file in pbar:
#     pbar.set_description(f"MAX: {MAX}")
#     mesh = trimesh.load(decomposed_obj_file)
#     bounds = mesh.bounds
#     max_abs = np.max(np.abs(bounds))
#     MAX = max(max_abs, MAX)
# %%
import trimesh
mesh = trimesh.load("/juno/u/tylerlum/github_repos/DexGraspN../data/rotated_meshdata_v2/core-jar-3dec5904337999561710801cae5dc529/coacd/decomposed.obj")
mesh.bounds

# %%
input_folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-05_rotated_grasps/raw_grasp_config_dicts/")
assert input_folder.exists()
output_subset_grasp_config_dicts = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-05_rotated_grasps/raw_grasp_config_dicts_subset/")
output_subset_grasp_config_dicts.mkdir(parents=True, exist_ok=False)

# %%
input_files = list(input_folder.glob("*.npy"))
for np_file in tqdm(input_files):
    np_file_name = np_file.name
    output_np_file = output_subset_grasp_config_dicts / np_file_name
    input_dict = np.load(np_file, allow_pickle=True).item()
    output_dict = {k: v[0:1] for k, v in input_dict.items()}
    np.save(output_np_file, output_dict, allow_pickle=True)

# %%
