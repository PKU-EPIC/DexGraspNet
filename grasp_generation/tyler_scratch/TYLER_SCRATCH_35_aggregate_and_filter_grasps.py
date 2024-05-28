# %%
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from transforms3d.euler import quat2euler

# %%
folders = [
    pathlib.Path(f"/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-13_rotated_grasps_{i}_bigger/raw_evaled_grasp_config_dicts")
    for i in range(10)
]
for folder in folders:
    assert folder.exists()

# %%
object_files = sum(
    [list(folder.glob("*.npy")) for folder in folders],
    []
)
assert len(object_files) > 0
print(f"Num object files: {len(object_files)}")

# %%
bad_files = []
for file in tqdm(object_files):
    data_dict = np.load(file, allow_pickle=True).item()
    object_states_before_grasp_array = data_dict["object_states_before_grasp"]
    quat_xyzw = object_states_before_grasp_array[..., 3:7].reshape(-1, 4)
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    quat_w = quat_wxyz[..., 0]

    lin_vel, ang_vel = object_states_before_grasp_array[..., 7:10].reshape(-1, 3), object_states_before_grasp_array[..., 10:13].reshape(-1, 3)
    lin_speed, ang_speed = np.linalg.norm(lin_vel, axis=-1), np.linalg.norm(ang_vel, axis=-1)

    rpy_list = []
    for i in range(quat_wxyz.shape[0]):
        rpy = quat2euler(quat_wxyz[i])
        rpy_list.append(rpy)
    rpy_array = np.array(rpy_list)
    abs_rpy_array = np.abs(rpy_array)

    passed_eval = data_dict["passed_eval"]
    if np.min(quat_w) < 0.95 or np.mean(lin_speed) > 0.05 or np.mean(ang_speed) > 1.0:
        bad_files.append(file)
        print(f"{file.name}, min_w: {np.min(quat_w)}, max_w: {np.max(quat_w)}")
print(f"bad_files: {len(bad_files)} {len(bad_files) / len(object_files) * 100:.2f}%")

# %%
object_files = [file for file in object_files if file not in bad_files]

# %%
n_passed_evals_list = []
bad_idxs = []
bad_files2 = []
for i, file in tqdm(enumerate(object_files), total=len(object_files)):
    data_dict = np.load(file, allow_pickle=True).item()
    # n_passed_evals = np.sum(data_dict["passed_eval"] > 0.9)
    n_passed_evals = data_dict["passed_eval"].max()
    n_passed_evals_list.append(n_passed_evals)
    if n_passed_evals < 0.99:
        bad_files2.append(file)
        bad_idxs.append(i)
        print(file.name)

print(f"Num with 0 passed evals: {len(bad_files2)} {len(bad_files2) / len(object_files) * 100:.2f}%")

# %%
object_files = [file for file in object_files if file not in bad_files2]

# %%
print(f"Num object files: {len(object_files)}")

# %%
import subprocess
new_experiment_folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-13_rotated_grasps_bigger_aggregated/raw_evaled_grasp_config_dicts")
new_experiment_folder.mkdir(parents=True, exist_ok=False)
for file in tqdm(object_files):
    new_output_file = new_experiment_folder / file.name
    ln_command = f"ln -rs {file} {new_output_file}"
    subprocess.run(ln_command, shell=True, check=True)


# %%
y_pos_list = []
for file in tqdm(object_files):
    data_dict = np.load(file, allow_pickle=True).item()
    object_states_before_grasp_array = data_dict["object_states_before_grasp"]
    y_pos = object_states_before_grasp_array[..., 1]
    y_pos_list.append(y_pos)

# %%
plt.hist(np.concatenate(y_pos_list).reshape(-1), bins=100)

# %%
plt.scatter(np.array(y_pos_list).mean(axis=1).mean(axis=1), n_passed_evals_list)
plt.xlabel("mean y pos")
plt.ylabel("n passed evals")

# %%
plt.scatter(np.array(y_pos_list).mean(axis=1).mean(axis=1)[bad_idxs], np.array(n_passed_evals_list)[bad_idxs], label="bad files")
plt.scatter(np.array(y_pos_list).mean(axis=1).mean(axis=1)[~np.array(bad_idxs)], np.array(n_passed_evals_list)[~np.array(bad_idxs)], label="good files")
plt.xlabel("mean y pos")
plt.ylabel("n passed evals")

# %%
np.array(y_pos_list).shape

# %%
len(n_passed_evals_list)

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
input_folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-05_rotated_grasps/raw_grasp_config_dicts_subset/")
input_meshdata = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata/")
assert input_folder.exists()
output_subset_grasp_config_dicts = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-05_rotated_grasps/raw_grasp_config_dicts_subset_all_objects/")
output_subset_grasp_config_dicts.mkdir(parents=True, exist_ok=False)

# %%
input_files = list(input_folder.glob("*.npy"))
input_file = input_files[0]
input_dict = np.load(input_file, allow_pickle=True).item()
object_codes = [file.name for file in input_meshdata.iterdir()]
for object_code in tqdm(object_codes):
    output_file = output_subset_grasp_config_dicts / f"{object_code}_0_0750.npy"
    output_dict = {k: v[0:1] for k, v in input_dict.items()}
    np.save(output_file, output_dict, allow_pickle=True)

# %%
object_states_before_grasp_array.shape
# %%
lin_speed.shape, ang_speed.shape
# %%
lin_speed, ang_speed
# %%
plt.hist(lin_speed, bins=100)

# %%
plt.hist(ang_speed, bins=100)

# %%
