# %%
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from transforms3d.euler import quat2euler

# %%
# folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-05_rotated_grasps/raw_evaled_grasp_config_dicts_subset_all_objects_DEBUG/")
folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-08_rotated_grasps_6/raw_evaled_grasp_config_dicts")
assert folder.exists()

# %%
object_files = list(folder.glob("*.npy"))
assert len(object_files) > 0

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
    # if np.min(quat_w) < 0.99:
    # if np.max(np.rad2deg(abs_rpy_array)) > 10:
    # if np.min(quat_w) < 0.95 or np.max(lin_speed) > 0.03 or np.max(ang_speed) > 0.5:
    if np.min(quat_w) < 0.95 or np.mean(lin_speed) > 0.05 or np.mean(ang_speed) > 1.0:
        bad_files.append(file)
        print(f"{file.name}, min_w: {np.min(quat_w)}, max_w: {np.max(quat_w)}")
        print(f"max(passed_eval): {np.max(passed_eval)}")
        print(f"max_rpy (deg): {np.max(np.rad2deg(abs_rpy_array[..., 0]))}, {np.max(np.rad2deg(abs_rpy_array[..., 1]))}, {np.max(np.rad2deg(abs_rpy_array[..., 2]))}")
        print(f"mean_lin_speed: {np.mean(lin_speed)}, mean_ang_speed: {np.mean(ang_speed)}")
        print()
print(f"bad_files: {len(bad_files)} {len(bad_files) / len(object_files) * 100:.2f}%")

# %%
lin_speeds, ang_speeds = [], []
for file in tqdm(object_files):
    data_dict = np.load(file, allow_pickle=True).item()
    object_states_before_grasp_array = data_dict["object_states_before_grasp"]
    lin_vel, ang_vel = object_states_before_grasp_array[..., 7:10].reshape(-1, 3), object_states_before_grasp_array[..., 10:13].reshape(-1, 3)
    lin_speed, ang_speed = np.linalg.norm(lin_vel, axis=-1), np.linalg.norm(ang_vel, axis=-1)
    mean_lin_speed, mean_ang_speed = np.mean(lin_speed), np.mean(ang_speed)
    lin_speeds.append(lin_speed)
    ang_speeds.append(ang_speed)

lin_speeds = np.concatenate(lin_speeds).reshape(-1)
ang_speeds = np.concatenate(ang_speeds).reshape(-1)

# %%
plt.hist(lin_speeds, bins=100)

# %%
plt.hist(ang_speeds, bins=100)

# %%
lin_speeds.max(), ang_speeds.max()

# %%
(ang_speeds > 0.5).nonzero(), (ang_speeds > 0.3).nonzero()

# %%
(lin_speeds > 0.03).nonzero(), (lin_speeds > 0.02).nonzero()

# %%
object_files[175].name

# %%
lin_speeds[lin_speeds > 0.05].shape, ang_speeds[ang_speeds > 0.5].shape


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
    object_states_before_grasp_array = data_dict["object_states_before_grasp"]
    y_pos = object_states_before_grasp_array[..., 1]
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
