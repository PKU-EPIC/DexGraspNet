# %%
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# %%
data_folder = pathlib.Path("../data/2024-04-13_rotated_grasps_big_aggregated/raw_evaled_grasp_config_dicts")
assert data_folder.exists()
new_data_folder = pathlib.Path("../data/2024-04-13_rotated_grasps_big_aggregated_augmented_pose_HALTON_50/grasp_config_dicts")
new_data_folder.mkdir(parents=True, exist_ok=True)
data_paths = sorted(list(data_folder.glob("*.npy")))

# %%
N_other_objects_per_grasp = 2
N_noise = 50
trans_max_noise = 0
rot_deg_max_noise = 0
N_SUCCESSFUL = 1

# %%
new_datas_by_name = {}
num_successes_list = []

# Combine successful grasp data with noise.

# Get N_SUCCESSFUL successful grasps from each object.
for data_path in data_paths:
    filename = data_path.name

    # Get success_idxs
    data_dict = np.load(data_path, allow_pickle=True).item()
    all_success_idxs = (data_dict['passed_eval'] == 1).nonzero()[0]
    num_successes = len(all_success_idxs)
    num_successes_list.append(num_successes)
    if num_successes < N_SUCCESSFUL:
        print(f"{filename} has {num_successes} successful grasps")
        continue

    success_idxs = [all_success_idxs[i] for i in range(N_SUCCESSFUL)]

    # Get N_SUCCESSFUL successful grasps
    new_data = {}
    for k, v in data_dict.items():
        new_data[k] = np.stack([v[i] for i in success_idxs], axis=0)
    assert new_data["passed_eval"].sum() == N_SUCCESSFUL
    new_datas_by_name[filename] = new_data

# %%
some_filename = list(new_datas_by_name.keys())[:5]
print(f"some_filename = {some_filename}")
for f in some_filename:
    print(f"new_datas_by_name[f]['trans'] = {new_datas_by_name[f]['trans']}")

# %%
FILENAME_IDX = -1

(np.load(data_folder / some_filename[FILENAME_IDX], allow_pickle=True).item()['passed_eval'] == 1).nonzero()

# %%
GOOD_IDX = (np.load(data_folder / some_filename[FILENAME_IDX], allow_pickle=True).item()['passed_eval'] == 1).nonzero()[0][0]

# %%
np.load(data_folder / some_filename[FILENAME_IDX], allow_pickle=True).item()['trans'][GOOD_IDX]

# %%
new_datas_by_name[some_filename[FILENAME_IDX]]['trans']

# %%
np.load(new_data_folder  / some_filename[FILENAME_IDX], allow_pickle=True).item()['trans'][::50]

# %%
new_data['trans'][0]


# %%
import matplotlib.pyplot as plt
plt.hist(num_successes_list, bins=100)

# %%
print(f"num_successful_total = {sum(num_successes_list)}")

# %%
for new_data in new_datas_by_name.values():
    assert (new_data["passed_eval"] == 1).all()
    assert new_data["passed_eval"].shape[0] == N_SUCCESSFUL

# %%
N_DATAPOINTS = N_SUCCESSFUL * (1 + N_other_objects_per_grasp) * N_noise

# %%
N_DATAPOINTS

# %%
for data_path in tqdm(data_paths):
    filename = data_path.name
    # if filename not in some_filename:
        # continue

    # For each object, copy over N_other_objects_per_grasp other objects' poses.
    other_random_filenames = np.random.choice(list(new_datas_by_name.keys()), size=N_other_objects_per_grasp, replace=False)
    filenames = [filename] + list(other_random_filenames)
    new_data = {k: np.concatenate([new_datas_by_name[f][k] for f in filenames], axis=0) for k in new_datas_by_name[filename].keys()}
    new_data = {k: v.repeat(N_noise, axis=0) for k, v in new_data.items()}

    # Check that the data is repeated correctly.
    for k, v in new_data.items():
        assert v.shape[0] == N_DATAPOINTS
        for i in range(N_SUCCESSFUL * (1 + N_other_objects_per_grasp)):
            rows_equal = np.all(v[i*N_noise:(i+1)*N_noise] == v[i*N_noise][None, ...].repeat(N_noise, axis=0)).all()
            assert rows_equal

    new_data_copy = new_data.copy()
    xyz = new_data["trans"]
    rpy = R.from_matrix(new_data["rot"]).as_euler('xyz', degrees=True)
    assert xyz.shape == (N_DATAPOINTS, 3)
    assert rpy.shape == (N_DATAPOINTS, 3)

    # Sample noise
    from scipy.stats.qmc import Halton
    USE_HALTON = True
    if USE_HALTON:
        xyz_noise = (Halton(d=3, scramble=True).random(n=N_DATAPOINTS) * 2 - 1) * trans_max_noise
        rpy_noise = (Halton(d=3, scramble=True).random(n=N_DATAPOINTS) * 2 - 1) * rot_deg_max_noise
    else:
        xyz_noise = np.random.uniform(low=-trans_max_noise, high=trans_max_noise, size=(N_DATAPOINTS, 3))
        rpy_noise = np.random.uniform(low=-rot_deg_max_noise, high=rot_deg_max_noise, size=(N_DATAPOINTS, 3))

    new_xyz = xyz + xyz_noise
    new_rpy = rpy + rpy_noise
    new_data_copy["trans"] = new_xyz
    new_data_copy["rot"] = R.from_euler('xyz', new_rpy, degrees=True).as_matrix()

    new_data_path = new_data_folder / filename
    print(f"new_data_path = {new_data_path}")
    np.save(new_data_path, new_data_copy)

# %%
xyz[0], xyz_noise[0]

# %%

# %%
v.shape

# %%
new_data['trans'][:50]

# %%
x = new_datas_by_name[data_paths[-1].name]
print(x['trans'])

# %%
for k, v in new_data.items():
    assert v.shape[0] == N_DATAPOINTS
    for i in range(N_SUCCESSFUL * (1 + N_other_objects_per_grasp)):
        rows_equal = np.all(v[i*N_noise:(i+1)*N_noise] == v[i*N_noise][None, ...].repeat(N_noise, axis=0)).all()
        assert rows_equal

# %%
from localscope import localscope
from typing import Tuple
import matplotlib.pyplot as plt
@localscope.mfc
def plot_labels(trans: np.ndarray, labels: np.ndarray, title: str) -> Tuple[plt.Figure, np.ndarray]:
    N = trans.shape[0]
    assert trans.shape == (N, 3)
    assert labels.shape == (N,)
    z_min, z_max = np.min(trans[:, 2]), np.max(trans[:, 2])
    n_plots = 10
    z_list = np.linspace(z_min, z_max, n_plots + 1)
    nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.flatten()
    for i, (z_low, z_high) in enumerate(zip(z_list[:-1], z_list[1:])):
        points_to_plot = np.logical_and(
            trans[:, 2] > z_low,
            trans[:, 2] < z_high,
        )
        axes[i].scatter(trans[points_to_plot, 0], trans[points_to_plot, 1], s=1, c=labels[points_to_plot])
        axes[i].set_title(f"z in [{z_low:.2f}, {z_high:.2f}]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

# plot_labels(trans=new_data_copy["trans"][-N:], labels=new_data_copy["passed_eval"][-N:], title=new_data_path.name)
# %%
plot_labels(trans=new_data_copy["trans"], labels=new_data_copy["passed_eval"], title=new_data_path.name)
# %%
