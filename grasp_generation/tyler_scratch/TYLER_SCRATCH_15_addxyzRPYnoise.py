# %%
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R
data_folder = pathlib.Path("../data/2024-03-06_10mugs/raw_evaled_grasp_config_dicts")
assert data_folder.exists()
data_paths = sorted(list(data_folder.glob("*.npy")))

N = 1000
trans_max_noise = 0.03
rot_deg_max_noise = 5
N_SUCCESSFUL = 3

# %%
new_datas = []
# Combine successful grasp data with noise.
for data_path in data_paths:
    data_dict = np.load(data_path, allow_pickle=True).item()

    success_idxs = [(data_dict['passed_eval'] == 1).nonzero()[0][i] for i in range(N_SUCCESSFUL)]
    new_data = {}
    for k, v in data_dict.items():
        new_data[k] = np.stack([v[i] for i in success_idxs], axis=0)
    assert new_data["passed_eval"].sum() == N_SUCCESSFUL
    new_datas.append(new_data)

# %%
for new_data in new_datas:
    assert (new_data["passed_eval"] == 1).all()
    assert new_data["passed_eval"].shape[0] == N_SUCCESSFUL
new_data = {k: np.concatenate([d[k] for d in new_datas], axis=0) for k in new_datas[0].keys()}

for k, v in new_data.items():
    assert v.shape[0] == N_SUCCESSFUL * len(new_datas)

new_data = {k: v.repeat(N, axis=0) for k, v in new_data.items()}
N_DATAPOINTS = N_SUCCESSFUL * len(new_datas) * N
for k, v in new_data.items():
    assert v.shape[0] == N_DATAPOINTS
    for i in range(N_SUCCESSFUL):
        rows_equal = np.all(v[i*N:(i+1)*N] == v[i*N][None, ...].repeat(N, axis=0)).all()
        assert rows_equal

# %%
for k, v in new_data.items():
    print(f"{k} = {v.shape} (v[:10] = {v[:10]})")

# %%
for data_path in data_paths:
    filename = data_path.name
    from scipy.stats.qmc import Halton
    USE_HALTON = True
    if USE_HALTON:
        xyz_noise = (Halton(d=3, scramble=True).random(n=N_DATAPOINTS) * 2 - 1) * trans_max_noise
        rpy_noise = (Halton(d=3, scramble=True).random(n=N_DATAPOINTS) * 2 - 1) * rot_deg_max_noise
    else:
        xyz_noise = np.random.uniform(low=-trans_max_noise, high=trans_max_noise, size=(N_DATAPOINTS, 3))
        rpy_noise = np.random.uniform(low=-rot_deg_max_noise, high=rot_deg_max_noise, size=(N_DATAPOINTS, 3))

    new_data_copy = {k: v.copy() for k, v in new_data.items()}
    xyz = new_data_copy["trans"]
    rpy = R.from_matrix(new_data_copy["rot"]).as_euler('xyz', degrees=True)
    assert xyz.shape == (N_DATAPOINTS, 3)
    assert rpy.shape == (N_DATAPOINTS, 3)

    new_xyz = xyz + xyz_noise
    new_rpy = rpy + rpy_noise
    new_data_copy["trans"][:] = new_xyz
    new_data_copy["rot"][:] = R.from_euler('xyz', new_rpy, degrees=True).as_matrix()

    new_data_path = pathlib.Path(f"../data/2024-03-07_10mugs_augmented_pose_HALTON_no-rot_1k_v2/grasp_config_dicts/{filename}")
    print(f"new_data_path = {new_data_path}")
    new_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(new_data_path, new_data_copy)

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
