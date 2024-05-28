# %%
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R
from localscope import localscope
from typing import Tuple
import matplotlib.pyplot as plt

# %%
data_path = pathlib.Path("../data/2024-03-07_10mugs_augmented_pose_HALTON_no-rot_1k/evaled_grasp_config_dicts/core-mug-8f6c86feaa74698d5c91ee20ade72edc_0_0765.npy")
data_path2 = pathlib.Path("../data/2024-03-07_10mugs_augmented_pose_HALTON_no-rot_1k/grasp_config_dicts/core-mug-8f6c86feaa74698d5c91ee20ade72edc_0_0765.npy")
assert data_path.exists()

# %%
data_dict = np.load(data_path, allow_pickle=True).item()

# %%
passed_sim = data_dict["passed_simulation"]
passed_penetration = data_dict["passed_new_penetration_test"]
passed_eval = data_dict["passed_eval"]
trans = data_dict["trans"]
rot = data_dict["rot"]

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
i = 10
N = 1000
plot_labels(trans=data_dict["trans"][i*N:(i+1)*N], labels=data_dict["passed_eval"][i*N:(i+1)*N], title="Passed Eval")

# %%

