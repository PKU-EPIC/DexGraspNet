# %%
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R

data_path_bad = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose_HALTON_no-rot/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
data_path_good = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose_HALTON/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
assert data_path_bad.exists()
assert data_path_good.exists()

# %%
data_dict_bad = np.load(data_path_bad, allow_pickle=True).item()
data_dict_good = np.load(data_path_good, allow_pickle=True).item()


# %%
trans_bad = data_dict_bad["trans"]
trans_good = data_dict_good["trans"]
rpy_bad = R.from_matrix(data_dict_bad["rot"]).as_euler('xyz', degrees=True)
rpy_good = R.from_matrix(data_dict_good["rot"]).as_euler('xyz', degrees=True)

# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=trans_bad[:, 0],
    y=trans_bad[:, 1],
    z=trans_bad[:, 2],
    mode="markers",
    marker=dict(size=1),
)
)
fig.show()


# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=trans_good[:, 0],
    y=trans_good[:, 1],
    z=trans_good[:, 2],
    mode="markers",
    marker=dict(size=1),
)
)
fig.show()


# %%
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=rpy_bad[:, 0],
    y=rpy_bad[:, 1],
    z=rpy_bad[:, 2],
    mode="markers",
    marker=dict(size=1),
)
)
fig.show()
# %%
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=rpy_good[:, 0],
    y=rpy_good[:, 1],
    z=rpy_good[:, 2],
    mode="markers",
    marker=dict(size=1),
)
)
fig.show()


# %%

