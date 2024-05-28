# %%
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R
data_path = pathlib.Path("../data/2024-03-05_softballs/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
assert data_path.exists()

# %%
data_dict = np.load(data_path, allow_pickle=True).item()

# %%
print(f"data_dict.keys() = {data_dict.keys()}")

# %%
data_dict['passed_eval'] == 1

# %%
(data_dict['passed_eval'] == 1).nonzero()

# %%
data_dict['passed_eval']

# %%
IDX = 11
data = {k: v[IDX] for k, v in data_dict.items()}
assert data["passed_eval"] == 1

# %%
N = 30000
trans_max_noise = 0.03
rot_deg_max_noise = 0

# %%
from scipy.stats.qmc import Halton
USE_HALTON = True
if USE_HALTON:
    xyz_noise = (Halton(d=3, scramble=True).random(n=N) * 2 - 1) * trans_max_noise
    rpy_noise = (Halton(d=3, scramble=True).random(n=N) * 2 - 1) * rot_deg_max_noise
else:
    xyz_noise = np.random.uniform(low=-trans_max_noise, high=trans_max_noise, size=(N, 3))
    rpy_noise = np.random.uniform(low=-rot_deg_max_noise, high=rot_deg_max_noise, size=(N, 3))

# No noise for the first element.
xyz_noise[0, :] = 0
rpy_noise[0, :] = 0

# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=xyz_noise[:, 0], y=xyz_noise[:, 1], z=xyz_noise[:, 2], mode="markers", marker=dict(size=1)))


# %%
new_data_dict = {k: v[None, ...].repeat(N, axis=0) for k, v in data.items()}
xyz = new_data_dict["trans"]
rpy = R.from_matrix(new_data_dict["rot"]).as_euler('xyz', degrees=True)
assert xyz.shape == (N, 3)
assert rpy.shape == (N, 3)
new_xyz = xyz + xyz_noise
new_rpy = rpy + rpy_noise
new_data_dict["trans"][:] = new_xyz
new_data_dict["rot"][:] = R.from_euler('xyz', new_rpy, degrees=True).as_matrix()

# %%
new_data_path = pathlib.Path(f"../data/2024-03-05_softballs_idx{IDX}_augmented_pose_HALTON_no-rot_30k/grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
print(f"new_data_path = {new_data_path}")
new_data_path.parent.mkdir(parents=True, exist_ok=True)
np.save(new_data_path, new_data_dict)

# %%
XYZ = new_data_dict["trans"]
RPY = R.from_matrix(new_data_dict["rot"]).as_euler('xyz', degrees=True)

# %%
def trans_diff(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz1 - xyz2, axis=1)

def rot_diff(rpy1: np.ndarray, rpy2: np.ndarray) -> np.ndarray:
    q1 = R.from_euler('xyz', rpy1, degrees=True).as_quat()
    q2 = R.from_euler('xyz', rpy2, degrees=True).as_quat()
    dot_product = np.sum(q1 * q2, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta = 2 * np.arccos(np.abs(dot_product))
    return np.degrees(theta)

# %%
my_trans_diff = trans_diff(XYZ[0:1].repeat(N, axis=0), XYZ)
my_rot_diff = rot_diff(RPY[0:1].repeat(N, axis=0), RPY)

# %%
import matplotlib.pyplot as plt
plt.hist(my_trans_diff)

# %%
plt.hist(my_rot_diff)