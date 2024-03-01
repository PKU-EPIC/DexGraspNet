# %%
import numpy as np
import pathlib

data_path = pathlib.Path("../data/PROBE_2024-02-07_softball_0-051_5random/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0510.npy")
# data_path = pathlib.Path(f"../data/PROBE_7_2024-02-07_50mugs_0-075_5random/evaled_grasp_config_dicts_train/core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy")
assert data_path.exists()

# %%
data_dict = np.load(data_path, allow_pickle=True).item()

# %%
print(f"data_dict.keys() = {data_dict.keys()}")

import matplotlib.pyplot as plt

# %%
passed_sim = data_dict["passed_simulation"]
passed_penetration = data_dict["passed_new_penetration_test"]
passed_eval = data_dict["passed_eval"]

# %%
plt.hist(passed_sim)

# %%
# Color each point by whether it passed the simulation.
plt.scatter(data_dict["trans"][:, 0], data_dict["trans"][:, 1], s=1, c=passed_sim)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Passed simulation')
# Add a colorbar.
plt.colorbar()

# %%
import plotly.graph_objects as go
fig = go.Figure()
# fig.add_trace(go.Scatter(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], mode="markers", marker=dict(size=1, color=passed_sim)))
fig.add_trace(go.Scatter3d(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], z=passed_sim, mode="markers", marker=dict(size=1, color=passed_sim)))

fig.show()


# %%
