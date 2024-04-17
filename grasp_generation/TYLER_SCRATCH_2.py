# %%
import numpy as np
import pathlib
# ../data/2024-02-07_softball_0-051_5random/evaled_grasp_config_dicts/ --meshdata_root_path ../data/2024-01-22_softball_meshdata/ --object_code_and_scale_str ddg-ycb_054_softball_0_0510 --idx_to_visualize 8

# data_path = pathlib.Path("../data/2024-02-07_softball_0-051_5random/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0510.npy")
data_path = pathlib.Path("../data/2024-02-07_50mugs_0-075_5random/evaled_grasp_config_dicts_train/core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy")
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
# IDX = 8
IDX = 7
data = {k: v[IDX] for k, v in data_dict.items()}

# %%
N = 100
x_noise = np.linspace(-0.1, 0.1, N)
y_noise = np.linspace(-0.1, 0.1, N)
xy_noise = np.array(np.meshgrid(x_noise, y_noise)).T.reshape(-1, 2)
print(f"xy_noise.shape = {xy_noise.shape}")

# %%
import matplotlib.pyplot as plt
plt.scatter(xy_noise[:, 0], xy_noise[:, 1], s=0.1)

# %%
data['trans']

# %%
new_data_dict = {k: v[None, ...].repeat(N**2, axis=0) for k, v in data.items()}
new_data_dict["trans"][..., :2] += xy_noise

# %%
plt.scatter(new_data_dict["trans"][:, 0], new_data_dict["trans"][:, 1], s=0.1)


# %%
# new_data_path = pathlib.Path(f"../data/PROBE_{IDX}_2024-02-07_softball_0-051_5random/grasp_config_dicts/ddg-ycb_054_softball_0_0510.npy")
new_data_path = pathlib.Path(f"../data/PROBE_{IDX}_2024-02-07_50mugs_0-075_5random/grasp_config_dicts_train/core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy")

new_data_path.parent.mkdir(parents=True, exist_ok=True)
np.save(new_data_path, new_data_dict)

# %%
new_data_dict.keys()

# %%
new_data_dict["passed_eval"]

# %%
