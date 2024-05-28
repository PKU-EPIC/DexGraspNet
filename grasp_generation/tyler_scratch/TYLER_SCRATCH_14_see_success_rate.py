# %%
import numpy as np
import pathlib

folder = pathlib.Path("../data/2024-03-06_10mugs/raw_evaled_grasp_config_dicts")
assert folder.exists()

# %%
filepaths = list(folder.glob("*.npy"))
assert len(filepaths) > 0

# %%
for filepath in filepaths:
    data_dict = np.load(filepath, allow_pickle=True).item()
    num_pts = len(data_dict["passed_eval"])
    num_successful = sum(data_dict["passed_eval"] == 1)
    print(f"{filepath.name}: {num_successful}/{num_pts} ({num_successful/num_pts:.2f})")
# %%
