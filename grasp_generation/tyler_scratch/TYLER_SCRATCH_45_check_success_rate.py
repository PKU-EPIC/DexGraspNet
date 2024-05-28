# %%
import numpy as np
import pathlib
from tqdm import tqdm

# %%
path_str = "../data/2024-05-06_rotated_stable_grasps_0/raw_evaled_grasp_config_dicts/"
path_bigger_str = "../data/2024-05-06_rotated_stable_grasps_bigger_0/raw_evaled_grasp_config_dicts/"
path_smaller_str = "../data/2024-05-06_rotated_stable_grasps_smaller_0/raw_evaled_grasp_config_dicts/"

paths = [pathlib.Path(path_str.replace("_0", f"_{i}")) for i in range(7)]
path_biggers = [pathlib.Path(path_bigger_str.replace("_0", f"_{i}")) for i in range(7)]
path_smallers = [pathlib.Path(path_smaller_str.replace("_0", f"_{i}")) for i in range(7)]

# %%
dicts = []
for path in tqdm(paths):
    data_paths = sorted(list(path.glob("*.npy")))
    for data_path in data_paths:
        data_dict = np.load(data_path, allow_pickle=True).item()
        dicts.append(data_dict)

# %%
dict_biggers = []
for path_bigger in tqdm(path_biggers):
    data_paths = sorted(list(path_bigger.glob("*.npy")))
    for data_path in data_paths:
        data_dict = np.load(data_path, allow_pickle=True).item()
        dict_biggers.append(data_dict)

# %%
dict_smallers = []
for path_smaller in tqdm(path_smallers):
    data_paths = sorted(list(path_smaller.glob("*.npy")))
    for data_path in data_paths:
        data_dict = np.load(data_path, allow_pickle=True).item()
        dict_smallers.append(data_dict)

# %%
######### BASE ##################
success_rates = [d['passed_eval'].mean() for d in dicts]
num_good_grasps = [(d['passed_eval'] > 0.9).sum() for d in dicts]
num_no_grasps = [(d['passed_eval'] == 0).sum() for d in dicts]

# %%
print(f"num objects 0% success rates = {(np.array(success_rates) == 0).sum()}")
print(f"num objects no good grasps = {(np.array(num_good_grasps) == 0).sum()}")
print(f"num objects = {len(success_rates)}")

# %%
print(f"num good grasps = {np.array(num_good_grasps).sum()}")
print(f"num grasps = {sum([len(d['passed_eval']) for d in dicts])}")

# %%
import matplotlib.pyplot as plt

# %%
plt.hist(success_rates)
plt.xlabel("Success Rate")

# %%
plt.hist(num_good_grasps)
plt.xlabel("Number of Good Grasps")

# %%
plt.hist(num_no_grasps)
plt.xlabel("Number of No Grasps")

# %%
######### BIGGER ##################
success_rates = [d['passed_eval'].mean() for d in dict_biggers]
num_good_grasps = [(d['passed_eval'] > 0.9).sum() for d in dict_biggers]
num_no_grasps = [(d['passed_eval'] == 0).sum() for d in dict_biggers]

# %%
print(f"num objects 0% success rates = {(np.array(success_rates) == 0).sum()}")
print(f"num objects no good grasps = {(np.array(num_good_grasps) == 0).sum()}")
print(f"num objects = {len(success_rates)}")

# %%
print(f"num good grasps = {np.array(num_good_grasps).sum()}")
print(f"num grasps = {sum([len(d['passed_eval']) for d in dict_biggers])}")

# %%
import matplotlib.pyplot as plt

# %%
plt.hist(success_rates)
plt.xlabel("Success Rate")

# %%
plt.hist(num_good_grasps)
plt.xlabel("Number of Good Grasps")

# %%
plt.hist(num_no_grasps)
plt.xlabel("Number of No Grasps")

# %%
######### SMALLER ##################
success_rates = [d['passed_eval'].mean() for d in dict_smallers]
num_good_grasps = [(d['passed_eval'] > 0.9).sum() for d in dict_smallers]
num_no_grasps = [(d['passed_eval'] == 0).sum() for d in dict_smallers]

# %%
print(f"num objects 0% success rates = {(np.array(success_rates) == 0).sum()}")
print(f"num objects no good grasps = {(np.array(num_good_grasps) == 0).sum()}")
print(f"num objects = {len(success_rates)}")

# %%
print(f"num good grasps = {np.array(num_good_grasps).sum()}")
print(f"num grasps = {sum([len(d['passed_eval']) for d in dict_smallers])}")

# %%
import matplotlib.pyplot as plt

# %%
plt.hist(success_rates)
plt.xlabel("Success Rate")

# %%
plt.hist(num_good_grasps)
plt.xlabel("Number of Good Grasps")

# %%
plt.hist(num_no_grasps)
plt.xlabel("Number of No Grasps")


# %%
