# %%
import numpy as np
import pathlib

# %%
grasp_config_folder = pathlib.Path("../data/2024-04-16_rotated_grasps_aggregated_augmented_pose_HALTON_50/evaled_grasp_config_dicts_train")
assert grasp_config_folder.exists()

# %%
npy_filepaths = sorted(list(grasp_config_folder.glob("*.npy")))
assert len(npy_filepaths) > 0
print(f"Found {len(npy_filepaths)} npy files.")

# %%
from collections import defaultdict
new_dict = defaultdict(list)
for npy_filepath in npy_filepaths:
    grasp_config_dict = np.load(npy_filepath, allow_pickle=True).item()
    for key, value in grasp_config_dict.items():
        new_dict[key].append(value[0:1])


# %%
new_dict.keys()

# %%
len(new_dict['trans'])

# %%
new_dict['trans'][0].shape

# %%
new_dict_final = {}
for key, value in new_dict.items():
    new_dict_final[key] = np.concatenate(value, axis=0)

# %%
for key, value in new_dict_final.items():
    print(key, value.shape)

# %%
output_dir = pathlib.Path("../data/2024-04-16_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train")
output_dir.mkdir(exist_ok=True, parents=True)
np.save(
    output_dir / "aggregated_evaled_grasp_config_dict_train.npy",
    new_dict_final,
    allow_pickle=True,
)

# %%
