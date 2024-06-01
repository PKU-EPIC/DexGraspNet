# %%
from tqdm import tqdm
import numpy as np
import pathlib

# %%
data_folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data")
assert data_folder.exists()

# %%
experiment_folders = sorted(list(data_folder.glob("2024-05-06_rotated_stable_grasps_*")))
print(f"Found {len(experiment_folders)} experiment folders")

# %%
npy_file_to_config_dict = {}
for folder in tqdm(experiment_folders, desc="Loading config dicts"):
    npy_files = list((folder / "raw_evaled_grasp_config_dicts").glob("*.npy"))
    for npy_file in npy_files:
        config_dict = np.load(npy_file, allow_pickle=True).item()
        npy_file_to_config_dict[npy_file] = config_dict
print(f"Found {len(npy_file_to_config_dict)} npy files")

# %%
npy_file_to_noisy_config_dict = {}
noisy_npy_files = list((data_folder / "2024-05-09_rotated_stable_grasps_noisy_TUNED" / "raw_evaled_grasp_config_dicts").glob("*.npy"))
for npy_file in tqdm(noisy_npy_files, desc="Loading config dicts"):
    config_dict = np.load(npy_file, allow_pickle=True).item()
    npy_file_to_noisy_config_dict[npy_file] = config_dict
print(f"Found {len(npy_file_to_noisy_config_dict)} noisy npy files")

# %%
experiment2_folders = sorted(list(data_folder.glob("2024-05-26_rotated_v2_only_grasps_*")))
print(f"Found {len(experiment2_folders)} experiment folders")

# %%
npy_file_to_config_dict2 = {}
for folder in tqdm(experiment2_folders, desc="Loading config dicts"):
    npy_files = list((folder / "raw_evaled_grasp_config_dicts").glob("*.npy"))
    for npy_file in npy_files:
        config_dict = np.load(npy_file, allow_pickle=True).item()
        npy_file_to_config_dict2[npy_file] = config_dict
print(f"Found {len(npy_file_to_config_dict2)} npy files")

# %%
npy_file_to_noisy_config_dict2 = {}
noisy_npy_files2 = list((data_folder / "2024-05-27_rotated_v2_only_grasps_noisy_TUNED_NOSHAKE" / "raw_evaled_grasp_config_dicts").glob("*.npy"))
for npy_file in tqdm(noisy_npy_files2, desc="Loading config dicts"):
    config_dict = np.load(npy_file, allow_pickle=True).item()
    npy_file_to_noisy_config_dict2[npy_file] = config_dict
print(f"Found {len(npy_file_to_noisy_config_dict2)} noisy npy files")

# %%
failure_logs = [
    experiment2_folder / "NEW_nerfdata_100imgs_failures.txt"
    for experiment2_folder in experiment2_folders
    if (experiment2_folder / "NEW_nerfdata_100imgs_failures.txt").exists()
]
print(f"Found {len(failure_logs)} failure logs")

# %%
def read_and_extract_object_names(file_path):
    object_names = []
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                object_name = line.split(':')[0]
                object_names.append(object_name.strip())
    return object_names

# %%
failed_object_names = []
for failure_log in failure_logs:
    object_names = read_and_extract_object_names(failure_log)
    print(f"{failure_log.name}: {object_names}")
    failed_object_names.extend(object_names)
print(f"Found {len(failed_object_names)} object names")

# %%
config_dict_mean = np.mean([x for config_dict in npy_file_to_config_dict.values() for x in config_dict["passed_eval"]])
noisy_config_dict_mean = np.mean([x for config_dict in npy_file_to_noisy_config_dict.values() for x in config_dict["passed_eval"]])
config2_dict_mean = np.mean([x for config_dict in npy_file_to_config_dict2.values() for x in config_dict["passed_eval"]])
noisy_config2_dict_mean = np.mean([x for config_dict in npy_file_to_noisy_config_dict2.values() for x in config_dict["passed_eval"]])
print(f"Mean passed_eval for config_dicts: {config_dict_mean}")
print(f"Mean passed_eval for noisy_config_dicts: {noisy_config_dict_mean}")
print(f"Mean passed_eval for config2_dicts: {config2_dict_mean}")
print(f"Mean passed_eval for noisy_config2_dicts: {noisy_config2_dict_mean}")

# %%
config_dict_mean_sim = np.mean([x for config_dict in npy_file_to_config_dict.values() for x in config_dict["passed_simulation"]])
noisy_config_dict_mean_sim = np.mean([x for config_dict in npy_file_to_noisy_config_dict.values() for x in config_dict["passed_simulation"]])
config2_dict_mean_sim = np.mean([x for config_dict in npy_file_to_config_dict2.values() for x in config_dict["passed_simulation"]])
noisy_config2_dict_mean_sim = np.mean([x for config_dict in npy_file_to_noisy_config_dict2.values() for x in config_dict["passed_simulation"]])
print(f"Mean passed_simulation for config_dicts: {config_dict_mean_sim}")
print(f"Mean passed_simulation for noisy_config_dicts: {noisy_config_dict_mean_sim}")
print(f"Mean passed_simulation for config2_dicts: {config2_dict_mean_sim}")
print(f"Mean passed_simulation for noisy_config2_dicts: {noisy_config2_dict_mean_sim}")

# %%
from collections import defaultdict
npy_file_to_all_config_dict = defaultdict(list)
for npy_file, config_dict in npy_file_to_config_dict.items():
    npy_file_to_all_config_dict[npy_file].append(config_dict)
for npy_file, config_dict in npy_file_to_noisy_config_dict.items():
    npy_file_to_all_config_dict[npy_file].append(config_dict)
for npy_file, config_dict in npy_file_to_config_dict2.items():
    npy_file_to_all_config_dict[npy_file].append(config_dict)
for npy_file, config_dict in npy_file_to_noisy_config_dict2.items():
    npy_file_to_all_config_dict[npy_file].append(config_dict)

npy_file_to_all_config_dict = {
    k: {
        k: np.concatenate([d[k] for d in v], axis=0)
        for k in v[0].keys()
    }
    for k, v in npy_file_to_all_config_dict.items()
}

all_npy_files, all_config_dicts = zip(*npy_file_to_all_config_dict.items())
print(f"Found {len(all_npy_files)} all npy files")
print(f"Found {len(all_config_dicts)} all config dicts")

# %%
npy_file_to_all_nonoise_config_dict = defaultdict(list)
for npy_file, config_dict in npy_file_to_config_dict.items():
    npy_file_to_all_nonoise_config_dict[npy_file].append(config_dict)
for npy_file, config_dict in npy_file_to_config_dict2.items():
    npy_file_to_all_nonoise_config_dict[npy_file].append(config_dict)

npy_file_to_all_nonoise_config_dict = {
    k: {
        k: np.concatenate([d[k] for d in v], axis=0)
        for k in v[0].keys()
    }
    for k, v in npy_file_to_all_nonoise_config_dict.items()
}
all_nonoise_npy_files, all_nonoise_config_dicts = zip(*npy_file_to_all_nonoise_config_dict.items())
print(f"Found {len(all_nonoise_npy_files)} all nonoise npy files")
print(f"Found {len(all_nonoise_config_dicts)} all nonoise config dicts")

# %%
set(failed_object_names).issubset(set([x.stem for x in all_npy_files]))

# %%
failed_object_names[0], failed_object_names[0] in [x.stem for x in all_npy_files]

# %%
set(failed_object_names) - set([x.stem for x in all_npy_files])

# %%
filtered_npy_file_to_all_config_dict = {
    k: v
    for k, v in npy_file_to_all_config_dict.items()
    if k.stem not in failed_object_names
}
filtered_all_npy_files, filtered_all_config_dicts = zip(*filtered_npy_file_to_all_config_dict.items())

filtered_npy_file_to_all_nonoise_config_dict = {
    k: v
    for k, v in npy_file_to_all_nonoise_config_dict.items()
    if k.stem not in failed_object_names
}
filtered_all_nonoise_npy_files, filtered_all_nonoise_config_dicts = zip(*filtered_npy_file_to_all_nonoise_config_dict.items())

# %%
print(f"Filtered {len(all_npy_files) - len(filtered_all_npy_files)} npy files")
print(f"Now have {len(filtered_all_npy_files)} npy files and {len(filtered_all_config_dicts)} config dicts")

# %%
print(f"Filtered {len(all_nonoise_npy_files) - len(filtered_all_nonoise_npy_files)} npy files")
print(f"Now have {len(filtered_all_nonoise_npy_files)} npy files and {len(filtered_all_nonoise_config_dicts)} config dicts")

# %%
1700*3 + 3000

# %%
for npy_file, config_dict in filtered_npy_file_to_all_config_dict.items():
    print(f"{npy_file.stem}: {config_dict['passed_eval'].shape[0]}")

# %%
num_all_grasps = np.array([v['passed_eval'].shape[0] for v in filtered_npy_file_to_all_config_dict.values()]).sum()
print(f"num_all_grasps = {num_all_grasps}")

# %%
num_all_grasps_v2 = 0
for d in filtered_all_config_dicts:
    num_all_grasps_v2 += d['passed_eval'].shape[0]
print(f"num_all_grasps_v2 = {num_all_grasps_v2}")

# %%
num_all_nonoise_grasps = np.array([v['passed_eval'].shape[0] for v in filtered_npy_file_to_all_nonoise_config_dict.values()]).sum()
print(f"num_all_nonoise_grasps = {num_all_nonoise_grasps}")

# %%
num_all_nonoise_grasps_v2 = 0
for d in filtered_all_nonoise_config_dicts:
    num_all_nonoise_grasps_v2 += d['passed_eval'].shape[0]
print(f"num_all_nonoise_grasps_v2 = {num_all_nonoise_grasps_v2}")

# %%
filtered_object_code_and_scale_strs = list(set([x.stem for x in filtered_all_npy_files]))
filtered_object_codes = list(set([
    obj[:obj.index("_0_")]
    for obj in filtered_object_code_and_scale_strs
]))
print(f"Found {len(filtered_object_code_and_scale_strs)} unique object code and scale strs")
print(f"Unique object codes: {len(filtered_object_codes)}")
print(f"filtered_object_codes[:10] = {filtered_object_codes[:10]}")
print(f"filtered_object_code_and_scale_strs[:10] = {filtered_object_code_and_scale_strs[:10]}")

# %%
for obj in filtered_object_code_and_scale_strs:
    if "core-bottle-908e85e13c6fbde0a1ca08763d503f0e" in obj:
        print(obj)


# %%
for obj in filtered_object_codes:
    if "core-bottle-908e85e13c6fbde0a1ca08763d503f0e" in obj:
        print(obj)

# %%
import localscope
@localscope.localscope.mfc
def get_one_good_config_dict(npy_file_to_config_dict):
    good_config_dicts = []
    for npy_file, config_dict in npy_file_to_config_dict.items():
        good_idxs = np.where(config_dict["passed_eval"] > 0.8)[0]
        if len(good_idxs) == 0:
            continue
        one_good_idx = good_idxs[0]
        one_good_config_dict = {
            k: v[one_good_idx:one_good_idx+1]
            for k, v in config_dict.items()
        }
        good_config_dicts.append(one_good_config_dict)
    good_config_dict = {
        k: np.concatenate([d[k] for d in good_config_dicts], axis=0)
        for k in good_config_dicts[0].keys()
    }
    return good_config_dict

@localscope.localscope.mfc
def get_all_good_config_dict(npy_file_to_config_dict):
    good_config_dicts = []
    for npy_file, config_dict in npy_file_to_config_dict.items():
        good_idxs = np.where(config_dict["passed_eval"] > 0.8)[0]
        if len(good_idxs) == 0:
            continue
        all_good_config_dicts = [
            {
                k: v[good_idx:good_idx+1]
                for k, v in config_dict.items()
            }
            for good_idx in good_idxs
        ]
        all_good_config_dict = {
            k: np.concatenate([d[k] for d in all_good_config_dicts], axis=0)
            for k in config_dict.keys()
        }
        good_config_dicts.append(all_good_config_dict)
    good_config_dict = {
        k: np.concatenate([d[k] for d in good_config_dicts], axis=0)
        for k in good_config_dicts[0].keys()
    }
    return good_config_dict

# %%
one_good_config_dict = get_one_good_config_dict(filtered_npy_file_to_all_config_dict)
all_good_config_dict = get_all_good_config_dict(filtered_npy_file_to_all_config_dict)

# %%
one_good_config_dict['passed_eval'].shape, all_good_config_dict['passed_eval'].shape

# %%
one_good_nonoise_config_dict = get_one_good_config_dict(filtered_npy_file_to_all_nonoise_config_dict)
all_good_nonoise_config_dict = get_all_good_config_dict(filtered_npy_file_to_all_nonoise_config_dict)

# %%
one_good_nonoise_config_dict['passed_eval'].shape, all_good_nonoise_config_dict['passed_eval'].shape

# %%
OUTPUT_DIR = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-05-30_FINAL_AGGREGATED_RESULTS")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
ALL_DIR = OUTPUT_DIR / "all"
ALL_DIR.mkdir(exist_ok=True, parents=True)


# %%
for npy_file, config_dict in tqdm(filtered_npy_file_to_all_config_dict.items(), total=len(filtered_npy_file_to_all_config_dict)):
    np.save(ALL_DIR / f"{npy_file.stem}.npy", config_dict)

# %%
ALL_NONOISE_DIR = OUTPUT_DIR / "all_nonoise"
ALL_NONOISE_DIR.mkdir(exist_ok=True, parents=True)

# %%
for npy_file, config_dict in tqdm(filtered_npy_file_to_all_nonoise_config_dict.items(), total=len(filtered_npy_file_to_all_nonoise_config_dict)):
    np.save(ALL_NONOISE_DIR / f"{npy_file.stem}.npy", config_dict)

# %%
INFERENCE_DIR = OUTPUT_DIR / "inference"
INFERENCE_DIR.mkdir(exist_ok=True, parents=True)

# %%
ALL_GOOD_DIR = INFERENCE_DIR / "all_good"
ALL_GOOD_DIR.mkdir(exist_ok=True, parents=True)
np.save(ALL_GOOD_DIR / "grasps.npy", all_good_config_dict)

# %%
ONE_GOOD_DIR = INFERENCE_DIR / "one_good"
ONE_GOOD_DIR.mkdir(exist_ok=True, parents=True)
np.save(ONE_GOOD_DIR / "grasps.npy", one_good_config_dict)

# %%
ALL_GOOD_NONOISE_DIR = INFERENCE_DIR / "all_good_nonoise"
ALL_GOOD_NONOISE_DIR.mkdir(exist_ok=True, parents=True)
np.save(ALL_GOOD_NONOISE_DIR / "grasps.npy", all_good_nonoise_config_dict)

# %%
ONE_GOOD_NONOISE_DIR = INFERENCE_DIR / "one_good_nonoise"
ONE_GOOD_NONOISE_DIR.mkdir(exist_ok=True, parents=True)
np.save(ONE_GOOD_NONOISE_DIR / "grasps.npy", one_good_nonoise_config_dict)

# %%
for npy_file, config_dict in tqdm(filtered_npy_file_to_all_nonoise_config_dict.items(), total=len(filtered_npy_file_to_all_nonoise_config_dict)):
    if "dog" in str(npy_file).lower():
        print(str(npy_file).lower())
        print(config_dict["passed_eval"].shape)
        print(f"passed_eval = {config_dict['passed_eval'].mean()}")
        print(f"passed_eval = {config_dict['passed_eval'].max()}")

# %%
all_passed_evals = [x for config_dict in filtered_npy_file_to_all_config_dict.values() for x in config_dict["passed_eval"]]
all_passed_simulations = [x for config_dict in filtered_npy_file_to_all_config_dict.values() for x in config_dict["passed_simulation"]]
all_passed_new_penetration_tests = [x for config_dict in filtered_npy_file_to_all_config_dict.values() for x in config_dict["passed_new_penetration_test"]]

# %%
print(f"Mean passed_eval for all: {np.mean(all_passed_evals)}")
print(f"Mean passed_simulation for all: {np.mean(all_passed_simulations)}")
print(f"Mean passed_new_penetration_test for all: {np.mean(all_passed_new_penetration_tests)}")

# %%
import matplotlib.pyplot as plt
plt.hist(all_passed_evals, bins=20, alpha=0.5, label="passed_eval")
plt.hist(all_passed_simulations, bins=20, alpha=0.5, label="passed_simulation")
plt.hist(all_passed_new_penetration_tests, bins=20, alpha=0.5, label="passed_new_penetration_test")
plt.legend()
plt.show()

# %%
(np.array(all_passed_evals) == 1).sum(), (np.array(all_passed_evals) > 0.8).sum()

# %%
