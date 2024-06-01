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
npy_files = []
for folder in experiment_folders:
    npy_files.extend(list((folder / "raw_evaled_grasp_config_dicts").glob("*.npy")))
print(f"Found {len(npy_files)} npy files")

# %%
config_dicts = [
    np.load(npy_file, allow_pickle=True).item()
    for npy_file in tqdm(npy_files, desc="Loading config dicts")
]

# %%
noisy_npy_files = list((data_folder / "2024-05-09_rotated_stable_grasps_noisy_TUNED" / "raw_evaled_grasp_config_dicts").glob("*.npy"))
print(f"Found {len(noisy_npy_files)} noisy npy files")

# %%
noisy_config_dicts = [
    np.load(npy_file, allow_pickle=True).item()
    for npy_file in tqdm(noisy_npy_files, desc="Loading config dicts")
]

# %%
experiment2_folders = sorted(list(data_folder.glob("2024-05-26_rotated_v2_only_grasps_*")))
print(f"Found {len(experiment2_folders)} experiment folders")

# %%
npy2_files = []
for folder in experiment2_folders:
    npy2_files.extend(list((folder / "raw_evaled_grasp_config_dicts").glob("*.npy")))
print(f"Found {len(npy2_files)} npy files")

# %%
config2_dicts = [
    np.load(npy_file, allow_pickle=True).item()
    for npy_file in tqdm(npy2_files, desc="Loading config dicts")
]

# %%
noisy_npy2_files = list((data_folder / "2024-05-27_rotated_v2_only_grasps_noisy_TUNED_NOSHAKE" / "raw_evaled_grasp_config_dicts").glob("*.npy"))
print(f"Found {len(noisy_npy2_files)} noisy npy files")

# %%
noisy_config2_dicts = [
    np.load(npy_file, allow_pickle=True).item()
    for npy_file in tqdm(noisy_npy2_files, desc="Loading config dicts")
]

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

# %%
config_dict_mean = np.mean([x for config_dict in config_dicts for x in config_dict["passed_eval"]])
noisy_config_dict_mean = np.mean([x for config_dict in noisy_config_dicts for x in config_dict["passed_eval"]])
config2_dict_mean = np.mean([x for config_dict in config2_dicts for x in config_dict["passed_eval"]])
noisy_config2_dict_mean = np.mean([x for config_dict in noisy_config2_dicts for x in config_dict["passed_eval"]])
print(f"Mean passed_eval for config_dicts: {config_dict_mean}")
print(f"Mean passed_eval for noisy_config_dicts: {noisy_config_dict_mean}")
print(f"Mean passed_eval for config2_dicts: {config2_dict_mean}")
print(f"Mean passed_eval for noisy_config2_dicts: {noisy_config2_dict_mean}")

# %%
config_dict_mean = np.mean([x for config_dict in config_dicts for x in config_dict["passed_simulation"]])
noisy_config_dict_mean = np.mean([x for config_dict in noisy_config_dicts for x in config_dict["passed_simulation"]])
config2_dict_mean = np.mean([x for config_dict in config2_dicts for x in config_dict["passed_simulation"]])
noisy_config2_dict_mean = np.mean([x for config_dict in noisy_config2_dicts for x in config_dict["passed_simulation"]])
print(f"Mean passed_simulation for config_dicts: {config_dict_mean}")
print(f"Mean passed_simulation for noisy_config_dicts: {noisy_config_dict_mean}")
print(f"Mean passed_simulation for config2_dicts: {config2_dict_mean}")
print(f"Mean passed_simulation for noisy_config2_dicts: {noisy_config2_dict_mean}")

# %%

all_passed_evals = ([x for config_dict in config_dicts for x in config_dict["passed_eval"]] + 
[x for config_dict in noisy_config_dicts for x in config_dict["passed_eval"]] +
[x for config_dict in config2_dicts for x in config_dict["passed_eval"]] +
[x for config_dict in noisy_config2_dicts for x in config_dict["passed_eval"]])

# %%
print(f"Mean passed_eval for all: {np.mean(all_passed_evals)}")

# %%
import matplotlib.pyplot as plt

plt.hist(all_passed_evals, bins=20)


# %%

all_passed_simulations = ([x for config_dict in config_dicts for x in config_dict["passed_simulation"]] + 
[x for config_dict in noisy_config_dicts for x in config_dict["passed_simulation"]] +
[x for config_dict in config2_dicts for x in config_dict["passed_simulation"]] +
[x for config_dict in noisy_config2_dicts for x in config_dict["passed_simulation"]])

# %%
print(f"Mean passed_simulation for all: {np.mean(all_passed_simulations)}")

# %%
import matplotlib.pyplot as plt

plt.hist(all_passed_simulations, bins=20)

# %%
all_passed_new_penetration_tests = ([x for config_dict in config_dicts for x in config_dict["passed_new_penetration_test"]] + 
[x for config_dict in noisy_config_dicts for x in config_dict["passed_new_penetration_test"]] +
[x for config_dict in config2_dicts for x in config_dict["passed_new_penetration_test"]] +
[x for config_dict in noisy_config2_dicts for x in config_dict["passed_new_penetration_test"]])

# %%
print(f"Mean passed_new_penetration_test for all: {np.mean(all_passed_new_penetration_tests)}")

# %%
import matplotlib.pyplot as plt
plt.hist(all_passed_new_penetration_tests, bins=20)

# %%
OUTPUT_DIR = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/2024-05-30_FINAL_AGGREGATED_RESULTS")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
ALL_DIR = OUTPUT_DIR / "all"
ALL_DIR.mkdir(exist_ok=True, parents=True)

# %%
npy_files[0].name
# %%
assert len(config_dicts) == len(npy_files)
for npy_file, config_dict in zip(npy_files, config_dicts):
    np.save(ALL_DIR / npy_file.name, config_dict)

# %%
assert len(noisy_config_dicts) == len(noisy_npy_files)
for npy_file, config_dict in zip(noisy_npy_files, noisy_config_dicts):
    np.save(ALL_DIR / npy_file.name, config_dict)

# %%
assert len(config2_dicts) == len(npy2_files)
for npy_file, config_dict in zip(npy2_files, config2_dicts):
    np.save(ALL_DIR / npy_file.name, config_dict)

# %%
assert len(noisy_config2_dicts) == len(noisy_npy2_files)
for npy_file, config_dict in zip(noisy_npy2_files, noisy_config2_dicts):
    np.save(ALL_DIR / npy_file.name, config_dict)

# %%
(np.array(all_passed_evals) == 1).sum()

# %%
np.array([x == 1 for config_dict in config_dicts for x in config_dict["passed_eval"]]).sum()

# %%
all_npy_files = npy_files + noisy_npy_files + npy2_files + noisy_npy2_files
all_config_dicts = config_dicts + noisy_config_dicts + config2_dicts + noisy_config2_dicts
print(f"Found {len(all_npy_files)} all npy files")
print(f"Found {len(all_config_dicts)} all config dicts")

# %%
set(failed_object_names).issubset(set([x.stem for x in all_npy_files]))

# %%
failed_object_names[0], failed_object_names[0] in [x.stem for x in all_npy_files]

# %%
set(failed_object_names) - set([x.stem for x in all_npy_files])

# %%
filtered_npy_files = []
filtered_config_dicts = []
for npy_file, config_dict in zip(all_npy_files, all_config_dicts):
    if npy_file.stem in failed_object_names:
        print(f"Found {npy_file.stem} in failed_object_names")
    else:
        filtered_npy_files.append(npy_file)
        filtered_config_dicts.append(config_dict)

# %%
print(f"Filtered {len(all_npy_files) - len(filtered_npy_files)} npy files")
print(f"Now have {len(filtered_npy_files)} npy files and {len(filtered_config_dicts)} config dicts")

# %%
1700*3 + 3000

# %%
len(npy_files), len(noisy_npy_files), len(npy2_files), len(noisy_npy2_files)

# %%
from collections import defaultdict
obj_to_config_dicts = defaultdict(list)
for npy_file, config_dict in zip(filtered_npy_files, filtered_config_dicts):
    obj_to_config_dicts[npy_file.stem].append(config_dict)

# %%
print(f"Found {len(obj_to_config_dicts)} objects")

# %%
obj_to_config_dict = {}
for obj, config_dicts in obj_to_config_dicts.items():
    if len(config_dicts) == 0:
        print(f"Skipping {obj}")
        continue
    for d in config_dicts:
        for k, v in d.items():
            assert v.shape[0] > 0

    obj_to_config_dict[obj] = {
        k: np.concatenate([d[k] for d in config_dicts], axis=0)
        for k in config_dicts[0].keys()
    }

# %%
for obj, config_dict in obj_to_config_dict.items():
    print(f"{obj}: {config_dict['passed_eval'].shape[0]}")

# %%
num_grasps = np.array([v['passed_eval'].shape[0] for v in obj_to_config_dict.values()])
print(f"num_grasps = {np.sum(num_grasps)}")

# %%
num_grasps = 0
for d in all_config_dicts:
    num_grasps += d['passed_eval'].shape[0]
print(f"num_grasps = {num_grasps}")

# %%
num_grasps = 0
for d in filtered_config_dicts:
    num_grasps += d['passed_eval'].shape[0]
print(f"num_grasps = {num_grasps}")

# %%
object_codes = [
    obj[:obj.index("_0_")]
    for obj in obj_to_config_dict.keys()
]
print(f"Found {len(object_codes)} object codes")
unique_object_codes = set(object_codes)
print(f"Unique object codes: {len(unique_object_codes)}")
print(f"object_codes[:10] = {object_codes[:10]}")
print(f"obj_code_and_scale_strs = {list(obj_to_config_dict.keys())[:10]}")

# %%
for obj in obj_to_config_dict.keys():
    if "core-bottle-908e85e13c6fbde0a1ca08763d503f0e" in obj:
        print(obj)


# %%
for obj in object_codes:
    if "core-bottle-908e85e13c6fbde0a1ca08763d503f0e" in obj:
        print(obj)

# %%
for obj, config_dict in tqdm(obj_to_config_dict.items(), total=len(obj_to_config_dict)):
    np.save(ALL_DIR / f"{obj}.npy", config_dict)

# %%
INFERENCE_DIR = OUTPUT_DIR / "inference"
INFERENCE_DIR.mkdir(exist_ok=True, parents=True)

# %%
num_good_objects = 0
num_good_grasps = 0
obj_to_one_good_config_dict = {}
obj_to_all_good_config_dict = {}
for obj, config_dict in tqdm(obj_to_config_dict.items(), total=len(obj_to_config_dict)):
    good_idxs = np.where(config_dict["passed_eval"] > 0.8)[0]
    num_good_objects += len(good_idxs) > 0
    num_good_grasps += len(good_idxs)
    if len(good_idxs) == 0:
        continue
    one_good_idx = good_idxs[0]
    obj_to_one_good_config_dict[obj] = {
        k: v[one_good_idx:one_good_idx+1]
        for k, v in config_dict.items()
    }

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
    obj_to_all_good_config_dict[obj] = all_good_config_dict

# %%
print(f"len(obj_to_one_good_config_dict) = {len(obj_to_one_good_config_dict)}")
print(f"len(obj_to_all_good_config_dict) = {len(obj_to_all_good_config_dict)}")
print(f"num_good_objects = {num_good_objects}")
print(f"num_good_grasps = {num_good_grasps}")

# %%
all_good_include_noisy_config_dicts = [
    obj_to_all_good_config_dict[obj]
    for obj in obj_to_all_good_config_dict.keys()
]
merged_all_good_include_noisy_config_dict = {
    k: np.concatenate([d[k] for d in all_good_include_noisy_config_dicts], axis=0)
    for k in all_good_include_noisy_config_dicts[0].keys()
}

# %%
merged_all_good_include_noisy_config_dict['passed_eval'].shape

# %%
ALL_GOOD_INCLUDE_NOISY_DIR = INFERENCE_DIR / "all_good_include_noisy"
ALL_GOOD_INCLUDE_NOISY_DIR.mkdir(exist_ok=True, parents=True)
np.save(ALL_GOOD_INCLUDE_NOISY_DIR / "grasps.npy", merged_all_good_include_noisy_config_dict)

# %%
one_good_include_noisy_config_dicts = [
    obj_to_one_good_config_dict[obj]
    for obj in obj_to_one_good_config_dict.keys()
]
merged_one_good_include_noisy_config_dict = {
    k: np.concatenate([d[k] for d in one_good_include_noisy_config_dicts], axis=0)
    for k in one_good_include_noisy_config_dicts[0].keys()
}
print(f"merged_one_good_include_noisy_config_dict['passed_eval'].shape = {merged_one_good_include_noisy_config_dict['passed_eval'].shape}")

# %%
ONE_GOOD_INCLUDE_NOISY_DIR = INFERENCE_DIR / "one_good_include_noisy"
ONE_GOOD_INCLUDE_NOISY_DIR.mkdir(exist_ok=True, parents=True)
np.save(ONE_GOOD_INCLUDE_NOISY_DIR / "grasps.npy", merged_one_good_include_noisy_config_dict)

# %%

# %%
for o, d in obj_to_config_dict.items():
    if "dog" in o.lower():
        print(o)
        print(d["passed_eval"].shape)
        print(f"passed_eval = {d['passed_eval'].mean()}")
        print(f"passed_eval = {d['passed_eval'].max()}")

# %%
all_nonoise_npy_files = npy_files + npy2_files
all_nonoise_config_dicts = config_dicts + config2_dicts
print(f"Found {len(all_nonoise_npy_files)} all nonoise npy files")
print(f"Found {len(all_nonoise_config_dicts)} all nonoise config dicts")

# %%
set(failed_object_names).issubset(set([x.stem for x in all_nonoise_npy_files]))

# %%
failed_object_names[0], failed_object_names[0] in [x.stem for x in all_nonoise_npy_files]

# %%
set(failed_object_names) - set([x.stem for x in all_nonoise_npy_files])

# %%
filtered_nonoise_npy_files = []
filtered_nonoise_config_dicts = []
for npy_file, config_dict in zip(all_nonoise_npy_files, all_nonoise_config_dicts):
    if npy_file.stem in failed_object_names:
        print(f"Found {npy_file.stem} in failed_object_names")
    else:
        filtered_nonoise_npy_files.append(npy_file)
        filtered_nonoise_config_dicts.append(config_dict)

# %%
print(f"Filtered {len(all_nonoise_npy_files) - len(filtered_nonoise_npy_files)} npy files")
print(f"Now have {len(filtered_nonoise_npy_files)} npy files and {len(filtered_nonoise_config_dicts)} config dicts")

# %%
from collections import defaultdict
nonoise_obj_to_config_dicts = defaultdict(list)
for npy_file, config_dict in zip(filtered_nonoise_npy_files, filtered_nonoise_config_dicts):
    nonoise_obj_to_config_dicts[npy_file.stem].append(config_dict)

# %%
print(f"Found {len(nonoise_obj_to_config_dicts)} objects")

# %%
nonoise_obj_to_config_dict = {}
for obj, config_dicts in nonoise_obj_to_config_dicts.items():
    if len(config_dicts) == 0:
        print(f"Skipping {obj}")
        continue
    for d in config_dicts:
        for k, v in d.items():
            assert v.shape[0] > 0

    nonoise_obj_to_config_dict[obj] = {
        k: np.concatenate([d[k] for d in config_dicts], axis=0)
        for k in config_dicts[0].keys()
    }

# %%
for obj, config_dict in nonoise_obj_to_config_dict.items():
    print(f"{obj}: {config_dict['passed_eval'].shape[0]}")

# %%
num_grasps = np.array([v['passed_eval'].shape[0] for v in nonoise_obj_to_config_dict.values()])
print(f"num_grasps = {np.sum(num_grasps)}")

