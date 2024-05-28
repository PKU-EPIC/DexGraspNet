# %%
import pathlib
import subprocess
from tqdm import tqdm

# %%
rotated_meshdata_path = pathlib.Path("../data/rotated_meshdata")
assert rotated_meshdata_path.exists()

rotated_meshdata_v2_path = pathlib.Path("../data/rotated_meshdata_v2")
rotated_meshdata_v2_path.mkdir(exist_ok=False)

good_object_folder = pathlib.Path("../data/2024-04-09_rotated_grasps_aggregated/raw_evaled_grasp_config_dicts")
assert good_object_folder.exists()

# %%
good_object_code_and_scale_strs = [x.stem for x in good_object_folder.iterdir()]
good_object_codes = [x.split("_0_")[0] for x in good_object_code_and_scale_strs]

assert len(good_object_codes) > 0
print(f"len(good_object_codes) = {len(good_object_codes)}")
print(f"good_object_codes[:10] = {good_object_codes[:10]}")

# %%
for good_object_code in good_object_codes:
    assert (rotated_meshdata_path / good_object_code).exists()

# %%
for good_object_code in tqdm(good_object_codes, desc="Copying"):
    cp_command = f"cp -r {rotated_meshdata_path / good_object_code} {rotated_meshdata_v2_path / good_object_code}"
    subprocess.run(cp_command, shell=True, check=True)

# %%
