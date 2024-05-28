# %%
import subprocess
import pathlib

# %%
path = pathlib.Path("../data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/nerfdata/")
assert path.exists()

output_path = pathlib.Path("/home/tylerlum/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/nerfdata/")
assert output_path.exists()

# %%
input_nerfdata_files = list(path.iterdir())
for input_nerfdata_file in input_nerfdata_files:
    output_nerfdata_file = output_path / input_nerfdata_file.name
    subprocess.run(f"ln -s {input_nerfdata_file.absolute()} {output_nerfdata_file.absolute()}", shell=True, check=True)
# %%

