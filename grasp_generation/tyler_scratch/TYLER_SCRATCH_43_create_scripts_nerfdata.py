# %%
import subprocess
import pathlib

# %%
path_bigger = pathlib.Path("../cluster_scripts/2024-05-07_generate_nerf_data_bigger_0.sh")
path_smaller = pathlib.Path("../cluster_scripts/2024-05-07_generate_nerf_data_smaller_0.sh")
path = pathlib.Path("../cluster_scripts/2024-05-07_generate_nerf_data_0.sh")
assert path_bigger.exists()
assert path_smaller.exists()
assert path.exists()

# %%
N_SCRIPTS = 7
for i in range(1, N_SCRIPTS):
    new_path = pathlib.Path(f"../cluster_scripts/2024-05-07_generate_nerf_data_{i}.sh")
    subprocess.run(f"cp {path} {new_path}", shell=True, check=True)
    subprocess.run(f"sed -i 's:_0/:_{i}/:g' {new_path}", shell=True, check=True)

    new_path_bigger = pathlib.Path(f"../cluster_scripts/2024-05-07_generate_nerf_data_bigger_{i}.sh")
    subprocess.run(f"cp {path_bigger} {new_path_bigger}", shell=True, check=True)
    subprocess.run(f"sed -i 's:_0/:_{i}/:g' {new_path_bigger}", shell=True, check=True)

    new_path_smaller = pathlib.Path(f"../cluster_scripts/2024-05-07_generate_nerf_data_smaller_{i}.sh")
    subprocess.run(f"cp {path_smaller} {new_path_smaller}", shell=True, check=True)
    subprocess.run(f"sed -i 's:_0/:_{i}/:g' {new_path_smaller}", shell=True, check=True)

# %%
