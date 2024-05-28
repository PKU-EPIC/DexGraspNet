# %%
import subprocess
import pathlib

# %%
path = pathlib.Path("../cluster_scripts/2024-04-13_relabel_augmented_big_0.sh")
assert path.exists()

# %%
N_SCRIPTS = 5
for i in range(1, N_SCRIPTS):
    new_path = pathlib.Path(f"../cluster_scripts/2024-04-13_relabel_augmented_big_{i}.sh")
    subprocess.run(f"cp {path} {new_path}", shell=True, check=True)
    subprocess.run(f"sed -i 's/randomize_order_seed 0/randomize_order_seed {i}/g' {new_path}", shell=True, check=True)
# %%
