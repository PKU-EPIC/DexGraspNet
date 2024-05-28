# %%
import pathlib
bottles_path = pathlib.Path("../data/rotated_meshdata_stable_by_category/0509_bottle")
assert bottles_path.exists()
bottle_codes = sorted([filepath.name for filepath in bottles_path.iterdir()])
assert len(bottle_codes) > 0
print(f"Found {len(bottle_codes)} bottle codes")
print(f"First 10: {bottle_codes[:10]}")

# %%
import subprocess
N_BOTTLES = 75
new_bottles_path = pathlib.Path(f"../data/2024-03-10_{N_BOTTLES}bottle_meshdata")
new_bottles_path.mkdir(parents=True, exist_ok=False)

# %%
meshdata_root_path = pathlib.Path("../data/rotated_meshdata_stable")
assert meshdata_root_path.exists()

for i in range(N_BOTTLES):
    bottle_code = bottle_codes[i]

    existing_path = meshdata_root_path / bottle_code
    assert existing_path.exists()

    new_path = new_bottles_path / bottle_code

    # Create symlink
    symlink_command = f"ln -rs {existing_path} {new_path}"
    print(f"Running command: {symlink_command}")
    subprocess.run(symlink_command, shell=True, check=True)


# %%
