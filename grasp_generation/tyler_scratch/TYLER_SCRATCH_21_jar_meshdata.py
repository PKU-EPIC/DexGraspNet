# %%
import pathlib
jars_path = pathlib.Path("../data/rotated_meshdata_v2_by_category/0336_jar")
assert jars_path.exists()
jar_codes = sorted([filepath.name for filepath in jars_path.iterdir()])
assert len(jar_codes) > 0
print(f"Found {len(jar_codes)} jar codes")
print(f"First 10: {jar_codes[:10]}")

# %%
import subprocess
N_JARS = 10
new_jars_path = pathlib.Path(f"../data/2024-03-13_{N_JARS}jar_meshdata")
new_jars_path.mkdir(parents=True, exist_ok=False)

# %%
meshdata_root_path = pathlib.Path("../data/rotated_meshdata_v2")
assert meshdata_root_path.exists()

for i in range(N_JARS):
    jar_code = jar_codes[i]

    existing_path = meshdata_root_path / jar_code
    assert existing_path.exists()

    new_path = new_jars_path / jar_code

    # Create symlink
    symlink_command = f"ln -rs {existing_path} {new_path}"
    print(f"Running command: {symlink_command}")
    subprocess.run(symlink_command, shell=True, check=True)


# %%
