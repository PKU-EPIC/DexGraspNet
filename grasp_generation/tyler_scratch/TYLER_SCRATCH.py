# %%
import trimesh
import pathlib
from tqdm import tqdm

path = pathlib.Path("../data/rotated_meshdata_v2")
assert path.exists()

# %%
MAIN_min_x, MAIN_min_y, MAIN_min_z = None, None, None
MAIN_max_x, MAIN_max_y, MAIN_max_z = None, None, None

filepaths = list(path.rglob("*.obj"))
for filepath in tqdm(filepaths):
    try:
        mesh = trimesh.load(filepath)
        mins, maxs = mesh.bounds
        min_x, min_y, min_z = mins
        max_x, max_y, max_z = maxs
        if MAIN_min_x is None:
            MAIN_min_x = min_x
            MAIN_min_y = min_y
            MAIN_min_z = min_z
            MAIN_max_x = max_x
            MAIN_max_y = max_y
            MAIN_max_z = max_z
        else:
            MAIN_min_x = min(MAIN_min_x, min_x)
            MAIN_min_y = min(MAIN_min_y, min_y)
            MAIN_min_z = min(MAIN_min_z, min_z)
            MAIN_max_x = max(MAIN_max_x, max_x)
            MAIN_max_y = max(MAIN_max_y, max_y)
            MAIN_max_z = max(MAIN_max_z, max_z)
        print(f"{MAIN_min_x} - {MAIN_max_x}, {MAIN_min_y} - {MAIN_max_y}, {MAIN_min_z} - {MAIN_max_z}")
    except:
        print(f"filepath = {filepath} failed")

# %%
print(f"MAIN_min_x = {MAIN_min_x}")
print(f"MAIN_min_y = {MAIN_min_y}")
print(f"MAIN_min_z = {MAIN_min_z}")
print(f"MAIN_max_x = {MAIN_max_x}")
print(f"MAIN_max_y = {MAIN_max_y}")
print(f"MAIN_max_z = {MAIN_max_z}")

# %%
