# %%
import pathlib
import random
path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_TUNED/raw_grasp_config_dicts")
assert path.exists()

# %%
object_paths = list(path.iterdir())
assert len(object_paths) > 0, object_paths
print(f"Found {len(object_paths)} object paths")

# %%
object_codes = random.sample([x.stem for x in object_paths], k=10)

# %%
print("Selected object codes: ")
print('\n'.join(object_codes))
# %%
