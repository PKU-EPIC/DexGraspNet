# %%
import pathlib
import random
path = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-04-16_rotated_grasps_aggregated_augmented_pose_HALTON_50/evaled_grasp_config_dicts_test")
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
