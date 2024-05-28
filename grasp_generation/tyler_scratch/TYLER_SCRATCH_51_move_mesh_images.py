# %%
import pathlib
import subprocess
from tqdm import tqdm

# %%
path = pathlib.Path("/home/tylerlum/mesh_images_by_category/")
output_path = pathlib.Path("/home/tylerlum/mesh_images")
assert path.exists()

# %%
category_paths = [p for p in path.iterdir() if p.is_dir()]
for category_path in tqdm(category_paths):
    subprocess.run(f"cp {category_path}/* {output_path}", shell=True, check=True)
# %%
