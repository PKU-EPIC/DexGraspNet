# %%
import pathlib
import subprocess
from tqdm.notebook import tqdm
import json
from copy import deepcopy

# %%
path_str = "../data/2024-05-06_rotated_stable_grasps_0/nerfdata/"
path_bigger_str = (
    "../data/2024-05-06_rotated_stable_grasps_bigger_0/nerfdata/"
)
path_smaller_str = (
    "../data/2024-05-06_rotated_stable_grasps_smaller_0/nerfdata/"
)

paths = [pathlib.Path(path_str.replace("_0", f"_{i}")) for i in range(7)]
path_biggers = [pathlib.Path(path_bigger_str.replace("_0", f"_{i}")) for i in range(7)]
path_smallers = [
    pathlib.Path(path_smaller_str.replace("_0", f"_{i}")) for i in range(7)
]

all_paths = paths + path_biggers + path_smallers
for path in all_paths:
    assert path.exists()


# %%
all_paths[:10]

# %%
for path in tqdm(all_paths, desc="Processing paths"):
    assert path.stem == "nerfdata", path.stem

    # Make output dir
    output_path = path.parent / "nerfdata_100imgs"
    output_path.mkdir(exist_ok=True)

    object_paths = list(path.iterdir())
    for object_path in tqdm(object_paths, desc="Processing objects"):
        assert (object_path / "transforms.json").exists(), object_path / "transforms.json"
        assert (object_path / "images").exists(), object_path / "images"

        output_object_path = output_path / object_path.name
        output_object_path.mkdir(exist_ok=True)

        # Extract the first 100 frames from the transforms.json
        transforms = json.load(open(object_path / "transforms.json"))
        sorted_frames = sorted(transforms["frames"], key=lambda x: int(pathlib.Path(x["file_path"]).stem))

        assert len(sorted_frames) >= 100, len(sorted_frames)

        output_transforms = deepcopy(transforms)
        output_transforms["frames"] = sorted_frames[:100]

        # Write new transforms.json
        with open(output_object_path / "transforms.json", "w") as f:
            json.dump(output_transforms, f, indent=4)

        # Copy over the first 100 images
        (output_object_path / "images").mkdir(exist_ok=True)
        for i in range(100):
            img_path = object_path / "images" / f"{i}.png"
            assert img_path.exists(), img_path

            cp_command = f"cp {img_path} {output_object_path / 'images'}"
            subprocess.run(cp_command, shell=True, check=True)
# %%
