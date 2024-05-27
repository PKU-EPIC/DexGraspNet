import pathlib
from collections import Counter, defaultdict
from tqdm import tqdm
import subprocess

input_meshdata_path = pathlib.Path("/home/tylerlum/meshdata")
output_meshdata_path = pathlib.Path("/home/tylerlum/meshdata_by_category_v2")
assert input_meshdata_path.exists()
output_meshdata_path.mkdir(parents=True, exist_ok=False)

keywords_counter = Counter()
keyword_to_object_code_paths = defaultdict(set)

object_code_paths = list(input_meshdata_path.iterdir())
for object_code_path in tqdm(object_code_paths):
    object_code = object_code_path.name
    keywords = object_code.lower().replace("_", "-").split("-")
    keywords_counter.update(keywords)
    for keyword in keywords:
        keyword_to_object_code_paths[keyword].add(object_code_path)

print("!" * 80)
print(f"keywords_counter: {keywords_counter}")
print("!" * 80 + "\n")

MIN_COUNT = 4
keywords_counter = {k: v for k, v in keywords_counter.items() if v >= MIN_COUNT}
for keyword, count in tqdm(keywords_counter.items()):
    new_folder = output_meshdata_path / f"{count:04d}_{keyword}"
    new_folder.mkdir(parents=True, exist_ok=True)
    object_code_paths_with_keyword = list(keyword_to_object_code_paths[keyword])
    for object_code_path in object_code_paths_with_keyword:
        if object_code_path == (new_folder / object_code_path.name):
            print(f"Skipping {object_code_path} because it already exists in {new_folder}")
            continue

        symlink_command = f"cp -r {object_code_path} {new_folder / object_code_path.name}"
        subprocess.run(symlink_command, shell=True, check=True)