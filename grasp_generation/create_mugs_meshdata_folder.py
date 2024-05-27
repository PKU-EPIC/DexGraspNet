import subprocess
import pathlib

input_meshdata_folder = pathlib.Path("../data/rotated_meshdata_v2").absolute()
output_meshdata_folder = pathlib.Path("../data/2023-11-17_meshdata_mugs").absolute()
mug_folder_names = [
    "core-mug-1038e4eac0e18dcce02ae6d2a21d494a",
    "core-mug-10f6e09036350e92b3f21f1137c3c347",
    "core-mug-127944b6dabee1c9e20e92c5b8147e4a",
    "core-mug-128ecbc10df5b05d96eaf1340564a4de",
    "core-mug-1305b9266d38eb4d9f818dd0aa1a251",
    "core-mug-141f1db25095b16dcfb3760e4293e310",
    "core-mug-159e56c18906830278d8f8c02c47cde0",
    "core-mug-15bd6225c209a8e3654b0ce7754570c8",
    "core-mug-162201dfe14b73f0281365259d1cf342",
    "core-mug-17952a204c0a9f526c69dceb67157a66",
]

assert input_meshdata_folder.exists()
assert not output_meshdata_folder.exists()
for mug_folder_name in mug_folder_names:
    assert (input_meshdata_folder / mug_folder_name).exists()

output_meshdata_folder.mkdir()

for mug_folder_name in mug_folder_names:
    subprocess.run(
        f"ln -s {str(input_meshdata_folder / mug_folder_name)} {str(output_meshdata_folder / mug_folder_name)}",
        shell=True,
        check=True,
    )
