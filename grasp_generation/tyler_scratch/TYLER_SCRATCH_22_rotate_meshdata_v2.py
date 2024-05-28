# %%
import trimesh
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
# from tqdm import tqdm
from tqdm.notebook import tqdm
from typing import List
from localscope import localscope

# %%
# Prepare transforms
identity_transform = np.eye(4)
x_rotation_transform = np.eye(4)
x_rotation_transform[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix()
x_rotation_transform_neg = np.eye(4)
x_rotation_transform_neg[:3, :3] = R.from_euler("x", -90, degrees=True).as_matrix()
y_rotation_transform = np.eye(4)
y_rotation_transform[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()
y_rotation_transform_neg = np.eye(4)
y_rotation_transform_neg[:3, :3] = R.from_euler("y", -90, degrees=True).as_matrix()
z_rotation_transform = np.eye(4)
z_rotation_transform[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
z_rotation_transform_neg = np.eye(4)
z_rotation_transform_neg[:3, :3] = R.from_euler("z", -90, degrees=True).as_matrix()
all_transforms = [
    identity_transform,
    x_rotation_transform,
    x_rotation_transform_neg,
    y_rotation_transform,
    y_rotation_transform_neg,
    z_rotation_transform,
    z_rotation_transform_neg,
]
all_transform_names = [
    "identity",
    "x_rotation",
    "x_rotation_neg",
    "y_rotation",
    "y_rotation_neg",
    "z_rotation",
    "z_rotation_neg",
]

# %%
@localscope.mfc
def save_img(mesh_file: pathlib.Path, output_filename: pathlib.Path, transforms: List[np.ndarray], transform_names: List[str]):
    mesh = trimesh.load_mesh(mesh_file)
    fig = plt.figure(figsize=(25, 5))
    fig.suptitle(mesh_file.parent.name)
    for i, (transform, transform_name) in enumerate(zip(transforms, transform_names)):
        new_vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        triang = Triangulation(new_vertices[:, 0], new_vertices[:, 1], mesh.faces)
        ax = fig.add_subplot(1, len(transforms), i + 1, projection='3d')
        ax.plot_trisurf(triang, new_vertices[:, 2], edgecolor='k', linewidth=0.5, antialiased=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis("equal")
        ax.set_title(f"{i}. {transform_name}")
    fig.tight_layout()
    plt.savefig(output_filename)


# %%
mesh_by_category_folder = pathlib.Path("/home/tylerlum/meshdata_by_category")
assert mesh_by_category_folder.exists()

output_imgs_folder = pathlib.Path("/home/tylerlum/mesh_images_by_category")
output_imgs_folder.mkdir(parents=True, exist_ok=True)

sorted_mesh_by_category_folder = sorted(mesh_by_category_folder.iterdir(), reverse=True)
for category_folder in tqdm(sorted_mesh_by_category_folder):
    if not category_folder.is_dir():
        continue
    (output_imgs_folder / category_folder.name).mkdir(parents=True, exist_ok=True)

    sorted_object_code_folders = sorted(category_folder.iterdir())
    for object_code_folder in tqdm(sorted_object_code_folders):
        if not object_code_folder.is_dir():
            continue
        for mesh_file in object_code_folder.rglob("decomposed.obj"):
            output_filename = output_imgs_folder / category_folder.name / f"{object_code_folder.name}.png"
            if output_filename.exists():
                continue
            save_img(mesh_file=mesh_file, output_filename=output_filename, transforms=all_transforms, transform_names=all_transform_names)

# %%
mesh_labels_filename = pathlib.Path("/home/tylerlum/mesh_labels.txt")
assert mesh_labels_filename.exists()

mesh_labels = {}
# Each line of file is:
# <object_code>: <int label>
with open(mesh_labels_filename, "r") as f:
    for line in f:
        try:
            object_code, label = line.strip().split(": ")
            mesh_labels[object_code] = int(label)
        except:
            print(f"Error parsing line: {line}")

# %%
output_mesh_dir = pathlib.Path("/home/tylerlum/rotated_meshdata_by_category")
assert output_mesh_dir.exists()

# %%
category_folders = sorted(list(output_mesh_dir.iterdir()))
for category_folder in tqdm(category_folders):
    if not category_folder.is_dir():
        continue

    object_code_folders = sorted(list(category_folder.iterdir()))
    for object_code_folder in tqdm(object_code_folders):
        if not object_code_folder.is_dir():
            continue
        assert object_code_folder.name in mesh_labels, f"{object_code_folder.name} not in mesh_labels"

# %%
category_folders = sorted(list(output_mesh_dir.iterdir()))
for category_folder in tqdm(category_folders):
    if not category_folder.is_dir():
        continue

    object_code_folders = sorted(list(category_folder.iterdir()))
    for object_code_folder in tqdm(object_code_folders):
        if not object_code_folder.is_dir():
            continue
        assert object_code_folder.name in mesh_labels, f"{object_code_folder.name} not in mesh_labels"

        label = mesh_labels[object_code_folder.name]
        transform_for_z_up = all_transforms[label]
        transform_for_y_up = x_rotation_transform_neg

        obj_files = list(object_code_folder.rglob("*.obj"))
        assert len(obj_files) > 0
        for obj_file in obj_files:
            mesh = trimesh.load_mesh(obj_file)
            mesh.apply_transform(transform_for_z_up)
            mesh.apply_transform(transform_for_y_up)
            mesh.export(obj_file)
# %%
rotated_mesh_by_category_folder = pathlib.Path("/home/tylerlum/rotated_meshdata_by_category")
assert rotated_mesh_by_category_folder.exists()

rotated_output_imgs_folder = pathlib.Path("/home/tylerlum/rotated_mesh_images_by_category")
rotated_output_imgs_folder.mkdir(parents=True, exist_ok=True)

sorted_rotated_mesh_by_category_folder = sorted(rotated_mesh_by_category_folder.iterdir(), reverse=True)
for category_folder in tqdm(sorted_rotated_mesh_by_category_folder):
    if not category_folder.is_dir():
        continue
    (rotated_output_imgs_folder / category_folder.name).mkdir(parents=True, exist_ok=True)

    sorted_object_code_folders = sorted(category_folder.iterdir())
    for object_code_folder in tqdm(sorted_object_code_folders):
        if not object_code_folder.is_dir():
            continue
        for mesh_file in object_code_folder.rglob("decomposed.obj"):
            output_filename = rotated_output_imgs_folder / category_folder.name / f"{object_code_folder.name}.png"
            if output_filename.exists():
                continue
            save_img(mesh_file=mesh_file, output_filename=output_filename, transforms=all_transforms, transform_names=all_transform_names)
# %%
