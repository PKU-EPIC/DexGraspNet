# %%
import trimesh
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation

# %%
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
mesh_folder = pathlib.Path("/juno/u/tylerlum/Downloads/meshdata")
assert mesh_folder.exists()

mesh_files = []
for object_code in mesh_folder.iterdir():
    if not object_code.is_dir():
        continue
    for mesh_file in object_code.rglob("decomposed.obj"):
        mesh_files.append(mesh_file)
print(f"Found {len(mesh_files)} mesh files")
assert len(mesh_files) > 0


# %%
output_dir = pathlib.Path("2024-04-04_mesh_images")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
mesh_files[:10]

# %%
from tqdm import tqdm
mesh_files = sorted(mesh_files)
pbar = tqdm(mesh_files)
for mesh_file in pbar:
    name = mesh_file.parent.parent.name
    pbar.set_description(name)
    mesh = trimesh.load_mesh(mesh_file)
    output_filename = output_dir / f"{name}.png"
    if output_filename.exists():
        continue

    fig = plt.figure(figsize=(25, 5))
    fig.suptitle(name)
    for i, (transform, transform_name) in enumerate(zip(all_transforms, all_transform_names)):
        new_vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        triang = Triangulation(new_vertices[:, 0], new_vertices[:, 1], mesh.faces)
        ax = fig.add_subplot(1, len(all_transforms), i + 1, projection='3d')
        ax.plot_trisurf(triang, new_vertices[:, 2], edgecolor='k', linewidth=0.5, antialiased=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis("equal")
        ax.set_title(f"{i}. {transform_name}")
    fig.tight_layout()
    plt.savefig(output_filename)

# %%
mesh_labels_filename = pathlib.Path("mesh_labels.txt")
assert mesh_labels_filename.exists()

mesh_labels = {}
# Each line of file is:
# <object_code>: <int label>
with open(mesh_labels_filename, "r") as f:
    for line in f:
        object_code, label = line.strip().split(": ")
        mesh_labels[object_code] = int(label)

# %%
output_mesh_dir = pathlib.Path("/juno/u/tylerlum/Downloads/rotated_meshdata")
assert output_mesh_dir.exists()

# %%
object_codes = list(output_mesh_dir.iterdir())
for object_code in tqdm(object_codes):
    assert object_code.name in mesh_labels
    label = mesh_labels[object_code.name]
    transform_for_z_up = all_transforms[label]
    transform_for_y_up = x_rotation_transform_neg

    obj_files = list(object_code.rglob("*.obj"))
    assert len(obj_files) > 0
    for obj_file in obj_files:
        mesh = trimesh.load_mesh(obj_file)
        mesh.apply_transform(transform_for_z_up)
        mesh.apply_transform(transform_for_y_up)
        mesh.export(obj_file)