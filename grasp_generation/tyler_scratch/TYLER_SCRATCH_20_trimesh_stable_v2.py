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
all_transforms = [
    identity_transform,
    x_rotation_transform,
    x_rotation_transform_neg,
    y_rotation_transform,
    y_rotation_transform_neg,
]
all_transform_names = [
    "identity",
    "x_rotation",
    "x_rotation_neg",
    "y_rotation",
    "y_rotation_neg",
]

# %%
mesh_folder = pathlib.Path("../data/rotated_meshdata_v2_by_category/0509_bottle/")
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
output_dir = pathlib.Path("2024-03-22_bottle_images_v2")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
from tqdm import tqdm
for mesh_file in tqdm(mesh_files):
    name = mesh_file.parent.parent.name
    mesh = trimesh.load_mesh(mesh_file)

    fig = plt.figure(figsize=(15, 5))
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
        ax.set_title(transform_name)
    fig.tight_layout()
    plt.savefig(output_dir / f"{name}.png")