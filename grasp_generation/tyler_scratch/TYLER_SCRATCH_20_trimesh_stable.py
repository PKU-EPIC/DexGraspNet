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
# mesh_path = pathlib.Path("../data/rotated_meshdata_v2_by_category/0509_bottle/sem-Bottle-fa44223c6f785c60e71da2487cb2ee5b/coacd/decomposed.obj")
# mesh_path = pathlib.Path("../data/rotated_meshdata_v2_by_category/0509_bottle/core-bottle-1071fa4cddb2da2fc8724d5673a063a6/coacd/decomposed.obj")
# mesh_path = pathlib.Path("../data/rotated_meshdata_v2_by_category/0509_bottle/sem-Bottle-e8b48d395d3d8744e53e6e0633163da8/coacd/decomposed.obj")
mesh_path = pathlib.Path("../data/rotated_meshdata_v2_by_category/0509_bottle/sem-Bottle-2d3f7082ad6d293daf6843c094838b08/coacd/decomposed.obj")
assert mesh_path.exists()

# %%
mesh = trimesh.load_mesh(mesh_path)

# %%
# transforms, probs = trimesh.poses.compute_stable_poses(mesh, n_samples=10)
# print(f"transforms.shape = {transforms.shape}")
# print(f"probs.shape = {probs.shape}")

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
fig = go.Figure()

# Mesh
fig.add_trace(
    go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color='lightpink',
        opacity=0.50,
    )
)
fig.show()


# %%
# Transform mesh plot
fig = go.Figure()

# new_vertices = mesh.vertices @ x_rotation_transform[:3, :3].T + x_rotation_transform[:3, 3]
new_vertices = mesh.vertices @ y_rotation_transform_neg[:3, :3].T + y_rotation_transform_neg[:3, 3]
fig.add_trace(
    go.Mesh3d(
        x=new_vertices[:, 0],
        y=new_vertices[:, 1],
        z=new_vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color='lightpink',
        opacity=0.50,
    )
)
fig.show()

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
output_dir = pathlib.Path("2024-03-13_bottle_images_v3")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
USE_PLOTLY = False
from tqdm import tqdm
for mesh_file in tqdm(mesh_files):
    name = mesh_file.parent.parent.name
    mesh = trimesh.load_mesh(mesh_file)
    for transform, transform_name in zip(all_transforms, all_transform_names):
        if USE_PLOTLY:
            fig = go.Figure()
            new_vertices = mesh.vertices @ x_rotation_transform[:3, :3].T + x_rotation_transform[:3, 3]
            fig.add_trace(
                go.Mesh3d(
                    x=new_vertices[:, 0],
                    y=new_vertices[:, 1],
                    z=new_vertices[:, 2],
                    i=mesh.faces[:, 0],
                    j=mesh.faces[:, 1],
                    k=mesh.faces[:, 2],
                    color='lightpink',
                    opacity=0.50,
                )
            )
            fig.write_image(output_dir / f"{name}_{transform_name}.png")
        else:
            new_vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
            triang = Triangulation(new_vertices[:, 0], new_vertices[:, 1], mesh.faces)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(triang, new_vertices[:, 2], edgecolor='k', linewidth=0.5, antialiased=True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig(output_dir / f"{name}_{transform_name}.png")

# # Create a Triangulation object
# triang = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
# 
# # Plot the 3D mesh
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(triang, vertices[:, 2], edgecolor='k', linewidth=0.5, antialiased=True)
# 
# # Label the axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# 
# # Save the plot to a file
# plt.savefig('/mnt/data/trimesh_3d_plot.png')
# %%
