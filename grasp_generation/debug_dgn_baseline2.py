# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import trimesh
import pathlib

# %%
meshdata_root_path = pathlib.Path("../data/meshdata")
nerf_meshdata_root_path = pathlib.Path("../../nerf_grasping/data/2023-11-17_01-27-23/nerf_meshdata_mugs_v8/")

object_code = "core-mug-1038e4eac0e18dcce02ae6d2a21d494a"

meshdata_filepath = meshdata_root_path / object_code / "coacd" / "decomposed.obj"
nerf_meshdata_filepath = nerf_meshdata_root_path / object_code / "coacd" / "decomposed.obj"

orig_mesh = trimesh.load(str(meshdata_filepath))
nerf_mesh = trimesh.load(str(nerf_meshdata_filepath))

# %%

def create_mesh_3d(
    vertices: np.ndarray, faces: np.ndarray, opacity: float = 1.0
) -> go.Mesh3d:
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
    )
# %%
nrows, ncols = 1, 2
fig = make_subplots(
    rows=nrows,
    cols=ncols,
    specs=[[{"type": "mesh3d"} for _ in range(ncols)] for _ in range(nrows)],
    subplot_titles=["orig", "nerf"],
)

fig.add_trace(
    create_mesh_3d(orig_mesh.vertices, orig_mesh.faces, opacity=0.3),
    row=1,
    col=1,
)
fig.add_trace(
    create_mesh_3d(nerf_mesh.vertices, nerf_mesh.faces, opacity=0.3),
    row=1,
    col=2,
)

# %%
print(f"orig_mesh.is_watertight: {orig_mesh.is_watertight}")
print(f"nerf_mesh.is_watertight: {nerf_mesh.is_watertight}")
print(f"orig_mesh.bounding_box.extents: {orig_mesh.bounding_box.extents}")
print(f"nerf_mesh.bounding_box.extents: {nerf_mesh.bounding_box.extents}")
# %%
from trimesh.proximity import signed_distance

orig_points = orig_mesh.bounding_box.sample_grid(step=0.1)
orig_sds = signed_distance(mesh=orig_mesh, points=orig_points)

# %%
orig_sds

# %%
nerf_sds = signed_distance(mesh=nerf_mesh, points=orig_points)

# %%
nerf_sds

# %%
inside_points = orig_points[orig_sds < 0]
outside_points = orig_points[orig_sds > 0]

nerf_inside_points = orig_points[nerf_sds < 0]
nerf_outside_points = orig_points[nerf_sds > 0]

# %%
fig.add_trace(
    go.Scatter3d(
        x=inside_points[:, 0],
        y=inside_points[:, 1],
        z=inside_points[:, 2],
        mode="markers",
        marker=dict(color="red", size=1),
        name="inside_points",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=outside_points[:, 0],
        y=outside_points[:, 1],
        z=outside_points[:, 2],
        mode="markers",
        marker=dict(color="blue", size=1),
        name="outside_points",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=nerf_inside_points[:, 0],
        y=nerf_inside_points[:, 1],
        z=nerf_inside_points[:, 2],
        mode="markers",
        marker=dict(color="red", size=1),
        name="nerf_inside_points",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter3d(
        x=nerf_outside_points[:, 0],
        y=nerf_outside_points[:, 1],
        z=nerf_outside_points[:, 2],
        mode="markers",
        marker=dict(color="blue", size=1),
        name="nerf_outside_points",
    ),
    row=1,
    col=2,
)

fig.show()

# %%
POINTS_TO_CHECK = np.array([[0.245586, -0.035945, -0.650089],
                            [0.445, 0.564, 0.749],
                            [0,0,0],
                            [0.145, -0.13, -0.75],
                            ]).reshape(-1, 3)
DISTS_MESH = signed_distance(mesh=orig_mesh, points=POINTS_TO_CHECK)
DISTS_NERF = signed_distance(mesh=nerf_mesh, points=POINTS_TO_CHECK)

dists_mesh = np.abs(DISTS_MESH)
dists_nerf = np.abs(DISTS_NERF)
insides_mesh = DISTS_MESH < 0
insides_nerf = DISTS_NERF < 0

# %%
def sample_spherical(npoints: int):
    DIM = 3
    vec = np.random.randn(npoints, DIM)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec

from typing import List
def create_scatter_3d_sphere(
    xyz_list: np.ndarray, radius_list: np.ndarray, colors: List[str], names: List[str]
) -> List[go.Scatter3d]:
    N = xyz_list.shape[0]
    assert xyz_list.shape == (N, 3)
    assert radius_list.shape == (N,)
    assert len(colors) == N
    assert len(names) == N

    N_PTS_PER_SPHERE = 1000
    centered_sphere_points = sample_spherical(N_PTS_PER_SPHERE * N).reshape(N, N_PTS_PER_SPHERE, 3)
    sphere_points = centered_sphere_points * radius_list.reshape(N, 1, 1) + xyz_list.reshape(N, 1, 3)
    scatters = []
    for i in range(N):
        scatters.append(
            go.Scatter3d(
                x=sphere_points[i, :, 0],
                y=sphere_points[i, :, 1],
                z=sphere_points[i, :, 2],
                mode="markers",
                marker=dict(color=colors[i], size=1),
                name=names[i],
            )
        )
    return scatters


scatters = create_scatter_3d_sphere(
        xyz_list=POINTS_TO_CHECK,
        radius_list=dists_mesh,
        colors=["red" if inside_mesh else "blue" for inside_mesh in insides_mesh],
        names=[f"dists_mesh_{i}" for i in range(len(POINTS_TO_CHECK))],
    )
for scatter in scatters:
    fig.add_trace(scatter, row=1, col=1)

scatters = create_scatter_3d_sphere(
        xyz_list=POINTS_TO_CHECK,
        radius_list=dists_nerf,
        colors=["red" if inside_nerf else "blue" for inside_nerf in insides_nerf],
        names=[f"dists_nerf_{i}" for i in range(len(POINTS_TO_CHECK))],
    )
for scatter in scatters:
    fig.add_trace(scatter, row=1, col=2)
fig.show()

# %%
DISTS_MESH
# %%
DISTS_NERF
# %%
