# %%
from utils.object_model import ObjectModel
import pathlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
batch_size = 1
meshdata_root_path = pathlib.Path("../data/meshdata")
nerf_meshdata_root_path = pathlib.Path("../../nerf_grasping/data/2023-11-17_01-27-23/nerf_meshdata_mugs_v9/")

object_code = "core-mug-1038e4eac0e18dcce02ae6d2a21d494a"
object_scale = 0.1

object_model = ObjectModel(
    meshdata_root_path=str(meshdata_root_path),
    batch_size_each=batch_size,
    num_samples=0,
    device="cuda",
)
object_model.initialize(object_code, object_scale)

nerf_object_model = ObjectModel(
    meshdata_root_path=str(nerf_meshdata_root_path),
    batch_size_each=batch_size,
    num_samples=0,
    device="cuda",
)
nerf_object_model.initialize(object_code, object_scale)

# %%
def create_fig_subplots(object_plotly_data: list, nerf_object_plotly_data: list) -> go.Figure:
    nrows, ncols = 1, 2
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "mesh3d"} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=["orig", "nerf"],
    )
    for d in object_plotly_data:
        fig.add_trace(d, row=1, col=1)
    for d in nerf_object_plotly_data:
        fig.add_trace(d, row=1, col=2)
    return fig

OBJECT_I = 0
assert 0 <= OBJECT_I < batch_size
object_plotly_data = object_model.get_plotly_data(
    i=OBJECT_I, color="lightgreen", opacity=0.5, with_surface_points=True
)
nerf_object_plotly_data = nerf_object_model.get_plotly_data(
    i=OBJECT_I, color="lightgreen", opacity=0.5, with_surface_points=True
)
create_fig_subplots(object_plotly_data, nerf_object_plotly_data).show()

# %%
# Get grid of points in [-0.1, 0.1]^3 then reshape to (N, 3), all 0.01 apart
import numpy as np
xyz_grid = np.mgrid[-0.1:0.1:0.01, -0.1:0.1:0.01, -0.1:0.1:0.01].reshape(3, -1).T

# %%
import torch
xyz_grid_torch = torch.from_numpy(xyz_grid).float().reshape(batch_size, -1, 3).clone().cuda().contiguous()
object_dists_interior_positive, object_normals, _ = object_model.cal_distance(xyz_grid_torch, with_closest_points=True)
nerf_object_dists_interior_positive, nerf_object_normals, _ = nerf_object_model.cal_distance(xyz_grid_torch, with_closest_points=True)

# %%
xyz_grid_torch.shape

# %%
object_dists_interior_positive.shape, object_normals.shape

# %%
nerf_object_dists_interior_positive.shape, nerf_object_normals.shape

# %%
object_inside_points = xyz_grid_torch[object_dists_interior_positive > 0].cpu()
object_outside_points = xyz_grid_torch[object_dists_interior_positive < 0].cpu()
fig = create_fig_subplots(object_plotly_data, nerf_object_plotly_data)
fig.add_trace(
    go.Scatter3d(
        x=object_inside_points[:, 0],
        y=object_inside_points[:, 1],
        z=object_inside_points[:, 2],
        mode="markers",
        marker=dict(size=1, color="blue"),
        name="inside",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=object_outside_points[:, 0],
        y=object_outside_points[:, 1],
        z=object_outside_points[:, 2],
        mode="markers",
        marker=dict(size=1, color="red"),
        name="outside",
    ),
    row=1,
    col=1,
)
nerf_object_inside_points = xyz_grid_torch[nerf_object_dists_interior_positive > 0].cpu()
nerf_object_outside_points = xyz_grid_torch[nerf_object_dists_interior_positive < 0].cpu()
fig.add_trace(
    go.Scatter3d(
        x=nerf_object_inside_points[:, 0],
        y=nerf_object_inside_points[:, 1],
        z=nerf_object_inside_points[:, 2],
        mode="markers",
        marker=dict(size=1, color="blue"),
        name="inside",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter3d(
        x=nerf_object_outside_points[:, 0],
        y=nerf_object_outside_points[:, 1],
        z=nerf_object_outside_points[:, 2],
        mode="markers",
        marker=dict(size=1, color="red"),
        name="outside",
    ),
    row=1,
    col=2,
)

fig.show()

# %%
# Randomly sample from inside points
N_RANDOM_POINTS = 3
random_points = xyz_grid_torch[:, np.random.choice(xyz_grid_torch.shape[1], N_RANDOM_POINTS, replace=False), :]
random_points

# %%
object_random_dists_interior_positive, object_random_normals, _ = object_model.cal_distance(random_points, with_closest_points=True)
nerf_object_random_dists_interior_positive, nerf_object_random_normals, _ = nerf_object_model.cal_distance(random_points, with_closest_points=True)

# %%
object_random_dists_interior_positive.shape, object_random_normals.shape

# %%
nerf_object_random_dists_interior_positive.shape, nerf_object_random_normals.shape

# %%
object_scaled_random_normals = object_random_normals * object_random_dists_interior_positive.unsqueeze(-1).abs()
nerf_object_scaled_random_normals = nerf_object_random_normals * nerf_object_random_dists_interior_positive.unsqueeze(-1).abs()
object_scaled_random_normals.shape, nerf_object_scaled_random_normals.shape

# %%
object_lines = torch.stack([random_points, random_points + object_scaled_random_normals], dim=-2).cpu()
nerf_object_lines = torch.stack([random_points, random_points + nerf_object_scaled_random_normals], dim=-2).cpu()
object_lines.shape, nerf_object_lines.shape

# %%
assert object_lines.shape == nerf_object_lines.shape == (batch_size, N_RANDOM_POINTS, 2, 3)

# %%
fig = create_fig_subplots(object_plotly_data, nerf_object_plotly_data)

# Create scatter3d line from random_points to random_points + scaled_normals
for i in range(N_RANDOM_POINTS):
    fig.add_trace(
        go.Scatter3d(
            x=object_lines[OBJECT_I, i, :, 0],
            y=object_lines[OBJECT_I, i, :, 1],
            z=object_lines[OBJECT_I, i, :, 2],
            mode="lines",
            marker=dict(size=1, color="red"),
            name=f"random_points_{i}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=nerf_object_lines[OBJECT_I, i, :, 0],
            y=nerf_object_lines[OBJECT_I, i, :, 1],
            z=nerf_object_lines[OBJECT_I, i, :, 2],
            mode="lines",
            marker=dict(size=1, color="red"),
            name=f"nerf_random_points_{i}",
        ),
        row=1,
        col=2,
    )
fig.show()
# %%
object_dists_interior_positive, nerf_object_dists_interior_positive
# %%
