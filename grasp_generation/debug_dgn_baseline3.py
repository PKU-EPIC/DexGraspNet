# %%
from utils.object_model import ObjectModel
import pathlib
import plotly.graph_objects as go

# %%
batch_size = 1
meshdata_root_path = pathlib.Path("../data/meshdata")
nerf_meshdata_root_path = pathlib.Path("../../nerf_grasping/data/2023-11-17_01-27-23/nerf_meshdata_mugs_v8/")

object_code = "core-mug-1038e4eac0e18dcce02ae6d2a21d494a"

object_model = ObjectModel(
    # meshdata_root_path=str(meshdata_root_path),
    meshdata_root_path=str(nerf_meshdata_root_path),
    batch_size_each=batch_size,
    num_samples=0,
    device="cuda",
)
object_model.initialize(object_code, 0.1)
# %%
plotly = object_model.get_plotly_data(
    i=0, color="lightgreen", opacity=0.5, with_surface_points=True
)
fig = go.Figure(data=plotly)
fig.show()

# %%
# Get grid of points in [-0.1, 0.1]^3 then reshape to (N, 3), all 0.01 apart
import numpy as np
xyz_grid = np.mgrid[-0.1:0.1:0.01, -0.1:0.1:0.01, -0.1:0.1:0.01].reshape(3, -1).T

# %%
xyz_grid.shape
fig.add_trace(
    go.Scatter3d(
        x=xyz_grid[:, 0],
        y=xyz_grid[:, 1],
        z=xyz_grid[:, 2],
        mode="markers",
        marker=dict(size=1, color="red"),
    )
)
fig.show()

# %%
import torch
xyz_grid_torch = torch.from_numpy(xyz_grid).float().reshape(batch_size, -1, 3).clone().cuda().contiguous()
dists, normals, _ = object_model.cal_distance(xyz_grid_torch, with_closest_points=True)

# %%
xyz_grid_torch.shape

# %%
dists.shape, normals.shape

# %%
fig = go.Figure()
inside_points = xyz_grid_torch[dists < 0].cpu()
outside_points = xyz_grid_torch[dists >= 0].cpu()
fig.add_trace(
    go.Scatter3d(
        x=inside_points[:, 0],
        y=inside_points[:, 1],
        z=inside_points[:, 2],
        mode="markers",
        marker=dict(size=1, color="blue"),
    )
)
fig.add_trace(
    go.Scatter3d(
        x=outside_points[:, 0],
        y=outside_points[:, 1],
        z=outside_points[:, 2],
        mode="markers",
        marker=dict(size=1, color="red"),
    )
)
for d in plotly:
    fig.add_trace(d)
fig.show()



# %%
