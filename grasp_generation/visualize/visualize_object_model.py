"""
Last modified date: 2023.08.24
Author: Tyler Lum
Description: visualize object model using plotly.graph_objects
"""

import os
import sys
import pathlib

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath("."))

import torch
import plotly.graph_objects as go
from utils.object_model import ObjectModel
from utils.seed import set_seed
from tap import Tap


set_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class VisualizeObjectModelArgumentParser(Tap):
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_stable")
    object_code: str = "sem-Mug-10f6e09036350e92b3f21f1137c3c347"
    object_scale: float = 0.1


if __name__ == "__main__":
    device = torch.device("cpu")
    args = VisualizeObjectModelArgumentParser().parse_args()

    # object model
    object_model = ObjectModel(
        meshdata_root_path=str(args.meshdata_root_path),
        batch_size_each=1,
        scale=args.object_scale,
        num_samples=2000,
        device="cpu",
    )
    object_model.initialize([args.object_code])

    # visualize
    object_plotly = object_model.get_plotly_data(
        i=0,
        with_surface_points=True,
    )
    fig = go.Figure(
        data=object_plotly,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title=f"Object Model: {args.object_code}",
        ),
    )
    fig.update_layout(scene_aspectmode="data")
    fig.show()
