"""
Last modified date: 2023.07.01
Author: Tyler Lum
Description: visualize hand model grasp optimization
"""

import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.realpath("."))

import plotly.graph_objects as go
import wandb
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple

import torch
import numpy as np
from utils.qpos_pose_conversion import qpos_to_pose
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
)
from utils.hand_model import HandModel
from utils.object_model import ObjectModel

path_to_this_file = os.path.dirname(os.path.realpath(__file__))

# Need this to play all: https://github.com/plotly/plotly.js/issues/1221
PLAY_BUTTON_ARG = None

# Need this for pause button: https://github.com/plotly/plotly.js/issues/1221
PAUSE_BUTTON_ARG = [None]


@dataclass
class Bounds3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def max_bounds(self, other):
        assert isinstance(other, Bounds3D)
        return Bounds3D(
            x_min=min(self.x_min, other.x_min),
            x_max=max(self.x_max, other.x_max),
            y_min=min(self.y_min, other.y_min),
            y_max=max(self.y_max, other.y_max),
            z_min=min(self.z_min, other.z_min),
            z_max=max(self.z_max, other.z_max),
        )

    @property
    def x_range(self):
        return self.x_max - self.x_min

    @property
    def y_range(self):
        return self.y_max - self.y_min

    @property
    def z_range(self):
        return self.z_max - self.z_min


## From Wandb ##
def download_plotly_files_from_wandb(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)

    # Store in folder
    folder_path = os.path.join(path_to_this_file, "wandb_files", run_path)
    os.makedirs(folder_path, exist_ok=True)

    unsorted_plotly_files = [f for f in run.files() if "plotly" in f.name]
    sorted_plotly_files = sorted(
        unsorted_plotly_files, key=lambda f: datetime.fromisoformat(f.updated_at)
    )
    for f in tqdm(sorted_plotly_files, desc="Downloading plotly files"):
        f.download(root=folder_path, exist_ok=True)
    print(f"Got {len(sorted_plotly_files)} files")

    plotly_file_paths = [os.path.join(folder_path, f.name) for f in sorted_plotly_files]
    print(f"First files: {plotly_file_paths[:3]}")
    return plotly_file_paths


def get_visualization_freq_from_wandb(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    if not "visualization_freq" in run.config:
        default_freq = 1
        print(
            f"WARNING: visualization_freq not in run config, defaulting to {default_freq}"
        )
        return default_freq
    return run.config["visualization_freq"]


## From Folder ##


def create_grasp_fig(
    hand_model: HandModel, object_model: ObjectModel, idx_to_visualize: int
) -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title="Grasp Visualization",
        )
    )
    plots = [
        *hand_model.get_plotly_data(
            i=idx_to_visualize,
            opacity=1.0,
            with_contact_points=False,  # No contact points after optimization
            with_contact_candidates=True,
        ),
        *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    ]
    for plot in plots:
        fig.add_trace(plot)
    return fig


def get_hand_and_object_model_from_data_dict(
    data_dict: np.ndarray, object_code: str
) -> Tuple[HandModel, ObjectModel]:
    HAND_MODEL_TYPE = HandModelType.ALLEGRO_HAND
    MESH_PATH = "../data/meshdata"

    joint_names = handmodeltype_to_joint_names[HAND_MODEL_TYPE]
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_pose_array = []
    for i in range(batch_size):
        qpos = data_dict[i]["qpos"]
        hand_pose_array.append(
            qpos_to_pose(qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False)
        )
        scale = data_dict[i]["scale"]
        scale_array.append(scale)

    GPU = 0
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    batch_size = len(hand_pose_array)

    # hand model
    hand_model = HandModel(hand_model_type=HAND_MODEL_TYPE, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # object model
    object_model = ObjectModel(
        data_root_path=MESH_PATH,
        batch_size_each=batch_size,
        num_samples=0,
        device=device,
    )
    object_model.initialize(object_code)
    object_model.object_scale_tensor = (
        torch.tensor(scale_array).reshape(1, -1).to(device)
    )  # 1 because 1 object code
    return hand_model, object_model


## Shared ##
def get_scene_dict(bounds: Bounds3D):
    return dict(
        xaxis=dict(title="X", range=[bounds.x_min, bounds.x_max]),
        yaxis=dict(title="Y", range=[bounds.y_min, bounds.y_max]),
        zaxis=dict(title="Z", range=[bounds.z_min, bounds.z_max]),
        aspectratio=dict(x=bounds.x_range, y=bounds.y_range, z=bounds.z_range),
    )


def get_bounds(fig: go.Figure):
    x_min = min([min(d.x) for d in fig.data])
    x_max = max([max(d.x) for d in fig.data])
    y_min = min([min(d.y) for d in fig.data])
    y_max = max([max(d.y) for d in fig.data])
    z_min = min([min(d.z) for d in fig.data])
    z_max = max([max(d.z) for d in fig.data])
    return Bounds3D(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
    )


def get_title(idx: int, visualization_freq: int):
    return f"Grasp Optimization Step {idx * visualization_freq}"


def get_fig_name(idx: int):
    return f"Fig {idx}"


def create_figure_with_buttons_and_slider(
    input_figs: List[go.Figure],
    visualization_freq: int,
    frame_duration: int,
    transition_duration: int,
) -> go.Figure:
    # Get bounds
    assert len(input_figs) > 0, "No files read"
    bounds = get_bounds(input_figs[0])
    for i in range(1, len(input_figs)):
        bounds = bounds.max_bounds(get_bounds(input_figs[i]))

    # Will create slider with each step being a frame
    # Each frame is one of the input figs, which is a single optimization step
    FIG_TO_SHOW_FIRST = 0
    REDRAW = True  # Needed for animation to work with 3D: https://github.com/plotly/plotly.js/issues/1221
    slider_steps = [
        dict(
            args=[
                [
                    get_fig_name(fig_idx)
                ],  # Draw this one frame: https://github.com/plotly/plotly.js/issues/1221
                {
                    "frame": {"duration": frame_duration, "redraw": REDRAW},
                    "mode": "immediate",
                    "transition": {"duration": transition_duration},
                },
            ],
            label=get_fig_name(fig_idx),
            method="animate",
        )
        for fig_idx in range(len(input_figs))
    ]
    sliders_dict = dict(
        active=FIG_TO_SHOW_FIRST,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=12),
            prefix="Step: ",  # Prefix for the slider label
            xanchor="right",
            visible=True,
        ),
        transition=dict(duration=transition_duration, easing="cubic-in-out"),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=slider_steps,
    )

    play_button_dict = dict(
        label="Play",
        method="animate",
        args=[
            PLAY_BUTTON_ARG,
            {
                "frame": {"duration": frame_duration, "redraw": REDRAW},
                "fromcurrent": True,
                "transition": {
                    "duration": transition_duration,
                    "easing": "quadratic-in-out",
                },
            },
        ],
    )
    PAUSE_BUTTON_REDRAW = False
    PAUSE_BUTTON_DURATION = 0
    pause_button_dict = dict(
        label="Pause",
        method="animate",
        args=[
            PAUSE_BUTTON_ARG,
            {
                "frame": {
                    "duration": PAUSE_BUTTON_DURATION,
                    "redraw": PAUSE_BUTTON_REDRAW,
                },
                "mode": "immediate",
                "transition": {"duration": PAUSE_BUTTON_DURATION},
            },
        ],
    )
    new_fig = go.Figure(
        data=input_figs[FIG_TO_SHOW_FIRST].data,
        layout=go.Layout(
            scene=get_scene_dict(bounds),
            title=get_title(
                idx=FIG_TO_SHOW_FIRST, visualization_freq=visualization_freq
            ),
            showlegend=True,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[play_button_dict, pause_button_dict],
                    direction="left",
                    pad={"r": 10, "t": 65},
                    showactive=False,
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top",
                ),
            ],
            sliders=[sliders_dict],
        ),
        frames=[
            go.Frame(
                data=fig.data,
                layout=go.Layout(
                    scene=get_scene_dict(bounds),
                    title=get_title(idx=fig_idx, visualization_freq=visualization_freq),
                    showlegend=True,
                ),
                name=get_fig_name(fig_idx),  # Important to match with slider label
            )
            for fig_idx, fig in enumerate(input_figs)
        ],
    )
    return new_fig
