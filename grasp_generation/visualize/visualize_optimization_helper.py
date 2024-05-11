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
from typing import List


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
    if "visualization_freq" not in run.config:
        default_freq = 1
        print(
            f"WARNING: visualization_freq not in run config, defaulting to {default_freq}"
        )
        return default_freq
    return run.config["visualization_freq"]


## Shared ##
def get_scene_dict(bounds: Bounds3D):
    return dict(
        xaxis=dict(title="X", range=[bounds.x_min, bounds.x_max]),
        yaxis=dict(title="Y", range=[bounds.y_min, bounds.y_max]),
        zaxis=dict(title="Z", range=[bounds.z_min, bounds.z_max]),
        aspectratio=dict(x=bounds.x_range, y=bounds.y_range, z=bounds.z_range),
    )


def get_bounds(fig: go.Figure):
    # d.<x, y, z> may be empty, need to handle
    x_min = min([min(d.x) for d in fig.data if len(d.x) > 0])
    x_max = max([max(d.x) for d in fig.data if len(d.x) > 0])
    y_min = min([min(d.y) for d in fig.data if len(d.y) > 0])
    y_max = max([max(d.y) for d in fig.data if len(d.y) > 0])
    z_min = min([min(d.z) for d in fig.data if len(d.z) > 0])
    z_max = max([max(d.z) for d in fig.data if len(d.z) > 0])
    return Bounds3D(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
    )


def get_title(fig: go.Figure, idx: int, visualization_freq: int):
    try:
        original_title = fig.layout.title.text
    except AttributeError:
        original_title = ""

    return f"Grasp Optimization Step {idx * visualization_freq} {original_title}"


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
                fig=input_figs[FIG_TO_SHOW_FIRST],
                idx=FIG_TO_SHOW_FIRST,
                visualization_freq=visualization_freq,
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
            scene_camera=input_figs[FIG_TO_SHOW_FIRST].layout.scene.camera,
        ),
        frames=[
            go.Frame(
                data=fig.data,
                layout=go.Layout(
                    scene=get_scene_dict(bounds),
                    title=get_title(
                        fig=fig, idx=fig_idx, visualization_freq=visualization_freq
                    ),
                    showlegend=True,
                    # scene_camera=fig.layout.scene.camera,  # Actually doing this resets the camera every time, so changing frames is jarring, leave empty
                ),
                name=get_fig_name(fig_idx),  # Important to match with slider label
            )
            for fig_idx, fig in enumerate(input_figs)
        ],
    )
    return new_fig
