"""
Last modified date: 2023.07.01
Author: Tyler Lum
Description: visualize hand model grasp optimization
"""

import os
import sys
from dataclasses import dataclass

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath("."))

import plotly.graph_objects as go
import plotly
from utils.seed import set_seed
import wandb
from tqdm import tqdm
from datetime import datetime
from typing import List
from tap import Tap

# Get path to this file
path_to_this_file = os.path.dirname(os.path.realpath(__file__))

set_seed(1)

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


class VisualizeOptimizationArgumentParser(Tap):
    wandb_entity: str = "tylerlum"
    wandb_project: str = "DexGraspNet_v1"
    run_id: str = "drv5njep"
    max_files_to_read: int = 100
    frame_duration: int = 200
    transition_duration: int = 100


def download_plotly_files(run_path: str):
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


def get_visualization_freq(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    if not "visualization_freq" in run.config:
        default_freq = 1
        print(
            f"WARNING: visualization_freq not in run config, defaulting to {default_freq}"
        )
        return default_freq
    return run.config["visualization_freq"]


def get_scene_dict(bounds: Bounds3D):
    return dict(
        xaxis=dict(title="X", range=[bounds.x_min, bounds.x_max]),
        yaxis=dict(title="Y", range=[bounds.y_min, bounds.y_max]),
        zaxis=dict(title="Z", range=[bounds.z_min, bounds.z_max]),
        aspectmode="cube",
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


def main(args: VisualizeOptimizationArgumentParser):
    # Specify run
    run_path = f"{args.wandb_entity}/{args.wandb_project}/{args.run_id}"
    print(f"Run path: {run_path}")

    # Download plotly files
    plotly_file_paths = download_plotly_files(run_path)
    visualization_freq = get_visualization_freq(run_path)

    # Read in json files
    plotly_file_paths = plotly_file_paths[: args.max_files_to_read]
    orig_figs = [
        plotly.io.read_json(file=plotly_file_path)
        for plotly_file_path in plotly_file_paths
    ]

    # Get bounds
    assert len(orig_figs) > 0, "No files read"
    bounds = get_bounds(orig_figs[0])
    for i in range(1, len(orig_figs)):
        bounds = bounds.max_bounds(get_bounds(orig_figs[i]))

    # Will create slider with each step being a frame
    # Each frame is one of the orig figs, which is a single optimization step
    FIG_TO_SHOW_FIRST = 0
    REDRAW = True  # Needed for animation to work with 3D: https://github.com/plotly/plotly.js/issues/1221
    slider_steps = [
        dict(
            args=[
                [
                    get_fig_name(fig_idx)
                ],  # Draw this one frame: https://github.com/plotly/plotly.js/issues/1221
                {
                    "frame": {"duration": args.frame_duration, "redraw": REDRAW},
                    "mode": "immediate",
                    "transition": {"duration": args.transition_duration},
                },
            ],
            label=get_fig_name(fig_idx),
            method="animate",
        )
        for fig_idx in range(len(orig_figs))
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
        transition=dict(duration=args.transition_duration, easing="cubic-in-out"),
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
                "frame": {"duration": args.frame_duration, "redraw": REDRAW},
                "fromcurrent": True,
                "transition": {
                    "duration": args.transition_duration,
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
        data=orig_figs[FIG_TO_SHOW_FIRST].data,
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
            for fig_idx, fig in enumerate(orig_figs)
        ],
    )
    new_fig.show()


if __name__ == "__main__":
    args = VisualizeOptimizationArgumentParser().parse_args()
    main(args)
