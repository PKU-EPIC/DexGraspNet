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


class VisualizeOptimizationArgumentParser(Tap):
    wandb_entity: str = "tylerlum"
    wandb_project: str = "DexGraspNet_v1"
    run_id: str = "drv5njep"
    max_files_to_read: int = 100


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

@dataclass
class Bounds3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

def get_bounds(figs: List[go.Figure]):
    x_min = min([min(d.x) for fig in figs for d in fig.data])
    x_max = max([max(d.x) for fig in figs for d in fig.data])
    y_min = min([min(d.y) for fig in figs for d in fig.data])
    y_max = max([max(d.y) for fig in figs for d in fig.data])
    z_min = min([min(d.z) for fig in figs for d in fig.data])
    z_max = max([max(d.z) for fig in figs for d in fig.data])
    return Bounds3D(x_min, x_max, y_min, y_max, z_min, z_max)

def main(args: VisualizeOptimizationArgumentParser):
    # Specify run
    run_path = f"{args.wandb_entity}/{args.wandb_project}/{args.run_id}"
    print(f"Run path: {run_path}")

    # Get files from wandb
    plotly_file_paths = download_plotly_files(run_path)

    # Read in json files
    plotly_file_paths = plotly_file_paths[: args.max_files_to_read]
    orig_figs = [
        plotly.io.read_json(file=plotly_file_path)
        for plotly_file_path in plotly_file_paths
    ]
    bounds = get_bounds(orig_figs)

    FIG_TO_SHOW_FIRST = 0

    slider_steps = [
        dict(
            args=[
                [fig_idx],
                {
                    # "frame": {"duration": 1000, "redraw": False},
                    # "mode": "immediate",
                    # "transition": {"duration": 300},
                },
            ],
            label=fig_idx,
            method="animate",
        )
        for fig_idx, f in enumerate(orig_figs)
    ]
    sliders_dict = dict(
        active=FIG_TO_SHOW_FIRST,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=12),
            prefix="Optimization Iter",  # Prefix for the slider label
            xanchor="right",
            visible=True,
        ),
        # transition=dict(duration=300, easing="cubic-in-out"),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=slider_steps,
    )

    play_button_dict = dict(
        label="Play From Start",
        method="animate",
        args=[
            None,
            {
                # "frame": {"duration": 1000, "redraw": False},
                # "fromcurrent": True,
                # "transition": {"duration": 1000, "easing": "quadratic-in-out"},
            },
        ],
    )
    pause_button_dict = dict(
        label="Pause",
        method="animate",
        args=[
            None,
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
    )
    new_fig = go.Figure(
        data=orig_figs[FIG_TO_SHOW_FIRST].data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X", range=[bounds.x_min, bounds.x_max]),
                yaxis=dict(title="Y", range=[bounds.y_min, bounds.y_max]),
                zaxis=dict(title="Z", range=[bounds.z_min, bounds.z_max]),
                aspectmode="cube",
            ),
            showlegend=True,
            title="new_fig",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[play_button_dict, pause_button_dict],
                    # direction="left",
                    # pad={"r": 10, "t": 87},
                    # showactive=False,
                    # x=0.1,
                    # y=0,
                    # xanchor="right",
                    # yanchor="top",
                ),
            ],
            sliders=[sliders_dict],
        ),
        frames=[
            go.Frame(
                data=fig.data,
                layout=go.Layout(
                    scene=dict(
                        xaxis=dict(title="X", range=[bounds.x_min, bounds.x_max]),
                        yaxis=dict(title="Y", range=[bounds.y_min, bounds.y_max]),
                        zaxis=dict(title="Z", range=[bounds.z_min, bounds.z_max]),
                        aspectmode="cube",
                    ),
                    showlegend=True,
                    title=fig_idx,
                ),
                name=fig_idx
            )
            for fig_idx, fig in enumerate(orig_figs)
        ],
    )
    new_fig.show()


if __name__ == "__main__":
    args = VisualizeOptimizationArgumentParser().parse_args()
    main(args)
