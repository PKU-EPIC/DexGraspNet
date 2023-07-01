"""
Last modified date: 2023.07.01
Author: Tyler Lum
Description: visualize hand model grasp optimization
"""

import os
import sys

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

    # Create new figure with all plots
    new_fig = go.Figure(
        data=orig_figs[0].data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title="new_fig",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[dict(label="Play", method="animate", args=[None])],
                )
            ],
        ),
        frames=[
            go.Frame(
                data=fig.data,
                layout=go.Layout(
                    scene=dict(
                        xaxis=dict(title="X"),
                        yaxis=dict(title="Y"),
                        zaxis=dict(title="Z"),
                        aspectmode="data",
                    ),
                    showlegend=True,
                    title=f"new_fig frame {i}",
                ),
            )
            for i, fig in enumerate(orig_figs)
        ],
    )
    #     fig_idx_per_trace = []
    #     for fig_idx_to_visualize, fig in enumerate(orig_figs):
    #         for d in fig.data:
    #             new_fig.add_trace(d)
    #             fig_idx_per_trace.append(fig_idx_to_visualize)
    #
    #     # Setup slider to show one figure per step
    #     slider_steps = []
    #     for fig_idx_to_visualize in range(len(orig_figs)):
    #         # Only visualize the traces for this figure
    #         visible_list = [
    #             fig_idx_to_visualize == fig_idx for fig_idx in fig_idx_per_trace
    #         ]
    #
    #         # Add a step to the slider for each figure
    #         step = {
    #             "method": "update",
    #             "args": [{"visible": visible_list}],  # Layout attribute
    #             "label": f"Plot {fig_idx_to_visualize + 1}",
    #         }
    #         slider_steps.append(step)
    #
    #     # Show one fig first
    #     fig_to_show_first = 0
    #     for i, fig_idx_to_visualize in enumerate(fig_idx_per_trace):
    #         new_fig.data[i].visible = fig_idx_to_visualize == fig_to_show_first
    #
    #     new_fig.update_layout(
    #         sliders=[
    #             dict(
    #                 steps=slider_steps,
    #                 active=fig_to_show_first,  # Initial active index
    #                 currentvalue=dict(
    #                     font=dict(size=12),
    #                     prefix="Optimization Iter",  # Prefix for the slider label
    #                     xanchor="center",
    #                     visible=True,
    #                 ),
    #                 len=1.0,  # Length of the slider
    #             )
    #         ],
    #     )

    new_fig.show()


if __name__ == "__main__":
    args = VisualizeOptimizationArgumentParser().parse_args()
    main(args)
