"""
Last modified date: 2023.07.01
Author: Tyler Lum
Description: visualize hand model grasp optimization
"""

import os
import sys

sys.path.append(os.path.realpath("."))

import plotly
from tap import Tap
from visualize_optimization_helper import (
    create_figure_with_buttons_and_slider,
    download_plotly_files_from_wandb,
    get_visualization_freq_from_wandb,
)


class VisualizeOptimizationArgumentParser(Tap):
    wandb_entity: str = "tylerlum"
    wandb_project: str = "DexGraspNet_v1"
    run_id: str = "qg17990t"
    max_files_to_read: int = 100
    frame_duration: int = 200
    transition_duration: int = 100


def main(args: VisualizeOptimizationArgumentParser):
    # Specify run
    run_path = f"{args.wandb_entity}/{args.wandb_project}/{args.run_id}"
    print(f"Run path: {run_path}")

    # Download plotly files
    plotly_file_paths = download_plotly_files_from_wandb(run_path)

    # Read in json files
    if len(plotly_file_paths) > args.max_files_to_read:
        print(f"Limiting to {args.max_files_to_read} files")
        plotly_file_paths = plotly_file_paths[: args.max_files_to_read]
    input_figs = [
        plotly.io.read_json(file=plotly_file_path)
        for plotly_file_path in plotly_file_paths
    ]

    new_fig = create_figure_with_buttons_and_slider(
        input_figs=input_figs,
        visualization_freq=get_visualization_freq_from_wandb(run_path),
        frame_duration=args.frame_duration,
        transition_duration=args.transition_duration,
    )

    new_fig.show()


if __name__ == "__main__":
    main(VisualizeOptimizationArgumentParser().parse_args())
