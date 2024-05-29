import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from tap import Tap
import os
import subprocess
import sys
import pathlib
from typing import Optional
from datetime import datetime

sys.path.append(os.path.realpath("."))


class ArgParser(Tap):
    evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )
    experiment_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main() -> None:
    args = ArgParser().parse_args()
    assert (
        args.evaled_grasp_config_dicts_path.exists()
    ), f"Path does not exist: {args.evaled_grasp_config_dicts_path}"

    # Get all object codes
    npy_files = sorted(list(args.evaled_grasp_config_dicts_path.glob("*.npy")))[:30]
    evaled_grasp_config_dicts = [
        np.load(path, allow_pickle=True).item()
        for path in tqdm(npy_files, desc="Loading grasp config dicts")
    ]

    # Get success rate for each object
    obj_to_success_rate = {
        npy_file.stem: evaled_grasp_config_dict["passed_eval"].mean()
        for npy_file, evaled_grasp_config_dict in zip(
            npy_files, evaled_grasp_config_dicts
        )
    }
    obj_to_num_successes = {
        npy_file.stem: evaled_grasp_config_dict["passed_eval"].sum()
        for npy_file, evaled_grasp_config_dict in zip(
            npy_files, evaled_grasp_config_dicts
        )
    }
    print(f"obj_to_success_rate: {obj_to_success_rate}")
    print(f"obj_to_num_successes: {obj_to_num_successes}")

    # Get total success rate
    total_success_rate = np.mean(list(obj_to_success_rate.values()))
    print(f"Total success rate: {total_success_rate}")

    df = pd.DataFrame(
        {
            "object": list(obj_to_success_rate.keys()),
            "success_rate": list(obj_to_success_rate.values()),
            "num_successes": list(obj_to_num_successes.values()),
        }
    )
    # Success rate for each object
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(df["object"], df["success_rate"], color="green", edgecolor="black")
    ax.set_xlabel("Object")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate for Each Object")
    plt.xticks(rotation=90)
    plt.tight_layout()

    img_filename = f"success_rate_{args.experiment_name}.png"
    plt.savefig(img_filename, dpi=300)
    print(f"Saved image to {img_filename}")

    csv_filename = f"success_rate_{args.experiment_name}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV to {csv_filename}")

    # Success rate histogram
    fig, ax = plt.subplots(figsize=(12, 8))
    counts, bins, patches = ax.hist(
        df["success_rate"],
        bins=np.linspace(0, 1, 11),
        alpha=0.7,
        rwidth=0.85,
        color="blue",
        edgecolor="black",
    )
    ax.set_title("Success Rate Histogram")
    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Frequency")
    ax.grid()
    ax.set_xlim(left=0, right=1)
    X_DELTA = 0.1
    ax.set_xticks(np.arange(0, 1 + X_DELTA, X_DELTA))

    Y_DELTA = 50
    ax.set_yticks(np.arange(0, max(counts) + Y_DELTA, Y_DELTA))
    img2_filename = f"success_rate_hist_{args.experiment_name}.png"
    plt.savefig(img2_filename, dpi=300)
    print(f"Saved image to {img2_filename}")


if __name__ == "__main__":
    main()
