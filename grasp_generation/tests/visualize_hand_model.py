"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize hand model using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath("."))

import torch
import plotly.graph_objects as go
from utils.hand_model import HandModel
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_angles_mu,
    handmodeltype_to_rotation_hand,
)
from utils.seed import set_seed
from tap import Tap


set_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class VisualizeHandModelArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND


if __name__ == "__main__":
    device = torch.device("cpu")
    args = VisualizeHandModelArgumentParser().parse_args()

    # hand model
    hand_model_type = args.hand_model_type

    hand_model = HandModel(
        hand_model_type=hand_model_type, n_surface_points=2000, device=device
    )
    joint_angles = handmodeltype_to_joint_angles_mu[hand_model_type].to(device)

    rotation = handmodeltype_to_rotation_hand[hand_model_type].to(device)
    hand_pose = torch.cat(
        [
            torch.tensor([0, 0, 0], dtype=torch.float, device=device),
            rotation.T.ravel()[:6],
            joint_angles,
        ]
    )
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # visualize
    hand_plotly = hand_model.get_plotly_data(
        i=0,
        opacity=0.5,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=True,
        with_surface_points=True,
        with_penetration_keypoints=True,
    )
    fig = go.Figure(
        data=(hand_plotly),
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title=f"Hand Model: {hand_model_type}",
        ),
    )
    fig.update_layout(scene_aspectmode="data")
    fig.show()
