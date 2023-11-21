# %%
import pathlib
input_hand_config_dicts_path = pathlib.Path(
    "../data/eval_results/2023-11-17_21-39-17/hand_config_dicts/"
)

# %%

hand_config_dict_filepaths = [
    path for path in list(input_hand_config_dicts_path.glob("*.npy"))
]
print(f"len(input_hand_config_dict_filepaths): {len(hand_config_dict_filepaths)}")
print(f"First 10: {[path for path in hand_config_dict_filepaths[:10]]}")
# %%
CONFIG_DICT_IDX = 0
hand_config_dict_filepath = hand_config_dict_filepaths[CONFIG_DICT_IDX]
print(f"hand_config_dict_filepath: {hand_config_dict_filepath}")
# %%
from utils.parse_object_code_and_scale import parse_object_code_and_scale
from utils.pose_conversion import (
    hand_config_to_pose,
)
from typing import Dict
import numpy as np
object_code_and_scale_str = hand_config_dict_filepath.stem
object_code, object_scale = parse_object_code_and_scale(
    object_code_and_scale_str
)

# Read in data
hand_config_dict: Dict[str, np.ndarray] = np.load(
    hand_config_dict_filepath, allow_pickle=True
).item()

hand_pose = hand_config_to_pose(
    hand_config_dict["trans"],
    hand_config_dict["rot"],
    hand_config_dict["joint_angles"],
)


# %%

from utils.joint_angle_targets import (
    compute_grasp_orientations as compute_grasp_orientations_external,
)
import torch
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import HandModelType
from dataclasses import dataclass

@dataclass
class Args:
    gpu: int = 0
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    # meshdata_root_path: pathlib.Path = pathlib.Path("../../nerf_grasping/data/2023-11-17_01-27-23/nerf_meshdata_mugs_v8/")


args = Args()
batch_size = hand_pose.shape[0]

device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"
hand_pose = hand_pose.to(device)

# hand model
hand_model = HandModel(
    hand_model_type=args.hand_model_type, device=hand_pose.device
)
hand_model.set_parameters(hand_pose)

# object model
object_model = ObjectModel(
    meshdata_root_path=str(args.meshdata_root_path),
    batch_size_each=batch_size,
    num_samples=0,
    device=hand_pose.device,
)
object_model.initialize(object_code, object_scale)

# %%
GRASP_IDX = 0
import plotly.graph_objects as go
hand_plotly = hand_model.get_plotly_data(
    i=GRASP_IDX,
    opacity=1,
    color="lightblue",
    with_contact_points=False,
    with_contact_candidates=True,
    with_surface_points=True,
    with_penetration_keypoints=False,
)

object_plotly = object_model.get_plotly_data(
    i=0, color="lightgreen", opacity=0.5, with_surface_points=True
)

fig = go.Figure(
    data=(hand_plotly + object_plotly),
)
fig.show()

# %%
grasp_orientations, hand_contact_nearest_points, nearest_object_to_hand_directions, nearest_distances = compute_grasp_orientations_external(
    joint_angles_start=hand_model.hand_pose[:, 9:],
    hand_model=hand_model,
    object_model=object_model,
    debug=True
)

# %%
grasp_orientations[GRASP_IDX].shape

# %%
hand_contact_nearest_points[GRASP_IDX].shape

# %%
nearest_object_to_hand_directions[GRASP_IDX].shape

# %%
nearest_distances[GRASP_IDX].shape

# %%
nearest_distances[GRASP_IDX]

# %%
object_nearest_points = hand_contact_nearest_points + nearest_object_to_hand_directions * nearest_distances.unsqueeze(-1)

# %%
object_nearest_points[GRASP_IDX].shape

# %%
# fig.add_trace(
#     go.Scatter3d(
#         x=hand_contact_nearest_points.cpu().numpy()[GRASP_IDX][:, 0],
#         y=hand_contact_nearest_points.cpu().numpy()[GRASP_IDX][:, 1],
#         z=hand_contact_nearest_points.cpu().numpy()[GRASP_IDX][:, 2],
#         mode="markers",
#         marker=dict(size=3, color="red"),
#         name="hand_contact_nearest_points",
#     )
# )

fig.add_trace(
    go.Scatter3d(
        x=object_nearest_points.cpu().numpy()[GRASP_IDX][:, 0],
        y=object_nearest_points.cpu().numpy()[GRASP_IDX][:, 1],
        z=object_nearest_points.cpu().numpy()[GRASP_IDX][:, 2],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="object_nearest_points",
    )
)
fig.show()

# %%
