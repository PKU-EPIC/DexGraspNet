# %%
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from collections import defaultdict

# %%
######### CREATE NEW DATASET ##################
path_str = "../data/2024-05-06_rotated_stable_grasps_0/raw_evaled_grasp_config_dicts/"
path_bigger_str = (
    "../data/2024-05-06_rotated_stable_grasps_bigger_0/raw_evaled_grasp_config_dicts/"
)
path_smaller_str = (
    "../data/2024-05-06_rotated_stable_grasps_smaller_0/raw_evaled_grasp_config_dicts/"
)

paths = [pathlib.Path(path_str.replace("_0", f"_{i}")) for i in range(7)]
path_biggers = [pathlib.Path(path_bigger_str.replace("_0", f"_{i}")) for i in range(7)]
path_smallers = [
    pathlib.Path(path_smaller_str.replace("_0", f"_{i}")) for i in range(7)
]

all_paths = paths + path_biggers + path_smallers
for path in all_paths:
    assert path.exists()

# %%
# Step 0: Get all data paths
all_data_paths = []
for path in tqdm(all_paths):
    data_paths = sorted(list(path.glob("*.npy")))
    all_data_paths += data_paths

# %%
# Step 1: Get obj_to_all_grasps
obj_to_all_grasps = {}
for data_path in tqdm(all_data_paths):
    data_dict = np.load(data_path, allow_pickle=True).item()
    obj = data_path.stem
    obj_to_all_grasps[obj] = data_dict

# %%
objs = list(obj_to_all_grasps.keys())

# %%
# Step 2: Get obj_to_good_grasps
obj_to_good_grasps = {}
for obj, all_grasps_dict in tqdm(obj_to_all_grasps.items()):
    good_idxs = all_grasps_dict["passed_eval"] > 0.9
    good_data_dict = {k: v[good_idxs] for k, v in all_grasps_dict.items()}
    obj_to_good_grasps[obj] = good_data_dict

# %%
def add_noise(data_dict: dict, N_noisy: int, trans_max_noise: float = 0.03, rot_deg_max_noise: float = 10, joint_pos_max_noise: float = 0.1, grasp_orientation_deg_max_noise: float = 10) -> dict:
    from utils.hand_model import HandModel
    from utils.hand_model_type import (
        HandModelType,
    )
    from utils.pose_conversion import (
        hand_config_to_pose,
    )
    from utils.joint_angle_targets import (
        compute_fingertip_dirs,
    )
    import torch
    N_FINGERS = 4
    from scipy.stats.qmc import Halton

    B = data_dict['trans'].shape[0]
    if B == 0:
        return {}
    xyz_noise = (Halton(d=3, scramble=True).random(n=N_noisy*B) * 2 - 1) * trans_max_noise
    rpy_noise = (Halton(d=3, scramble=True).random(n=N_noisy*B) * 2 - 1) * np.deg2rad(rot_deg_max_noise)
    joint_angles_noise = (Halton(d=16, scramble=True).random(n=N_noisy*B) * 2 - 1) * joint_pos_max_noise
    grasp_orientation_noise = (Halton(d=2, scramble=True).random(n=N_noisy*B*N_FINGERS) * 2 - 1) * np.deg2rad(grasp_orientation_deg_max_noise)

    orig_trans = data_dict["trans"]
    orig_rot = data_dict["rot"]
    orig_joint_angles = data_dict["joint_angles"]
    orig_grasp_orientations = data_dict["grasp_orientations"]

    assert orig_trans.shape == (B, 3)
    assert orig_rot.shape == (B, 3, 3)
    assert orig_joint_angles.shape == (B, 16)
    assert orig_grasp_orientations.shape == (B, N_FINGERS, 3, 3)

    new_data_dict = {}

    # trans
    new_trans = orig_trans[:, None, ...].repeat(N_noisy, axis=1).reshape(N_noisy * B, 3)
    new_trans += xyz_noise
    new_data_dict["trans"] = new_trans

    # rot
    R_x = np.eye(3)[None, ...].repeat(N_noisy*B, axis=0)
    R_y = np.eye(3)[None, ...].repeat(N_noisy*B, axis=0)
    R_z = np.eye(3)[None, ...].repeat(N_noisy*B, axis=0)

    R_x[:, 1, 1] = np.cos(rpy_noise[:, 0])
    R_x[:, 1, 2] = -np.sin(rpy_noise[:, 0])
    R_x[:, 2, 1] = np.sin(rpy_noise[:, 0])
    R_x[:, 2, 2] = np.cos(rpy_noise[:, 0])

    R_y[:, 0, 0] = np.cos(rpy_noise[:, 1])
    R_y[:, 0, 2] = np.sin(rpy_noise[:, 1])
    R_y[:, 2, 0] = -np.sin(rpy_noise[:, 1])
    R_y[:, 2, 2] = np.cos(rpy_noise[:, 1])

    R_z[:, 0, 0] = np.cos(rpy_noise[:, 2])
    R_z[:, 0, 1] = -np.sin(rpy_noise[:, 2])
    R_z[:, 1, 0] = np.sin(rpy_noise[:, 2])
    R_z[:, 1, 1] = np.cos(rpy_noise[:, 2])

    R_zy = np.einsum('ijk,ikl->ijl', R_z, R_y)
    R_zyx = np.einsum('ijk,ikl->ijl', R_zy, R_x)

    new_rot = orig_rot[:, None, ...].repeat(N_noisy, axis=1).reshape(N_noisy * B, 3, 3)
    new_rot = np.einsum('ijk,ikl->ijl', new_rot, R_zyx)
    new_data_dict["rot"] = new_rot

    # joint_angles
    new_joint_angles = orig_joint_angles[:, None, ...].repeat(N_noisy, axis=1).reshape(N_noisy * B, 16)
    new_joint_angles += joint_angles_noise

    # hand_model
    device = "cuda"
    hand_pose = hand_config_to_pose(new_trans, new_rot, new_joint_angles).to(device)
    hand_model_type = HandModelType.ALLEGRO_HAND
    hand_model = HandModel(hand_model_type=hand_model_type, device=device)
    hand_model.set_parameters(hand_pose)

    # Clamp
    joint_lowers = hand_model.joints_lower.detach().cpu().numpy()
    joint_uppers = hand_model.joints_upper.detach().cpu().numpy()
    new_joint_angles = np.clip(new_joint_angles, joint_lowers, joint_uppers)
    new_data_dict["joint_angles"] = new_joint_angles

    # grasp_orientations
    orig_z_dirs = orig_grasp_orientations[:, :, :, 2]
    new_z_dirs = orig_z_dirs[:, None, ...].repeat(N_noisy, axis=1).reshape(N_noisy * B * N_FINGERS, 3)

    cos_thetas = np.cos(grasp_orientation_noise[:, 0])
    sin_thetas = np.sin(grasp_orientation_noise[:, 0])
    cos_phis = np.cos(grasp_orientation_noise[:, 1])
    sin_phis = np.sin(grasp_orientation_noise[:, 1])

    # Rotation around z-axis
    RRz = np.eye(3)[None, ...].repeat(N_noisy*B*N_FINGERS, axis=0)
    RRz[:, 0, 0] = cos_thetas
    RRz[:, 0, 1] = -sin_thetas
    RRz[:, 1, 0] = sin_thetas
    RRz[:, 1, 1] = cos_thetas

    RRy = np.eye(3)[None, ...].repeat(N_noisy*B*N_FINGERS, axis=0)
    RRy[:, 0, 0] = cos_phis
    RRy[:, 0, 2] = sin_phis
    RRy[:, 2, 0] = -sin_phis
    RRy[:, 2, 2] = cos_phis

    RRyz = np.einsum('ijk,ikl->ijl', RRy, RRz)
    new_z_dirs = np.einsum('ik,ikl->il', new_z_dirs, RRyz)
    new_z_dirs_torch = torch.from_numpy(new_z_dirs).float().cuda().reshape(N_noisy * B, N_FINGERS, 3)

    (center_to_right_dirs, center_to_tip_dirs) = compute_fingertip_dirs(
        joint_angles=torch.from_numpy(new_joint_angles).float().cuda(),
        hand_model=hand_model,
    )
    option_1_ok = torch.cross(center_to_tip_dirs, new_z_dirs_torch).norm(dim=-1, keepdim=True) > 0

    y_dirs = torch.where(
        option_1_ok,
        center_to_tip_dirs
        - (center_to_tip_dirs * new_z_dirs_torch).sum(dim=-1, keepdim=True) * new_z_dirs_torch,
        center_to_right_dirs
        - (center_to_right_dirs * new_z_dirs_torch).sum(dim=-1, keepdim=True) * new_z_dirs_torch,
    )

    assert (y_dirs.norm(dim=-1).min() > 0).all()
    y_dirs = y_dirs / y_dirs.norm(dim=-1, keepdim=True)

    x_dirs = torch.cross(y_dirs, new_z_dirs_torch)
    assert (x_dirs.norm(dim=-1).min() > 0).all()
    x_dirs = x_dirs / x_dirs.norm(dim=-1, keepdim=True)
    new_grasp_orientations = torch.stack([x_dirs, y_dirs, new_z_dirs_torch], dim=-1)
    new_data_dict["grasp_orientations"] = new_grasp_orientations.cpu().numpy().reshape(N_noisy * B, N_FINGERS, 3, 3)

    for k, v in data_dict.items():
        if k in ["trans", "rot", "joint_angles", "grasp_orientations"]:
            continue

        new_data_dict[k] = v[:, None, ...].repeat(N_noisy, axis=1).reshape(N_noisy * B, *v.shape[1:])
    return new_data_dict

# %%
test_input = obj_to_good_grasps[objs[1]]
test_output = add_noise(data_dict=test_input, N_noisy=10)

# %%
import matplotlib.pyplot as plt
idx = 1
test_input_trans_0 = test_input["trans"][idx]
test_output_trans_0 = test_output["trans"][10 + idx]
diff = np.linalg.norm(test_input_trans_0 - test_output_trans_0)
print(diff)

# %%
from utils.hand_model import HandModel
from utils.hand_model_type import (
    HandModelType,
)
from utils.pose_conversion import (
    hand_config_to_pose,
)
from utils.joint_angle_targets import (
    compute_fingertip_dirs,
)
import torch

# hand_model
device = "cuda"
idx = 1
input_hand_pose = hand_config_to_pose(test_input["trans"], test_input["rot"], test_input["joint_angles"]).to(device)
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(hand_model_type=hand_model_type, device=device)
hand_model.set_parameters(input_hand_pose)
input_plotly = hand_model.get_plotly_data(i=idx, opacity=1.0)

output_hand_pose = hand_config_to_pose(test_output["trans"], test_output["rot"], test_output["joint_angles"]).to(device)
hand_model.set_parameters(output_hand_pose)
offset_idx = 1
output_plotly = hand_model.get_plotly_data(i=10*idx + offset_idx)

# %%
import plotly.graph_objects as go
fig = go.Figure(data=input_plotly + output_plotly)
fig.show()


# %%
# Step 3: Get obj_to_noisy_good_grasps, obj_to_noisy_other_grasps
obj_to_noisy_good_grasps = defaultdict(list)
obj_to_noisy_other_grasps = defaultdict(list)
N_noisy = 40
N_other = 10
for obj, good_grasps_dict in tqdm(obj_to_good_grasps.items()):
    # Add noise
    noisy_good_data_dict = add_noise(data_dict=good_grasps_dict, N_noisy=N_noisy)
    noisy_other_data_dict = add_noise(data_dict=good_grasps_dict, N_noisy=N_other)

    # Get other object
    while True:
        other_obj = np.random.choice(objs)
        if other_obj != obj:
            break

    obj_to_noisy_good_grasps[obj].append(noisy_good_data_dict)
    obj_to_noisy_other_grasps[other_obj].append(noisy_other_data_dict)

# %%
# Step 4: Aggregate
obj_to_new_grasps = {}
for obj in tqdm(objs):
    new_dicts = obj_to_noisy_good_grasps[obj] + obj_to_noisy_other_grasps[obj]
    new_dict = {
        k: np.concatenate([d[k] for d in new_dicts if k in d], axis=0)
        for k in new_dicts[0].keys()
    }
    obj_to_new_grasps[obj] = new_dict

# %%
# Step 5: Save
OUTPUT_PATH = pathlib.Path(
    "../data/2024-05-09_rotated_stable_grasps_noisy_less_fixed_allow_grasp_orientations/raw_grasp_config_dicts/"
)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

for obj, new_dict in tqdm(obj_to_new_grasps.items()):
    np.save(OUTPUT_PATH / f"{obj}.npy", new_dict)

# %%
