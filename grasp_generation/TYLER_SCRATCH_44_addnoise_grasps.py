# %%
from typing import Optional, Tuple
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from collections import defaultdict
from localscope import localscope

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

# %%
TRANS_MAX_NOISE = 0.01
ROT_DEG_MAX_NOISE = 2.5
JOINT_POS_MAX_NOISE = 0.1
GRASP_ORIENTATION_DEG_MAX_NOISE = 15

# %%
MAX_N_OBJECTS = None

# %%
OUTPUT_PATH = pathlib.Path(
    "../data/2024-05-09_rotated_stable_grasps_noisy_TUNED/raw_grasp_config_dicts/"
)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# %%
@localscope.mfc
def add_noise_to_rot_matrices(
    rot_matrices: np.ndarray,
    rpy_noise: np.ndarray,
) -> np.ndarray:
    N = rot_matrices.shape[0]
    assert rot_matrices.shape == (N, 3, 3)
    assert rpy_noise.shape == (N, 3)

    R_x = np.eye(3)[None, ...].repeat(N, axis=0)
    R_y = np.eye(3)[None, ...].repeat(N, axis=0)
    R_z = np.eye(3)[None, ...].repeat(N, axis=0)

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

    R_zy = np.einsum("ijk,ikl->ijl", R_z, R_y)
    R_zyx = np.einsum("ijk,ikl->ijl", R_zy, R_x)

    new_rot_matrices = np.einsum("ijk,ikl->ijl", rot_matrices, R_zyx)
    return new_rot_matrices


@localscope.mfc
def add_noise_to_dirs(dirs: np.ndarray, theta_phi_noise: np.ndarray) -> np.ndarray:
    N = dirs.shape[0]
    assert dirs.shape == (N, 3)
    assert theta_phi_noise.shape == (N, 2)

    cos_thetas = np.cos(theta_phi_noise[:, 0])
    sin_thetas = np.sin(theta_phi_noise[:, 0])
    cos_phis = np.cos(theta_phi_noise[:, 1])
    sin_phis = np.sin(theta_phi_noise[:, 1])

    # Rotation around z-axis
    RRz = np.eye(3)[None, ...].repeat(N, axis=0)
    RRz[:, 0, 0] = cos_thetas
    RRz[:, 0, 1] = -sin_thetas
    RRz[:, 1, 0] = sin_thetas
    RRz[:, 1, 1] = cos_thetas

    RRy = np.eye(3)[None, ...].repeat(N, axis=0)
    RRy[:, 0, 0] = cos_phis
    RRy[:, 0, 2] = sin_phis
    RRy[:, 2, 0] = -sin_phis
    RRy[:, 2, 2] = cos_phis

    RRyz = np.einsum("ijk,ikl->ijl", RRy, RRz)
    new_z_dirs = np.einsum("ik,ikl->il", dirs, RRyz)
    return new_z_dirs


@localscope.mfc
def clamp_joint_angles(
    joint_angles: np.ndarray,
    hand_model: HandModel,
) -> np.ndarray:
    N = joint_angles.shape[0]
    assert joint_angles.shape == (N, 16)

    joint_lowers = hand_model.joints_lower.detach().cpu().numpy()
    joint_uppers = hand_model.joints_upper.detach().cpu().numpy()
    new_joint_angles = np.clip(joint_angles, joint_lowers[None], joint_uppers[None])
    return new_joint_angles


# %%
device = "cuda"
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(hand_model_type=hand_model_type, device=device)

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
all_paths = all_paths[:MAX_N_OBJECTS] if MAX_N_OBJECTS is not None else all_paths

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
@localscope.mfc
def sample_noise(shape: Tuple[int], scale: float, mode: str = "normal") -> np.ndarray:
    batch_dims = shape[:-1]
    d = shape[-1]

    if mode == "halton":
        from scipy.stats.qmc import Halton

        N = np.prod(batch_dims)
        noise = (Halton(d=d, scramble=True).random(n=N) * 2 - 1) * scale
    elif mode == "uniform":
        noise = np.random.uniform(low=-scale, high=scale, size=(*batch_dims, d))
    elif mode == "normal":
        noise = np.random.normal(loc=0, scale=scale, size=(*batch_dims, d))
    else:
        raise ValueError(f"Invalid mode: {mode}")

    assert noise.shape == (*batch_dims, d)
    return noise


# %%
@localscope.mfc
def add_noise(
    data_dict: dict,
    N_noisy: int,
    hand_model: HandModel,
    trans_max_noise: float = TRANS_MAX_NOISE,
    rot_deg_max_noise: float = ROT_DEG_MAX_NOISE,
    joint_pos_max_noise: float = JOINT_POS_MAX_NOISE,
    grasp_orientation_deg_max_noise: float = GRASP_ORIENTATION_DEG_MAX_NOISE,
) -> dict:
    N_FINGERS = 4

    B = data_dict["trans"].shape[0]
    if B == 0:
        return {}

    xyz_noise = sample_noise(shape=(B, N_noisy, 3), scale=trans_max_noise)
    rpy_noise = sample_noise(shape=(B, N_noisy, 3), scale=np.deg2rad(rot_deg_max_noise))
    joint_angles_noise = sample_noise(shape=(B, N_noisy, 16), scale=joint_pos_max_noise)
    grasp_orientation_noise = sample_noise(
        shape=(B, N_noisy, N_FINGERS, 2),
        scale=np.deg2rad(grasp_orientation_deg_max_noise),
    )

    # 0 noise for the first noisy sample of each batch dim
    xyz_noise[:, 0] = 0
    rpy_noise[:, 0] = 0
    joint_angles_noise[:, 0] = 0
    grasp_orientation_noise[:, 0] = 0

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
    new_trans = orig_trans[:, None, ...].repeat(N_noisy, axis=1)
    new_trans = (new_trans + xyz_noise).reshape(N_noisy * B, 3)
    new_data_dict["trans"] = new_trans

    # rot
    new_rot = orig_rot[:, None, ...].repeat(N_noisy, axis=1)
    new_rot = add_noise_to_rot_matrices(
        rot_matrices=new_rot.reshape(B * N_noisy, 3, 3),
        rpy_noise=rpy_noise.reshape(B * N_noisy, 3),
    ).reshape(N_noisy * B, 3, 3)
    new_data_dict["rot"] = new_rot

    # joint_angles
    new_joint_angles = orig_joint_angles[:, None, ...].repeat(N_noisy, axis=1)
    new_joint_angles += joint_angles_noise
    new_joint_angles = clamp_joint_angles(
        joint_angles=new_joint_angles.reshape(N_noisy * B, 16), hand_model=hand_model
    )
    new_data_dict["joint_angles"] = new_joint_angles

    # hand_model
    hand_pose = hand_config_to_pose(new_trans, new_rot, new_joint_angles).to(
        hand_model.device
    )
    hand_model.set_parameters(hand_pose)

    # grasp_orientations
    orig_z_dirs = orig_grasp_orientations[:, :, :, 2]
    new_z_dirs = orig_z_dirs[:, None, ...].repeat(N_noisy, axis=1)
    new_z_dirs = add_noise_to_dirs(
        dirs=new_z_dirs.reshape(B * N_noisy * N_FINGERS, 3),
        theta_phi_noise=grasp_orientation_noise.reshape(B * N_noisy * N_FINGERS, 2),
    )
    new_z_dirs_torch = (
        torch.from_numpy(new_z_dirs).float().cuda().reshape(N_noisy * B, N_FINGERS, 3)
    )

    # Math to get x_dirs, y_dirs
    (center_to_right_dirs, center_to_tip_dirs) = compute_fingertip_dirs(
        joint_angles=torch.from_numpy(new_joint_angles).float().cuda(),
        hand_model=hand_model,
    )
    option_1_ok = (
        torch.cross(center_to_tip_dirs, new_z_dirs_torch).norm(dim=-1, keepdim=True) > 0
    )

    y_dirs = torch.where(
        option_1_ok,
        center_to_tip_dirs
        - (center_to_tip_dirs * new_z_dirs_torch).sum(dim=-1, keepdim=True)
        * new_z_dirs_torch,
        center_to_right_dirs
        - (center_to_right_dirs * new_z_dirs_torch).sum(dim=-1, keepdim=True)
        * new_z_dirs_torch,
    )

    assert (y_dirs.norm(dim=-1).min() > 0).all()
    y_dirs = y_dirs / y_dirs.norm(dim=-1, keepdim=True)

    x_dirs = torch.cross(y_dirs, new_z_dirs_torch)
    assert (x_dirs.norm(dim=-1).min() > 0).all()
    x_dirs = x_dirs / x_dirs.norm(dim=-1, keepdim=True)
    new_grasp_orientations = (
        torch.stack([x_dirs, y_dirs, new_z_dirs_torch], dim=-1).cpu().numpy()
    )
    new_data_dict["grasp_orientations"] = new_grasp_orientations

    for k, v in data_dict.items():
        if k in ["trans", "rot", "joint_angles", "grasp_orientations"]:
            continue

        new_data_dict[k] = (
            v[:, None, ...].repeat(N_noisy, axis=1).reshape(N_noisy * B, *v.shape[1:])
        )
    return new_data_dict


# %%
test_input = obj_to_good_grasps[objs[1]]
test_n_noisy = 10
test_output = add_noise(
    data_dict=test_input, N_noisy=test_n_noisy, hand_model=hand_model
)

# %%
idx = 1
offset_idx = 1
test_input_trans_0 = test_input["trans"][idx]
test_output_trans_0 = test_output["trans"][test_n_noisy * idx + offset_idx]
diff = np.linalg.norm(test_input_trans_0 - test_output_trans_0)
print(f"diff: {diff}")

# %%
# hand_model
idx = 2
offset_idx = 0

input_hand_pose = hand_config_to_pose(
    test_input["trans"], test_input["rot"], test_input["joint_angles"]
).to(device)
hand_model.set_parameters(input_hand_pose)
input_plotly = hand_model.get_plotly_data(i=idx, opacity=1.0)

output_hand_pose = hand_config_to_pose(
    test_output["trans"], test_output["rot"], test_output["joint_angles"]
).to(device)
hand_model.set_parameters(output_hand_pose)
output_plotly = hand_model.get_plotly_data(i=test_n_noisy * idx + offset_idx)

# %%
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
    noisy_good_data_dict = add_noise(
        data_dict=good_grasps_dict, N_noisy=N_noisy, hand_model=hand_model
    )
    noisy_other_data_dict = add_noise(
        data_dict=good_grasps_dict, N_noisy=N_other, hand_model=hand_model
    )

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
    if len(new_dict) == 0:
        continue
    obj_to_new_grasps[obj] = new_dict

# %%
# Step 5: Save
for obj, new_dict in tqdm(obj_to_new_grasps.items()):
    np.save(OUTPUT_PATH / f"{obj}.npy", new_dict)

# %%
