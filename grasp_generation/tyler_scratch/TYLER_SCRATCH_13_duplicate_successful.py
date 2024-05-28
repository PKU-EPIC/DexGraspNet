# %%
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R
data_folder = pathlib.Path("../data/2024-03-06_10mugs/raw_evaled_grasp_config_dicts")
assert data_folder.exists()
data_paths = list(data_folder.glob("*.npy"))

N = 30000
trans_max_noise = 0.03
rot_deg_max_noise = 0

# %%
for data_path in data_paths:
    filename = data_path.name
    data_dict = np.load(data_path, allow_pickle=True).item()

    success_idx = (data_dict['passed_eval'] == 1).nonzero()[0][0]
    data = {k: v[success_idx] for k, v in data_dict.items()}
    assert data["passed_eval"] == 1

    from scipy.stats.qmc import Halton
    USE_HALTON = True
    if USE_HALTON:
        xyz_noise = (Halton(d=3, scramble=True).random(n=N) * 2 - 1) * trans_max_noise
        rpy_noise = (Halton(d=3, scramble=True).random(n=N) * 2 - 1) * rot_deg_max_noise
    else:
        xyz_noise = np.random.uniform(low=-trans_max_noise, high=trans_max_noise, size=(N, 3))
        rpy_noise = np.random.uniform(low=-rot_deg_max_noise, high=rot_deg_max_noise, size=(N, 3))

    # No noise for the first element.
    xyz_noise[0, :] = 0
    rpy_noise[0, :] = 0

    # %%
    new_data_dict = {k: v[None, ...].repeat(N, axis=0) for k, v in data.items()}
    xyz = new_data_dict["trans"]
    rpy = R.from_matrix(new_data_dict["rot"]).as_euler('xyz', degrees=True)
    assert xyz.shape == (N, 3)
    assert rpy.shape == (N, 3)
    new_xyz = xyz + xyz_noise
    new_rpy = rpy + rpy_noise
    new_data_dict["trans"][:] = new_xyz
    new_data_dict["rot"][:] = R.from_euler('xyz', new_rpy, degrees=True).as_matrix()

    new_data_path = pathlib.Path(f"../data/2024-03-06_10mugs_augmented_pose_HALTON_no-rot_30k/grasp_config_dicts/{filename}")
    print(f"new_data_path = {new_data_path}")
    new_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(new_data_path, new_data_dict)
