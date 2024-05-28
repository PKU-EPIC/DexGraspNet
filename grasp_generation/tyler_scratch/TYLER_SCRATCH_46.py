# %%
import pathlib
import numpy as np

pose_only_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_pose_only/raw_evaled_grasp_config_dicts/")
pose_only_more_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_pose_only_more/raw_evaled_grasp_config_dicts/")
joints_only_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_joints_only/raw_evaled_grasp_config_dicts/")
joints_only_more_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_joints_only_more/raw_evaled_grasp_config_dicts/")
joints_only_more2_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_joints_only_more2/raw_evaled_grasp_config_dicts/")
pose_only_less_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_pose_only_less/raw_evaled_grasp_config_dicts/")
pose_joints_less_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_pose_joints_less/raw_evaled_grasp_config_dicts/")
go_only_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_go_only/raw_evaled_grasp_config_dicts/")
go_only_more_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_go_only_more/raw_evaled_grasp_config_dicts/")
go_only_more2_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_go_only_more2/raw_evaled_grasp_config_dicts/")
go_only_more3_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_go_only_more3/raw_evaled_grasp_config_dicts/")
TUNED_path = pathlib.Path("../data/2024-05-09_rotated_stable_grasps_noisy_TUNED/raw_evaled_grasp_config_dicts/")
assert pose_only_path.exists()
assert pose_only_more_path.exists()
assert joints_only_path.exists()
assert joints_only_more_path.exists()
assert joints_only_more2_path.exists()
assert pose_only_less_path.exists()
assert pose_joints_less_path.exists()
assert go_only_path.exists()
assert go_only_more_path.exists()
assert go_only_more2_path.exists()
assert go_only_more3_path.exists()
assert TUNED_path.exists()

# %%
pose_only_files = [f for f in pose_only_path.glob("*.npy")]
pose_only_more_files = [f for f in pose_only_more_path.glob("*.npy")]
joints_only_files = [f for f in joints_only_path.glob("*.npy")]
joints_only_more_files = [f for f in joints_only_more_path.glob("*.npy")]
joints_only_more2_files = [f for f in joints_only_more2_path.glob("*.npy")]
pose_only_less_files = [f for f in pose_only_less_path.glob("*.npy")]
pose_joints_less_files = [f for f in pose_joints_less_path.glob("*.npy")]
go_only_files = [f for f in go_only_path.glob("*.npy")]
go_only_more_files = [f for f in go_only_more_path.glob("*.npy")]
go_only_more2_files = [f for f in go_only_more2_path.glob("*.npy")]
go_only_more3_files = [f for f in go_only_more3_path.glob("*.npy")]
TUNED_files = [f for f in TUNED_path.glob("*.npy")]
print(f"len(pose_only_files) = {len(pose_only_files)}")
print(f"len(pose_only_more_files) = {len(pose_only_more_files)}")
print(f"len(joints_only_files) = {len(joints_only_files)}")
print(f"len(joints_only_more_files) = {len(joints_only_more_files)}")
print(f"len(joints_only_more2_files) = {len(joints_only_more2_files)}")
print(f"len(pose_only_less_files) = {len(pose_only_less_files)}")
print(f"len(pose_joints_less_files) = {len(pose_joints_less_files)}")
print(f"len(go_only_files) = {len(go_only_files)}")
print(f"len(go_only_more_files) = {len(go_only_more_files)}")
print(f"len(go_only_more2_files) = {len(go_only_more2_files)}")
print(f"len(go_only_more3_files) = {len(go_only_more3_files)}")
print(f"len(TUNED_files) = {len(TUNED_files)}")

# %%
pose_only_dicts = [np.load(f, allow_pickle=True).item() for f in pose_only_files]
pose_only_more_dicts = [np.load(f, allow_pickle=True).item() for f in pose_only_more_files]
joints_only_dicts = [np.load(f, allow_pickle=True).item() for f in joints_only_files]
joints_only_more_dicts = [np.load(f, allow_pickle=True).item() for f in joints_only_more_files]
joints_only_more2_dicts = [np.load(f, allow_pickle=True).item() for f in joints_only_more2_files]
pose_only_less_dicts = [np.load(f, allow_pickle=True).item() for f in pose_only_less_files]
pose_joints_less_dicts = [np.load(f, allow_pickle=True).item() for f in pose_joints_less_files]
go_only_dicts = [np.load(f, allow_pickle=True).item() for f in go_only_files]
go_only_more_dicts = [np.load(f, allow_pickle=True).item() for f in go_only_more_files]
go_only_more2_dicts = [np.load(f, allow_pickle=True).item() for f in go_only_more2_files]
go_only_more3_dicts = [np.load(f, allow_pickle=True).item() for f in go_only_more3_files]
TUNED_dicts = [np.load(f, allow_pickle=True).item() for f in TUNED_files]

# %%
pose_only_success_rates = [d['passed_eval'].mean() for d in pose_only_dicts]
pose_only_more_success_rates = [d['passed_eval'].mean() for d in pose_only_more_dicts]
joints_only_success_rates = [d['passed_eval'].mean() for d in joints_only_dicts]
joints_only_more_success_rates = [d['passed_eval'].mean() for d in joints_only_more_dicts]
joints_only_more2_success_rates = [d['passed_eval'].mean() for d in joints_only_more2_dicts]
pose_only_less_success_rates = [d['passed_eval'].mean() for d in pose_only_less_dicts]
pose_joints_less_success_rates = [d['passed_eval'].mean() for d in pose_joints_less_dicts]
go_only_success_rates = [d['passed_eval'].mean() for d in go_only_dicts]
go_only_more_success_rates = [d['passed_eval'].mean() for d in go_only_more_dicts]
go_only_more2_success_rates = [d['passed_eval'].mean() for d in go_only_more2_dicts]
go_only_more3_success_rates = [d['passed_eval'].mean() for d in go_only_more3_dicts]
TUNED_success_rates = [d['passed_eval'].mean() for d in TUNED_dicts]

# %%
print(f"np.mean(pose_only_success_rates) = {np.mean(pose_only_success_rates)}")
print(f"np.mean(pose_only_more_success_rates) = {np.mean(pose_only_more_success_rates)}")
print(f"np.mean(joints_only_success_rates) = {np.mean(joints_only_success_rates)}")
print(f"np.mean(joints_only_more_success_rates) = {np.mean(joints_only_more_success_rates)}")
print(f"np.mean(joints_only_more2_success_rates) = {np.mean(joints_only_more2_success_rates)}")
print(f"np.mean(pose_only_less_success_rates) = {np.mean(pose_only_less_success_rates)}")
print(f"np.mean(pose_joints_less_success_rates) = {np.mean(pose_joints_less_success_rates)}")
print(f"np.mean(go_only_success_rates) = {np.mean(go_only_success_rates)}")
print(f"np.mean(go_only_more_success_rates) = {np.mean(go_only_more_success_rates)}")
print(f"np.mean(go_only_more2_success_rates) = {np.mean(go_only_more2_success_rates)}")
print(f"np.mean(go_only_more3_success_rates) = {np.mean(go_only_more3_success_rates)}")
print(f"np.mean(TUNED_success_rates) = {np.mean(TUNED_success_rates)}")

# %%
pose_only_num_grasps = [len(d['passed_eval']) for d in pose_only_dicts]
pose_only_more_num_grasps = [len(d['passed_eval']) for d in pose_only_more_dicts]
joints_only_num_grasps = [len(d['passed_eval']) for d in joints_only_dicts]
joints_only_more_num_grasps = [len(d['passed_eval']) for d in joints_only_more_dicts]
joints_only_more2_num_grasps = [len(d['passed_eval']) for d in joints_only_more2_dicts]
pose_only_less_num_grasps = [len(d['passed_eval']) for d in pose_only_less_dicts]
pose_joints_less_num_grasps = [len(d['passed_eval']) for d in pose_joints_less_dicts]
go_only_num_grasps = [len(d['passed_eval']) for d in go_only_dicts]
go_only_more_num_grasps = [len(d['passed_eval']) for d in go_only_more_dicts]
go_only_more2_num_grasps = [len(d['passed_eval']) for d in go_only_more2_dicts]
go_only_more3_num_grasps = [len(d['passed_eval']) for d in go_only_more3_dicts]
TUNED_num_grasps = [len(d['passed_eval']) for d in TUNED_dicts]
print(f"np.sum(pose_only_num_grasps) = {np.sum(pose_only_num_grasps)}")
print(f"np.sum(pose_only_more_num_grasps) = {np.sum(pose_only_more_num_grasps)}")
print(f"np.sum(joints_only_num_grasps) = {np.sum(joints_only_num_grasps)}")
print(f"np.sum(joints_only_more_num_grasps) = {np.sum(joints_only_more_num_grasps)}")
print(f"np.sum(joints_only_more2_num_grasps) = {np.sum(joints_only_more2_num_grasps)}")
print(f"np.sum(pose_only_less_num_grasps) = {np.sum(pose_only_less_num_grasps)}")
print(f"np.sum(pose_joints_less_num_grasps) = {np.sum(pose_joints_less_num_grasps)}")
print(f"np.sum(go_only_num_grasps) = {np.sum(go_only_num_grasps)}")
print(f"np.sum(go_only_more_num_grasps) = {np.sum(go_only_more_num_grasps)}")
print(f"np.sum(go_only_more2_num_grasps) = {np.sum(go_only_more2_num_grasps)}")
print(f"np.sum(go_only_more3_num_grasps) = {np.sum(go_only_more3_num_grasps)}")
print(f"np.sum(TUNED_num_grasps) = {np.sum(TUNED_num_grasps)}")


# %%
pose_only_passed_evals = [x for d in pose_only_dicts for x in d['passed_eval']]
pose_only_more_passed_evals = [x for d in pose_only_more_dicts for x in d['passed_eval']]
joints_only_passed_evals = [x for d in joints_only_dicts for x in d['passed_eval']]
joints_only_more_passed_evals = [x for d in joints_only_more_dicts for x in d['passed_eval']]
joints_only_more2_passed_evals = [x for d in joints_only_more2_dicts for x in d['passed_eval']]
pose_only_less_passed_evals = [x for d in pose_only_less_dicts for x in d['passed_eval']]
pose_joints_less_passed_evals = [x for d in pose_joints_less_dicts for x in d['passed_eval']]
go_only_passed_evals = [x for d in go_only_dicts for x in d['passed_eval']]
go_only_more_passed_evals = [x for d in go_only_more_dicts for x in d['passed_eval']]
go_only_more2_passed_evals = [x for d in go_only_more2_dicts for x in d['passed_eval']]
go_only_more3_passed_evals = [x for d in go_only_more3_dicts for x in d['passed_eval']]
TUNED_passed_evals = [x for d in TUNED_dicts for x in d['passed_eval']]

# %%
print(f"np.mean(pose_only_passed_evals) = {np.mean(pose_only_passed_evals)}")
print(f"np.mean(pose_only_more_passed_evals) = {np.mean(pose_only_more_passed_evals)}")
print(f"np.mean(joints_only_passed_evals) = {np.mean(joints_only_passed_evals)}")
print(f"np.mean(joints_only_more_passed_evals) = {np.mean(joints_only_more_passed_evals)}")
print(f"np.mean(joints_only_more2_passed_evals) = {np.mean(joints_only_more2_passed_evals)}")
print(f"np.mean(pose_only_less_passed_evals) = {np.mean(pose_only_less_passed_evals)}")
print(f"np.mean(pose_joints_less_passed_evals) = {np.mean(pose_joints_less_passed_evals)}")
print(f"np.mean(go_only_passed_evals) = {np.mean(go_only_passed_evals)}")
print(f"np.mean(go_only_more_passed_evals) = {np.mean(go_only_more_passed_evals)}")
print(f"np.mean(go_only_more2_passed_evals) = {np.mean(go_only_more2_passed_evals)}")
print(f"np.mean(go_only_more3_passed_evals) = {np.mean(go_only_more3_passed_evals)}")
print(f"np.mean(TUNED_passed_evals) = {np.mean(TUNED_passed_evals)}")

# %%
TUNED_dicts[7]["passed_eval"][:80]

# %%
TUNED_dicts[7]["passed_simulation"].mean()

# %%
