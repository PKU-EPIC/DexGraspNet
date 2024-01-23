"""
Last modified date: 2023.02.19
Author: Ruicheng Wang
Description: Class IsaacValidator
"""

from isaacgym import gymapi, torch_utils, gymutil
import math
from time import sleep
from tqdm import tqdm
from utils.hand_model_type import (
    handmodeltype_to_allowedcontactlinknames,
    handmodeltype_to_joint_names,
    HandModelType,
    handmodeltype_to_hand_root_hand_file,
    handmodeltype_to_hand_root_hand_file_with_virtual_joints,
)
from collections import defaultdict
import torch
from enum import Enum, auto
from typing import List, Optional, Tuple
import transforms3d

## NERF GRASPING START ##

import numpy as np
import os
import json
from pathlib import Path
import shutil
from PIL import Image
import imageio
import pathlib

from utils.quaternions import Quaternion
from datetime import datetime

CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH = 400, 400
CAMERA_HORIZONTAL_FOV_DEG = 35.0
CAMERA_VERTICAL_FOV_DEG = (
    CAMERA_IMG_HEIGHT / CAMERA_IMG_WIDTH
) * CAMERA_HORIZONTAL_FOV_DEG
OBJ_SEGMENTATION_ID = 1
TABLE_SEGMENTATION_ID = 2

# collision_filter is a bit mask that lets you filter out collision between bodies. Two bodies will not collide if their collision filters have a common bit set. 
HAND_COLLISION_FILTER = 0  # 0 means turn off collisions
OBJ_COLLISION_FILTER = 0  # 0 means don't turn off collisions
TABLE_COLLISION_FILTER = 0  # 0 means don't turn off collisions
RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE = 1
ISAAC_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_fixed_camera_transform(
    gym, sim, env, camera
) -> Tuple[torch.Tensor, Quaternion]:
    # OLD: currently x+ is pointing down camera view axis - other degree of freedom is messed up
    # NEW: currently z- is pointing down camera view axis - other degree of freedom is messed up
    # output will have x+ be optical axis, y+ pointing left (looking down camera) and z+ pointing up
    t = gym.get_camera_transform(sim, env, camera)
    pos = torch.tensor([t.p.x, t.p.y, t.p.z])
    quat = Quaternion.fromWLast([t.r.x, t.r.y, t.r.z, t.r.w])

    # x_axis = torch.tensor([1.0, 0, 0])
    y_axis = torch.tensor([0, 1.0, 0])
    z_axis = torch.tensor([0, 0, 1.0])

    optical_axis = quat.rotate(-z_axis)
    side_left_axis = y_axis.cross(optical_axis)
    up_axis = optical_axis.cross(side_left_axis)

    optical_axis /= torch.norm(optical_axis)
    side_left_axis /= torch.norm(side_left_axis)
    up_axis /= torch.norm(up_axis)

    new_x_axis = optical_axis
    new_y_axis = side_left_axis
    new_z_axis = up_axis

    rot_matrix = torch.stack([new_x_axis, new_y_axis, new_z_axis], dim=-1)
    fixed_quat = Quaternion.fromMatrix(rot_matrix)

    return pos, fixed_quat


## NERF GRASPING END ##


gym = gymapi.acquire_gym()


def get_link_idx_to_name_dict(env, actor_handle) -> dict:
    link_idx_to_name_dict = {}
    num_links = gym.get_actor_rigid_body_count(env, actor_handle)
    link_names = gym.get_actor_rigid_body_names(env, actor_handle)
    assert len(link_names) == num_links
    for i in range(num_links):
        link_idx = gym.get_actor_rigid_body_index(
            env, actor_handle, i, gymapi.DOMAIN_ENV
        )
        link_name = link_names[i]
        link_idx_to_name_dict[link_idx] = link_name
    return link_idx_to_name_dict


class AutoName(Enum):
    # https://docs.python.org/3.9/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


class ValidationType(AutoName):
    GRAVITY_IN_6_DIRS = auto()
    NO_GRAVITY_SHAKING = auto()
    GRAVITY_AND_TABLE = auto()


class IsaacValidator:
    def __init__(
        self,
        hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND,
        mode: str = "direct",
        hand_friction: float = 0.6,
        obj_friction: float = 0.6,
        num_sim_steps: int = 120,
        gpu: int = 0,
        debug_interval: float = 0.05,
        start_with_step_mode: bool = False,
        validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING,
        use_cpu: bool = True,
    ) -> None:
        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.debug_interval = debug_interval
        self.num_sim_steps = num_sim_steps
        self.gpu = gpu
        self.validation_type = validation_type

        self.joint_names = handmodeltype_to_joint_names[hand_model_type]
        self.allowed_contact_link_names = handmodeltype_to_allowedcontactlinknames[
            hand_model_type
        ]

        self._reset_state()

        # Need virtual joints to control hand position
        if self.validation_type == ValidationType.GRAVITY_IN_6_DIRS:
            self.hand_root, self.hand_file = handmodeltype_to_hand_root_hand_file[
                hand_model_type
            ]
            self.virtual_joint_names = []
        elif self.validation_type == ValidationType.NO_GRAVITY_SHAKING:
            (
                self.hand_root,
                self.hand_file,
            ) = handmodeltype_to_hand_root_hand_file_with_virtual_joints[
                hand_model_type
            ]
            # HACK: Hardcoded virtual joint names
            self.virtual_joint_names = [
                "virtual_joint_translation_x",
                "virtual_joint_translation_y",
                "virtual_joint_translation_z",
                "virtual_joint_rotation_z",
                "virtual_joint_rotation_y",
                "virtual_joint_rotation_x",
            ]
        elif self.validation_type == ValidationType.GRAVITY_AND_TABLE:
            (
                self.hand_root,
                self.hand_file,
            ) = handmodeltype_to_hand_root_hand_file_with_virtual_joints[
                hand_model_type
            ]
            # HACK: Hardcoded virtual joint names
            self.virtual_joint_names = [
                "virtual_joint_translation_x",
                "virtual_joint_translation_y",
                "virtual_joint_translation_z",
                "virtual_joint_rotation_z",
                "virtual_joint_rotation_y",
                "virtual_joint_rotation_x",
            ]
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")

        self.sim_params = gymapi.SimParams()

        # set common parameters
        self.sim_params.dt = 1 / 60
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)

        # set PhysX-specific parameters
        print("~" * 80)
        print(
            "NOTE: Tyler has had big discrepancy between using GPU vs CPU, hypothesize that CPU is safer"
        )
        print("~" * 80 + "\n")
        self.sim_params.physx.use_gpu = not use_cpu

        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 8
        self.sim_params.physx.contact_offset = 0.005
        self.sim_params.physx.rest_offset = 0.0

        self.sim_params.use_gpu_pipeline = False
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 800
        self.camera_props.height = 600
        self.camera_props.use_collision_geometry = (
            True  # TODO: Maybe change this to see true visual
        )

        # set viewer
        self.viewer = None
        if mode == "gui":
            self.has_viewer = True
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
            gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(0, 0, 1), gymapi.Vec3(0, 0, 0)
            )
            self.subscribe_to_keyboard_events()
        else:
            self.has_viewer = False

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.collapse_fixed_joints = True
        self.hand_asset_options.fix_base_link = True

        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 500

        if self.validation_type == ValidationType.GRAVITY_IN_6_DIRS:
            self.obj_asset_options.disable_gravity = False
        elif self.validation_type == ValidationType.NO_GRAVITY_SHAKING:
            self.obj_asset_options.disable_gravity = True
        elif self.validation_type == ValidationType.GRAVITY_AND_TABLE:
            self.obj_asset_options.disable_gravity = False
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")

        self.test_rotations = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1 * math.pi),
            ),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * math.pi),
            ),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -0.5 * math.pi),
            ),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi),
            ),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * math.pi),
            ),
        ]

        self.is_paused = False
        self.is_step_mode = self.has_viewer and start_with_step_mode

        self.hand_asset = gym.load_asset(
            self.sim, self.hand_root, self.hand_file, self.hand_asset_options
        )

    def set_obj_asset(self, obj_root: str, obj_file: str) -> None:
        self.obj_asset = gym.load_asset(
            self.sim, obj_root, obj_file, self.obj_asset_options
        )

    def add_env_all_test_rotations(
        self,
        hand_quaternion_wxyz: np.ndarray,
        hand_translation: np.ndarray,
        hand_qpos: np.ndarray,
        obj_scale: float,
        target_qpos: np.ndarray,
        add_random_pose_noise: bool = False,
    ) -> None:
        for test_rotation_idx in range(len(self.test_rotations)):
            self.add_env_single_test_rotation(
                hand_quaternion_wxyz=hand_quaternion_wxyz,
                hand_translation=hand_translation,
                hand_qpos=hand_qpos,
                obj_scale=obj_scale,
                target_qpos=target_qpos,
                add_random_pose_noise=add_random_pose_noise,
                test_rotation_index=test_rotation_idx,
            )

    def add_env_single_test_rotation(
        self,
        hand_quaternion_wxyz: np.ndarray,
        hand_translation: np.ndarray,
        hand_qpos: np.ndarray,
        obj_scale: float,
        target_qpos: np.ndarray,
        add_random_pose_noise: bool = False,
        test_rotation_index: int = 0,
        record: bool = False,
    ) -> None:
        collision_idx = len(self.envs)  # Should be unique for each env so envs don't collide

        # Set test rotation
        test_rot = self.test_rotations[test_rotation_index]

        # Create env
        env = gym.create_env(
            self.sim,
            gymapi.Vec3(-1, -1, -1),
            gymapi.Vec3(1, 1, 1),
            len(self.test_rotations),
        )
        self.envs.append(env)

        self._setup_hand(
            env=env,
            hand_quaternion_wxyz=hand_quaternion_wxyz,
            hand_translation=hand_translation,
            hand_qpos=hand_qpos,
            transformation=test_rot,
            target_qpos=target_qpos,
            collision_idx=collision_idx,
        )

        self._setup_obj(
            env,
            obj_scale,
            test_rot,
            collision_idx=collision_idx,
            add_random_pose_noise=add_random_pose_noise,
        )

        self.init_rel_obj_poses.append(
            self.init_hand_poses[-1].inverse() * self.init_obj_poses[-1]
        )

        if self.validation_type == ValidationType.GRAVITY_AND_TABLE:
            self._setup_table(env=env, transformation=test_rot, collision_idx=collision_idx, obj_scale=obj_scale)

        if record:
            self._setup_camera(env)

    def _setup_table(self, env, transformation: gymapi.Transform, collision_idx: int, obj_scale: int) -> None:
        OBJ_MAX_EXTENT_FROM_ORIGIN = 1.0 * obj_scale
        TABLE_THICKNESS = 0.1
        y_offset = OBJ_MAX_EXTENT_FROM_ORIGIN + TABLE_THICKNESS / 2
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, -y_offset, 0)
        table_pose.r = gymapi.Quat(0, 0, 0, 1)

        table_pose = transformation * table_pose

        # Create table
        table_actor_handle = gym.create_actor(
            env,
            self.table_asset,
            table_pose,
            "table",
            collision_idx,
            TABLE_COLLISION_FILTER,
            TABLE_SEGMENTATION_ID,
        )

        # Set table shape props
        table_shape_props = gym.get_actor_rigid_shape_properties(env, table_actor_handle)
        for i in range(len(table_shape_props)):
            table_shape_props[i].friction = 1.0
        gym.set_actor_rigid_shape_properties(env, table_actor_handle, table_shape_props)
        return

    # TODO: be less lazy, integrate with NeRF datagen.
    def _setup_camera(self, env) -> None:
        camera_properties = gymapi.CameraProperties()  # type: ignore

        camera_properties.width = int(
            CAMERA_IMG_WIDTH / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )
        camera_properties.height = int(
            CAMERA_IMG_HEIGHT / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )

        camera_handle = gym.create_camera_sensor(
            env,
            camera_properties,
        )
        self.camera_handles.append(camera_handle)
        self.camera_properties_list.append(camera_properties)

        self.camera_envs.append(env)

        cam_target = gymapi.Vec3(0, 0, 0)  # type: ignore  # where object s
        cam_pos = cam_target + gymapi.Vec3(-0.25, 0.1, 0.1)  # Define offset

        self.video_frames.append([])
        gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

    def _setup_hand(
        self,
        env,
        hand_quaternion_wxyz: np.ndarray,
        hand_translation: np.ndarray,
        hand_qpos: np.ndarray,
        transformation: gymapi.Transform,
        target_qpos: np.ndarray,
        collision_idx: int,
    ) -> None:
        # Set hand pose
        hand_pose = gymapi.Transform()
        hand_pose.r = gymapi.Quat(*hand_quaternion_wxyz[1:], hand_quaternion_wxyz[0])
        hand_pose.p = gymapi.Vec3(*hand_translation)
        hand_pose = transformation * hand_pose
        self.init_hand_poses.append(hand_pose)

        # Create hand
        hand_actor_handle = gym.create_actor(
            env,
            self.hand_asset,
            hand_pose,
            "hand",
            collision_idx,
            HAND_COLLISION_FILTER,
        )
        self.hand_handles.append(hand_actor_handle)

        # Store target hand qpos for later
        self.target_qpos_list.append(target_qpos)
        self.init_qpos_list.append(hand_qpos)

        # Set hand dof props
        hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)

        # TODO: Consider making finger joints pos controlled and virtual joints vel controlled
        hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)

        # Finger joints
        for joint in self.joint_names:
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            hand_props["stiffness"][joint_idx] = 50.0
            hand_props["damping"][joint_idx] = 0.0

        # Virtual joints
        for joint in self.virtual_joint_names:
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            HARD_SHAKE_STIFFNESS = 500.0
            LIGHT_SHAKE_STIFFNESS = 200.0
            hand_props["stiffness"][joint_idx] = LIGHT_SHAKE_STIFFNESS
            hand_props["damping"][joint_idx] = 10.0

        gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)

        # Set hand dof states
        dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
        for i, joint in enumerate(self.joint_names):
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            dof_states["pos"][joint_idx] = hand_qpos[i]
        gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)

        # Set hand dof targets to current pos
        self._set_dof_pos_targets(
            env=env,
            hand_actor_handle=hand_actor_handle,
            target_qpos=hand_qpos,
        )

        # Store hand link_idx_to_name_dict
        self.hand_link_idx_to_name_dicts.append(
            get_link_idx_to_name_dict(env=env, actor_handle=hand_actor_handle)
        )

        # Set hand shape props
        hand_shape_props = gym.get_actor_rigid_shape_properties(env, hand_actor_handle)
        for i in range(len(hand_shape_props)):
            hand_shape_props[i].friction = self.hand_friction
        gym.set_actor_rigid_shape_properties(env, hand_actor_handle, hand_shape_props)
        return

    def _set_dof_pos_targets(
        self,
        env,
        hand_actor_handle,
        target_qpos: np.ndarray,
    ) -> None:
        dof_pos_targets = gym.get_actor_dof_position_targets(env, hand_actor_handle)
        for i, joint in enumerate(self.joint_names):
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            dof_pos_targets[joint_idx] = target_qpos[i]
        gym.set_actor_dof_position_targets(env, hand_actor_handle, dof_pos_targets)

    def _setup_obj(
        self,
        env,
        obj_scale: float,
        transformation: gymapi.Transform,
        collision_idx: int,
        add_random_pose_noise: bool = False,
    ) -> None:
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0, 0, 0)
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)

        if add_random_pose_noise:
            TRANSLATION_NOISE_CM = 0.5
            TRANSLATION_NOISE_M = TRANSLATION_NOISE_CM / 100
            ROTATION_NOISE_DEG = 5
            xyz_noise = np.random.uniform(
                -TRANSLATION_NOISE_M, TRANSLATION_NOISE_M, 3
            )
            rpy_noise = np.random.uniform(
                -ROTATION_NOISE_DEG, ROTATION_NOISE_DEG, 3
            ) * math.pi / 180
            quat_wxyz = transforms3d.euler.euler2quat(*rpy_noise)
            assert xyz_noise.shape == (3,)
            assert rpy_noise.shape == (3,)
            assert quat_wxyz.shape == (4,)

            obj_pose.p = gymapi.Vec3(*xyz_noise)
            obj_pose.r = gymapi.Quat(*quat_wxyz[1:], quat_wxyz[0])

        obj_pose = transformation * obj_pose
        self.init_obj_poses.append(obj_pose)

        # Create obj
        obj_actor_handle = gym.create_actor(
            env,
            self.obj_asset,
            obj_pose,
            "obj",
            collision_idx,
            OBJ_COLLISION_FILTER,
            OBJ_SEGMENTATION_ID,
        )
        self.obj_handles.append(obj_actor_handle)
        gym.set_actor_scale(env, obj_actor_handle, obj_scale)

        # Store obj link_idx_to_name_dict
        self.obj_link_idx_to_name_dicts.append(
            get_link_idx_to_name_dict(env=env, actor_handle=obj_actor_handle)
        )

        # Set obj shape props
        obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_actor_handle)
        for i in range(len(obj_shape_props)):
            obj_shape_props[i].friction = self.obj_friction
        gym.set_actor_rigid_shape_properties(env, obj_actor_handle, obj_shape_props)
        return

    def run_sim(self) -> Tuple[List[bool], List[bool]]:
        gym.prepare_sim(self.sim)  # TODO: Check if this is needed?

        objs_stationary_before_hand_joint_closed = self._run_sim_steps()

        # Render out all videos.
        if self.camera_handles:
            for ii, _ in enumerate(self.camera_envs):
                video_path = pathlib.Path(f"videos/{ISAAC_DATETIME_STR}_video_{ii}.mp4")
                if not video_path.parent.exists():
                    video_path.parent.mkdir(parents=True)
                print(f"Rendering camera {ii} to video at path {video_path}.")
                self._render_video(
                    video_frames=self.video_frames[ii],
                    video_path=video_path,
                    fps=int(1 / self.sim_params.dt),
                )
                print(f"Done rendering camera {ii}.")

        successes = self._check_successes()
        return successes, objs_stationary_before_hand_joint_closed

    def _check_successes(self) -> List[bool]:
        successes = []
        for i, (
            env,
            hand_link_idx_to_name,
            obj_link_idx_to_name,
            obj_handle,
            init_rel_obj_pose,
        ) in enumerate(
            zip(
                self.envs,
                self.hand_link_idx_to_name_dicts,
                self.obj_link_idx_to_name_dicts,
                self.obj_handles,
                self.init_rel_obj_poses,
            )
        ):
            contacts = gym.get_env_rigid_contacts(env)

            # Find hand object contacts
            hand_object_contacts = []
            for contact in contacts:
                body0 = contact["body0"]
                body1 = contact["body1"]
                is_hand_object_contact = (
                    body0 in hand_link_idx_to_name and body1 in obj_link_idx_to_name
                ) or (body1 in hand_link_idx_to_name and body0 in obj_link_idx_to_name)
                if is_hand_object_contact:
                    hand_object_contacts.append(contact)

            # Count hand link contacts
            hand_link_contact_count = defaultdict(int)
            for contact in hand_object_contacts:
                body0 = contact["body0"]
                body1 = contact["body1"]
                hand_link_name = (
                    hand_link_idx_to_name[body0]
                    if body0 in hand_link_idx_to_name
                    else hand_link_idx_to_name[body1]
                )
                hand_link_contact_count[hand_link_name] += 1

            # Success conditions
            not_allowed_contacts = set(hand_link_contact_count.keys()) - set(
                self.allowed_contact_link_names
            )

            # palm_link_idx
            palm_link_idxs = [
                idx
                for idx, name in hand_link_idx_to_name.items()
                if name == "palm_link"
            ]
            assert (
                len(palm_link_idxs) == 1
            ), f"len(palm_link_idxs) = {len(palm_link_idxs)}"
            palm_link_idx = palm_link_idxs[0]

            final_hand_pose = gymapi.Transform()
            final_hand_pose.p, final_hand_pose.r = gym.get_actor_rigid_body_states(
                env, self.hand_handles[i], gymapi.STATE_POS
            )[palm_link_idx]["pose"]

            OBJ_BASE_LINK_IDX = 0
            final_obj_pose = gymapi.Transform()
            final_obj_pose.p, final_obj_pose.r = gym.get_actor_rigid_body_states(
                env, obj_handle, gymapi.STATE_POS
            )[OBJ_BASE_LINK_IDX]["pose"]
            init_rel_obj_pos = torch.tensor(
                [init_rel_obj_pose.p.x, init_rel_obj_pose.p.y, init_rel_obj_pose.p.z]
            )
            init_rel_obj_quat = torch.tensor(
                [
                    init_rel_obj_pose.r.x,
                    init_rel_obj_pose.r.y,
                    init_rel_obj_pose.r.z,
                    init_rel_obj_pose.r.w,
                ]
            )
            final_rel_obj_pose = final_hand_pose.inverse() * final_obj_pose

            final_rel_obj_pos = torch.tensor(
                [final_rel_obj_pose.p.x, final_rel_obj_pose.p.y, final_rel_obj_pose.p.z]
            )
            final_rel_obj_quat = torch.tensor(
                [
                    final_rel_obj_pose.r.x,
                    final_rel_obj_pose.r.y,
                    final_rel_obj_pose.r.z,
                    final_rel_obj_pose.r.w,
                ]
            )

            quat_diff = torch_utils.quat_mul(
                final_rel_obj_quat, torch_utils.quat_conjugate(init_rel_obj_quat)
            )
            pos_change = torch.linalg.norm(final_rel_obj_pos - init_rel_obj_pos).item()

            euler_change = torch.stack(
                torch_utils.get_euler_xyz(quat_diff[None, ...])
            ).abs()
            euler_change = torch.where(
                euler_change > math.pi, 2 * math.pi - euler_change, euler_change
            )
            max_euler_change = euler_change.max().rad2deg().item()

            success = (
                len(hand_object_contacts) > 0
                and len(hand_link_contact_count.keys()) >= 3
                and len(not_allowed_contacts) == 0
                and pos_change < 0.1
                and max_euler_change < 30
            )

            successes.append(success)

            DEBUG = True
            if DEBUG:
                print(f"i = {i}")
                print(f"success = {success}")
                print(f"pos_change = {pos_change}")
                print(f"max_euler_change = {max_euler_change}")
                print(f"len(contacts) = {len(contacts)}")
                print(f"len(hand_object_contacts) = {len(hand_object_contacts)}")
                print(f"hand_link_contact_count = {hand_link_contact_count}")
                print(f"not_allowed_contacts = {not_allowed_contacts}")
                print(
                    f"len(hand_link_contact_count.keys()) = {len(hand_link_contact_count.keys())}"
                )
                print("-------------")

        return successes

    def _run_sim_steps(self) -> List[bool]:
        sim_step_idx = 0
        default_desc = "Simulating"
        pbar = tqdm(total=self.num_sim_steps, desc=default_desc, dynamic_ncols=True)

        objs_stationary_before_hand_joint_closed = [True for _ in range(len(self.envs))]

        while sim_step_idx < self.num_sim_steps:
            # Set hand joint targets only after first few steps
            #   Heard that first few steps may be less deterministic because of isaacgym state
            #   Eg. contact buffers, so not moving for the first few steps may resolve this by clearing buffers
            #   Move hand joints to target qpos linearly over a few steps
            NUM_STEPS_TO_NOT_MOVE_HAND_JOINTS = 10
            NUM_STEPS_TO_CLOSE_HAND_JOINTS = 15
            if sim_step_idx >= NUM_STEPS_TO_NOT_MOVE_HAND_JOINTS:
                frac_progress = (
                    sim_step_idx - NUM_STEPS_TO_NOT_MOVE_HAND_JOINTS
                ) / NUM_STEPS_TO_CLOSE_HAND_JOINTS
                frac_progress = np.clip(
                    frac_progress,
                    0,
                    1,
                )
                for env, hand_actor_handle, target_qpos, init_qpos in zip(
                    self.envs,
                    self.hand_handles,
                    self.target_qpos_list,
                    self.init_qpos_list,
                ):
                    current_target_qpos = (
                        target_qpos * frac_progress + init_qpos * (1 - frac_progress)
                    )
                    self._set_dof_pos_targets(
                        env=env,
                        hand_actor_handle=hand_actor_handle,
                        target_qpos=current_target_qpos,
                    )
            else:
                # Check if object has velocity before hand joints start moving
                OBJ_BASE_LINK_IDX = 0
                for i, (env, obj_handle) in enumerate(zip(self.envs, self.obj_handles)):
                    if not objs_stationary_before_hand_joint_closed[i]:
                        continue

                    obj_vel, _ = gym.get_actor_rigid_body_states(
                        env, obj_handle, gymapi.STATE_VEL
                    )[OBJ_BASE_LINK_IDX]["vel"]
                    obj_speed = np.linalg.norm(
                        [obj_vel["x"], obj_vel["y"], obj_vel["z"]]
                    )
                    if obj_speed > 0.01:
                        objs_stationary_before_hand_joint_closed[i] = False

            # Set virtual joint targets (wrist pose) only after a few steps
            #   Let hand close and object settle before moving wrist
            #   Only do this when virtual joints exist
            NUM_STEPS_TO_NOT_MOVE_WRIST_POSE = 30
            assert NUM_STEPS_TO_NOT_MOVE_WRIST_POSE > (NUM_STEPS_TO_NOT_MOVE_HAND_JOINTS + NUM_STEPS_TO_CLOSE_HAND_JOINTS)
            if (
                sim_step_idx >= NUM_STEPS_TO_NOT_MOVE_WRIST_POSE
                and len(self.virtual_joint_names) > 0
            ):
                frac_progress = (sim_step_idx - NUM_STEPS_TO_NOT_MOVE_WRIST_POSE) / (
                    self.num_sim_steps - NUM_STEPS_TO_NOT_MOVE_WRIST_POSE
                )
                virtual_joint_dof_pos_targets = (
                    self._compute_virtual_joint_dof_pos_targets(
                        frac_progress=frac_progress
                    )
                )
                self._set_virtual_joint_dof_pos_targets(virtual_joint_dof_pos_targets)
            else:
                virtual_joint_dof_pos_targets = None

            # Step physics if not paused
            if not self.is_paused:
                gym.simulate(self.sim)
                gym.fetch_results(
                    self.sim, True
                )  # TODO: Check if this slows things down

                if self.camera_handles and sim_step_idx > 0:
                    gym.step_graphics(self.sim)
                    gym.render_all_camera_sensors(self.sim)
                    for ii, env in enumerate(self.camera_envs):
                        self.video_frames[ii].append(
                            gym.get_camera_image(
                                self.sim,
                                env,
                                self.camera_handles[ii],
                                gymapi.IMAGE_COLOR,
                            ).reshape(
                                self.camera_properties_list[ii].height,
                                self.camera_properties_list[ii].width,
                                4,  # RGBA
                            )
                        )
                sim_step_idx += 1
                pbar.update(1)

                # Step mode
                if self.is_step_mode:
                    self.is_paused = True

            # Update viewer
            if self.has_viewer:
                sleep(self.debug_interval)
                if gym.query_viewer_has_closed(self.viewer):
                    break

                # Check for keyboard events
                for event in gym.query_viewer_action_events(self.viewer):
                    if event.value > 0 and event.action in self.event_to_function:
                        self.event_to_function[event.action]()

                gym.clear_lines(self.viewer)

                # Visualize origin lines
                self._visualize_origin_lines()

                # Visualize virtual joint targets
                if virtual_joint_dof_pos_targets is not None:
                    self._visualize_virtual_joint_dof_pos_targets(
                        virtual_joint_dof_pos_targets
                    )

                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)

                # Update progress bar text
                desc = default_desc
                desc += ". 'KEY_SPACE' = toggle pause. 'KEY_S' = toggle step mode"
                if self.is_paused:
                    desc += ". Paused"
                if self.is_step_mode:
                    desc += ". Step mode on"
                pbar.set_description(desc)

        return objs_stationary_before_hand_joint_closed

    @property
    def table_asset(self):
        if not hasattr(self, "_table_asset"):
            table_asset_options = gymapi.AssetOptions()
            table_asset_options.fix_base_link = True
            table_root, table_file = (
                "table",
                "table.urdf",
            )
            self._table_asset = gym.load_asset(
                self.sim, table_root, table_file, table_asset_options
            )
        return self._table_asset

    def _render_video(
        self, video_frames: List[torch.Tensor], video_path: pathlib.Path, fps: int
    ):
        print(f"number of frames: {len(video_frames)}")
        imageio.mimsave(video_path, video_frames, fps=fps)

    def _compute_virtual_joint_dof_pos_targets(
        self,
        frac_progress: float,
    ) -> List[torch.Tensor]:
        assert len(self.virtual_joint_names) == 6

        # Shaking / perturbation parameters for virtual joint targets.

        # Set dof pos targets [+x, -x]*N, 0, [+y, -y]*N, 0, [+z, -z]*N
        dist_to_move = 0.05
        N = 2
        directions_sequence = [
            *([[dist_to_move, 0.0, 0.0], [-dist_to_move, 0.0, 0.0]] * N),
            [0.0, 0.0, 0.0],
            *([[0.0, dist_to_move, 0.0], [0.0, -dist_to_move, 0.0]] * N),
            [0.0, 0.0, 0.0],
            *([[0.0, 0.0, dist_to_move], [0.0, 0.0, -dist_to_move]] * N),
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

        direction_idx = int(frac_progress * len(directions_sequence))
        direction = directions_sequence[direction_idx]

        # direction in global frame
        # dof_pos_targets in hand frame
        # so need to perform inverse hand frame rotation
        rotation_transforms = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), init_hand_pose.r)
            for init_hand_pose in self.init_hand_poses
        ]
        dof_pos_targets = [
            rotation_transform.inverse().transform_point(gymapi.Vec3(*direction))
            for rotation_transform in rotation_transforms
        ]
        dof_pos_targets = [
            torch.tensor([dof_pos_target.x, dof_pos_target.y, dof_pos_target.z])
            for dof_pos_target in dof_pos_targets
        ]

        # Add target angles
        target_angles = torch.tensor([0.0, 0.0, 0.0])
        dof_pos_targets = [
            torch.cat([dof_pos_target, target_angles])
            for dof_pos_target in dof_pos_targets
        ]

        return dof_pos_targets

    def _set_virtual_joint_dof_pos_targets(
        self, dof_pos_targets: List[torch.Tensor]
    ) -> None:
        for env, hand_handle, dof_pos_target in zip(
            self.envs, self.hand_handles, dof_pos_targets
        ):
            actor_dof_pos_targets = gym.get_actor_dof_position_targets(env, hand_handle)

            for i, joint in enumerate(self.virtual_joint_names):
                joint_idx = gym.find_actor_dof_index(
                    env, hand_handle, joint, gymapi.DOMAIN_ACTOR
                )
                actor_dof_pos_targets[joint_idx] = dof_pos_target[i]
            gym.set_actor_dof_position_targets(env, hand_handle, actor_dof_pos_targets)

    def _visualize_virtual_joint_dof_pos_targets(
        self, dof_pos_targets: List[torch.Tensor]
    ) -> None:
        if not self.has_viewer:
            return
        visualization_sphere_green = gymutil.WireframeSphereGeometry(
            radius=0.01, num_lats=10, num_lons=10, color=(0, 1, 0)
        )
        for env, init_hand_pose, dof_pos_target in zip(
            self.envs, self.init_hand_poses, dof_pos_targets
        ):
            # dof_pos_targets in hand frame
            # direction in global frame
            # so need to perform hand frame rotation
            dof_pos_target = gymapi.Vec3(
                dof_pos_target[0], dof_pos_target[1], dof_pos_target[2]
            )
            rotation_transform = gymapi.Transform(
                gymapi.Vec3(0, 0, 0), init_hand_pose.r
            )
            direction = rotation_transform.transform_point(dof_pos_target)

            sphere_pose = gymapi.Transform(
                gymapi.Vec3(
                    init_hand_pose.p.x + direction.x,
                    init_hand_pose.p.y + direction.y,
                    init_hand_pose.p.z + direction.z,
                ),
                r=None,
            )
            gymutil.draw_lines(
                visualization_sphere_green, gym, self.viewer, env, sphere_pose
            )

    def _visualize_origin_lines(self) -> None:
        if not self.has_viewer:
            return

        origin_pos = gymapi.Vec3(0, 0, 0)
        x_pos = gymapi.Vec3(0.1, 0, 0)
        y_pos = gymapi.Vec3(0, 0.1, 0)
        z_pos = gymapi.Vec3(0, 0, 0.1)
        red = gymapi.Vec3(1, 0, 0)
        green = gymapi.Vec3(0, 1, 0)
        blue = gymapi.Vec3(0, 0, 1)
        for pos, color in zip([x_pos, y_pos, z_pos], [red, green, blue]):
            for env in self.envs:
                gymutil.draw_line(origin_pos, pos, color, gym, self.viewer, env)

    def reset_simulator(self) -> None:
        self.destroy()

        if self.has_viewer:
            self.viewer = gym.create_viewer(self.sim, self.camera_props)

        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)

        # Recreate hand asset in new sim.
        self.hand_asset = gym.load_asset(
            self.sim, self.hand_root, self.hand_file, self.hand_asset_options
        )

        self._reset_state()

    def destroy(self) -> None:
        for env in self.envs:
            gym.destroy_env(env)
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)

    def _reset_state(self) -> None:
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_link_idx_to_name_dicts = []
        self.obj_link_idx_to_name_dicts = []
        self.init_obj_poses = []
        self.init_hand_poses = []
        self.init_rel_obj_poses = []
        self.target_qpos_list = []
        self.init_qpos_list = []

        self.camera_handles = []
        self.camera_envs = []
        self.camera_properties_list = []
        self.video_frames = []
        self.obj_asset = None

    def subscribe_to_keyboard_events(self) -> None:
        if self.has_viewer:
            self.event_to_key = {
                "STEP_MODE": gymapi.KEY_S,
                "PAUSE_SIM": gymapi.KEY_SPACE,
            }
            self.event_to_function = {
                "STEP_MODE": self._step_mode_callback,
                "PAUSE_SIM": self._pause_sim_callback,
            }
        else:
            self.event_to_key = {}
            self.event_to_function = {}
        assert set(self.event_to_key.keys()) == set(self.event_to_function.keys())

        if self.has_viewer and self.viewer is not None:
            for event, key in self.event_to_key.items():
                gym.subscribe_viewer_keyboard_event(self.viewer, key, event)

    ## KEYBOARD EVENT SUBSCRIPTIONS START ##
    def _step_mode_callback(self):
        self.is_step_mode = not self.is_step_mode
        print(f"Simulation is in {'step' if self.is_step_mode else 'continuous'} mode")
        self._pause_sim_callback()

    def _pause_sim_callback(self):
        self.is_paused = not self.is_paused
        print(f"Simulation is {'paused' if self.is_paused else 'unpaused'}")

    ## KEYBOARD EVENT SUBSCRIPTIONS END ##

    ## NERF DATA COLLECTION START ##
    def add_env_nerf_data_collection(
        self,
        obj_scale: float,
    ) -> None:
        # Set test rotation
        identity_transform = gymapi.Transform(
            gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)
        )

        # Create env
        spacing = 1.0
        env = gym.create_env(
            self.sim,
            gymapi.Vec3(-spacing, -spacing, 0.0),
            gymapi.Vec3(spacing, spacing, spacing),
            0,  # TODO: Should it be 0?
        )
        self.envs.append(env)

        self._setup_obj(env, obj_scale, identity_transform, collision_idx=0)

    def save_images(self, folder: str, overwrite: bool = False) -> None:
        assert len(self.envs) == 1
        self._setup_cameras(self.envs[0])

        gym.step_graphics(self.sim)
        gym.render_all_camera_sensors(self.sim)
        path = self._setup_save_dir(folder, overwrite)

        for ii, camera_handle in enumerate(self.camera_handles):
            self._save_single_image(path, ii, camera_handle)

        # Avoid segfault if run multiple times by destroying camera sensors
        self._destroy_cameras(self.envs[0])

    def save_images_lightweight(
        self,
        folder: str,
        obj_scale: float,
        overwrite: bool = False,
        generate_seg: bool = False,
        generate_depth: bool = False,
    ) -> None:
        assert len(self.envs) == 1
        camera_radius = 3 * obj_scale
        self._setup_cameras(self.envs[0], radius=camera_radius)

        gym.step_graphics(self.sim)
        gym.render_all_camera_sensors(self.sim)
        print("rendered!")
        path = self._setup_save_dir(folder, overwrite)

        for ii, camera_handle in enumerate(self.camera_handles):
            self._save_single_image_lightweight(
                path,
                ii,
                camera_handle,
                generate_seg=generate_seg,
                generate_depth=generate_depth,
            )

        # Avoid segfault if run multiple times by destroying camera sensors
        self._destroy_cameras(self.envs[0])

    def _setup_cameras(self, env, num_cameras=250, radius=0.3):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = CAMERA_HORIZONTAL_FOV_DEG
        camera_props.width = CAMERA_IMG_WIDTH
        camera_props.height = CAMERA_IMG_HEIGHT

        # Generates camera positions uniformly sampled from sphere around object.
        u_vals = np.random.uniform(-1.0, 1.0, num_cameras)
        th_vals = np.random.uniform(0.0, 2 * np.pi, num_cameras)

        x_vals = radius * np.sqrt(1 - np.square(u_vals)) * np.cos(th_vals)
        y_vals = radius * np.sqrt(1 - np.square(u_vals)) * np.sin(th_vals)
        z_vals = radius * u_vals

        for xx, yy, zz in zip(x_vals, y_vals, z_vals):
            camera_handle = gym.create_camera_sensor(env, camera_props)
            gym.set_camera_location(
                camera_handle, env, gymapi.Vec3(xx, yy, zz), gymapi.Vec3(0.0, 0.0, 0.0)
            )
            self.camera_handles.append(camera_handle)

        # # generates camera positions along rings around object
        # heights = [0.1, 0.3, 0.25, 0.35, 0.0]
        # distances = [0.05, 0.125, 0.3, 0.3, 0.2]
        # counts = [56, 104, 96, 1, 60]
        # target_ys = [0.0, 0.1, 0.0, 0.1, 0.0]

        # # compute camera positions
        # camera_positions = []
        # for height, distance, count, target_y in zip(
        #     heights, distances, counts, target_ys
        # ):
        #     for alpha in np.linspace(0, 2 * np.pi, count, endpoint=False):
        #         pos = [distance * np.sin(alpha), height, distance * np.cos(alpha)]
        #         camera_positions.append((pos, target_y))
        # # repeat all from under since there is no ground plane
        # for height, distance, count, target_y in zip(
        #     heights, distances, counts, target_ys
        # ):
        #     if height == 0.0:
        #         print(f"Continuing because height == 0.0")
        #         continue
        #     height = -height
        #     target_y = -target_y
        #     for alpha in np.linspace(0, 2 * np.pi, count, endpoint=False):
        #         pos = [distance * np.sin(alpha), height, distance * np.cos(alpha)]
        #         camera_positions.append((pos, target_y))

        # self.camera_handles = []
        # for pos, target_y in camera_positions:
        #     camera_handle = gym.create_camera_sensor(env, camera_props)
        #     gym.set_camera_location(
        #         camera_handle, env, gymapi.Vec3(*pos), gymapi.Vec3(0, target_y, 0)
        #     )

        #     self.camera_handles.append(camera_handle)

        # self.overhead_camera_handle = gym.create_camera_sensor(env, camera_props)
        # gym.set_camera_location(
        #     self.overhead_camera_handle,
        #     env,
        #     gymapi.Vec3(0, 0.5, 0.001),
        #     gymapi.Vec3(0, 0.01, 0),
        # )

    def _destroy_cameras(self, env):
        for camera_handle in self.camera_handles:
            gym.destroy_camera_sensor(self.sim, env, camera_handle)

    def _setup_save_dir(self, folder, overwrite=False):
        path = Path(folder)

        if path.exists():
            print(path, "already exists!")
            if overwrite:
                shutil.rmtree(path)
            elif input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(path)

        path.mkdir()
        return path

    def _save_single_image(
        self, path, ii, camera_handle, numpy_depth=False, debug=False
    ):
        if debug:
            print(f"saving camera {ii}")
        env_idx = 0
        env = self.envs[env_idx]

        color_image = gym.get_camera_image(
            self.sim, env, camera_handle, gymapi.IMAGE_COLOR
        )
        color_image = color_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH, -1)
        Image.fromarray(color_image).save(path / f"col_{ii}.png")

        segmentation_image = gym.get_camera_image(
            self.sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION
        )
        segmentation_image = segmentation_image == OBJ_SEGMENTATION_ID
        segmentation_image = (
            segmentation_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH) * 255
        ).astype(np.uint8)
        Image.fromarray(segmentation_image).convert("L").save(path / f"seg_{ii}.png")

        # TODO: get_camera_image has -inf values, which can't be cast to int, should fix this for depth supervision
        depth_image = gym.get_camera_image(
            self.sim, env, camera_handle, gymapi.IMAGE_DEPTH
        )
        # distance in units I think
        depth_image = -1000 * depth_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH)
        if numpy_depth:
            np.save(path / f"dep_{ii}.npy", depth_image)
        else:
            depth_image = (depth_image).astype(np.uint8)
            Image.fromarray(depth_image).convert("L").save(path / f"dep_{ii}.png")

        pos, quat = get_fixed_camera_transform(gym, self.sim, env, camera_handle)

        with open(path / f"pos_xyz_quat_xyzw_{ii}.txt", "w+") as f:
            data = [*pos.tolist(), *quat.q[1:].tolist(), quat.q[0].tolist()]
            json.dump(data, f)

    def _save_single_image_lightweight(
        self,
        path,
        ii,
        camera_handle,
        generate_seg=False,
        generate_depth=False,
        numpy_depth=False,
        debug=False,
    ):
        if debug:
            print(f"saving camera {ii}")
        env_idx = 0
        env = self.envs[env_idx]

        # COLOR IMAGE
        # NEED THESE TEMPORARILY FOR transforms.json
        color_image = gym.get_camera_image(
            self.sim, env, camera_handle, gymapi.IMAGE_COLOR
        )
        color_image = color_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH, -1)
        Image.fromarray(color_image).save(path / f"col_{ii}.png")

        # SEGMENTATION IMAGE
        if generate_seg:
            segmentation_image = gym.get_camera_image(
                self.sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION
            )
            segmentation_image = segmentation_image == OBJ_SEGMENTATION_ID
            segmentation_image = (
                segmentation_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH) * 255
            ).astype(np.uint8)
            Image.fromarray(segmentation_image).convert("L").save(
                path / f"seg_{ii}.png"
            )

        # DEPTH IMAGE
        if generate_depth:
            depth_image = gym.get_camera_image(
                self.sim, env, camera_handle, gymapi.IMAGE_DEPTH
            )
            depth_image = -1000 * depth_image.reshape(
                CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH
            )
            if numpy_depth:
                np.save(path / f"dep_{ii}.npy", depth_image)
            else:
                depth_image = (depth_image).astype(np.uint8)
                Image.fromarray(depth_image).convert("L").save(path / f"dep_{ii}.png")

        # NEED THESE TEMPORARILY FOR transforms.json
        pos, quat = get_fixed_camera_transform(gym, self.sim, env, camera_handle)
        with open(path / f"pos_xyz_quat_xyzw_{ii}.txt", "w+") as f:
            data = [*pos.tolist(), *quat.q[1:].tolist(), quat.q[0].tolist()]
            json.dump(data, f)

    def create_train_val_test_split(
        self, folder: str, train_frac: float, val_frac: float
    ) -> None:
        assert train_frac + val_frac < 1.0
        num_imgs = len(self.camera_handles)
        num_train = int(train_frac * num_imgs)
        num_val = int(val_frac * num_imgs)
        num_test = num_imgs - num_train - num_val
        print()
        print(f"num_imgs = {num_imgs}")
        print(f"num_train = {num_train}")
        print(f"num_val = {num_val}")
        print(f"num_test = {num_test}")
        print()

        img_range = np.arange(num_imgs)

        np.random.shuffle(img_range)
        train_range = img_range[:num_train]
        test_range = img_range[num_train : (num_train + num_test)]
        val_range = img_range[(num_train + num_test) :]

        self._create_one_split(
            split_name="train", split_range=train_range, folder=folder
        )
        self._create_one_split(split_name="val", split_range=val_range, folder=folder)
        self._create_one_split(split_name="test", split_range=test_range, folder=folder)

    def create_no_split_data(self, folder: str, generate_depth: bool = False) -> None:
        # create the images folder and transforms.json
        num_imgs = len(self.camera_handles)
        print()
        print(f"num_imgs = {num_imgs}")
        print()
        img_range = np.arange(num_imgs)
        self._create_one_split(
            split_name="images",
            split_range=img_range,
            folder=folder,
            generate_depth=generate_depth,
        )

        # delete all the .txt and .png files
        directory = os.listdir(folder)
        for item in directory:
            if item.endswith(".txt"):
                os.remove(os.path.join(folder, item))
            elif item.endswith(".png"):
                os.remove(os.path.join(folder, item))

    def _run_sanity_check_proj_matrices_all_same(self):
        proj_matrix = gym.get_camera_proj_matrix(
            self.sim, self.envs[0], self.camera_handles[0]
        )
        for camera_handle in self.camera_handles:
            next_proj_matrix = gym.get_camera_proj_matrix(
                self.sim, self.envs[0], camera_handle
            )
            assert np.allclose(proj_matrix, next_proj_matrix)

    def _get_camera_intrinsics(self) -> Tuple[float, float, float, float]:
        self._run_sanity_check_proj_matrices_all_same()

        proj_matrix = gym.get_camera_proj_matrix(
            self.sim, self.envs[0], self.camera_handles[0]
        )
        fx = proj_matrix[0, 0]
        fy = proj_matrix[1, 1]
        cx = proj_matrix[0, 2]
        cy = proj_matrix[1, 2]

        assert math.isclose(fx, fy)
        assert math.isclose(cx, cy) and math.isclose(cx, 0) and math.isclose(cy, 0)
        return fx, fy, cx, cy

    def _create_one_split(
        self,
        split_name: str,
        split_range: np.ndarray,
        folder: str,
        generate_depth: bool = False,
    ):
        import scipy

        USE_TORCH_NGP = False
        USE_NERF_STUDIO = True
        assert sum([USE_TORCH_NGP, USE_NERF_STUDIO]) == 1

        # Sanity check
        if USE_TORCH_NGP:
            json_dict = {
                "camera_angle_x": math.radians(CAMERA_HORIZONTAL_FOV_DEG),
                "camera_angle_y": math.radians(CAMERA_VERTICAL_FOV_DEG),
                "frames": [],
            }
        elif USE_NERF_STUDIO:
            fx, fy, cx, cy = self._get_camera_intrinsics()
            json_dict = {
                "fl_x": fx * CAMERA_IMG_WIDTH / 2,  # SUSPICIOUS
                "fl_y": fy * CAMERA_IMG_HEIGHT / 2,
                # "cx": cx * CAMERA_IMG_WIDTH,
                # "cy": cy * CAMERA_IMG_HEIGHT,
                "cx": CAMERA_IMG_WIDTH // 2,
                "cy": CAMERA_IMG_HEIGHT // 2,
                "h": CAMERA_IMG_HEIGHT,
                "w": CAMERA_IMG_WIDTH,
                "frames": [],
            }
        else:
            raise ValueError()

        for ii in split_range:
            pose_file = os.path.join(folder, f"pos_xyz_quat_xyzw_{ii}.txt")
            with open(pose_file) as file:
                raw_pose_str = file.readline()[1:-1]  # Remove brackets
                pose = np.fromstring(raw_pose_str, sep=",")

                transform_mat = np.eye(4)
                pos, quat = pose[:3], pose[-4:]
                R = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
                R = (
                    R
                    @ scipy.spatial.transform.Rotation.from_euler(
                        "YZ", [-np.pi / 2, -np.pi / 2]
                    ).as_matrix()
                )
                transform_mat[:3, :3] = R
                transform_mat[:3, -1] = pos

                source_img = "col_" + str(ii)

                new_folder = os.path.join(folder, split_name)
                os.makedirs(new_folder, exist_ok=True)

                source_img = os.path.join(folder, f"col_{ii}.png")
                target_img = os.path.join(new_folder, f"{ii}.png")
                depth_img = f"dep_{ii}.png"
                shutil.copyfile(source_img, target_img)

                # Remove the first part of the path
                target_img_split = target_img.split("/")
                target_img = os.path.join(
                    *target_img_split[target_img_split.index(split_name) :]
                )

                if USE_TORCH_NGP:
                    # Exclude ext because adds it in load
                    target_img, _ = os.path.splitext(target_img)
                elif USE_NERF_STUDIO:
                    target_img = target_img
                else:
                    raise ValueError()

                if generate_depth:
                    json_dict["frames"].append(
                        {
                            "transform_matrix": transform_mat.tolist(),
                            "file_path": target_img,
                            "depth_file_path": depth_img,
                        }
                    )
                else:
                    json_dict["frames"].append(
                        {
                            "transform_matrix": transform_mat.tolist(),
                            "file_path": target_img,
                        }
                    )

        with open(os.path.join(folder, "transforms.json"), "w") as outfile:
            outfile.write(json.dumps(json_dict))

    ## NERF DATA COLLECTION END ##
