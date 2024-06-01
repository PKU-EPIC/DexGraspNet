"""
Last modified date: 2023.02.19
Author: Ruicheng Wang
Description: Class IsaacValidator
"""

from isaacgym import gymapi, torch_utils, gymutil, gymtorch
import math
import trimesh
from time import sleep
from tqdm import tqdm
from utils.hand_model_type import (
    handmodeltype_to_allowedcontactlinknames,
    handmodeltype_to_joint_names,
    HandModelType,
    handmodeltype_to_hand_root_hand_file,
    handmodeltype_to_hand_root_hand_file_with_virtual_joints,
)
from utils.torch_quat_utils import (
    pose_to_T,
    T_to_pose,
)
from collections import defaultdict
import torch
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict
import transforms3d
from datetime import datetime

# collision_filter is a bit mask that lets you filter out collision between bodies. Two bodies will not collide if their collision filters have a common bit set.
HAND_COLLISION_FILTER = 0  # 0 means turn off collisions
OBJ_COLLISION_FILTER = 0  # 0 means don't turn off collisions
TABLE_COLLISION_FILTER = 0  # 0 means don't turn off collisions
RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE = 1
ISAAC_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ARBITRARY_INIT_HAND_POS = gymapi.Vec3(0, 1, 0)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


########## NERF GRASPING START ##########

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


########## NERF GRASPING END ##########


gym = gymapi.acquire_gym()


def get_link_idx_to_name_dict(env, actor_handle) -> Dict[int, str]:
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
    NO_GRAVITY_SHAKING = auto()
    GRAVITY_AND_TABLE = auto()
    GRAVITY_AND_TABLE_AND_SHAKING = auto()


class IsaacValidator:
    def __init__(
        self,
        hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND,
        mode: str = "headless",
        gpu: int = 0,
        start_with_step_mode: bool = False,
        validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING,
        use_cpu: bool = True,
    ) -> None:
        self.gpu = gpu
        self.validation_type = validation_type

        self.joint_names = handmodeltype_to_joint_names[hand_model_type]
        self.allowed_contact_link_names = handmodeltype_to_allowedcontactlinknames[
            hand_model_type
        ]

        self._init_or_reset_state()

        # Need virtual joints to control hand position
        (
            self.hand_root,
            self.hand_file,
        ) = handmodeltype_to_hand_root_hand_file_with_virtual_joints[hand_model_type]
        # HACK: Hardcoded virtual joint names
        self.virtual_joint_names = [
            "virtual_joint_translation_x",
            "virtual_joint_translation_y",
            "virtual_joint_translation_z",
            "virtual_joint_rotation_z",
            "virtual_joint_rotation_y",
            "virtual_joint_rotation_x",
        ]

        # Try to keep num_sim_steps as small as possible to save sim time, but not so short to not lift and shake well
        if self.validation_type == ValidationType.NO_GRAVITY_SHAKING:
            self.num_sim_steps = 200
        elif self.validation_type == ValidationType.GRAVITY_AND_TABLE:
            self.num_sim_steps = 200
        elif self.validation_type == ValidationType.GRAVITY_AND_TABLE_AND_SHAKING:
            # Need more steps to shake
            self.num_sim_steps = 400
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
        self.sim_params.physx.contact_offset = 0.001  # Want this to be very close to 0 so no unneeded collisions, but need some for sim stability
        self.sim_params.physx.rest_offset = (
            0.0  # Want this to be 0 so that objects don't float when on table
        )
        self.sim_params.physx.max_gpu_contact_pairs = (
            8 * 1024 * 1024
        )  # Default is 1024 * 1024
        self.sim_params.physx.default_buffer_size_multiplier = 20.0  # Default is 2.0

        self.sim_params.use_gpu_pipeline = False
        # self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self.sim = gym.create_sim(self.gpu, -1, gymapi.SIM_PHYSX, self.sim_params)
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

        # Set lighting
        light_index = 0
        intensity = gymapi.Vec3(0.75, 0.75, 0.75)
        ambient = gymapi.Vec3(0.75, 0.75, 0.75)
        direction = gymapi.Vec3(0.0, 0.0, -1.0)
        gym.set_light_parameters(self.sim, light_index, intensity, ambient, direction)

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.collapse_fixed_joints = True
        self.hand_asset_options.fix_base_link = True

        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 500

        if self.validation_type == ValidationType.NO_GRAVITY_SHAKING:
            self.obj_asset_options.disable_gravity = True
        elif self.validation_type in [
            ValidationType.GRAVITY_AND_TABLE,
            ValidationType.GRAVITY_AND_TABLE_AND_SHAKING,
        ]:
            self.obj_asset_options.disable_gravity = False
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")

        # Simulation state
        self.is_paused = False
        self.is_step_mode = self.has_viewer and start_with_step_mode

        self.hand_asset = gym.load_asset(
            self.sim, self.hand_root, self.hand_file, self.hand_asset_options
        )

    def set_obj_asset(
        self, obj_root: str, obj_file: str, vhacd_enabled: bool = True
    ) -> None:
        self.obj_asset_options.vhacd_enabled = vhacd_enabled  # Convex decomposition is better than convex hull, but slower, not 100% sure why it's not working
        self.obj_asset = gym.load_asset(
            self.sim, obj_root, obj_file, self.obj_asset_options
        )
        self.obj_root = obj_root
        self.obj_file = obj_file

    ########## ENV SETUP START ##########
    def add_env(
        self,
        hand_quaternion_wxyz: np.ndarray,
        hand_translation: np.ndarray,
        hand_qpos: np.ndarray,
        obj_scale: float,
        target_qpos: np.ndarray,
        add_random_pose_noise: bool = False,
        record: bool = False,
    ) -> None:
        # collision_idx should be unique for each env so envs don't collide
        collision_idx = len(self.envs)

        # Create env
        ENVS_PER_ROW = 6
        env = gym.create_env(
            self.sim,
            gymapi.Vec3(-1, -1, -1),
            gymapi.Vec3(1, 1, 1),
            ENVS_PER_ROW,
        )
        self.envs.append(env)

        self._setup_hand(
            env=env,
            hand_quaternion_wxyz=hand_quaternion_wxyz,
            hand_translation=hand_translation,
            hand_qpos=hand_qpos,
            target_qpos=target_qpos,
            collision_idx=collision_idx,
            add_random_pose_noise=add_random_pose_noise,
        )

        if self.validation_type == ValidationType.NO_GRAVITY_SHAKING:
            obj_pose = gymapi.Transform()

            self._setup_obj(
                env,
                obj_pose=obj_pose,
                obj_scale=obj_scale,
                collision_idx=collision_idx,
            )
        elif self.validation_type in [
            ValidationType.GRAVITY_AND_TABLE,
            ValidationType.GRAVITY_AND_TABLE_AND_SHAKING,
        ]:
            obj_pose = self._compute_init_obj_pose_above_table(obj_scale)

            self._setup_obj(
                env,
                obj_pose=obj_pose,
                obj_scale=obj_scale,
                collision_idx=collision_idx,
            )
            self._setup_table(
                env=env,
                collision_idx=collision_idx,
            )
        else:
            raise ValueError(f"Unknown validation type: {self.validation_type}")

        if record:
            self._setup_camera(env)

    def _compute_init_obj_pose_above_table(self, obj_scale: float) -> gymapi.Transform:
        USE_HARDCODED_MAX_EXTENT = False
        if USE_HARDCODED_MAX_EXTENT:
            # All objs are assumed to be centered in a bounding box, with the max width being 2.0m (unscaled)
            # Thus max extent from origin is 1.0m (unscaled)
            # So we want to place the obj above the table a bit more then rescale
            # TODO: Make this better by reading bounding box
            OBJ_MAX_EXTENT_FROM_ORIGIN = 1.0
            BUFFER_SCALE = 1.2
            y_above_table = OBJ_MAX_EXTENT_FROM_ORIGIN * obj_scale * BUFFER_SCALE
        else:
            assert self.obj_root is not None
            assert self.obj_file is not None
            obj_root = pathlib.Path(self.obj_root)
            assert obj_root.exists()

            # obj_file is a urdf, but we want a obj
            obj_path = obj_root / "decomposed.obj"
            assert obj_path.exists(), f"{obj_path} does not exist"

            # Get bounds and use -min_y
            # For example: if min_y = -0.1, then y_above_table = 0.11
            mesh = trimesh.load_mesh(obj_path)
            mesh.apply_scale(obj_scale)
            bounds = mesh.bounds
            assert bounds.shape == (2, 3)
            min_y = bounds[0, 1]
            BUFFER = 0.02
            y_above_table = -min_y + BUFFER

        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0.0, y_above_table, 0.0)
        obj_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        return obj_pose

    def _setup_table(
        self,
        env,
        collision_idx: int,
    ) -> None:
        TABLE_THICKNESS = 0.1

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(
            0, -TABLE_THICKNESS / 2, 0
        )  # Table surface is at y=0
        table_pose.r = gymapi.Quat(0, 0, 0, 1)

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

        # Store table link_idx_to_name_dict
        self.table_link_idx_to_name_dicts.append(
            get_link_idx_to_name_dict(env=env, actor_handle=table_actor_handle)
        )

        # Set table shape props
        table_shape_props = gym.get_actor_rigid_shape_properties(
            env, table_actor_handle
        )
        for i in range(len(table_shape_props)):
            table_shape_props[i].friction = 1.0
        gym.set_actor_rigid_shape_properties(env, table_actor_handle, table_shape_props)

        # Set table texture
        if not hasattr(self, "table_texture"):
            self.table_texture = gym.create_texture_from_file(
                self.sim, "table/wood.png"
            )
        RB_IDX = 0
        gym.set_rigid_body_texture(
            env,
            table_actor_handle,
            RB_IDX,
            gymapi.MESH_VISUAL_AND_COLLISION,
            self.table_texture,
        )
        return

    def _setup_hand(
        self,
        env,
        hand_quaternion_wxyz: np.ndarray,
        hand_translation: np.ndarray,
        hand_qpos: np.ndarray,
        target_qpos: np.ndarray,
        collision_idx: int,
        add_random_pose_noise: bool = False,
    ) -> None:
        # Set hand pose
        # For now, move hand to arbitrary offset from origin
        # Will move hand to object later
        hand_pose = gymapi.Transform()
        hand_pose.p = ARBITRARY_INIT_HAND_POS

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

        # Store for later
        self.target_qpos_list.append(target_qpos)
        self.init_qpos_list.append(hand_qpos)
        hand_quaternion_xyzw = np.concatenate(
            [hand_quaternion_wxyz[1:], hand_quaternion_wxyz[:1]]
        )

        desired_hand_pose_object_frame = gymapi.Transform(
            p=gymapi.Vec3(*hand_translation), r=gymapi.Quat(*hand_quaternion_xyzw)
        )
        if add_random_pose_noise:
            TRANSLATION_NOISE_CM = 0.5
            TRANSLATION_NOISE_M = TRANSLATION_NOISE_CM / 100
            ROTATION_NOISE_DEG = 5
            xyz_noise = np.random.uniform(-TRANSLATION_NOISE_M, TRANSLATION_NOISE_M, 3)
            rpy_noise = (
                np.random.uniform(-ROTATION_NOISE_DEG, ROTATION_NOISE_DEG, 3)
                * math.pi
                / 180
            )
            quat_wxyz = transforms3d.euler.euler2quat(*rpy_noise)
            assert xyz_noise.shape == (3,)
            assert rpy_noise.shape == (3,)
            assert quat_wxyz.shape == (4,)

            pose_noise_transform = gymapi.Transform(
                p=gymapi.Vec3(*xyz_noise),
                r=gymapi.Quat(*quat_wxyz[1:], quat_wxyz[0]),
            )
            desired_hand_pose_object_frame = (
                pose_noise_transform * desired_hand_pose_object_frame
            )

        self.desired_hand_poses_object_frame.append(desired_hand_pose_object_frame)

        # Set hand dof props
        hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)

        # TODO: Consider making finger joints pos controlled and virtual joints vel controlled
        hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)

        # Finger joints
        for joint in self.joint_names:
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            hand_props["stiffness"][joint_idx] = 5.0 * 10
            hand_props["damping"][joint_idx] = 0.1 * 10

        # Virtual joints
        for joint in self.virtual_joint_names:
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            HARD_SHAKE_STIFFNESS = 20000.0
            LIGHT_SHAKE_STIFFNESS = 200.0
            hand_props["stiffness"][joint_idx] = HARD_SHAKE_STIFFNESS
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
            hand_shape_props[i].friction = 0.9
        gym.set_actor_rigid_shape_properties(env, hand_actor_handle, hand_shape_props)
        return

    def _setup_obj(
        self,
        env,
        obj_pose: gymapi.Transform,
        obj_scale: float,
        collision_idx: int,
    ) -> None:
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

        # Print
        PRINT_MASS = False
        if PRINT_MASS:
            obj_rb_props = gym.get_actor_rigid_body_properties(env, obj_actor_handle)
            masses = [rb_prop.mass for rb_prop in obj_rb_props]
            print(f"total_mass = {sum(masses)}")

        # Set obj shape props
        obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_actor_handle)
        for i in range(len(obj_shape_props)):
            obj_shape_props[i].friction = 0.9
        gym.set_actor_rigid_shape_properties(env, obj_actor_handle, obj_shape_props)
        return

    ########## ENV SETUP START ##########

    ########## RUN SIM END ##########
    def run_sim(self) -> Tuple[List[bool], List[bool], List[bool], np.ndarray]:
        gym.prepare_sim(self.sim)

        # Prepare tensors
        root_state_tensor = gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(root_state_tensor)

        (
            hand_not_penetrate_object_list,
            hand_not_penetrate_table_list,
            object_states_before_grasp,
        ) = self._run_sim_steps()

        # Render out all videos.
        self._save_video_if_needed()

        successes = self._check_successes()
        return (
            successes,
            hand_not_penetrate_object_list,
            hand_not_penetrate_table_list,
            object_states_before_grasp,
        )

    def _check_successes(self) -> List[bool]:
        successes = []
        for i, (
            env,
            hand_link_idx_to_name,
            obj_link_idx_to_name,
            obj_handle,
            hand_handle,
        ) in enumerate(
            zip(
                self.envs,
                self.hand_link_idx_to_name_dicts,
                self.obj_link_idx_to_name_dicts,
                self.obj_handles,
                self.hand_handles,
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

            # Find not allowed contacts
            not_allowed_contacts = set(hand_link_contact_count.keys()) - set(
                self.allowed_contact_link_names
            )

            # Change from init to final obj pose relative to hand
            final_hand_pose = self._get_palm_pose(
                env=env,
                hand_handle=hand_handle,
                hand_link_idx_to_name=hand_link_idx_to_name,
            )
            final_obj_pose = self._get_object_pose(env=env, obj_handle=obj_handle)
            final_rel_obj_pose = final_hand_pose.inverse() * final_obj_pose
            init_rel_obj_pose = self.desired_hand_poses_object_frame[i].inverse()
            pos_change, max_euler_change = self._get_pos_and_euler_change(
                pose1=init_rel_obj_pose, pose2=final_rel_obj_pose
            )

            # Success conditions
            success = (
                len(hand_object_contacts) > 0
                and len(hand_link_contact_count.keys()) >= 3
                and len(not_allowed_contacts) == 0
                and pos_change < 0.1
                and max_euler_change < 45
            )

            successes.append(success)

            DEBUG = False
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

    def _get_palm_pose(
        self, env, hand_handle, hand_link_idx_to_name: Dict[int, str]
    ) -> gymapi.Transform:
        palm_link_idxs = [
            idx for idx, name in hand_link_idx_to_name.items() if name == "palm_link"
        ]
        assert len(palm_link_idxs) == 1, f"len(palm_link_idxs) = {len(palm_link_idxs)}"
        palm_link_idx = palm_link_idxs[0]

        final_hand_pose = gymapi.Transform()
        final_hand_pose.p, final_hand_pose.r = gym.get_actor_rigid_body_states(
            env, hand_handle, gymapi.STATE_POS
        )[palm_link_idx]["pose"]
        return final_hand_pose

    def _get_object_pose(self, env, obj_handle: int) -> gymapi.Transform:
        OBJ_BASE_LINK_IDX = 0
        final_obj_pose = gymapi.Transform()
        final_obj_pose.p, final_obj_pose.r = gym.get_actor_rigid_body_states(
            env, obj_handle, gymapi.STATE_POS
        )[OBJ_BASE_LINK_IDX]["pose"]
        return final_obj_pose

    def _get_pos_and_euler_change(
        self, pose1: gymapi.Transform, pose2: gymapi.Transform
    ) -> Tuple[float, float]:
        pos1 = torch.tensor([pose1.p.x, pose1.p.y, pose1.p.z])
        quat1_xyzw = torch.tensor(
            [
                pose1.r.x,
                pose1.r.y,
                pose1.r.z,
                pose1.r.w,
            ]
        )

        pos2 = torch.tensor([pose2.p.x, pose2.p.y, pose2.p.z])
        quat2_xyzw = torch.tensor(
            [
                pose2.r.x,
                pose2.r.y,
                pose2.r.z,
                pose2.r.w,
            ]
        )

        quat_diff = torch_utils.quat_mul(
            quat2_xyzw, torch_utils.quat_conjugate(quat1_xyzw)
        )
        pos_change = torch.linalg.norm(pos2 - pos1).item()

        euler_change = torch.stack(
            torch_utils.get_euler_xyz(quat_diff[None, ...])
        ).abs()
        euler_change = torch.where(
            euler_change > math.pi, 2 * math.pi - euler_change, euler_change
        )
        max_euler_change = euler_change.max().rad2deg().item()
        return pos_change, max_euler_change

    def _is_hand_colliding_with_table(self) -> List[bool]:
        assert_equals(len(self.table_link_idx_to_name_dicts), len(self.envs))

        is_hand_colliding_with_table = []
        for env, table_link_idx_to_name, hand_link_idx_to_name in zip(
            self.envs,
            self.table_link_idx_to_name_dicts,
            self.hand_link_idx_to_name_dicts,
        ):
            hand_table_contacts = []
            contacts = gym.get_env_rigid_contacts(env)
            for contact in contacts:
                body0 = contact["body0"]
                body1 = contact["body1"]
                is_hand_table_contact = (
                    body0 in hand_link_idx_to_name and body1 in table_link_idx_to_name
                ) or (
                    body1 in hand_link_idx_to_name and body0 in table_link_idx_to_name
                )
                if is_hand_table_contact:
                    hand_table_contacts.append(contact)

            DEBUG = False
            if len(hand_table_contacts) > 0:
                if DEBUG:
                    print(f"HAND COLLIDES TABLE for {env}")
                    print(
                        f"Collisions between hand and table: {[(c['body0'], c['body1']) for c in hand_table_contacts]}, {hand_link_idx_to_name}, {table_link_idx_to_name}"
                    )
                is_hand_colliding_with_table.append(True)
            else:
                is_hand_colliding_with_table.append(False)

        assert_equals(len(is_hand_colliding_with_table), len(self.envs))
        return is_hand_colliding_with_table

    def _is_hand_colliding_with_obj(self) -> List[bool]:
        assert_equals(len(self.obj_link_idx_to_name_dicts), len(self.envs))

        is_hand_colliding_with_obj = []
        for env, obj_link_idx_to_name, hand_link_idx_to_name in zip(
            self.envs,
            self.obj_link_idx_to_name_dicts,
            self.hand_link_idx_to_name_dicts,
        ):
            hand_obj_contacts = []
            contacts = gym.get_env_rigid_contacts(env)
            for contact in contacts:
                body0 = contact["body0"]
                body1 = contact["body1"]
                is_hand_obj_contact = (
                    body0 in hand_link_idx_to_name and body1 in obj_link_idx_to_name
                ) or (body1 in hand_link_idx_to_name and body0 in obj_link_idx_to_name)
                if is_hand_obj_contact:
                    hand_obj_contacts.append(contact)

            DEBUG = False
            if len(hand_obj_contacts) > 0:
                if DEBUG:
                    print(f"HAND COLLIDES OBJECT for {env}")
                    print(
                        f"Collisions between hand and object: {[(c['body0'], c['body1']) for c in hand_obj_contacts]}, {hand_link_idx_to_name}, {obj_link_idx_to_name}"
                    )
                is_hand_colliding_with_obj.append(True)
            else:
                is_hand_colliding_with_obj.append(False)

        assert_equals(len(is_hand_colliding_with_obj), len(self.envs))
        return is_hand_colliding_with_obj

    def _move_hands_to_objects(self) -> None:
        # Get current object poses
        hand_indices = self._get_actor_indices(
            envs=self.envs, actors=self.hand_handles
        ).to(self.root_state_tensor.device)
        object_indices = self._get_actor_indices(
            envs=self.envs, actors=self.obj_handles
        ).to(self.root_state_tensor.device)
        current_object_poses = self.root_state_tensor[object_indices, :7].clone()

        N = current_object_poses.shape[0]
        assert_equals(current_object_poses.shape, (N, 7))

        # Currently: hand_pose = ARBITRARY_INIT_HAND_POS
        # Next: hand_pose = desired_hand_pose_in_world_frame = desired_hand_pose_in_object_frame + object_pose
        # world_to_hand_transform = world_to_object_transform @ object_to_hand_transform
        desired_hand_poses_in_object_frame = self._gymapi_transforms_to_poses(
            self.desired_hand_poses_object_frame
        )

        object_to_hand_transforms = pose_to_T(desired_hand_poses_in_object_frame)
        world_to_object_transforms = pose_to_T(current_object_poses)
        assert_equals(object_to_hand_transforms.shape, (N, 4, 4))
        assert_equals(world_to_object_transforms.shape, (N, 4, 4))
        world_to_hand_transforms = torch.bmm(
            world_to_object_transforms, object_to_hand_transforms
        )
        assert_equals(world_to_hand_transforms.shape, (N, 4, 4))
        new_hand_poses = T_to_pose(world_to_hand_transforms)
        assert_equals(new_hand_poses.shape, (N, 7))

        self.root_state_tensor[hand_indices, :7] = new_hand_poses
        gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor)
        )

    def _gymapi_transforms_to_poses(
        self, transforms: List[gymapi.Transform]
    ) -> torch.Tensor:
        poses = torch.zeros((len(transforms), 7)).float()
        for i, transform in enumerate(transforms):
            poses[i, :3] = torch.tensor([transform.p.x, transform.p.y, transform.p.z])
            poses[i, 3:] = torch.tensor(
                [transform.r.x, transform.r.y, transform.r.z, transform.r.w]
            )
        return poses

    def _run_sim_steps(self) -> Tuple[List[bool], List[bool], np.ndarray]:
        sim_step_idx = 0
        default_desc = "Simulating"
        pbar = tqdm(total=self.num_sim_steps, desc=default_desc, dynamic_ncols=True)

        hand_not_penetrate_object_list = [True for _ in range(len(self.envs))]
        hand_not_penetrate_table_list = [True for _ in range(len(self.envs))]
        object_states_before_grasp = None

        while sim_step_idx < self.num_sim_steps:
            # Phase 1: Do nothing, hand far away
            #   * For NO_GRAVITY_SHAKING: object should stay in place
            #   * For GRAVITY_AND_TABLE and GRAVITY_AND_TABLE_AND_SHAKING: object should fall to table and settle
            # Phase 2: Move hand to object
            #   * For NO_GRAVITY_SHAKING: check if hand collides with object
            #   * For GRAVITY_AND_TABLE and GRAVITY_AND_TABLE_AND_SHAKING: check if hand collides with object AND if hand collides with table
            # Phase 3: Close hand
            # Phase 4: Shake hand
            #   * For NO_GRAVITY_SHAKING: shake from this position
            #   * For GRAVITY_AND_TABLE and GRAVITY_AND_TABLE_AND_SHAKING: lift from table first, then shake
            PHASE_1_LAST_STEP = (
                50  # From analysis, takes about 40 steps for ball to settle
            )
            PHASE_2_LAST_STEP = PHASE_1_LAST_STEP + 10
            PHASE_3_LAST_STEP = PHASE_2_LAST_STEP + 15
            PHASE_4_LAST_STEP = self.num_sim_steps
            assert_equals(PHASE_4_LAST_STEP, self.num_sim_steps)

            if sim_step_idx < PHASE_1_LAST_STEP:
                self._run_phase_1(step=sim_step_idx, length=PHASE_1_LAST_STEP)
            elif sim_step_idx < PHASE_2_LAST_STEP:
                if sim_step_idx == PHASE_1_LAST_STEP:
                    # Get current object poses
                    object_indices = self._get_actor_indices(
                        envs=self.envs, actors=self.obj_handles
                    ).to(self.root_state_tensor.device)
                    object_states_before_grasp = self.root_state_tensor[
                        object_indices, :13
                    ].clone()

                (
                    TEMP_hand_not_penetrate_object_list,
                    TEMP_hand_not_penetrate_table_list,
                ) = self._run_phase_2(
                    step=sim_step_idx - PHASE_1_LAST_STEP,
                    length=PHASE_2_LAST_STEP - PHASE_1_LAST_STEP,
                )
                hand_not_penetrate_object_list = [
                    hand_not_penetrate_obj and TEMP_hand_not_penetrate_obj
                    for hand_not_penetrate_obj, TEMP_hand_not_penetrate_obj in zip(
                        hand_not_penetrate_object_list,
                        TEMP_hand_not_penetrate_object_list,
                    )
                ]
                hand_not_penetrate_table_list = [
                    hand_not_penetrate_table and TEMP_hand_not_penetrate_table
                    for hand_not_penetrate_table, TEMP_hand_not_penetrate_table in zip(
                        hand_not_penetrate_table_list,
                        TEMP_hand_not_penetrate_table_list,
                    )
                ]
            elif sim_step_idx < PHASE_3_LAST_STEP:
                self._run_phase_3(
                    step=sim_step_idx - PHASE_2_LAST_STEP,
                    length=PHASE_3_LAST_STEP - PHASE_2_LAST_STEP,
                )
            elif sim_step_idx < PHASE_4_LAST_STEP:
                self._run_phase_4(
                    step=sim_step_idx - PHASE_3_LAST_STEP,
                    length=PHASE_4_LAST_STEP - PHASE_3_LAST_STEP,
                )
            else:
                raise ValueError(f"Unknown sim_step_idx: {sim_step_idx}")

            # Step physics if not paused
            if not self.is_paused:
                gym.simulate(self.sim)
                gym.fetch_results(self.sim, True)
                gym.refresh_actor_root_state_tensor(self.sim)

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
                sleep(0.05)
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
                self._visualize_virtual_joint_dof_pos_targets()

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

        return (
            hand_not_penetrate_object_list,
            hand_not_penetrate_table_list,
            object_states_before_grasp.cpu().numpy(),
        )

    def _run_phase_1(self, step: int, length: int) -> None:
        assert step < length, f"{step} >= {length}"
        return

    def _run_phase_2(self, step: int, length: int) -> Tuple[List[bool], List[bool]]:
        assert step < length, f"{step} >= {length}"
        if step == 0:
            self._move_hands_to_objects()
            return [True for _ in range(len(self.envs))], [
                True for _ in range(len(self.envs))
            ]
        else:
            # Can only check the collisions after taking a sim step
            hand_colliding_obj = self._is_hand_colliding_with_obj()
            hand_not_colliding_object_list = [
                not colliding_obj for colliding_obj in hand_colliding_obj
            ]

            if self.validation_type in [
                ValidationType.GRAVITY_AND_TABLE,
                ValidationType.GRAVITY_AND_TABLE_AND_SHAKING,
            ]:
                hand_colliding_table = self._is_hand_colliding_with_table()
                hand_not_colliding_table_list = [
                    not colliding_table for colliding_table in hand_colliding_table
                ]
            else:
                hand_not_colliding_table_list = [True for _ in range(len(self.envs))]
            return hand_not_colliding_object_list, hand_not_colliding_table_list

    def _run_phase_3(self, step: int, length: int) -> None:
        assert step < length, f"{step} >= {length}"
        frac_progress = (step + 1) / length
        for env, hand_actor_handle, target_qpos, init_qpos in zip(
            self.envs,
            self.hand_handles,
            self.target_qpos_list,
            self.init_qpos_list,
        ):
            current_target_qpos = target_qpos * frac_progress + init_qpos * (
                1 - frac_progress
            )
            self._set_dof_pos_targets(
                env=env,
                hand_actor_handle=hand_actor_handle,
                target_qpos=current_target_qpos,
            )

    def _run_phase_4(self, step: int, length: int) -> None:
        assert step < length, f"{step} >= {length}"
        frac_progress = (step + 1) / length
        if len(self.virtual_joint_names) > 0:
            virtual_joint_dof_pos_targets = self._compute_virtual_joint_dof_pos_targets(
                frac_progress=frac_progress
            )
            self._set_virtual_joint_dof_pos_targets(virtual_joint_dof_pos_targets)

    ########## RUN SIM END ##########

    ########## HELPERS START ##########
    def _get_hand_poses(self) -> List[gymapi.Transform]:
        # Get current pose of hand
        hand_indices = self._get_actor_indices(
            envs=self.envs, actors=self.hand_handles
        ).to(self.root_state_tensor.device)
        hand_poses = self.root_state_tensor[hand_indices, :7].clone()
        assert_equals(hand_poses.shape, (len(self.envs), 7))
        hand_poses = [
            gymapi.Transform(p=gymapi.Vec3(*pose[:3]), r=gymapi.Quat(*pose[3:]))
            for pose in hand_poses
        ]
        return hand_poses

    def _get_actor_indices(self, envs, actors) -> torch.Tensor:
        assert_equals(len(envs), len(actors))
        actor_indices = torch_utils.to_torch(
            [
                gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)  # type: ignore
                for env, actor in zip(envs, actors)
            ],
            dtype=torch.long,
        )
        return actor_indices

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

    ########## HELPERS END ##########

    ########## VIDEO START ##########
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

        cam_target = gymapi.Vec3(0, 0.1, 0)  # type: ignore  # where object s

        cam_pos = cam_target + gymapi.Vec3(0.3, 0.3, 0.0)  # Define offset

        self.video_frames.append([])
        gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

    def _save_video_if_needed(self) -> None:
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

    def _render_video(
        self, video_frames: List[torch.Tensor], video_path: pathlib.Path, fps: int
    ):
        print(f"number of frames: {len(video_frames)}")
        imageio.mimsave(video_path, video_frames, fps=fps)

    ########## VIDEO END ##########

    ########## DOF TARGETS START ##########
    def _compute_virtual_joint_dof_pos_targets(
        self,
        frac_progress: float,
    ) -> List[torch.Tensor]:
        assert len(self.virtual_joint_names) == 6

        # Shaking / perturbation parameters for virtual joint targets.

        # Set dof pos targets [+x, -x]*N, 0, [+y, -y]*N, 0, [+z, -z]*N
        dist_to_move = 0.03
        N_SHAKES = 1
        if self.validation_type == ValidationType.NO_GRAVITY_SHAKING:
            targets_sequence = [
                *([[dist_to_move, 0.0, 0.0], [-dist_to_move, 0.0, 0.0]] * N_SHAKES),
                [0.0, 0.0, 0.0],
                *([[0.0, dist_to_move, 0.0], [0.0, -dist_to_move, 0.0]] * N_SHAKES),
                [0.0, 0.0, 0.0],
                *([[0.0, 0.0, dist_to_move], [0.0, 0.0, -dist_to_move]] * N_SHAKES),
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        elif self.validation_type in [
            ValidationType.GRAVITY_AND_TABLE,
            ValidationType.GRAVITY_AND_TABLE_AND_SHAKING,
        ]:
            Y_LIFT = 0.2
            INCLUDE_SHAKE = (
                self.validation_type == ValidationType.GRAVITY_AND_TABLE_AND_SHAKING
            )
            targets_sequence = [
                [0.0, -Y_LIFT, 0.0],
                [0.0, -Y_LIFT * 7 / 8, 0.0],
                [0.0, -Y_LIFT * 6 / 8, 0.0],
                [0.0, -Y_LIFT * 5 / 8, 0.0],
                [0.0, -Y_LIFT * 4 / 8, 0.0],
                [0.0, -Y_LIFT * 3 / 8, 0.0],
                [0.0, -Y_LIFT * 2 / 8, 0.0],
                [0.0, -Y_LIFT * 1 / 8, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
            if INCLUDE_SHAKE:
                targets_sequence += [
                    *(
                        [
                            [dist_to_move / 2, 0.0, 0.0],
                            [dist_to_move, 0.0, 0.0],
                            [dist_to_move / 2, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [-dist_to_move / 2, 0.0, 0.0],
                            [-dist_to_move, 0.0, 0.0],
                            [-dist_to_move / 2, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ]
                        * N_SHAKES
                    ),
                    [0.0, 0.0, 0.0],
                    *(
                        [
                            [0.0, dist_to_move / 2, 0.0],
                            [0.0, dist_to_move, 0.0],
                            [0.0, dist_to_move / 2, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, -dist_to_move / 2, 0.0],
                            [0.0, -dist_to_move, 0.0],
                            [0.0, -dist_to_move / 2, 0.0],
                            [0.0, 0.0, 0.0],
                        ]
                        * N_SHAKES
                    ),
                    [0.0, 0.0, 0.0],
                    *(
                        [
                            [0.0, 0.0, dist_to_move / 2],
                            [0.0, 0.0, dist_to_move],
                            [0.0, 0.0, dist_to_move / 2],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, -dist_to_move / 2],
                            [0.0, 0.0, -dist_to_move],
                            [0.0, 0.0, -dist_to_move / 2],
                            [0.0, 0.0, 0.0],
                        ]
                        * N_SHAKES
                    ),
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]

            targets_sequence = [
                [target[0], target[1] + Y_LIFT, target[2]]
                for target in targets_sequence
            ]
        else:
            raise ValueError(f"Unknown validation_type: {self.validation_type}")

        target_idx = int(frac_progress * len(targets_sequence))
        target_idx = np.clip(target_idx, 0, len(targets_sequence) - 1)
        target = targets_sequence[target_idx]

        # Smooth out target so that it doesn't jump around
        prev_target = targets_sequence[target_idx - 1] if target_idx > 0 else target
        alpha = frac_progress * len(targets_sequence) - target_idx
        target_smoothed = np.array(target) * alpha + np.array(prev_target) * (1 - alpha)

        # direction in global frame
        # virtual_joint_dof_pos_targets in hand frame
        # so need to perform inverse hand frame rotation
        rotation_transforms = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), hand_pose.r)
            for hand_pose in self._get_hand_poses()
        ]
        virtual_joint_dof_pos_targets = [
            rotation_transform.inverse().transform_point(gymapi.Vec3(*target_smoothed))
            for rotation_transform in rotation_transforms
        ]
        virtual_joint_dof_pos_targets = [
            torch.tensor([dof_pos_target.x, dof_pos_target.y, dof_pos_target.z])
            for dof_pos_target in virtual_joint_dof_pos_targets
        ]

        # Add target angles
        target_angles = torch.tensor([0.0, 0.0, 0.0])
        virtual_joint_dof_pos_targets = [
            torch.cat([dof_pos_target, target_angles])
            for dof_pos_target in virtual_joint_dof_pos_targets
        ]

        return virtual_joint_dof_pos_targets

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

    ########## DOF TARGETS END ##########

    ########## VISUALIZE START ##########
    def _visualize_virtual_joint_dof_pos_targets(self) -> None:
        if not self.has_viewer:
            return

        if len(self.virtual_joint_names) == 0:
            return

        virtual_joint_dof_pos_list = self._get_virtual_joint_dof_pos_list()
        virtual_joint_dof_pos_targets = self._get_virtual_joint_dof_pos_targets_list()
        visualization_sphere_green = gymutil.WireframeSphereGeometry(
            radius=0.05, num_lats=10, num_lons=10, color=(0, 1, 0)
        )
        visualization_sphere_blue = gymutil.WireframeSphereGeometry(
            radius=0.05, num_lats=10, num_lons=10, color=(1, 0, 0)
        )

        # Get current pose of hand
        hand_poses = self._get_hand_poses()

        for env, hand_pose, dof_pos_target, dof_pos in zip(
            self.envs,
            hand_poses,
            virtual_joint_dof_pos_targets,
            virtual_joint_dof_pos_list,
        ):
            rotation_transform = gymapi.Transform(gymapi.Vec3(0, 0, 0), hand_pose.r)

            # virtual_joint_dof_pos_targets in hand frame
            # direction in global frame
            # so need to perform hand frame rotation
            dof_pos_target = gymapi.Vec3(
                dof_pos_target[0], dof_pos_target[1], dof_pos_target[2]
            )
            dof_pos_target_world = rotation_transform.transform_point(dof_pos_target)
            green_sphere_pose = gymapi.Transform(
                p=hand_pose.p + dof_pos_target_world,
                r=None,
            )
            gymutil.draw_lines(
                visualization_sphere_green, gym, self.viewer, env, green_sphere_pose
            )

            dof_pos = gymapi.Vec3(dof_pos[0], dof_pos[1], dof_pos[2])
            dof_pos_world = rotation_transform.transform_point(dof_pos)
            blue_sphere_pose = gymapi.Transform(
                p=hand_pose.p + dof_pos_world,
                r=None,
            )
            gymutil.draw_lines(
                visualization_sphere_blue, gym, self.viewer, env, blue_sphere_pose
            )

    def _get_joint_dof_pos_list(self) -> List[torch.Tensor]:
        dof_pos_list = []
        for env, hand_actor_handle in zip(self.envs, self.hand_handles):
            dof_pos = []
            dof_states = gym.get_actor_dof_states(
                env, hand_actor_handle, gymapi.STATE_ALL
            )
            for i, joint in enumerate(self.joint_names):
                joint_idx = gym.find_actor_dof_index(
                    env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
                )
                dof_pos.append(dof_states["pos"][joint_idx])
            dof_pos_list.append(torch.tensor(dof_pos))
        return dof_pos_list

    def _get_virtual_joint_dof_pos_list(self) -> List[torch.Tensor]:
        dof_pos_list = []
        for env, hand_actor_handle in zip(self.envs, self.hand_handles):
            dof_pos = []
            dof_states = gym.get_actor_dof_states(
                env, hand_actor_handle, gymapi.STATE_ALL
            )
            for i, joint in enumerate(self.virtual_joint_names):
                joint_idx = gym.find_actor_dof_index(
                    env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
                )
                dof_pos.append(dof_states["pos"][joint_idx])
            dof_pos_list.append(torch.tensor(dof_pos))
        return dof_pos_list

    def _get_virtual_joint_dof_pos_targets_list(self) -> List[torch.Tensor]:
        dof_pos_targets_list = []
        for env, hand_actor_handle in zip(self.envs, self.hand_handles):
            dof_pos_targets = []
            all_dof_pos_targets = gym.get_actor_dof_position_targets(
                env, hand_actor_handle
            )
            for i, joint in enumerate(self.virtual_joint_names):
                joint_idx = gym.find_actor_dof_index(
                    env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
                )
                dof_pos_targets.append(all_dof_pos_targets[joint_idx])
            dof_pos_targets_list.append(torch.tensor(dof_pos_targets))
        return dof_pos_targets_list

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

    ########## VISUALIZE END ##########

    ########## RESET START ##########
    def reset_simulator(self) -> None:
        self.destroy()

        if self.has_viewer:
            self.viewer = gym.create_viewer(self.sim, self.camera_props)

        # self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self.sim = gym.create_sim(self.gpu, -1, gymapi.SIM_PHYSX, self.sim_params)

        # Recreate hand asset in new sim.
        self.hand_asset = gym.load_asset(
            self.sim, self.hand_root, self.hand_file, self.hand_asset_options
        )

        self._init_or_reset_state()

    def destroy(self) -> None:
        for env in self.envs:
            gym.destroy_env(env)
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)

    def _init_or_reset_state(self) -> None:
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []

        self.hand_link_idx_to_name_dicts = []
        self.obj_link_idx_to_name_dicts = []
        self.table_link_idx_to_name_dicts = []

        self.init_obj_poses = []
        self.target_qpos_list = []
        self.init_qpos_list = []
        self.desired_hand_poses_object_frame = []

        self.camera_handles = []
        self.camera_envs = []
        self.camera_properties_list = []
        self.video_frames = []
        self.obj_asset = None
        self.obj_root = None
        self.obj_file = None

    ########## RESET END ##########

    ########## KEYBOARD SUBSCRIPTIONS START ##########
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

    def _step_mode_callback(self):
        self.is_step_mode = not self.is_step_mode
        print(f"Simulation is in {'step' if self.is_step_mode else 'continuous'} mode")
        self._pause_sim_callback()

    def _pause_sim_callback(self):
        self.is_paused = not self.is_paused
        print(f"Simulation is {'paused' if self.is_paused else 'unpaused'}")

    ########## KEYBOARD SUBSCRIPTIONS END ##########

    ########## NERF DATA COLLECTION START ##########
    def add_env_nerf_data_collection(
        self,
        obj_scale: float,
    ) -> None:
        # Create env
        spacing = 1.0
        env = gym.create_env(
            self.sim,
            gymapi.Vec3(-spacing, -spacing, 0.0),
            gymapi.Vec3(spacing, spacing, spacing),
            0,  # TODO: Should it be 0?
        )
        self.envs.append(env)

        if self.validation_type == ValidationType.NO_GRAVITY_SHAKING:
            obj_pose = gymapi.Transform()

            self._setup_obj(
                env,
                obj_pose=obj_pose,
                obj_scale=obj_scale,
                collision_idx=0,
            )
        elif self.validation_type in [
            ValidationType.GRAVITY_AND_TABLE,
            ValidationType.GRAVITY_AND_TABLE_AND_SHAKING,
        ]:
            obj_pose = self._compute_init_obj_pose_above_table(obj_scale)

            self._setup_obj(
                env,
                obj_pose=obj_pose,
                obj_scale=obj_scale,
                collision_idx=0,
            )
            self._setup_table(
                env=env,
                collision_idx=0,
            )
        else:
            raise ValueError(f"Unknown validation type: {self.validation_type}")

    def run_sim_till_object_settles_upright(
        self, max_sim_steps: int = 200, n_consecutive_steps: int = 15
    ) -> Tuple[bool, str]:
        gym.prepare_sim(self.sim)

        # Prepare tensors
        root_state_tensor = gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(root_state_tensor)

        pbar = tqdm(range(max_sim_steps), dynamic_ncols=True)
        xyzs, rpys = [], []
        for sim_step_idx in pbar:
            # Get object state
            object_indices = self._get_actor_indices(
                envs=self.envs, actors=self.obj_handles
            ).to(self.root_state_tensor.device)
            object_states = self.root_state_tensor[object_indices, :13].clone()

            xyz = object_states[:, :3].squeeze(dim=0)
            quat_xyzw = object_states[:, 3:7].squeeze(dim=0)
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            rpy = torch.tensor(transforms3d.euler.quat2euler(quat_wxyz)).to(
                device=xyz.device, dtype=xyz.dtype
            )
            xyzs.append(xyz)
            rpys.append(rpy)

            # Check if object has settled for n_consecutive_steps steps
            if len(xyzs) >= n_consecutive_steps:
                recent_xyzs = torch.stack(xyzs[-n_consecutive_steps:])
                recent_rpys = torch.stack(rpys[-n_consecutive_steps:])
                assert recent_xyzs.shape == (
                    n_consecutive_steps,
                    3,
                ), recent_xyzs.shape
                assert recent_rpys.shape == (
                    n_consecutive_steps,
                    3,
                ), recent_rpys.shape

                max_xyz_diff = torch.abs(recent_xyzs - xyz).max().item()
                max_rpy_diff = torch.abs(recent_rpys - rpy).max().item()
                is_object_settled = max_xyz_diff < 1e-3 and max_rpy_diff < 1e-2
            else:
                max_xyz_diff = np.inf
                max_rpy_diff = np.inf
                is_object_settled = False

            # Check if object is upright
            quat_w = quat_wxyz[0]
            is_object_upright = quat_w >= 0.95

            if is_object_settled and is_object_upright:
                log_text = f"Object settled at step {sim_step_idx}"
                print(log_text)
                return True, log_text

            # Step physics
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)
            gym.refresh_actor_root_state_tensor(self.sim)
            if self.has_viewer:
                # No need to step graphics until we need to render
                # Unless we are using viewer
                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)
                sleep(0.05)
            pbar.set_description(
                f"q_wxyz: {np.round(quat_wxyz.tolist(), 4)}, xyz_diff: {max_xyz_diff:.4f}, rpy_diff: {max_rpy_diff:.4f}"
            )

        log_text = f"Object did not settle after max steps {max_sim_steps}, quat_wxyz: {quat_wxyz}, xyz_diff: {max_xyz_diff}, rpy_diff: {max_rpy_diff}"
        print(log_text)
        return False, log_text

    def save_images(
        self, folder: str, overwrite: bool = False, num_cameras: int = 250
    ) -> None:
        assert len(self.envs) == 1
        self._setup_cameras(self.envs[0], num_cameras=num_cameras)

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
        overwrite: bool = False,
        generate_seg: bool = False,
        generate_depth: bool = False,
        num_cameras: int = 250,
    ) -> None:
        assert len(self.envs) == 1
        self._setup_cameras(self.envs[0], num_cameras=num_cameras)

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

    def _setup_cameras(self, env, num_cameras: int):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = CAMERA_HORIZONTAL_FOV_DEG
        camera_props.width = CAMERA_IMG_WIDTH
        camera_props.height = CAMERA_IMG_HEIGHT

        # Sample num_cameras points on sphere with points away from table surface using phi_degrees
        # May 2024: Radius is approximately 0.45, phi_degrees is 45 in real world
        xyz_vals = sample_points_on_sphere(N=num_cameras, radius=0.45, phi_degrees=45)
        x_vals, y_vals, z_vals = xyz_vals[:, 0], xyz_vals[:, 1], xyz_vals[:, 2]

        camera_target = gymapi.Vec3(0, 0, 0)

        for xx, yy, zz in zip(x_vals, y_vals, z_vals):
            camera_handle = gym.create_camera_sensor(env, camera_props)
            gym.set_camera_location(
                camera_handle, env, gymapi.Vec3(xx, yy, zz), camera_target
            )
            self.camera_handles.append(camera_handle)

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


def sample_points_on_sphere(N: int, radius: float, phi_degrees: float) -> np.ndarray:
    """
    Generate N random points on the surface of a sphere of given radius.
    Points are sampled in a region where the polar angle phi (from the y-axis)
    is in the range [0, phi_degrees].

    Parameters:
    N (int): Number of points to sample.
    radius (float): Radius of the sphere.
    phi_degrees (float): Maximum angle in degrees from the y-axis.

    Returns:
    np.ndarray: Array of shape (N, 3) containing the xyz coordinates of the sampled points.
    """
    # Convert maximum phi from degrees to radians
    phi_max = np.radians(phi_degrees)

    # Sample phi and theta values
    # phi is sampled uniformly in cosine space to maintain uniformity on the sphere's surface
    cos_phi = np.random.uniform(np.cos(phi_max), 1, size=N)
    phi = np.arccos(cos_phi)  # Invert cosine to get angles
    theta = np.random.uniform(
        0, 2 * np.pi, size=N
    )  # Uniformly sample theta around the sphere

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.cos(phi)  # y is up, so it's linked to the cos(phi)
    z = radius * np.sin(phi) * np.sin(theta)

    # Stack into a (N, 3) array
    points = np.vstack((x, y, z)).T

    return points

    ########## NERF DATA COLLECTION END ##########
