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

## NERF GRASPING START ##

import numpy as np
import os
import json
from pathlib import Path
import shutil
from PIL import Image

from utils.quaternions import Quaternion

CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH = 400, 400
CAMERA_HORIZONTAL_FOV_DEG = 35.0
CAMERA_VERTICAL_FOV_DEG = (
    CAMERA_IMG_HEIGHT / CAMERA_IMG_WIDTH
) * CAMERA_HORIZONTAL_FOV_DEG
OBJ_SEGMENTATION_ID = 1


def get_fixed_camera_transform(gym, sim, env, camera):
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


def get_link_idx_to_name_dict(env, actor_handle):
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


class IsaacValidator:
    def __init__(
        self,
        hand_model_type=HandModelType.ALLEGRO_HAND,
        mode="direct",
        hand_friction=3.0,
        obj_friction=3.0,
        threshold_dis=0.1,
        env_batch=1,
        sim_step=100,
        gpu=0,
        debug_interval=0.05,
        start_with_step_mode=False,
        validation_type=ValidationType.NO_GRAVITY_SHAKING,
    ):
        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.debug_interval = debug_interval
        self.threshold_dis = threshold_dis
        self.env_batch = env_batch
        self.sim_step = sim_step
        self.gpu = gpu
        self.validation_type = validation_type

        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_link_idx_to_name_dicts = []
        self.obj_link_idx_to_name_dicts = []
        self.init_hand_poses = []
        self.init_obj_poses = []
        self.joint_names = handmodeltype_to_joint_names[hand_model_type]
        self.allowed_contact_link_names = handmodeltype_to_allowedcontactlinknames[
            hand_model_type
        ]

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
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")

        self.hand_asset = None
        self.obj_asset = None

        self.sim_params = gymapi.SimParams()

        # set common parameters
        self.sim_params.dt = 1 / 60
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)

        # set PhysX-specific parameters
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
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

    def set_obj_asset(self, obj_root, obj_file):
        self.obj_asset = gym.load_asset(
            self.sim, obj_root, obj_file, self.obj_asset_options
        )

    def add_env_all_test_rotations(
        self, hand_quaternion, hand_translation, hand_qpos, obj_scale, target_qpos=None
    ):
        for test_rotation_idx in range(len(self.test_rotations)):
            self.add_env_single_test_rotation(
                hand_quaternion,
                hand_translation,
                hand_qpos,
                obj_scale,
                test_rotation_idx,
                target_qpos,
            )

    def add_env_single_test_rotation(
        self,
        hand_quaternion,
        hand_translation,
        hand_qpos,
        obj_scale,
        test_rotation_index=0,
        target_qpos=None,
    ):
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
            env, hand_quaternion, hand_translation, hand_qpos, test_rot, target_qpos
        )

        self._setup_obj(env, obj_scale, test_rot)

    def _setup_hand(
        self,
        env,
        hand_quaternion,
        hand_translation,
        hand_qpos,
        transformation,
        target_qpos=None,
    ):
        # Set hand pose
        hand_pose = gymapi.Transform()
        hand_pose.r = gymapi.Quat(*hand_quaternion[1:], hand_quaternion[0])
        hand_pose.p = gymapi.Vec3(*hand_translation)
        hand_pose = transformation * hand_pose
        self.init_hand_poses.append(hand_pose)

        # Create hand
        hand_actor_handle = gym.create_actor(
            env, self.hand_asset, hand_pose, "hand", 0, -1
        )
        self.hand_handles.append(hand_actor_handle)

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
            hand_props["stiffness"][joint_idx] = 200.0
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

        # Set hand dof targets
        if target_qpos is not None:
            dof_pos_targets = gym.get_actor_dof_position_targets(env, hand_actor_handle)
            for i, joint in enumerate(self.joint_names):
                joint_idx = gym.find_actor_dof_index(
                    env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
                )
                dof_pos_targets[joint_idx] = target_qpos[i]
        else:
            dof_pos_targets = dof_states["pos"]
        gym.set_actor_dof_position_targets(env, hand_actor_handle, dof_pos_targets)

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

    def _setup_obj(self, env, obj_scale, transformation):
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0, 0, 0)
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)
        obj_pose = transformation * obj_pose
        self.init_obj_poses.append(obj_pose)

        # Create obj
        obj_actor_handle = gym.create_actor(
            env, self.obj_asset, obj_pose, "obj", 0, 1, OBJ_SEGMENTATION_ID
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

    def run_sim(self):
        sim_step_idx = 0
        default_desc = "Simulating"
        pbar = tqdm(total=self.sim_step, desc=default_desc, dynamic_ncols=True)
        while sim_step_idx < self.sim_step:
            # Set virtual joint targets
            virtual_joint_dof_pos_targets = self._compute_virtual_joint_dof_pos_targets(
                sim_step_idx
            )
            if virtual_joint_dof_pos_targets is not None:
                self._set_virtual_joint_dof_pos_targets(virtual_joint_dof_pos_targets)

            # Step physics if not paused
            if not self.is_paused:
                gym.simulate(self.sim)
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

        successes = []
        for i, (
            env,
            hand_link_idx_to_name,
            obj_link_idx_to_name,
            obj_handle,
            init_obj_pose,
        ) in enumerate(
            zip(
                self.envs,
                self.hand_link_idx_to_name_dicts,
                self.obj_link_idx_to_name_dicts,
                self.obj_handles,
                self.init_obj_poses,
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

            obj_pose = gym.get_actor_rigid_body_states(
                env, obj_handle, gymapi.STATE_POS
            )[0]["pose"]
            init_obj_pos = torch.tensor(
                [init_obj_pose.p.x, init_obj_pose.p.y, init_obj_pose.p.z]
            )
            init_obj_quat = torch.tensor(
                [
                    init_obj_pose.r.x,
                    init_obj_pose.r.y,
                    init_obj_pose.r.z,
                    init_obj_pose.r.w,
                ]
            )
            obj_pos = torch.tensor([obj_pose["p"][s] for s in "xyz"])
            obj_quat = torch.tensor([obj_pose["r"][s] for s in "xyzw"])

            quat_diff = torch_utils.quat_mul(
                obj_quat, torch_utils.quat_conjugate(init_obj_quat)
            )
            pos_change = torch.linalg.norm(obj_pos - init_obj_pos).item()
            euler_change = torch.stack(
                torch_utils.get_euler_xyz(quat_diff[None, ...])
            ).abs()
            euler_change = torch.where(
                euler_change > math.pi, 2 * math.pi - euler_change, euler_change
            )
            max_euler_change = euler_change.max().rad2deg().item()

            success = (
                len(hand_object_contacts) > 0
                and len(not_allowed_contacts) == 0
                and pos_change < 0.1
                and max_euler_change < 30
            )

            successes.append(success)

            DEBUG = False
            if DEBUG and len(hand_object_contacts) > 0:
                print(f"i = {i}")
                print(f"success = {success}")
                print(f"pos_change = {pos_change}")
                print(f"max_euler_change = {max_euler_change}")
                print(f"len(contacts) = {len(contacts)}")
                print(f"len(hand_object_contacts) = {len(hand_object_contacts)}")
                print(f"hand_link_contact_count = {hand_link_contact_count}")
                print(f"not_allowed_contacts = {not_allowed_contacts}")
                print("-------------")

        return successes

    def _compute_virtual_joint_dof_pos_targets(
        self, sim_step_idx: int
    ) -> Optional[List[torch.Tensor]]:
        # Only when virtual joints exist
        if len(self.virtual_joint_names) == 0:
            return None

        assert len(self.virtual_joint_names) == 6

        # First do nothing
        fraction_do_nothing = 0.1
        total_steps_not_moving = int(self.sim_step * fraction_do_nothing)
        if sim_step_idx < total_steps_not_moving:
            return None

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

        num_steps_moving_so_far = sim_step_idx - total_steps_not_moving
        total_steps_moving = self.sim_step - total_steps_not_moving
        direction_idx = int(
            (num_steps_moving_so_far / total_steps_moving) * len(directions_sequence)
        )
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

    def _visualize_origin_lines(self):
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

    def reset_simulator(self):
        for env in self.envs:
            gym.destroy_env(env)
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.sim)
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_link_idx_to_name_dicts = []
        self.obj_link_idx_to_name_dicts = []
        self.hand_asset = None
        self.obj_asset = None

    def destroy(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)

    def subscribe_to_keyboard_events(self):
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
        obj_scale,
    ):
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

        self._setup_obj(env, obj_scale, identity_transform)

    def save_images(self, folder, overwrite=False):
        assert len(self.envs) == 1
        self._setup_cameras(self.envs[0])

        gym.step_graphics(self.sim)
        gym.render_all_camera_sensors(self.sim)
        path = self._setup_save_dir(folder, overwrite)

        for ii, camera_handle in enumerate(self.camera_handles):
            self._save_single_image(path, ii, camera_handle)
        self._save_single_image(
            path, "overhead", self.overhead_camera_handle, numpy_depth=True
        )

        # Avoid segfault if run multiple times by destroying camera sensors
        self._destroy_cameras(self.envs[0])

    def _setup_cameras(self, env):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = CAMERA_HORIZONTAL_FOV_DEG
        camera_props.width = CAMERA_IMG_WIDTH
        camera_props.height = CAMERA_IMG_HEIGHT

        # generates camera positions along rings around object
        heights = [0.1, 0.3, 0.25, 0.35]
        distances = [0.2, 0.2, 0.3, 0.3]
        counts = [16, 16, 16, 16]
        target_ys = [0.0, 0.1, 0.0, 0.1]

        # compute camera positions
        camera_positions = []
        for height, distance, count, target_y in zip(heights, distances, counts, target_ys):
            for alpha in np.linspace(0, 2 * np.pi, count, endpoint=False):
                pos = [distance * np.sin(alpha), height, distance * np.cos(alpha)]
                camera_positions.append((pos, target_y))
        # repeat all from under since there is no ground plane
        for height, distance, count, target_y in zip(heights, distances, counts, target_ys):
            height = -height
            target_y = -target_y
            for alpha in np.linspace(0, 2 * np.pi, count, endpoint=False):
                pos = [distance * np.sin(alpha), height, distance * np.cos(alpha)]
                camera_positions.append((pos, target_y))

        self.camera_handles = []
        for pos, target_y in camera_positions:
            camera_handle = gym.create_camera_sensor(env, camera_props)
            gym.set_camera_location(
                camera_handle, env, gymapi.Vec3(*pos), gymapi.Vec3(0, target_y, 0)
            )

            self.camera_handles.append(camera_handle)

        self.overhead_camera_handle = gym.create_camera_sensor(env, camera_props)
        gym.set_camera_location(
            self.overhead_camera_handle,
            env,
            gymapi.Vec3(0, 0.5, 0.001),
            gymapi.Vec3(0, 0.01, 0),
        )

    def _destroy_cameras(self, env):
        for camera_handle in self.camera_handles:
            gym.destroy_camera_sensor(self.sim, env, camera_handle)
        gym.destroy_camera_sensor(self.sim, env, self.overhead_camera_handle)

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

    def _save_single_image(self, path, ii, camera_handle, numpy_depth=False, debug=False):
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

        depth_image = gym.get_camera_image(
            self.sim, env, camera_handle, gymapi.IMAGE_DEPTH
        )
        # distance in units I think
        depth_image = -depth_image.reshape(CAMERA_IMG_HEIGHT, CAMERA_IMG_WIDTH)
        if numpy_depth:
            np.save(path / f"dep_{ii}.npy", depth_image)
        else:
            depth_image = (np.clip(depth_image, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(depth_image).convert("L").save(path / f"dep_{ii}.png")

        pos, quat = get_fixed_camera_transform(gym, self.sim, env, camera_handle)

        with open(path / f"pos_xyz_quat_xyzw_{ii}.txt", "w+") as f:
            data = [*pos.tolist(), *quat.q[1:].tolist(), quat.q[0].tolist()]
            json.dump(data, f)

    def create_train_val_test_split(self, folder, train_frac, val_frac):
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

    def _run_sanity_check_proj_matrices_all_same(self):
        proj_matrix = gym.get_camera_proj_matrix(self.sim, self.envs[0], self.overhead_camera_handle)
        for camera_handle in self.camera_handles:
            next_proj_matrix = gym.get_camera_proj_matrix(self.sim, self.envs[0], camera_handle)
            assert np.allclose(proj_matrix, next_proj_matrix)

    def _get_camera_intrinsics(self) -> Tuple[float, float, float, float]:
        self._run_sanity_check_proj_matrices_all_same()

        proj_matrix = gym.get_camera_proj_matrix(self.sim, self.envs[0], self.camera_handles[0])
        fx = proj_matrix[0, 0]
        fy = proj_matrix[1, 1]
        cx = proj_matrix[0, 2]
        cy = proj_matrix[0, 2]

        assert math.isclose(fx, fy)
        assert math.isclose(cx, cy) and math.isclose(cx, 0) and math.isclose(cy, 0)
        return fx, fy, cx, cy

    def _create_one_split(self, split_name, split_range, folder):
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
                "fl_x": fx * CAMERA_IMG_WIDTH,
                "fl_y": fy * CAMERA_IMG_HEIGHT,
                # "cx": cx * CAMERA_IMG_WIDTH,
                # "cy": cy * CAMERA_IMG_HEIGHT,
                "cx": CAMERA_IMG_WIDTH//2,
                "cy": CAMERA_IMG_HEIGHT//2,
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

                json_dict["frames"].append(
                    {
                        "transform_matrix": transform_mat.tolist(),
                        "file_path": target_img,
                    }
                )

        with open(
            os.path.join(folder, f"transforms_{split_name}.json"), "w"
        ) as outfile:
            outfile.write(json.dumps(json_dict))

    ## NERF DATA COLLECTION END ##
