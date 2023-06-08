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
from typing import List, Optional

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


class ValidationType(Enum):
    GRAVITY_IN_6_DIRS = auto()
    NO_GRAVITY_SHAKING = auto()


class IsaacValidator:
    def __init__(
        self,
        hand_model_type=HandModelType.SHADOW_HAND,
        mode="direct",
        hand_friction=3.0,
        obj_friction=3.0,
        threshold_dis=0.1,
        env_batch=1,
        sim_step=500,
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

    def set_obj_asset(self, obj_root, obj_file):
        # TODO: Maybe don't need to make new hand_asset each time this is called
        self.hand_asset = gym.load_asset(
            self.sim, self.hand_root, self.hand_file, self.hand_asset_options
        )
        self.obj_asset = gym.load_asset(
            self.sim, obj_root, obj_file, self.obj_asset_options
        )

    def add_env_all_test_rotations(
        self, hand_rotation, hand_translation, hand_qpos, obj_scale, target_qpos=None
    ):
        for test_rotation_idx in range(len(self.test_rotations)):
            self.add_env_single_test_rotation(
                hand_rotation,
                hand_translation,
                hand_qpos,
                obj_scale,
                test_rotation_idx,
                target_qpos,
            )

    def add_env_single_test_rotation(
        self,
        hand_rotation,
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

        # Set hand pose
        hand_pose = gymapi.Transform()
        hand_pose.r = gymapi.Quat(*hand_rotation[1:], hand_rotation[0])
        hand_pose.p = gymapi.Vec3(*hand_translation)
        hand_pose = test_rot * hand_pose
        self.init_hand_poses.append(hand_pose)

        # Create hand
        hand_actor_handle = gym.create_actor(
            env, self.hand_asset, hand_pose, "hand", 0, -1
        )
        self.hand_handles.append(hand_actor_handle)

        # Set hand dof props
        hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
        hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)

        # Finger joints
        for joint in self.joint_names:
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            hand_props["stiffness"][joint_idx] = 1000.0
            hand_props["damping"][joint_idx] = 0.0

        # Virtual joints
        for joint in self.virtual_joint_names:
            joint_idx = gym.find_actor_dof_index(
                env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR
            )
            hand_props["stiffness"][joint_idx] = 100.0
            hand_props["damping"][joint_idx] = 1.0

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

        # Set obj pose
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0, 0, 0)
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)
        obj_pose = test_rot * obj_pose
        self.init_obj_poses.append(obj_pose)

        # Create obj
        obj_actor_handle = gym.create_actor(env, self.obj_asset, obj_pose, "obj", 0, 1)
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

    def run_sim(self):
        sim_step_idx = 0
        default_desc = "Simulating"
        pbar = tqdm(total=self.sim_step, desc=default_desc, dynamic_ncols=True)
        while sim_step_idx < self.sim_step:
            # Set virtual joint target
            virtual_joint_dof_pos_target = self._compute_virtual_joint_dof_pos_target(
                sim_step_idx
            )
            if virtual_joint_dof_pos_target is not None:
                self._set_virtual_joint_dof_pos_target(virtual_joint_dof_pos_target)

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

                # Visualize virtual joint target
                if virtual_joint_dof_pos_target is not None:
                    self._visualize_virtual_joint_dof_pos_target(
                        virtual_joint_dof_pos_target
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

    def _compute_virtual_joint_dof_pos_target(
        self, sim_step_idx: int
    ) -> Optional[List[float]]:
        # Only when virtual joints exist
        if len(self.virtual_joint_names) == 0:
            return None

        assert len(self.virtual_joint_names) == 6

        # First do nothing
        fraction_do_nothing = 0.5
        total_steps_not_moving = int(self.sim_step * fraction_do_nothing)
        if sim_step_idx < total_steps_not_moving:
            return None

        # Set dof pos target +x, -x, 0, +y, -y, 0, +z, -z
        dist_to_move = 0.1
        possible_dof_pos_targets = [
            [dist_to_move, 0.0, 0.0],
            [-dist_to_move, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, dist_to_move, 0.0],
            [0.0, -dist_to_move, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, dist_to_move],
            [0.0, 0.0, -dist_to_move],
        ]

        num_steps_moving_so_far = sim_step_idx - total_steps_not_moving
        total_steps_moving = self.sim_step - total_steps_not_moving
        dof_pos_target_idx = int(
            (num_steps_moving_so_far / total_steps_moving)
            * len(possible_dof_pos_targets)
        )
        print(f"dof_pos_target_idx = {dof_pos_target_idx}")
        dof_pos_target = possible_dof_pos_targets[dof_pos_target_idx]

        # Add target angles
        target_angles = [0.0, 0.0, 0.0]
        dof_pos_target = dof_pos_target + target_angles

        return dof_pos_target

    def _set_virtual_joint_dof_pos_target(self, dof_pos_target: List[float]) -> None:
        for env, hand_handle in zip(self.envs, self.hand_handles):
            actor_dof_pos_targets = gym.get_actor_dof_position_targets(env, hand_handle)

            for i, joint in enumerate(self.virtual_joint_names):
                joint_idx = gym.find_actor_dof_index(
                    env, hand_handle, joint, gymapi.DOMAIN_ACTOR
                )
                actor_dof_pos_targets[joint_idx] = dof_pos_target[i]
            gym.set_actor_dof_position_targets(env, hand_handle, actor_dof_pos_targets)

    def _visualize_virtual_joint_dof_pos_target(
        self, dof_pos_target: List[float]
    ) -> None:
        if not self.has_viewer:
            return
        visualization_sphere_green = gymutil.WireframeSphereGeometry(
            radius=0.01, num_lats=10, num_lons=10, color=(0, 1, 0)
        )
        for env, init_hand_pose in zip(self.envs, self.init_hand_poses):
            sphere_pose = gymapi.Transform(
                gymapi.Vec3(
                    init_hand_pose.p.x + dof_pos_target[0],
                    init_hand_pose.p.y + dof_pos_target[1],
                    init_hand_pose.p.z + dof_pos_target[2],
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
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.sim)
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        for env in self.envs:
            gym.destroy_env(env)
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
