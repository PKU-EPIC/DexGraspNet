"""
Last modified date: 2023.02.19
Author: Ruicheng Wang
Description: Class IsaacValidator
"""

from isaacgym import gymapi
from isaacgym import gymutil
import math
from time import sleep
from tqdm import tqdm
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType

gym = gymapi.acquire_gym()


class IsaacValidator():

    def __init__(self,
                 hand_model_type=HandModelType.SHADOW_HAND,
                 mode='direct',
                 hand_friction=3.,
                 obj_friction=3.,
                 threshold_dis=0.1,
                 env_batch=1,
                 sim_step=100,
                 gpu=0,
                 debug_interval=0.05):

        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.debug_interval = debug_interval
        self.threshold_dis = threshold_dis
        self.env_batch = env_batch
        self.gpu = gpu
        self.sim_step = sim_step
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.joint_names = handmodeltype_to_joint_names[hand_model_type]
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
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX,
                                  self.sim_params)
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 800
        self.camera_props.height = 600
        self.camera_props.use_collision_geometry = True

        # set viewer
        self.viewer = None
        if mode == "gui":
            self.has_viewer = True
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
            gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(0, 0, 1),
                                      gymapi.Vec3(0, 0, 0))
            self.subscribe_to_keyboard_events()
        else:
            self.has_viewer = False

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.fix_base_link = True
        self.hand_asset_options.collapse_fixed_joints = True
        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 500

        self.test_rotations = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            1 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            -0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            -0.5 * math.pi)),
        ]
        self.is_paused = False

    def set_asset(self, hand_root, hand_file, obj_root, obj_file):
        self.hand_asset = gym.load_asset(self.sim, hand_root, hand_file,
                                         self.hand_asset_options)
        self.obj_asset = gym.load_asset(self.sim, obj_root, obj_file,
                                        self.obj_asset_options)

    def add_env_all_test_rotations(self, hand_rotation, hand_translation, hand_qpos, obj_scale, target_qpos=None):
        for test_rotation_idx in range(len(self.test_rotations)):
            self.add_env_single_test_rotation(hand_rotation, hand_translation,
                                              hand_qpos, obj_scale,
                                              test_rotation_idx, target_qpos)

    def add_env_single_test_rotation(self, hand_rotation, hand_translation, hand_qpos, obj_scale, test_rotation_index=0, target_qpos=None):
        test_rot = self.test_rotations[test_rotation_index]

        # Create env
        env = gym.create_env(self.sim, gymapi.Vec3(-1, -1, -1),
                             gymapi.Vec3(1, 1, 1), 6)
        self.envs.append(env)

        # Set hand pose
        hand_pose = gymapi.Transform()
        hand_pose.r = gymapi.Quat(*hand_rotation[1:], hand_rotation[0])
        hand_pose.p = gymapi.Vec3(*hand_translation)
        hand_pose = test_rot * hand_pose

        # Create hand
        hand_actor_handle = gym.create_actor(
            env, self.hand_asset, hand_pose, "shand", 0, -1)
        self.hand_handles.append(hand_actor_handle)

        # Set hand dof props
        hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
        hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        hand_props["stiffness"].fill(1000)
        hand_props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)

        # Set hand dof states
        dof_states = gym.get_actor_dof_states(env, hand_actor_handle,
                                              gymapi.STATE_ALL)
        for i, joint in enumerate(self.joint_names):
            joint_idx = gym.find_actor_dof_index(env, hand_actor_handle,
                                                 joint,
                                                 gymapi.DOMAIN_ACTOR)
            dof_states["pos"][joint_idx] = hand_qpos[i]
        gym.set_actor_dof_states(env, hand_actor_handle, dof_states,
                                 gymapi.STATE_ALL)
        if target_qpos is not None:
            for i, joint in enumerate(self.joint_names):
                joint_idx = gym.find_actor_dof_index(env, hand_actor_handle,
                                                     joint,
                                                     gymapi.DOMAIN_ACTOR)
                dof_states["pos"][joint_idx] = target_qpos[i]
        gym.set_actor_dof_position_targets(env, hand_actor_handle,
                                           dof_states["pos"])

        # Store hand rigid body set
        hand_rigid_body_set = set()
        for i in range(
                gym.get_actor_rigid_body_count(env, hand_actor_handle)):
            hand_rigid_body_set.add(
                gym.get_actor_rigid_body_index(env, hand_actor_handle, i,
                                               gymapi.DOMAIN_ENV))
        self.hand_rigid_body_sets.append(hand_rigid_body_set)

        # Set hand shape props
        hand_shape_props = gym.get_actor_rigid_shape_properties(
            env, hand_actor_handle)
        for i in range(len(hand_shape_props)):
            hand_shape_props[i].friction = self.hand_friction
        gym.set_actor_rigid_shape_properties(env, hand_actor_handle,
                                             hand_shape_props)

        # Set obj pose
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0, 0, 0)
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)
        obj_pose = test_rot * obj_pose

        # Create obj
        obj_actor_handle = gym.create_actor(
            env, self.obj_asset, obj_pose, "obj", 0, 1)
        self.obj_handles.append(obj_actor_handle)
        gym.set_actor_scale(env, obj_actor_handle, obj_scale)

        # Store obj rigid body set
        obj_rigid_body_set = set()
        for i in range(
                gym.get_actor_rigid_body_count(env, obj_actor_handle)):
            obj_rigid_body_set.add(
                gym.get_actor_rigid_body_index(env, obj_actor_handle, i,
                                               gymapi.DOMAIN_ENV))
        self.obj_rigid_body_sets.append(obj_rigid_body_set)

        # Set obj shape props
        obj_shape_props = gym.get_actor_rigid_shape_properties(
            env, obj_actor_handle)
        for i in range(len(obj_shape_props)):
            obj_shape_props[i].friction = self.obj_friction
        gym.set_actor_rigid_shape_properties(env, obj_actor_handle,
                                             obj_shape_props)

    def run_sim(self):
        sim_step_idx = 0
        pbar = tqdm(total=self.sim_step, desc="Simulating")
        while sim_step_idx < self.sim_step:

            # Step physics if not paused
            if not self.is_paused:
                gym.simulate(self.sim)
                sim_step_idx += 1
                pbar.update(1)

            # Update viewer
            if self.has_viewer:
                sleep(self.debug_interval)
                if gym.query_viewer_has_closed(self.viewer):
                    break

                # Check for keyboard events
                for event in gym.query_viewer_action_events(self.viewer):
                    if event.value > 0 and event.action in self.event_to_function:
                        self.event_to_function[event.action]()

                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)

        success = []
        for i, env in enumerate(self.envs):
            contacts = gym.get_env_rigid_contacts(env)
            flag = False

            # TODO: Maybe count number of contacts
            for contact in contacts:
                body0 = contact['body0']
                body1 = contact['body1']
                hand_obj_in_contact = (
                    (body0 in self.hand_rigid_body_sets[i] and body1 in self.obj_rigid_body_sets[i])
                    or (body1 in self.hand_rigid_body_sets[i] and body0 in self.obj_rigid_body_sets[i])
                )
                if hand_obj_in_contact:
                    flag = True
                    break
            success.append(flag)
        return success

    def reset_simulator(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.sim)
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX,
                                  self.sim_params)
        for env in self.envs:
            gym.destroy_env(env)
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.hand_asset = None
        self.obj_asset = None

    def destroy(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)

    def subscribe_to_keyboard_events(self):
        if self.has_viewer:
            self.event_to_key = {
                "SLEEP": gymapi.KEY_S,
                "PAUSE_SIM": gymapi.KEY_P,
            }
            self.event_to_function  = {
                "SLEEP": self._sleep_callback,
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
    def _sleep_callback(self):
        N_SECONDS = 5
        print(f"Sleeping for {N_SECONDS} seconds...")
        sleep(N_SECONDS)

    def _pause_sim_callback(self):
        self.is_paused = not self.is_paused
    ## KEYBOARD EVENT SUBSCRIPTIONS END ##
