"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: Class HandModel
"""


import os
import json
import numpy as np
import torch
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import pytorch_kinematics as pk
import plotly.graph_objects as go
import trimesh as tm

import transforms3d
from urdf_parser_py.urdf import Robot, Box, Sphere
from utils.hand_model_type import HandModelType, handmodeltype_to_fingerkeywords
from collections import defaultdict


class HandModel:
    def __init__(
        self,
        hand_model_type=HandModelType.SHADOW_HAND,
        n_surface_points=0,
        device="cpu",
    ):
        """
        Create a Hand Model for a MJCF robot

        Parameters
        ----------
        hand_model_type: HandModelType
            type of hand model
        device: str | torch.Device
            device for torch tensors
        """
        self.hand_model_type = hand_model_type
        self.device = device

        # load articulation
        # should create:
        #   * self.chain: pytorch_kinematics.Chain
        #   * self.mesh dict with link_name keys and dict values with:
        #       * vertices: (N, 3) torch.FloatTensor
        #       * faces: (N, 3) torch.LongTensor
        #       * contact_candidates: (M, 3) torch.FloatTensor
        #       * penetration_keypoints: (K, 3) torch.FloatTensor
        #       * surface_points: (S, 3) torch.FloatTensor
        #       * some others that are type and link specific
        #   * self.areas dict with link_name keys and float values
        #   * self.n_dofs: int
        #   * self.joints_upper: (D,) torch.FloatTensor
        #   * self.joints_lower: (D,) torch.FloatTensor
        if self.hand_model_type == HandModelType.ALLEGRO_HAND:
            self._init_allegro(n_surface_points=n_surface_points)
        elif self.hand_model_type == HandModelType.SHADOW_HAND:
            self._init_shadow(n_surface_points=n_surface_points)
        else:
            raise ValueError(f"Unknown hand model type: {self.hand_model_type}")

        # indexing
        self.link_name_to_link_index = dict(
            zip([link_name for link_name in self.mesh], range(len(self.mesh)))
        )

        self.link_name_to_contact_candidates = {
            link_name: self.mesh[link_name]["contact_candidates"]
            for link_name in self.mesh
        }
        contact_candidates = [
            self.link_name_to_contact_candidates[link_name] for link_name in self.mesh
        ]
        self.global_index_to_link_index = sum(
            [
                [i] * len(contact_candidates)
                for i, contact_candidates in enumerate(contact_candidates)
            ],
            [],
        )
        self.link_index_to_global_indices = defaultdict(list)
        for global_idx, link_idx in enumerate(self.global_index_to_link_index):
            self.link_index_to_global_indices[link_idx].append(global_idx)

        self.contact_candidates = torch.cat(contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=device
        )
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = [
            self.mesh[link_name]["penetration_keypoints"] for link_name in self.mesh
        ]
        self.global_index_to_link_index_penetration = sum(
            [
                [i] * len(penetration_keypoints)
                for i, penetration_keypoints in enumerate(self.penetration_keypoints)
            ],
            [],
        )
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(
            self.global_index_to_link_index_penetration, dtype=torch.long, device=device
        )
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # parameters
        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

    def _init_allegro(
        self,
        urdf_path="allegro_hand_description/allegro_hand_description_right.urdf",
        contact_points_path="allegro_hand_description/contact_points_precision_grasp_dense.json",
        penetration_points_path="allegro_hand_description/penetration_points.json",
        n_surface_points=0,
    ):
        device = self.device

        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(
            dtype=torch.float, device=device
        )
        robot = Robot.from_xml_file(urdf_path)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        penetration_points = json.load(open(penetration_points_path, "r"))
        contact_points = json.load(open(contact_points_path, "r"))

        self.mesh = {}
        self.areas = {}
        for link in robot.links:
            if link.visual is None or link.collision is None:
                continue
            self.mesh[link.name] = {}

            # load collision mesh
            collision = link.collision
            if type(collision.geometry) == Sphere:
                link_mesh = tm.primitives.Sphere(radius=collision.geometry.radius)
                self.mesh[link.name]["radius"] = collision.geometry.radius
            elif type(collision.geometry) == Box:
                # link_mesh = tm.primitives.Box(extents=collision.geometry.size)
                link_mesh = tm.load_mesh(
                    os.path.join(os.path.dirname(urdf_path), "meshes", "box.obj"),
                    process=False,
                )
                link_mesh.vertices *= np.array(collision.geometry.size) / 2
            else:
                raise ValueError(
                    f"Unknown collision geometry: {type(collision.geometry)}"
                )
            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device
            )
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if (
                hasattr(collision.geometry, "scale")
                and collision.geometry.scale is None
            ):
                collision.geometry.scale = [1, 1, 1]
            scale = torch.tensor(
                getattr(collision.geometry, "scale", [1, 1, 1]),
                dtype=torch.float,
                device=device,
            )
            translation = torch.tensor(
                getattr(collision.origin, "xyz", [0, 0, 0]),
                dtype=torch.float,
                device=device,
            )
            rotation = torch.tensor(
                transforms3d.euler.euler2mat(
                    *getattr(collision.origin, "rpy", [0, 0, 0])
                ),
                dtype=torch.float,
                device=device,
            )
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update(
                {
                    "vertices": vertices,
                    "faces": faces,
                }
            )
            if "radius" not in self.mesh[link.name]:
                # ########################### #
                # DEBUG: comment this for now #
                # ########################### #
                print("WARNING: COMMENTING OUT THE INDEX VERTICES BY FACES FUNC")
                # self.mesh[link.name]["face_verts"] = index_vertices_by_faces(
                #     vertices, faces
                # )

            # load visual mesh
            visual = link.visual
            filename = os.path.join(
                os.path.dirname(os.path.dirname(urdf_path)),
                visual.geometry.filename[10:],
            )
            link_mesh = tm.load_mesh(filename)
            visual_vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device
            )
            visual_faces = torch.tensor(
                link_mesh.faces, dtype=torch.long, device=device
            )
            if hasattr(visual.geometry, "scale") and visual.geometry.scale is None:
                visual.geometry.scale = [1, 1, 1]
            visual_scale = torch.tensor(
                getattr(visual.geometry, "scale", [1, 1, 1]),
                dtype=torch.float,
                device=device,
            )
            visual_translation = torch.tensor(
                getattr(visual.origin, "xyz", [0, 0, 0]),
                dtype=torch.float,
                device=device,
            )
            visual_rotation = torch.tensor(
                transforms3d.euler.euler2mat(*getattr(visual.origin, "rpy", [0, 0, 0])),
                dtype=torch.float,
                device=device,
            )
            visual_vertices = visual_vertices * visual_scale
            visual_vertices = visual_vertices @ visual_rotation.T + visual_translation
            self.mesh[link.name].update(
                {
                    "visual_vertices": visual_vertices,
                    "visual_faces": visual_faces,
                }
            )
            # load contact candidates and penetration keypoints
            contact_candidates = torch.tensor(
                contact_points[link.name], dtype=torch.float32, device=device
            ).reshape(-1, 3)
            penetration_keypoints = torch.tensor(
                penetration_points[link.name], dtype=torch.float32, device=device
            ).reshape(-1, 3)
            self.mesh[link.name].update(
                {
                    "contact_candidates": contact_candidates,
                    "penetration_keypoints": penetration_keypoints,
                }
            )
            self.areas[link.name] = tm.Trimesh(
                vertices.cpu().numpy(), faces.cpu().numpy()
            ).area.item()

        self.joints_lower = torch.tensor(
            [
                joint.limit.lower
                for joint in robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )
        self.joints_upper = torch.tensor(
            [
                joint.limit.upper
                for joint in robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )

        self._sample_surface_points(n_surface_points)

    def _init_shadow(
        self,
        mjcf_path="mjcf/shadow_hand_wrist_free.xml",
        mesh_path="mjcf/meshes",
        contact_points_path="mjcf/contact_points_precision_grasp.json",
        penetration_points_path="mjcf/penetration_points.json",
        n_surface_points=0,
    ):
        """
        Create a Hand Model for a MJCF robot

        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of hand, use fps
        """
        device = self.device

        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(
            dtype=torch.float, device=device
        )
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # load contact points and penetration points

        contact_points = (
            json.load(open(contact_points_path, "r"))
            if contact_points_path is not None
            else None
        )
        penetration_points = (
            json.load(open(penetration_points_path, "r"))
            if penetration_points_path is not None
            else None
        )

        # build mesh

        self.mesh = {}
        self.areas = {}

        def build_mesh_recurse(body):
            if len(body.link.visuals) > 0:
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        # link_mesh = trimesh.primitives.Box(extents=2 * visual.geom_param)
                        link_mesh = tm.load_mesh(
                            os.path.join(mesh_path, "box.obj"), process=False
                        )
                        link_mesh.vertices *= visual.geom_param.detach().cpu().numpy()
                    elif visual.geom_type == "capsule":
                        link_mesh = tm.primitives.Capsule(
                            radius=visual.geom_param[0], height=visual.geom_param[1] * 2
                        ).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        link_mesh = tm.load_mesh(
                            os.path.join(
                                mesh_path, visual.geom_param[0].split(":")[1] + ".obj"
                            ),
                            process=False,
                        )
                        if visual.geom_param[1] is not None:
                            scale = torch.tensor(
                                visual.geom_param[1], dtype=torch.float, device=device
                            )
                    vertices = torch.tensor(
                        link_mesh.vertices, dtype=torch.float, device=device
                    )
                    faces = torch.tensor(
                        link_mesh.faces, dtype=torch.long, device=device
                    )
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                contact_candidates = (
                    torch.tensor(
                        contact_points[link_name], dtype=torch.float32, device=device
                    ).reshape(-1, 3)
                    if contact_points is not None
                    else None
                )
                penetration_keypoints = (
                    torch.tensor(
                        penetration_points[link_name],
                        dtype=torch.float32,
                        device=device,
                    ).reshape(-1, 3)
                    if penetration_points is not None
                    else None
                )
                self.mesh[link_name] = {
                    "vertices": link_vertices,
                    "faces": link_faces,
                    "contact_candidates": contact_candidates,
                    "penetration_keypoints": penetration_keypoints,
                }
                if link_name in [
                    "robot0:palm",
                    "robot0:palm_child",
                    "robot0:lfmetacarpal_child",
                ]:
                    from torchsdf import index_vertices_by_faces, compute_sdf
                    link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                    self.mesh[link_name]["face_verts"] = link_face_verts
                else:
                    self.mesh[link_name]["geom_param"] = body.link.visuals[0].geom_param
                self.areas[link_name] = tm.Trimesh(
                    link_vertices.cpu().numpy(), link_faces.cpu().numpy()
                ).area.item()
            for children in body.children:
                build_mesh_recurse(children)

        build_mesh_recurse(self.chain._root)

        # set joint limits
        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)

        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        self._sample_surface_points(n_surface_points)

    def _sample_surface_points(self, n_surface_points):
        if n_surface_points == 0:
            return
        import pytorch3d.structures
        import pytorch3d.ops
        device = self.device

        total_area = sum(self.areas.values())
        num_samples = dict(
            [
                (link_name, int(self.areas[link_name] / total_area * n_surface_points))
                for link_name in self.mesh
            ]
        )
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(
            num_samples.values()
        )
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=device
                ).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(
                self.mesh[link_name]["vertices"].unsqueeze(0),
                self.mesh[link_name]["faces"].unsqueeze(0),
            )
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=100 * num_samples[link_name]
            )
            surface_points = pytorch3d.ops.sample_farthest_points(
                dense_point_cloud, K=num_samples[link_name]
            )[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]["surface_points"] = surface_points

    def sample_contact_points(self, total_batch_size: int, n_contacts_per_finger: int):
        # Ensure that each finger gets sampled at least once
        # Goal: Output (B, n_fingers * n_contacts_per_finger) torch.LongTensor of sampled contact point indices
        # Each contact point is represented by a global index
        # Each contact point is sampled from a link
        # For each finger:
        #    Get the link indices that contain the finger keyword
        #    Get the possible contact point indices from these link indices
        #    Sample from these contact point indices

        finger_keywords = handmodeltype_to_fingerkeywords[self.hand_model_type]

        # Get link indices that contain the finger keyword
        finger_possible_link_idxs_list = [
            [
                link_idx
                for link_name, link_idx in self.link_name_to_link_index.items()
                if finger_keyword in link_name
            ]
            for finger_keyword in finger_keywords
        ]

        # Get the possible contact point indices from these link indices
        finger_possible_contact_point_idxs_list = [
            sum(
                [self.link_index_to_global_indices[link_idx] for link_idx in link_idxs],
                [],
            )
            for link_idxs in finger_possible_link_idxs_list
        ]

        # Sample from these contact point indices
        sampled_contact_point_idxs_list = []
        for (
            finger_possible_contact_point_idxs
        ) in finger_possible_contact_point_idxs_list:
            sampled_idxs = torch.randint(
                len(finger_possible_contact_point_idxs),
                size=[total_batch_size, n_contacts_per_finger],
                device=self.device,
            )
            sampled_contact_point_idxs = torch.tensor(
                finger_possible_contact_point_idxs, device=self.device, dtype=torch.long
            )[sampled_idxs]
            sampled_contact_point_idxs_list.append(sampled_contact_point_idxs)
        sampled_contact_point_idxs_list = torch.cat(
            sampled_contact_point_idxs_list, dim=1
        )

        assert sampled_contact_point_idxs_list.shape == (
            total_batch_size,
            len(finger_keywords) * n_contacts_per_finger,
        )

        return sampled_contact_point_idxs_list

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        assert len(hand_pose.shape) <= 2
        if len(hand_pose.shape) == 1:
            hand_pose = hand_pose.unsqueeze(0)
        assert hand_pose.shape[1] == 3 + 6 + self.n_dofs

        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9]
        )
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(
                batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device
            )
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = (
                    self.current_status[link_name]
                    .get_matrix()
                    .unsqueeze(1)
                    .expand(batch_size, n_contact, 4, 4)
                )
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat(
                [
                    self.contact_points,
                    torch.ones(
                        batch_size, n_contact, 1, dtype=torch.float, device=self.device
                    ),
                ],
                dim=2,
            )
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[
                :, :, :3, 0
            ]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(
                1, 2
            ) + self.global_translation.unsqueeze(1)

    def cal_distance(self, x):
        if self.hand_model_type == HandModelType.ALLEGRO_HAND:
            return self._cal_distance_allegro(x)
        elif self.hand_model_type == HandModelType.SHADOW_HAND:
            return self._cal_distance_shadow(x)
        else:
            raise ValueError(f"Unknown hand model type: {self.hand_model_type}")

    def _cal_distance_shadow(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes

        Interiors are positive, exteriors are negative

        Use analytical method and our modified Kaolin package

        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Consider each link seperately:
        #   First, transform x into each link's local reference frame using inversed fk, which gives us x_local
        #   Next, calculate point-to-mesh distances in each link's frame, this gives dis_local
        #   Finally, the maximum over all links is the final distance from one point to the entire ariticulation
        # In particular, the collision mesh of ShadowHand is only composed of Capsules and Boxes
        # We use analytical method to calculate Capsule sdf, and use our modified Kaolin package for other meshes
        # This practice speeds up the reverse penetration calculation
        # Note that we use a chamfer box instead of a primitive box to get more accurate signs
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            if link_name in [
                "robot0:forearm",
                "robot0:wrist_child",
                "robot0:ffknuckle_child",
                "robot0:mfknuckle_child",
                "robot0:rfknuckle_child",
                "robot0:lfknuckle_child",
                "robot0:thbase_child",
                "robot0:thhub_child",
            ]:
                continue
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if "geom_param" not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]["face_verts"]
                from torchsdf import index_vertices_by_faces, compute_sdf
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                height = self.mesh[link_name]["geom_param"][1] * 2
                radius = self.mesh[link_name]["geom_param"][0]
                nearest_point = x_local.detach().clone()
                nearest_point[:, :2] = 0
                nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
                dis_local = radius - (x_local - nearest_point).norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def _cal_distance_allegro(self, x):
        from torchsdf import index_vertices_by_faces, compute_sdf

        # x: (total_batch_size, num_samples, 3)
        # 单独考虑每个link
        # 先把x变换到link的局部坐标系里面，得到x_local: (total_batch_size, num_samples, 3)
        # 然后计算dis，按照内外取符号，内部是正号
        # 最后的dis就是所有link的dis的最大值
        # 对于sphere的link，使用解析方法计算dis，否则用mesh的方法计算dis
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if "radius" not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]["face_verts"]
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                dis_local = self.mesh[link_name]["radius"] - x_local.norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def cal_self_penetration_energy(self):
        """
        Calculate self penetration energy

        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(
            batch_size, 1
        )
        transforms = torch.zeros(
            batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device
        )
        for link_name in self.mesh:
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = (
                self.current_status[link_name]
                .get_matrix()
                .unsqueeze(1)
                .expand(batch_size, self.n_keypoints, 4, 4)
            )
            transforms[mask] = cur[mask]
        points = torch.cat(
            [
                points,
                torch.ones(
                    batch_size,
                    self.n_keypoints,
                    1,
                    dtype=torch.float,
                    device=self.device,
                ),
            ],
            dim=2,
        )
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1, 2))

    def cal_joint_limit_energy(self):
        joint_limit_energy = torch.sum(
            (self.hand_pose[:, 9:] > self.joints_upper)
            * (self.hand_pose[:, 9:] - self.joints_upper),
            dim=-1,
        ) + torch.sum(
            (self.hand_pose[:, 9:] < self.joints_lower)
            * (self.joints_lower - self.hand_pose[:, 9:]),
            dim=-1,
        )
        return joint_limit_energy

    def cal_finger_finger_distance_energy(self):
        batch_size = self.contact_points.shape[0]
        finger_finger_distance_energy = (
            -torch.cdist(self.contact_points, self.contact_points, p=2)
            .reshape(batch_size, -1)
            .sum(dim=-1)
        )
        return finger_finger_distance_energy

    def cal_palm_finger_distance_energy(self):
        palm_position = self.global_translation[:, None, :]
        palm_finger_distance_energy = (
            -(palm_position - self.contact_points).norm(dim=-1).sum(dim=-1)
        )
        return palm_finger_distance_energy

    def get_surface_points(self):
        """
        Get surface points

        Returns
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["surface_points"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_contact_candidates(self):
        """
        Get all contact candidates

        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_contact_candidates = self.mesh[link_name]["contact_candidates"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["contact_candidates"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_contact_candidates, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_penetration_keypoints(self):
        """
        Get penetration keypoints

        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_keypoints = self.mesh[link_name]["penetration_keypoints"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["penetration_keypoints"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_keypoints, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_plotly_data(
        self,
        i,
        opacity=0.5,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=False,
        with_surface_points=False,
        with_penetration_keypoints=False,
        pose=None,
        visual=True,
    ):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        with_contact_candidates: bool
            whether to visualize contact candidates
        with_surface_points: bool
            whether to visualize surface points
        with_penetration_keypoints: bool
            whether to visualize penetration keypoints
        pose: (4, 4) matrix
            homogeneous transformation matrix
        visual: bool
            whether to visualize the hand with visual components

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]["visual_vertices"]
                if visual and "visual_vertices" in self.mesh[link_name]
                else self.mesh[link_name]["vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = (
                (
                    self.mesh[link_name]["visual_faces"]
                    if visual and "visual_faces" in self.mesh[link_name]
                    else self.mesh[link_name]["faces"]
                )
                .detach()
                .cpu()
            )
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                    name="hand",
                )
            )
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="contact points",
                )
            )
        if with_contact_candidates:
            contact_candidates = self.get_contact_candidates()[i].detach().cpu()
            if pose is not None:
                contact_candidates = contact_candidates @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=contact_candidates[:, 0],
                    y=contact_candidates[:, 1],
                    z=contact_candidates[:, 2],
                    mode="markers",
                    marker=dict(color="blue", size=5),
                    name="contact candidates",
                )
            )
        if with_surface_points:
            surface_points = self.get_surface_points()[i].detach().cpu()
            if pose is not None:
                surface_points = surface_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=surface_points[:, 0],
                    y=surface_points[:, 1],
                    z=surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="green", size=2),
                    name="surface points",
                )
            )

        if with_penetration_keypoints:
            penetration_keypoints = self.get_penetration_keypoints()[i].detach().cpu()
            if pose is not None:
                penetration_keypoints = (
                    penetration_keypoints @ pose[:3, :3].T + pose[:3, 3]
                )
            data.append(
                go.Scatter3d(
                    x=penetration_keypoints[:, 0],
                    y=penetration_keypoints[:, 1],
                    z=penetration_keypoints[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=3),
                    name="penetration_keypoints",
                )
            )
            for penetration_keypoint in penetration_keypoints:
                penetration_keypoint = penetration_keypoint.numpy()
                mesh = tm.primitives.Capsule(radius=0.01, height=0)
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                data.append(
                    go.Mesh3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        color="burlywood",
                        opacity=0.5,
                        name="penetration_keypoints_mesh",
                    )
                )

        return data
    
    @property
    def n_fingers(self):
        return len(handmodeltype_to_fingerkeywords[self.hand_model_type])

    def get_trimesh_data(self, i):
        """
        Get full mesh

        Returns
        -------
        data: trimesh.Trimesh
        """
        import trimesh

        data = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]["vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data

    @property
    def batch_size(self) -> int:
        if self.hand_pose is None:
            raise ValueError("Hand pose is not set")
        return self.hand_pose.shape[0]

    @property
    def num_fingers(self) -> int:
        return len(handmodeltype_to_fingerkeywords[self.hand_model_type])

    def cal_table_penetration(self, table_pos: torch.Tensor, table_normal: torch.Tensor) -> torch.Tensor:
        """
        Calculate table penetration energy

        Args
        ----
        table_pos: (B, 3) torch.Tensor
            position of table surface
        table_normal: (B, 3) torch.Tensor
            normal of table

        Returns
        -------
        E_tpen: (B,) torch.Tensor
            table penetration energy
        """
        # Two methods: use sampled points or meshes
        B1, D1 = table_pos.shape
        B2, D2 = table_normal.shape
        assert B1 == B2
        assert D1 == D2 == 3

        sampled_points_world_frame = self.get_surface_points()
        B, N, D = sampled_points_world_frame.shape
        assert B == B1
        assert D == 3

        # Positive = above table, negative = below table
        signed_distance_from_table = torch.sum(
            (sampled_points_world_frame - table_pos.unsqueeze(1)) * table_normal.unsqueeze(1), dim=-1
        )

        penetration = torch.clamp(signed_distance_from_table, max=0.0)
        penetration = -penetration
        assert penetration.shape == (B, N)

        return penetration.sum(-1)

