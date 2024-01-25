"""
Last modified date: 2023.02.23
Author: Ruicheng Wang, Jialiang Zhang
Description: Class ObjectModel
"""

import os
import trimesh as tm
import plotly.graph_objects as go
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np
from typing import Optional, Tuple

from torchsdf import index_vertices_by_faces, compute_sdf


class ObjectModel:
    def __init__(
        self,
        meshdata_root_path: str,
        batch_size_each: int,
        num_samples: int = 250,
        num_calc_samples: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Create a Object Model

        Parameters
        ----------
        meshdata_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        scale: float or list of floats
            scale of object meshes
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """
        self.meshdata_root_path = meshdata_root_path
        self.batch_size_each = batch_size_each
        self.num_samples = num_samples
        self.device = device

        if num_calc_samples is None:
            self.num_calc_samples = num_samples
        else:
            self.num_calc_samples = num_calc_samples

        self.object_code_list = None
        self.object_scale_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None

    def initialize(self, object_code_list, object_scale_list):
        """
        Initialize Object Model with list of objects

        Choose scales, load meshes, sample surface points

        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        if isinstance(object_scale_list, float):
            object_scale_list = [object_scale_list]
        self.object_scale_list = object_scale_list

        self.object_scale_tensor = (
            torch.tensor(self.object_scale_list, dtype=torch.float, device=self.device)
            .unsqueeze(-1)
            .expand(-1, self.batch_size_each)
        )

        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        for object_code in object_code_list:
            self.object_mesh_list.append(
                tm.load(
                    os.path.join(
                        self.meshdata_root_path, object_code, "coacd", "decomposed.obj"
                    ),
                    force="mesh",
                    process=False,
                )
            )
            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(
                self.device
            )
            object_faces = (
                torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
            )
            self.object_face_verts_list.append(
                index_vertices_by_faces(object_verts, object_faces)
            )
            if self.num_samples != 0:
                vertices = torch.tensor(
                    self.object_mesh_list[-1].vertices,
                    dtype=torch.float,
                    device=self.device,
                )
                faces = torch.tensor(
                    self.object_mesh_list[-1].faces,
                    dtype=torch.float,
                    device=self.device,
                )
                mesh = pytorch3d.structures.Meshes(
                    vertices.unsqueeze(0), faces.unsqueeze(0)
                )
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                    mesh, num_samples=100 * self.num_samples
                )
                surface_points = pytorch3d.ops.sample_farthest_points(
                    dense_point_cloud, K=self.num_samples
                )[0][0]
                surface_points.to(dtype=float, device=self.device)
                self.surface_points_tensor.append(surface_points)

        if self.num_samples != 0:
            self.surface_points_tensor = (
                torch.stack(self.surface_points_tensor, dim=0)
                .unsqueeze(1)
                .expand(-1, self.batch_size_each, -1, -1)
                .reshape(-1, self.num_samples, 3)
            )  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals

        Interiors are positive, exteriors are negative

        Use our modified Kaolin package

        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes

        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(
                -1, n_points, 3
            )
            return distance, normals, closest_points
        return distance, normals

    def get_plotly_data(
        self, i, color="lightgreen", opacity=0.5, pose=None, with_surface_points=False, with_table=False
    ):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        data = []
        model_index = i // self.batch_size_each
        model_code = self.object_code_list[model_index]
        model_scale = (
            self.object_scale_tensor[model_index, i % self.batch_size_each]
            .detach()
            .cpu()
            .numpy()
        )
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color=color,
                opacity=opacity,
                name=f"object: {model_code}",
            )
        )

        if with_surface_points and len(self.surface_points_tensor) > 0:
            object_surface_points = (
                self.surface_points_tensor[i].detach().cpu().numpy() * model_scale
            )  # (num_samples, 3)
            if pose is not None:
                object_surface_points = (
                    object_surface_points @ pose[:3, :3].T + pose[:3, 3]
                )
            data.append(
                go.Scatter3d(
                    x=object_surface_points[:, 0],
                    y=object_surface_points[:, 1],
                    z=object_surface_points[:, 2],
                    mode="markers",
                    marker=dict(size=5, color="red"),
                    name=f"object surface points: {model_code}",
                )
            )
        if with_table:
            table_mesh = self.get_hacky_table_mesh(i, scaled=True)
            table_vertices = table_mesh.vertices
            if pose is not None:
                pose = np.array(pose, dtype=np.float32)
                table_vertices = table_vertices @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Mesh3d(
                    x=table_vertices[:, 0],
                    y=table_vertices[:, 1],
                    z=table_vertices[:, 2],
                    i=table_mesh.faces[:, 0],
                    j=table_mesh.faces[:, 1],
                    k=table_mesh.faces[:, 2],
                    color=color,
                    opacity=opacity,
                    name="table",
                )
            )

        return data

    def get_bounds(self, scaled: bool = True) -> torch.Tensor:
        """
        Get bounds of object meshes

        Args
        ----
        scaled: bool
            whether to return scaled bounds

        Returns
        -------
        bounds: (n_objects * batch_size_each, 2, 3) torch.Tensor
            bounds of object meshes
        """
        n_objects = len(self.object_mesh_list)
        bounds = []
        for i in range(n_objects):
            mesh = self.object_mesh_list[i]
            bounds.append(
                torch.from_numpy(mesh.bounds).float().to(self.device)
            )
        bounds = torch.stack(bounds)
        assert bounds.shape == (n_objects, 2, 3)
        bounds = bounds.unsqueeze(1).expand(-1, self.batch_size_each, -1, -1).reshape(-1, 2, 3)
        assert bounds.shape == (n_objects * self.batch_size_each, 2, 3)

        if scaled:
            scale = self.object_scale_tensor.reshape(-1, 1, 1)
            assert scale.shape == (n_objects * self.batch_size_each, 1, 1)
            bounds = bounds * scale

        return bounds

    def get_hacky_table(self, scaled: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hacky way to get "table" position and normal (i.e. min object y)

        Args:
            scaled (bool, optional): whether to return scaled table position. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: table position and normal and parallel
        """
        n_objects = len(self.object_mesh_list)

        full_batch_size = n_objects * self.batch_size_each
        object_bounds = self.get_bounds(scaled=scaled)
        assert object_bounds.shape == (full_batch_size, 2, 3)

        min_object_y = object_bounds[:, 0, 1]

        Y_OFFSET = 0.025
        table_y = min_object_y + Y_OFFSET
        assert table_y.shape == (full_batch_size,)

        table_pos = torch.zeros(
            [full_batch_size, 3], dtype=torch.float, device=self.device
        )
        table_pos[:, 1] = table_y

        table_normal = (
            torch.tensor([0, 1, 0], dtype=torch.float, device=self.device)
            .reshape(1, 3)
            .expand(full_batch_size, -1)
        )
        table_parallel = (
            torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
            .reshape(1, 3)
            .expand(full_batch_size, -1)
        )
        assert table_pos.shape == table_normal.shape == table_parallel.shape == (full_batch_size, 3)

        return table_pos, table_normal, table_parallel

    def get_hacky_table_mesh(self, idx: int, scaled: bool = True) -> tm.Trimesh:
        table_pos, table_normal, table_parallel = self.get_hacky_table(scaled=scaled)
        table_pos, table_normal, table_parallel = (
            table_pos[idx].detach().cpu().numpy(),
            table_normal[idx].detach().cpu().numpy(),
            table_parallel[idx].detach().cpu().numpy(),
        )
        assert table_pos.shape == table_normal.shape == (3,)

        bounds = self.get_bounds(scaled=True)
        bounds = bounds[idx].detach().cpu().numpy()
        assert bounds.shape == (2, 3)
        SCALE_FACTOR = 2
        W, H = bounds[1, 0] - bounds[0, 0], bounds[1, 2] - bounds[0, 2]
        W, H = SCALE_FACTOR * W, SCALE_FACTOR * H

        table_parallel_2 = np.cross(table_normal, table_parallel)
        corner1 = table_pos + W / 2 * table_parallel + H / 2 * table_parallel_2
        corner2 = table_pos + W / 2 * table_parallel - H / 2 * table_parallel_2
        corner3 = table_pos - W / 2 * table_parallel + H / 2 * table_parallel_2
        corner4 = table_pos - W / 2 * table_parallel - H / 2 * table_parallel_2

        x = np.array([corner1[0], corner2[0], corner3[0], corner4[0]])
        y = np.array([corner1[1], corner2[1], corner3[1], corner4[1]])
        z = np.array([corner1[2], corner2[2], corner3[2], corner4[2]])

        i = [0, 0, 1]
        j = [1, 2, 2]
        k = [2, 3, 3]

        table_mesh = tm.Trimesh(vertices=np.stack([x, y, z], axis=1), faces=np.stack([i, j, k], axis=1))
        return table_mesh

