"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize object in world frame using plotly.graph_objects
"""

import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import trimesh as tm
import plotly.graph_objects as go


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default='../data/meshdata')
    parser.add_argument('--poses', type=str, default='../data/poses')
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0.1234567890)
    args = parser.parse_args()

    # load data
    pose_matrices = np.load(os.path.join(args.poses, args.object_code + '.npy'))
    print(f'n_data: {len(pose_matrices)}')
    pose_matrix = pose_matrices[args.num]
    pose_matrix[:3, 3] *= args.scale
    object_mesh = tm.load(os.path.join(args.data_root_path, args.object_code, 'coacd', 'decomposed.obj')).apply_scale(args.scale)

    # visualize
    v = object_mesh.vertices @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]
    f = object_mesh.faces
    object_plotly = go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='lightgreen', opacity=1)
    fig = go.Figure(object_plotly)
    fig.show()
