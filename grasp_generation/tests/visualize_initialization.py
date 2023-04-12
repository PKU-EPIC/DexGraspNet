"""
Last modified date: 2023.04.12
Author: Jialiang Zhang
Description: visualize convex hull initialization using plotly.graph_objects
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_table_top
import torch
import plotly.graph_objects as go
import argparse
import numpy as np
import random


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plane2pose(plane_parameters):
    r3 = plane_parameters[:3]
    r2 = torch.zeros_like(r3)
    r2[0], r2[1], r2[2] = (-r3[1], r3[0], 0) if r3[2] * r3[2] <= 0.5 else (-r3[2], 0, r3[0])
    r1 = torch.cross(r2, r3)
    pose = torch.zeros([4, 4], dtype=torch.float, device=plane_parameters.device)
    pose[0, :3] = r1
    pose[1, :3] = r2
    pose[2, :3] = r3
    pose[2, 3] = plane_parameters[3]
    pose[3, 3] = 1
    return pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--data_root_path', default='../data/meshdata', type=str)
    parser.add_argument('--poses', default='../data/poses', type=str)
    parser.add_argument('--object_code', default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_hand', default=30, type=int)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--num', default=2, type=int)
    # initialization settings
    parser.add_argument('--distance_lower', default=0.1, type=float)
    parser.add_argument('--distance_upper', default=0.1, type=float)
    parser.add_argument('--theta_lower', default=0, type=float)
    parser.add_argument('--theta_upper', default=0, type=float)
    parser.add_argument('--jitter_strength', default=0., type=float)
    parser.add_argument('--angle_upper', default=np.pi / 4, type=float)
    args = parser.parse_args()

    # seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cpu')
    
    # hand model

    hand_model = HandModel(
        mano_root='mano', 
        contact_indices_path='mano/contact_indices.json', 
        pose_distrib_path='mano/pose_distrib.pt', 
        device=device
    )
    
    # object model

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.n_hand,
        device=device
    )
    object_model.initialize(object_code_list=args.object_code)
    object_model.object_scale_tensor = torch.tensor([[0.06] * args.n_hand], dtype=torch.float, device=device)
    
    # initialize
    
    initialize_table_top(hand_model, object_model, args)
    
    # visualize
    
    pose = plane2pose(object_model.plane_parameters[args.num]).detach().cpu().numpy()

    object_plotly = object_model.get_plotly_data(0, opacity=1.0, color='lightgreen', pose=pose)

    mesh_convex = object_model.object_mesh_list[0].convex_hull
    v = mesh_convex.vertices * object_model.object_scale_tensor[0].max().detach().cpu().numpy()
    v += 0.2 * v / np.linalg.norm(v, axis=1, keepdims=True)
    f = mesh_convex.faces
    mesh_convex_plotly = [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='burlywood', opacity=0.5)]

    hand_plotly = hand_model.get_plotly_data(args.num, color='lightblue', opacity=1, pose=pose, with_contact_points=False)

    fig = go.Figure(object_plotly + mesh_convex_plotly + hand_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()
