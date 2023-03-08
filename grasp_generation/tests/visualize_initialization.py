"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize convex hull initialization using plotly.graph_objects
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
import torch
import plotly.graph_objects as go
import argparse
import math
import numpy as np
import trimesh as tm


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hand', default=30, type=int)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--distance_lower', default=0.1, type=float)
    parser.add_argument('--distance_upper', default=0.1, type=float)
    parser.add_argument('--theta_lower', default=0, type=float)
    parser.add_argument('--theta_upper', default=0, type=float)
    parser.add_argument('--jitter_strength', default=0., type=float)
    args = parser.parse_args()

    torch.manual_seed(1)

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
        data_root_path='../data/meshdata',
        batch_size_each=args.n_hand,
        device=device
    )
    object_model.initialize(object_code_list='core-bottle-1ffd7113492d375593202bf99dddc268')
    object_model.object_scale_tensor = torch.tensor([[0.06] * args.n_hand], dtype=torch.float, device=device)
    
    # initialize
    
    initialize_convex_hull(hand_model, object_model, args)
    
    # visualize

    object_plotly = object_model.get_plotly_data(0, opacity=1.0, color='lightgreen')

    data = object_plotly
    for i in range(args.n_hand):
        hand_plotly = hand_model.get_plotly_data(i, opacity=1.0, color='lightblue')
        data += hand_plotly
    
    mesh_convex = object_model.object_mesh_list[0].convex_hull
    v = mesh_convex.vertices * object_model.object_scale_tensor[0].max().detach().cpu().numpy()
    v += 0.2 * v / np.linalg.norm(v, axis=1, keepdims=True)
    f = mesh_convex.faces
    mesh_convex_plotly = [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='burlywood', opacity=0.5)]
    data += mesh_convex_plotly

    fig = go.Figure(data)
    fig.update_layout(scene_aspectmode='data')
    fig.show()
