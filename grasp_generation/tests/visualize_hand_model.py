"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize hand model using plotly.graph_objects
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

import numpy as np
import torch
import transforms3d
import plotly.graph_objects as go
from utils.hand_model import HandModel


torch.manual_seed(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    device = torch.device('cpu')

    # hand model

    hand_model = HandModel(
        mano_root='mano', 
        contact_indices_path='mano/contact_indices.json', 
        pose_distrib_path='mano/pose_distrib.pt', 
        device=device
    )

    vec, angle = transforms3d.euler.euler2axangle(-np.pi / 2, -np.pi / 2, np.pi / 6, axes='rzxz')
    hand_pose = torch.concat([
        torch.tensor([-0.1, -0.05, 0], dtype=torch.float, device=device), 
        torch.tensor(vec * angle, dtype=torch.float, device=device), 
        torch.tensor([
            0, 0, torch.pi / 6, 
            0, 0, 0, 
            0, 0, 0, 
            
            0, 0, torch.pi / 6, 
            0, 0, 0, 
            0, 0, 0, 
            
            0, 0, torch.pi / 6, 
            0, 0, 0, 
            0, 0, 0, 
            
            0, 0, torch.pi / 6, 
            0, 0, 0, 
            0, 0, 0, 
            
            *(torch.pi / 2 * torch.tensor([2, 1, 0], dtype=torch.float) / torch.tensor([2, 1, 0], dtype=torch.float).norm()), 
            0, 0, 0, 
            0, 0, 0, 
        ], dtype=torch.float, device=device), 
    ])
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # info
    contact_candidates = hand_model.get_contact_candidates()
    print(f'n_contact_candidates: {hand_model.n_contact_candidates}')

    # visualize

    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_keypoints=True)
    v = contact_candidates[0].detach().cpu().numpy()
    contact_candidates_plotly = [go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='white'))]
    
    fig = go.Figure(hand_plotly + contact_candidates_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()
