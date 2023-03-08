"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='core-mug-1a1c0a8d4bad82169f0594e65f756cf5')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='../data/experiments/exp_32/results')
    args = parser.parse_args()

    device = 'cpu'

    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    qpos = data_dict['qpos']
    hand_pose = torch.concat([torch.tensor(qpos[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])
    if 'contact_point_indices' in data_dict:
        contact_point_indices = torch.tensor(data_dict['contact_point_indices'], dtype=torch.long, device=device)
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        hand_pose_st = torch.concat([torch.tensor(qpos_st[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])

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
        batch_size_each=1,
        num_samples=2000, 
        device=device
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    # visualize

    if 'qpos_st' in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue')
    else:
        hand_st_plotly = []
    if 'contact_point_indices' in data_dict:
        hand_model.set_parameters(hand_pose.unsqueeze(0), contact_point_indices.unsqueeze(0))
        hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=True)
    else:
        hand_model.set_parameters(hand_pose.unsqueeze(0))
        hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue')
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    if 'energy' in data_dict:
        scale = round(data_dict['scale'], 2)
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_prior = round(data_dict['E_prior'], 3)
        E_spen = round(data_dict['E_spen'], 4)
        result = f'Index {args.num}  scale {scale}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}  E_prior {E_prior}  E_spen {E_spen}'
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    fig.update_layout(scene_aspectmode='data')
    fig.show()
