"""
Last modified date: 2023.04.12
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
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--num', type=int, default=9)
    parser.add_argument('--result_path', type=str, default='../data/experiments/demo/results')
    args = parser.parse_args()

    device = 'cpu'

    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    plane = torch.tensor(data_dict['plane'], dtype=torch.float, device=device)  # plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
    pose = plane2pose(plane)  # 4x4 homogeneous transformation matrix from object frame to world frame
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
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', pose=pose)
    else:
        hand_st_plotly = []
    if 'contact_point_indices' in data_dict:
        hand_model.set_parameters(hand_pose.unsqueeze(0), contact_point_indices.unsqueeze(0))
        hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=True, pose=pose)
    else:
        hand_model.set_parameters(hand_pose.unsqueeze(0))
        hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', pose=pose)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1, pose=pose)
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    if 'energy' in data_dict:
        scale = round(data_dict['scale'], 2)
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_prior = round(data_dict['E_prior'], 3)
        E_spen = round(data_dict['E_spen'], 4)
        E_tpen = round(data_dict['E_tpen'], 4)
        result = f'Index {args.num}  scale {scale}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}  E_prior {E_prior}  E_spen {E_spen}  E_tpen {E_tpen}'
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    fig.update_layout(scene_aspectmode='data')
    fig.show()
