"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType
from utils.qpos_pose_conversion import qpos_to_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand_model_type', default=HandModelType.SHADOW_HAND, type=HandModelType.from_string, choices=list(HandModelType))
    parser.add_argument('--object_code', type=str, default='sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='../data/dataset')
    args = parser.parse_args()

    device = 'cpu'

    joint_names = handmodeltype_to_joint_names[args.hand_model_type]

    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    qpos = data_dict['qpos']
    hand_pose = qpos_to_pose(qpos=qpos, joint_names=joint_names).to(device)
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        hand_pose_st = qpos_to_pose(qpos=qpos_st, joint_names=joint_names).to(device)

    # hand model
    hand_model = HandModel(
        hand_model_type=args.hand_model_type,
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
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_contact_points=False)
    else:
        hand_st_plotly = []
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    if 'energy' in data_dict:
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_spen = round(data_dict['E_spen'], 5)
        E_joints = round(data_dict['E_joints'], 5)
        result = f'Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}'
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    fig.update_layout(scene_aspectmode='data')
    fig.show()
