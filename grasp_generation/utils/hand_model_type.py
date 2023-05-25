from enum import Enum, auto
import torch
import transforms3d
import numpy as np

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]


class HandModelType(Enum):
    ALLEGRO_HAND = auto()
    SHADOW_HAND = auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return HandModelType[s]
        except KeyError:
            raise ValueError()


_allegro_joint_names = [
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
]

_shadow_joint_names = [
    "robot0:FFJ3",
    "robot0:FFJ2",
    "robot0:FFJ1",
    "robot0:FFJ0",
    "robot0:MFJ3",
    "robot0:MFJ2",
    "robot0:MFJ1",
    "robot0:MFJ0",
    "robot0:RFJ3",
    "robot0:RFJ2",
    "robot0:RFJ1",
    "robot0:RFJ0",
    "robot0:LFJ4",
    "robot0:LFJ3",
    "robot0:LFJ2",
    "robot0:LFJ1",
    "robot0:LFJ0",
    "robot0:THJ4",
    "robot0:THJ3",
    "robot0:THJ2",
    "robot0:THJ1",
    "robot0:THJ0",
]

_allegro_rotation_hand = torch.tensor(
    transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes="rzyz"),
    dtype=torch.float,
)
_shadow_rotation_hand = torch.tensor(
    transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes="rzxz"),
    dtype=torch.float,
)

_allegro_joint_angles_mu = torch.tensor(
    [
        0,
        0.5,
        0,
        0,
        0,
        0.5,
        0,
        0,
        0,
        0.5,
        0,
        0,
        1.4,
        0,
        0,
        0,
    ],
    dtype=torch.float,
)
_shadow_joint_angles_mu = torch.tensor(
    [
        0.1,
        0,
        0.6,
        0,
        0,
        0,
        0.6,
        0,
        -0.1,
        0,
        0.6,
        0,
        0,
        -0.2,
        0,
        0.6,
        0,
        0,
        1.2,
        0,
        -0.2,
        0,
    ],
    dtype=torch.float,
)

_allegro_hand_root_hand_file = ("allegro_hand_description", "allegro_hand_description_right.urdf")
_shadow_hand_root_hand_file = ("open_ai_assets", "hand/shadow_hand.xml")

handmodeltype_to_joint_names = {
    HandModelType.ALLEGRO_HAND: _allegro_joint_names,
    HandModelType.SHADOW_HAND: _shadow_joint_names,
}
handmodeltype_to_rotation_hand = {
    HandModelType.ALLEGRO_HAND: _allegro_rotation_hand,
    HandModelType.SHADOW_HAND: _shadow_rotation_hand,
}
handmodeltype_to_joint_angles_mu = {
    HandModelType.ALLEGRO_HAND: _allegro_joint_angles_mu,
    HandModelType.SHADOW_HAND: _shadow_joint_angles_mu,
}
handmodeltype_to_hand_root_hand_file = {
    HandModelType.ALLEGRO_HAND: _allegro_hand_root_hand_file,
    HandModelType.SHADOW_HAND: _shadow_hand_root_hand_file,
}
