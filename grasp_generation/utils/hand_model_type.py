from enum import Enum, auto
import torch
import transforms3d
import numpy as np

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]


class AutoName(Enum):
    # https://docs.python.org/3.9/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


class HandModelType(AutoName):
    ALLEGRO_HAND = auto()
    SHADOW_HAND = auto()


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

_allegro_hand_root_hand_file = (
    "allegro_hand_description",
    "allegro_hand_description_right.urdf",
)
_shadow_hand_root_hand_file = ("open_ai_assets", "hand/shadow_hand.xml")

_allegro_hand_root_hand_file_with_virtual_joints = (
    "allegro_hand_description",
    "allegro_hand_description_right_with_virtual_joints.urdf",
)
_shadow_hand_root_hand_file_with_virtual_joints = ("open_ai_assets", "NOT IMPLEMENTED")

_allegro_hand_allowed_contact_link_names = [
    "link_3.0",
    "link_7.0",
    "link_11.0",
    "link_15.0",
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
    # Allow some contact with parts before
    "link_2.0",
    "link_6.0",
    "link_10.0",
    "link_14.0",
    # Allow some contact with parts before
    "link_1.0",
    "link_5.0",
    "link_9.0",
    "link_13.0",
]

_shadow_hand_allowed_contact_link_names = [
    "robot0:ffdistal_child",
    "robot0:mfdistal_child",
    "robot0:rfdistal_child",
    "robot0:lfdistal_child",
    "robot0:thdistal_child",
]

_allegro_hand_finger_keywords = ["3.0", "7.0", "11.0", "15.0"]

_shadow_hand_finger_keywords = ["ff", "mf", "rf", "lf", "th"]

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
handmodeltype_to_hand_root_hand_file_with_virtual_joints = {
    HandModelType.ALLEGRO_HAND: _allegro_hand_root_hand_file_with_virtual_joints,
    HandModelType.SHADOW_HAND: _shadow_hand_root_hand_file_with_virtual_joints,
}

handmodeltype_to_fingerkeywords = {
    HandModelType.ALLEGRO_HAND: _allegro_hand_finger_keywords,
    HandModelType.SHADOW_HAND: _shadow_hand_finger_keywords,
}

# HACK: This is a list of allowed contact link names for each hand model type for precision grasps
handmodeltype_to_allowedcontactlinknames = {
    HandModelType.ALLEGRO_HAND: _allegro_hand_allowed_contact_link_names,
    HandModelType.SHADOW_HAND: _shadow_hand_allowed_contact_link_names,
}
