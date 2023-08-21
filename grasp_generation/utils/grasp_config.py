import numpy as np
import pathlib
import pytorch_kinematics as pk
import pypose as pp
import torch

import nerf_grasping

# from nerf_grasping import grasp_utils
import grasp_utils2

from typing import List

ALLEGRO_URDF_PATH = list(
    pathlib.Path(nerf_grasping.get_package_root()).rglob(
        "*allegro_hand_description_right.urdf"
    )
)[0]

Z_AXIS = torch.tensor([0, 0, 1], dtype=torch.float32)

FINGERTIP_LINK_NAMES = [
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
]


def load_allegro(allegro_path: pathlib.Path = ALLEGRO_URDF_PATH) -> pk.chain.Chain:
    return pk.build_chain_from_urdf(open(allegro_path).read())


class AllegroHandConfig(torch.nn.Module):
    """
    A container specifying a batch of configurations for an Allegro hand, i.e., the
    wrist pose and the joint configurations.
    """

    def __init__(
        self,
        batch_size: int = 1,  # TODO(pculbert): refactor for arbitrary batch sizes.
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
    ):
        super().__init__()
        self.chain = chain
        self.wrist_pose = pp.Parameter(
            pp.randn_SE3(batch_size), requires_grad=requires_grad
        )
        self.joint_angles = torch.nn.Parameter(
            torch.zeros(batch_size, 16), requires_grad=requires_grad
        )
        self.batch_size = batch_size

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.chain.to(device=torch.device(device), dtype=dtype)

    def set_wrist_pose(self, wrist_pose: pp.LieTensor):
        assert (
            wrist_pose.shape == self.wrist_pose.shape
        ), f"New wrist pose, shape {wrist_pose.shape} does not match current wrist pose shape {self.wrist_pose.shape}"
        self.wrist_pose.data = wrist_pose.data.clone()

    def set_joint_angles(self, joint_angles: torch.Tensor):
        assert (
            joint_angles.shape == self.joint_angles.shape
        ), f"New hand config, shape {joint_angles.shape}, does not match shape of current hand config, {self.joint_angles.shape}."
        self.joint_angles.data = joint_angles

    def get_fingertip_transforms(self) -> List[pp.LieTensor]:
        # Run batched FK from current hand config.
        link_poses_hand_frame = self.chain.forward_kinematics(self.joint_angles)

        # Pull out fingertip poses + cast to PyPose.
        fingertip_poses = [link_poses_hand_frame[ln] for ln in FINGERTIP_LINK_NAMES]
        fingertip_pyposes = [
            pp.from_matrix(fp.get_matrix(), pp.SE3_type) for fp in fingertip_poses
        ]

        # Apply wrist transformation to get world-frame fingertip poses.
        return torch.stack(
            [self.wrist_pose @ fp for fp in fingertip_pyposes], dim=1
        )  # shape [B, batch_size, 7]


class AllegroGraspConfig(torch.nn.Module):
    """Container defining a batch of grasps -- both pre-grasps
    and grasping directions -- for use in grasp optimization."""

    def __init__(
        self,
        batch_size: int = 1,
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
    ):
        self.batch_size = batch_size
        super().__init__()
        self.hand_config = AllegroHandConfig(batch_size, chain, requires_grad)
        self.grasp_orientations = pp.Parameter(
            pp.identity_SO3(batch_size, grasp_utils2.NUM_FINGERS),
            requires_grad=requires_grad,
        )

    @classmethod
    def from_path(cls, path: pathlib.Path):
        state_dict = torch.load(str(path))
        batch_size = state_dict["hand_config.wrist_pose"].shape[0]
        grasp_config = cls(batch_size)
        grasp_config.load_state_dict(state_dict)
        return grasp_config

    @classmethod
    def randn(
        cls,
        batch_size: int = 1,
        std_orientation: float = 0.1,
        std_wrist_pose: float = 0.1,
        std_joint_angles: float = 0.1,
    ):
        grasp_config = cls(batch_size)

        state_dict = {}
        state_dict["grasp_orientations"] = pp.so3(
            std_orientation
            * torch.randn(
                batch_size,
                grasp_utils2.NUM_FINGERS,
                3,
                device=grasp_config.grasp_orientations.device,
                dtype=grasp_config.grasp_orientations.dtype,
            )
        ).Exp()

        state_dict["hand_config.wrist_pose"] = pp.se3(
            std_wrist_pose
            * torch.randn(
                batch_size,
                6,
                dtype=grasp_config.grasp_orientations.dtype,
                device=grasp_config.grasp_orientations.device,
            )
        ).Exp()

        state_dict["hand_config.joint_angles"] = std_joint_angles * torch.randn(
            batch_size,
            16,
            dtype=grasp_config.grasp_orientations.dtype,
            device=grasp_config.grasp_orientations.device,
        )

        grasp_config.load_state_dict(state_dict)

        return grasp_config

    @classmethod
    def from_grasp_data(cls, grasp_data_path: pathlib.Path, batch_size: int = 1):
        # Load grasp data + instantiate correctly-sized config object.
        grasp_data = np.load(str(grasp_data_path), allow_pickle=True)
        grasp_config = cls(batch_size)

        # Sample (with replacement) random indices into grasp data.
        indices = np.random.choice(np.arange(len(grasp_data)), batch_size)
        grasp_data = grasp_data[indices]

        # Assemble these samples into the data we need for the grasp config.
        grasp_data_tuples = [
            grasp_utils2.get_grasp_config_from_grasp_data(gd) for gd in grasp_data
        ]

        # List of tuples -> tuple of lists.
        grasp_data_list = list(zip(*grasp_data_tuples))

        # Set the grasp config's data.
        state_dict = {}
        state_dict["hand_config.wrist_pose"] = torch.stack(
            grasp_data_list[0], dim=0
        ).to(
            device=grasp_config.grasp_orientations.device,
            dtype=grasp_config.grasp_orientations.dtype,
        )

        state_dict["hand_config.joint_angles"] = torch.stack(
            grasp_data_list[1], dim=0
        ).to(
            device=grasp_config.grasp_orientations.device,
            dtype=grasp_config.grasp_orientations.dtype,
        )

        state_dict["grasp_orientations"] = torch.stack(grasp_data_list[2], dim=0).to(
            device=grasp_config.grasp_orientations.device,
            dtype=grasp_config.grasp_orientations.dtype,
        )

        # Load state dict for module.
        grasp_config.load_state_dict(state_dict)

        return grasp_config

    @property
    def wrist_pose(self) -> pp.LieTensor:
        return self.hand_config.wrist_pose

    @property
    def joint_angles(self) -> torch.Tensor:
        return self.hand_config.joint_angles

    @property
    def fingertip_transforms(self) -> pp.LieTensor:
        """Returns finger-to-world transforms."""
        return self.hand_config.get_fingertip_transforms()

    @property
    def grasp_frame_transforms(self) -> pp.LieTensor:
        """Returns SE(3) transforms for ``grasp frame'', i.e.,
        z-axis pointing along grasp direction."""

        return self.fingertip_transforms @ pp.from_matrix(
            self.grasp_orientations.unsqueeze(1).matrix(), pp.SE3_type
        )

    @property
    def grasp_dirs(self) -> torch.Tensor:  # shape [B, 4, 3].
        return pp.from_matrix(
            self.grasp_frame_transforms.matrix(), pp.SO3_type
        ) @ Z_AXIS.to(
            device=self.grasp_orientations.device, dtype=self.grasp_orientations.dtype
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        )

    def get_qpos(self, i: int) -> dict:
        from grasp_utils2 import DEXGRASPNET_TRANS_NAMES as translation_names, DEXGRASPNET_ROT_NAMES as rot_names, ALLEGRO_JOINT_NAMES as joint_names
        import transforms3d
        joint_angles = self.joint_angles[i].tolist()
        euler = transforms3d.euler.mat2euler(self.wrist_pose[i].rotation().matrix(), axes="sxyz").tolist()
        translation = self.wrist_pose[i].translation().tolist()

        qpos = {}
        qpos.update(dict(zip(joint_names, joint_angles)))
        qpos.update(dict(zip(rot_names, euler)))
        qpos.update(dict(zip(translation_names, translation)))
        return qpos


def dry_run():
    # Some semi-hardcoded unit tests to make sure the code runs.

    batch_size = 32
    grasp_config = AllegroGraspConfig(batch_size=batch_size)
    print(f"grasp_config = {grasp_config}")


if __name__ == "__main__":
    dry_run()
