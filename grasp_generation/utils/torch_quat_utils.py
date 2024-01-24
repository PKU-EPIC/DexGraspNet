import torch
import torch.nn.functional as F


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))


def pose_to_T(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert a pose (position and quaternion) to a 4x4 transformation matrix.

    Args:
    pose (Tensor): A tensor of shape (N, 7) where N is the number of poses.

    Returns:
    Tensor: A tensor of shape (N, 4, 4) representing the transformation matrices.
    """
    N = pose.shape[0]
    assert_equals(pose.shape, (N, 7))
    position = pose[:, :3]
    quaternion_xyzw = pose[:, 3:]
    quaternion_wxyz = torch.cat(
        [quaternion_xyzw[:, -1:], quaternion_xyzw[:, :-1]], dim=-1
    )

    rotation_matrix = quaternion_to_matrix(quaternion_wxyz)

    transformation_matrix = torch.zeros((pose.shape[0], 4, 4), device=pose.device)
    transformation_matrix[:, :3, :3] = rotation_matrix
    transformation_matrix[:, :3, 3] = position
    transformation_matrix[:, 3, 3] = 1.0

    assert_equals(transformation_matrix.shape, (N, 4, 4))

    return transformation_matrix


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    subgradient is zero where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def T_to_pose(T: torch.Tensor) -> torch.Tensor:
    """
    Convert a 4x4 transformation matrix to a pose (position and quaternion).

    Args:
    T (Tensor): A tensor of shape (N, 4, 4) where N is the number of transformation matrices.

    Returns:
    Tensor: A tensor of shape (N, 7) representing the poses.
    """
    N = T.shape[0]
    assert_equals(T.shape, (N, 4, 4))
    pose = torch.zeros((N, 7), device=T.device)
    pose[:, :3] = T[:, :3, 3]
    quaternion_wxyz = matrix_to_quaternion(T[:, :3, :3])
    quaternion_xyzw = torch.cat(
        [quaternion_wxyz[:, 1:], quaternion_wxyz[:, :1]], dim=-1
    )
    pose[:, 3:] = quaternion_xyzw
    assert_equals(pose.shape, (N, 7))
    return pose
