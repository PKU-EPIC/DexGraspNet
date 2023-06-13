import torch


class Quaternion:
    @classmethod
    def Identity(cls):
        return Quaternion([1, 0, 0, 0])

    @classmethod
    def fromTangentSpace(cls, w):
        angle = torch.norm(w)
        return Quaternion.fromAxisAngle(w, angle)

    @classmethod
    def fromNudge(cls, nudge):
        w = torch.sqrt(1 - torch.norm(nudge) ** 2)
        return Quaternion([w, nudge[0], nudge[1], nudge[2]])

    @classmethod
    def fromAxisAngle(cls, axis, angle):
        if angle == 0:
            return Quaternion.Identity()
        if type(angle) != torch.Tensor:
            angle = torch.Tensor([angle])
        axis = axis / torch.norm(axis)
        c = torch.cos(angle / 2)
        s = torch.sin(angle / 2)
        return Quaternion([c, s * axis[0], s * axis[1], s * axis[2]])

    @classmethod
    def fromWLast(cls, array):
        x, y, z, w = array
        return Quaternion([w, x, y, z])

    @classmethod
    def fromMatrix(cls, matrix):
        # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        m = matrix.T
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                quat = Quaternion.fromWLast(
                    [t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2], m[1, 2] - m[2, 1]]
                )
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                quat = Quaternion.fromWLast(
                    [m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1], m[2, 0] - m[0, 2]]
                )
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                quat = Quaternion.fromWLast(
                    [m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t, m[0, 1] - m[1, 0]]
                )
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                quat = Quaternion.fromWLast(
                    [m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0], t]
                )

        quat.q *= 0.5 / torch.sqrt(t)
        return quat

    def __init__(self, array):
        self.q = torch.Tensor(array)

    def __matmul__(self, other):
        w0, x0, y0, z0 = other.q
        w1, x1, y1, z1 = self.q
        return Quaternion(
            [
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            ]
        )

    @property
    def T(self):
        w, x, y, z = self.q
        return Quaternion([w, -x, -y, -z])

    def __repr__(self):
        w, x, y, z = self.q
        return f"Quaternion(x={x:.5},y={y:.5},z={z:.5},w={w:.5})"

    def to_tangent_space(self):
        axis = self.q[1:] / torch.norm(self.q[1:])
        angle = 2 * torch.acos(self.q[0])
        return angle * axis

    def get_matrix(self):
        w = self.q[0]
        v = self.q[1:]

        return (
            (w**2 - torch.norm(v) ** 2) * torch.eye(3)
            + 2 * torch.outer(v, v)
            + 2 * w * self.skew_matrix(v)
        )

    @staticmethod
    def skew_matrix(v):
        # also the cross product matrix
        x, y, z = v
        return torch.Tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def normalize(self):
        return Quaternion(self.q / torch.norm(self.q))

    def rotate(self, vector):
        assert len(vector) == 3
        tmp = Quaternion([0, vector[0], vector[1], vector[2]])
        out = self @ tmp @ self.T
        return out.q[1:]
