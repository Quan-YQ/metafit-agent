"""AMASS 研究准备：用给定的静态关节计算轴角运动学，不读取模型文件。"""
import math


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _vector(value):
    if len(value) != 3 or any(not math.isfinite(float(v)) for v in value):
        raise ValueError("坐标或轴角必须是三个有限数字")
    return tuple(float(v) for v in value)


def _multiply(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _apply(matrix, vector):
    return tuple(sum(matrix[i][k] * vector[k] for k in range(3)) for i in range(3))


def axis_angle_matrix(axis_angle):
    x, y, z = _vector(axis_angle)
    angle = math.hypot(x, y, z)
    if angle < 1e-12:
        return IDENTITY
    x, y, z = x / angle, y / angle, z / angle
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    return ((c + x*x*t, x*y*t - z*s, x*z*t + y*s),
            (y*x*t + z*s, c + y*y*t, y*z*t - x*s),
            (z*x*t - y*s, z*y*t + x*s, c + z*z*t))


def forward_kinematics(rest_joints, parents, axis_angles, translation=(0, 0, 0)):
    """输入模型算出的绝对静态关节、父关节索引及局部弧度轴角；输出同坐标系关节。

    静态关节必须来自对应身体模型；该函数不加载 AMASS 或计算蒙皮、DMPL。
    """
    n = len(rest_joints)
    if not n or len(parents) != n or len(axis_angles) != n or parents[0] not in (-1, None):
        raise ValueError("关节树维度或根节点无效")
    rest = [_vector(point) for point in rest_joints]
    angles = [_vector(angle) for angle in axis_angles]
    trans = _vector(translation)
    positions, rotations = [], []
    for i in range(n):
        parent = parents[i]
        if i and (type(parent) is not int or not 0 <= parent < i):
            raise ValueError("父节点必须位于子节点之前")
        local = axis_angle_matrix(angles[i])
        if i == 0:
            positions.append(tuple(rest[0][j] + trans[j] for j in range(3)))
            rotations.append(local)
        else:
            offset = tuple(rest[i][j] - rest[parent][j] for j in range(3))
            delta = _apply(rotations[parent], offset)
            positions.append(tuple(positions[parent][j] + delta[j] for j in range(3)))
            rotations.append(_multiply(rotations[parent], local))
    return positions
