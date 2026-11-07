"""将本项目中选取的 6 帧 CMU BVH 动作绘制成两视角骨架图。"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
import zlib


SAMPLE = Path(__file__).with_name("cmu_22_14_sample.bvh")
WALK_SAMPLE = Path(__file__).with_name("cmu_02_01_sample.bvh")
ROUND_SAMPLES = (
    Path(__file__).with_name("cmu_23_14_sample.bvh"),
    Path(__file__).with_name("cmu_02_02_sample.bvh"),
    Path(__file__).with_name("cmu_111_28_sample.bvh"),
)
OUTPUT = Path(__file__).with_name("cmu_22_14_skeleton.png")
WIDTH, HEIGHT = 1200, 460


@dataclass
class Joint:
    name: str
    parent: int | None
    offset: tuple[float, float, float]
    channels: list[str]


def load_bvh(path: Path = SAMPLE) -> tuple[list[Joint], list[list[float]]]:
    return parse_bvh(path.read_text(encoding="utf-8"))


def parse_bvh(text: str) -> tuple[list[Joint], list[list[float]]]:
    hierarchy, motion = text.split("MOTION", 1)
    tokens = iter(re.findall(r"\{|\}|[^\s{}]+", hierarchy))

    def expect(value: str) -> None:
        if next(tokens) != value:
            raise ValueError("BVH 骨架结构异常")

    if next(tokens) != "HIERARCHY":
        raise ValueError("缺少 BVH 骨架")
    joints: list[Joint] = []

    def parse_joint(kind: str, parent: int | None) -> None:
        name = next(tokens)
        if kind == "End":
            if name != "Site":
                raise ValueError("无效的末端关节")
            name = "EndSite"
        expect("{")
        expect("OFFSET")
        offset = tuple(float(next(tokens)) for _ in range(3))
        index = len(joints)
        joints.append(Joint(name, parent, offset, []))
        for token in tokens:
            if token == "CHANNELS":
                count = int(next(tokens))
                joints[index].channels = [next(tokens) for _ in range(count)]
            elif token in ("JOINT", "End"):
                parse_joint(token, index)
            elif token == "}":
                return
            else:
                raise ValueError("不支持的 BVH 骨架字段")
        raise ValueError("BVH 骨架未闭合")

    expect("ROOT")
    parse_joint("ROOT", None)
    lines = [line.strip() for line in motion.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith("Frames:") or not lines[1].startswith("Frame Time:"):
        raise ValueError("BVH 动作信息异常")
    frame_count = int(lines[0].split(":", 1)[1])
    frames = [[float(v) for v in line.split()] for line in lines[2:]]
    channel_count = sum(len(j.channels) for j in joints)
    if len(frames) != frame_count or any(len(frame) != channel_count for frame in frames):
        raise ValueError("BVH 帧数或通道数不匹配")
    return joints, frames


def matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def transform(matrix, vector):
    return tuple(sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3))


def rotation(axis: str, degrees: float):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "X":
        return ((1, 0, 0), (0, c, -s), (0, s, c))
    if axis == "Y":
        return ((c, 0, s), (0, 1, 0), (-s, 0, c))
    if axis == "Z":
        return ((c, -s, 0), (s, c, 0), (0, 0, 1))
    raise ValueError("不支持的旋转轴")


def world_positions(joints: list[Joint], frame: list[float]) -> list[tuple[float, float, float]]:
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    positions, rotations = [], []
    cursor = 0
    for joint in joints:
        local = identity
        translation = list(joint.offset)
        for channel in joint.channels:
            value = frame[cursor]
            cursor += 1
            if channel.endswith("position") and channel[0] in "XYZ":
                translation["XYZ".index(channel[0])] += value
            elif channel.endswith("rotation") and channel[0] in "XYZ":
                local = matmul(local, rotation(channel[0], value))
            else:
                raise ValueError("不支持的 BVH 通道")
        parent_rotation = rotations[joint.parent] if joint.parent is not None else identity
        parent_position = positions[joint.parent] if joint.parent is not None else (0, 0, 0)
        delta = transform(parent_rotation, translation)
        positions.append(tuple(parent_position[i] + delta[i] for i in range(3)))
        rotations.append(matmul(parent_rotation, local))
    if cursor != len(frame):
        raise ValueError("BVH 通道数不匹配")
    return positions


def render(path: Path = SAMPLE, *, reverse: bool = False) -> bytes:
    joints, frames = load_bvh(path)
    return render_frames(joints, frames, reverse=reverse)


def render_frames(joints: list[Joint], frames: list[list[float]], *, reverse: bool = False, draw_root: bool = False) -> bytes:
    if reverse:
        frames = list(reversed(frames))
    all_positions = [world_positions(joints, frame) for frame in frames]
    if len(all_positions) != 6:
        raise ValueError("实验样本需要 6 帧")
    pixels = bytearray([255, 255, 255] * WIDTH * HEIGHT)

    def dot(x, y, radius, color):
        for yy in range(max(0, y - radius), min(HEIGHT, y + radius + 1)):
            for xx in range(max(0, x - radius), min(WIDTH, x + radius + 1)):
                if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:
                    pos = (yy * WIDTH + xx) * 3
                    pixels[pos:pos + 3] = bytes(color)

    def line(a, b, color, radius=2):
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
        for step in range(steps + 1):
            dot(round(a[0] + (b[0] - a[0]) * step / steps),
                round(a[1] + (b[1] - a[1]) * step / steps), radius, color)

    # 各帧只去掉根节点水平移动；使用统一比例和竖直位置，以保留姿态变化。
    normalized = [[(p[0] - pose[0][0], p[1], p[2] - pose[0][2]) for p in pose] for pose in all_positions]
    low_y = min(p[1] for pose in normalized for p in pose)
    high_y = max(p[1] for pose in normalized for p in pose)
    extent = max(max(abs(p[0]), abs(p[2])) for pose in normalized for p in pose)
    scale = min(155 / max(high_y - low_y, 1), 78 / max(extent, 1))
    blue, orange = (37, 73, 130), (223, 95, 62)
    for row, horizontal_axis in enumerate((0, 2)):
        for col, pose in enumerate(normalized):
            center_x = 100 + col * 200
            bottom = row * 230 + 205
            points = [(round(center_x + p[horizontal_axis] * scale), round(bottom - (p[1] - low_y) * scale)) for p in pose]
            line((col * 200 + 20, bottom), (col * 200 + 180, bottom), (220, 225, 230), 1)
            for index, joint in enumerate(joints):
                if joint.parent is not None and joint.name != "EndSite" and not any(w in joint.name.lower() for w in ("finger", "thumb")):
                    line(points[joint.parent], points[index], blue)
                    dot(*points[index], 3, orange)
            if draw_root:
                dot(*points[0], 4, orange)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    rows = b"".join(b"\0" + pixels[y * WIDTH * 3:(y + 1) * WIDTH * 3] for y in range(HEIGHT))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="保存公开动作样本的骨架图")
    parser.add_argument("--walk", action="store_true", help="使用走路样本")
    args = parser.parse_args()
    output = OUTPUT.with_name("cmu_02_01_skeleton.png") if args.walk else OUTPUT
    output.write_bytes(render(WALK_SAMPLE if args.walk else SAMPLE))
    print(f"已生成骨架图：{output}")
