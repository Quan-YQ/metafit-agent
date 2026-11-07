"""生成一张人工绘制的骨架示意图；不依赖第三方库。"""

from pathlib import Path
import struct
import zlib


WIDTH, HEIGHT = 600, 340
OUTPUT = Path(__file__).with_name("vision_sample.png")


def make_image() -> bytes:
    pixels = bytearray([255, 255, 255] * WIDTH * HEIGHT)

    def dot(x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
        for yy in range(max(0, y - radius), min(HEIGHT, y + radius + 1)):
            for xx in range(max(0, x - radius), min(WIDTH, x + radius + 1)):
                if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:
                    offset = (yy * WIDTH + xx) * 3
                    pixels[offset:offset + 3] = bytes(color)

    def line(a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int], width: int = 4) -> None:
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
        for step in range(steps + 1):
            dot(round(a[0] + (b[0] - a[0]) * step / steps),
                round(a[1] + (b[1] - a[1]) * step / steps), width, color)

    blue, joint, ground = (37, 73, 130), (223, 95, 62), (190, 198, 206)
    line((30, 285), (570, 285), ground, 1)
    # 人工示意：左侧较直立，右侧髋膝弯曲。不是人体测量或真实动作。
    poses = [
        [(150, 83), (150, 115), (150, 176), (150, 224), (150, 276), (175, 276), (126, 151)],
        [(375, 96), (382, 126), (411, 190), (356, 211), (381, 276), (405, 276), (341, 158)],
    ]
    for head, shoulder, hip, knee, ankle, foot, hand in poses:
        dot(*head, 15, blue)
        for a, b in [(head, shoulder), (shoulder, hip), (hip, knee), (knee, ankle),
                     (ankle, foot), (shoulder, hand)]:
            line(a, b, blue)
        for point in (shoulder, hip, knee, ankle):
            dot(*point, 6, joint)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    scanlines = b"".join(b"\0" + pixels[y * WIDTH * 3:(y + 1) * WIDTH * 3] for y in range(HEIGHT))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    OUTPUT.write_bytes(make_image())
    print(f"已生成模拟骨架图：{OUTPUT}")
