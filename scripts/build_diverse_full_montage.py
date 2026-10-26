"""Make a compact, slide-shaped GIF of the eight v2 Minecraft environments."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build(review, output, colors=32):
    summary = json.loads((review / "summary.json").read_text())
    environments = summary["environments"]
    if [env["id"] for env in environments] != [f"env-{i:02d}" for i in range(8)]:
        raise ValueError("Expected the complete eight-environment v2 review")
    names = [
        "INDOOR ROOM", "OUTDOOR AREA", "INDOOR MAZE", "OUTDOOR MAZE",
        "NATURE", "MINE", "HILLS", "VILLAGE",
    ]
    clips = [Image.open(review / f"env-{i:02d}-route-0.gif") for i in range(8)]
    try:
        if {clip.info.get("duration") for clip in clips} != {100}:
            raise ValueError("Expected 100 ms source frames")
        count = max(clip.n_frames for clip in clips)
        frames = []
        for tick in range(count):
            canvas = Image.new("RGB", (1600, 664), "#101820")
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 14), "Eight Minecraft environments | dataset v2", font=font(29), fill="white")
            draw.text((1280, 21), f"sim {tick * 0.4:5.1f} s · 4×", font=font(18), fill="#c8d9e8")
            for i, clip in enumerate(clips):
                clip.seek(min(tick, clip.n_frames - 1))
                # The source GIF includes a review overlay; keep its real RGB viewport.
                rgb = clip.convert("RGB").crop((0, 35, 448, 315))
                rgb = rgb.resize((384, 240), Image.Resampling.LANCZOS)
                x = (i % 4) * 400 + 8
                y = (i // 4) * 286 + 96
                canvas.paste(rgb, (x, y))
                draw.text((x + 3, y - 29), f"{i + 1}. {names[i]}", font=font(20), fill="#e7f0f4")
            draw.text((20, 642), "Real Minecraft recordings · route 1 per environment · all captured frames", font=font(16), fill="#9eb2c1")
            frames.append(canvas.quantize(colors=colors, method=Image.Quantize.FASTOCTREE))
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    finally:
        for clip in clips:
            clip.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colors", type=int, default=32)
    args = parser.parse_args()
    build(args.review, args.output, args.colors)
