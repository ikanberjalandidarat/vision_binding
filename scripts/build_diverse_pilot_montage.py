"""Combine the eight real Minecraft route GIFs in a pilot review for slides."""

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


def build(review, output):
    summary = json.loads((review / "summary.json").read_text())
    environments = summary["environments"]
    if len(environments) != 4:
        raise ValueError("Expected the four-environment v2 pilot review")
    clips = []
    try:
        for env in environments:
            clips.append(
                [Image.open(review / f"{env['id']}-route-{route}.gif") for route in range(2)]
            )
        durations = {clip.info.get("duration") for pair in clips for clip in pair}
        if durations != {100}:
            raise ValueError(f"Expected 100 ms GIF frames; found {durations}")
        n_frames = max(clip.n_frames for pair in clips for clip in pair)
        names = {
            "indoor_room": "INDOOR ROOM",
            "outdoor_maze": "OUTDOOR MAZE",
            "forest_and_pond": "FOREST + POND",
            "village": "VILLAGE",
        }
        frames = []
        for t in range(n_frames):
            canvas = Image.new("RGB", (1600, 700), "#101820")
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 16), "Minecraft environments  |  diverse dataset v2 pilot", font=font(29), fill="white")
            draw.text((1220, 25), f"sim {t * 0.4:5.1f} s  ·  4×", font=font(19), fill="#c8d9e8")
            for col, (env, pair) in enumerate(zip(environments, clips)):
                x = col * 400
                draw.rectangle((x + 2, 67, x + 398, 655), outline="#334653", width=2)
                draw.text((x + 12, 78), names.get(env["name"], env["name"].upper()), font=font(22), fill="#e7f0f4")
                for route, clip in enumerate(pair):
                    clip.seek(min(t, clip.n_frames - 1))
                    # Keep the real first-person RGB region, excluding the earlier review overlay.
                    rgb = clip.convert("RGB").crop((0, 35, 448, 315))
                    rgb = rgb.resize((384, 240), Image.Resampling.LANCZOS)
                    y = 116 + route * 270
                    canvas.paste(rgb, (x + 8, y))
                    draw.text((x + 12, y + 244), f"Route {route + 1}", font=font(17), fill="#c8d9e8")
            draw.text((24, 668), "Real Minecraft recordings · all captured frames · shorter routes hold their last view", font=font(17), fill="#9eb2c1")
            frames.append(canvas.quantize(colors=96, method=Image.Quantize.FASTOCTREE))
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    finally:
        for pair in clips:
            for clip in pair:
                clip.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.review, args.output)
