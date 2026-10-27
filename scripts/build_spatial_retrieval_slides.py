"""Build an evidence-labelled PDF explanation of the within-world V-JEPA pilot."""

import json
import math
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/oscar/spatial-video-withinworld-7983593/report.json"
META = ROOT / "runs/oscar/diverse-dataset-v2-metadata/env-02/sequences.json"
REVIEW = ROOT / "runs/oscar/diverse-dataset-v2-review"
OUTPUT = ROOT / "output/pdf/spatial-video/within-world-retrieval-explained.pdf"
W, H = 960, 540
INK = colors.HexColor("#14283a")
BLUE = colors.HexColor("#2368a0")
PINK = colors.HexColor("#c63672")
GREEN = colors.HexColor("#17856c")
GRAY = colors.HexColor("#526678")
LIGHT = colors.HexColor("#e8eef3")


def fonts():
    base = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("ArialSlide", str(base / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("ArialSlideBold", str(base / "Arial Bold.ttf")))


def txt(c, x, y, text, size=17, color=INK, bold=False):
    c.setFillColor(color)
    c.setFont("ArialSlideBold" if bold else "ArialSlide", size)
    c.drawString(x, y, text)


def lines(c, x, y, items, size=17, leading=25, color=INK):
    for i, line in enumerate(items):
        txt(c, x, y - i * leading, line, size, color)


def base(c, number, title, subtitle=None):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(BLUE)
    c.rect(0, H - 7, W, 7, stroke=0, fill=1)
    txt(c, 42, 482, title, 29, INK, True)
    if subtitle:
        txt(c, 43, 451, subtitle, 16, GRAY)
    c.setStrokeColor(LIGHT)
    c.line(42, 38, W - 42, 38)
    txt(c, 42, 19, "Frozen V-JEPA 2 / Minecraft / job 7983593", 10, GRAY)
    txt(c, W - 60, 19, str(number), 10, GRAY)


def still(route, tick):
    im = Image.open(REVIEW / f"env-02-route-{route}.gif")
    im.seek(tick // 8)
    return ImageReader(im.convert("RGB"))


def media(c, image, x, y, w, h):
    c.drawImage(image, x, y, width=w, height=h, preserveAspectRatio=False, mask="auto")


def endpoint(route, tick):
    entries = json.loads(META.read_text())
    return next(f["pose"] for f in entries[route]["frames"] if f["elapsed_ticks"] == tick)


def chart_bar(c, label, value, y, color):
    x0, scale = 310, 38
    txt(c, 47, y + 7, label, 18, INK, label == "Video features")
    c.setFillColor(color)
    c.rect(x0, y, value * scale, 27, stroke=0, fill=1)
    txt(c, x0 + value * scale + 12, y + 6, f"{value:.2f}", 18, INK, True)


def make():
    fonts()
    report = json.loads(REPORT.read_text())
    assert report["state"] == "complete" and len(report["results"]) == 48
    rows = report["results"]
    by = {(r["world"], r["representation"]): r for r in rows}
    import statistics
    mean = lambda field, rep: statistics.mean(by[(f"env-{i:02d}", rep)][field] for i in range(8))
    numbers = {
        "oracle": mean("mean_oracle_distance", "video_layer3"),
        "random": mean("mean_random_key_distance", "video_layer3"),
        "video": mean("mean_retrieved_distance", "video_layer3"),
        "repeat": mean("mean_retrieved_distance", "repeat_last_frame_layer3"),
    }
    assert all(abs(numbers[k] - v) < .01 for k, v in
               {"oracle": 2.752, "random": 13.132, "video": 9.575, "repeat": 11.023}.items())

    # An actual route-1 endpoint and its coordinate-nearest route-0 endpoint.
    # The saved report has no per-query feature-selected key, so this is an
    # oracle example, explicitly distinguished from the model's mean error.
    query, key = endpoint(1, 1472), endpoint(0, 832)
    xyz = lambda p: (p["x"], p["y"], p["z"])
    d = math.dist(xyz(query), xyz(key))
    h = abs((query["yaw"] - key["yaw"] + 180) % 360 - 180)
    assert 1.5 < d < 1.7 and 100 < h < 120

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("Within-world place retrieval in Minecraft - explained")

    base(c, 1, "Can the video model recognize a revisited place?",
         "Actual Minecraft views from two routes through the same indoor maze")
    media(c, still(1, 1472), 42, 176, 425, 232)
    media(c, still(0, 832), 493, 176, 425, 232)
    txt(c, 42, 146, "Query: route 1, tick 1472", 19, PINK, True)
    txt(c, 493, 146, "Closest recorded place: route 0, tick 832", 18, BLUE, True)
    txt(c, 42, 103, f"These two recorded poses are {d:.2f} blocks apart, but face {h:.0f} degrees apart.", 19)
    txt(c, 42, 69, "The right image is the coordinate oracle, NOT a claimed model match.", 16, GRAY)
    c.showPage()

    base(c, 2, "Where are the two views in the world?",
         "A real query and its nearest available route-0 reference")
    map_image = ImageReader(str(REVIEW / "env-02-topdown.png"))
    mx, my, mw, mh = 60, 70, 320, 340
    media(c, map_image, mx, my, mw, mh)
    def point(p):
        px = 70 + (p["x"] + 16) * 20
        py = 90 + (17 - p["z"]) * 20
        return mx + px * mw / 800, my + (850 - py) * mh / 850
    qx, qy = point(query)
    kx, ky = point(key)
    c.setLineWidth(2)
    c.setStrokeColor(INK)
    c.line(qx, qy, kx, ky)
    for x, y, color, label, offx, offy in [
        (qx, qy, PINK, "Q", -38, 32), (kx, ky, BLUE, "K", 30, -36)]:
        c.setFillColor(color)
        c.circle(x, y, 7, stroke=0, fill=1)
        c.setStrokeColor(color)
        c.line(x, y, x + offx * .65, y + offy * .65)
        txt(c, x + offx, y + offy, label, 16, color, True)
    txt(c, 420, 376, "Q: where route 1 really was", 20, PINK, True)
    txt(c, 420, 339, f"X {query['x']:.2f}    Y {query['y']:.0f}    Z {query['z']:.2f}", 19)
    txt(c, 420, 271, "K: nearest route-0 recording", 20, BLUE, True)
    txt(c, 420, 234, f"X {key['x']:.2f}    Y {key['y']:.0f}    Z {key['z']:.2f}", 19)
    txt(c, 420, 159, f"Actual Q-to-K distance: {d:.2f} blocks", 20, INK, True)
    lines(c, 420, 119, ["Minecraft coordinates score the match.",
                         "They were not inputs to V-JEPA."], 15, 22, GRAY)
    txt(c, 60, 51, "Top-down: authored build + recorded paths, not an aerial image.", 11, GRAY)
    c.showPage()

    base(c, 3, "What the retrieval test found",
         "Mean 3D distance between each route-1 query and its chosen route-0 clip")
    chart_bar(c, "Random route-0 clip", numbers["random"], 351, GRAY)
    chart_bar(c, "Repeated final frame", numbers["repeat"], 285, colors.HexColor("#9aaaba"))
    chart_bar(c, "Video features", numbers["video"], 219, PINK)
    chart_bar(c, "Nearest available (oracle)", numbers["oracle"], 153, BLUE)
    txt(c, 47, 99, "Lower is better. Each world contributes equally to the mean.", 18, INK)
    txt(c, 47, 66, "The model beats random retrieval, but is far from the oracle.", 18, INK, True)
    c.showPage()

    base(c, 4, "The different-heading test is still small",
         "Query and reference positions within 2 blocks, with at least 90 degrees of heading change")
    labels = [("Indoor maze", "env-02", "3 / 7"),
              ("Forest", "env-04", "0 / 4"),
              ("Mine", "env-05", "5 / 10")]
    for i, (name, world, ratio) in enumerate(labels):
        y = 366 - i * 76
        txt(c, 52, y, name, 24, INK, True)
        txt(c, 465, y, ratio, 24, PINK, True)
        txt(c, 579, y, "queries matched within 2 blocks", 17, GRAY)
    c.setStrokeColor(LIGHT)
    c.line(52, 130, W - 52, 130)
    lines(c, 52, 103, ["These are correlated clips from only three worlds with enough qualifying revisits.",
                        "The report does not establish view-invariant place cells or a cognitive map."], 16, 25)
    c.showPage()

    base(c, 5, "Why test sliding windows next?",
         "More evaluation endpoints from the SAME recordings; frozen model and clip length unchanged")
    txt(c, 53, 379, "Recorded frames", 17, INK, True)
    for i in range(16):
        x = 254 + i * 39
        c.setFillColor(LIGHT)
        c.rect(x, 363, 27, 32, stroke=0, fill=1)
        txt(c, x + (4 if i >= 9 else 8), 372, str(i + 1), 11)
    txt(c, 53, 299, "Old: one endpoint per 16-frame block", 18)
    txt(c, 53, 258, "Proposed: move the 16-frame window by one recorded frame", 18)
    txt(c, 53, 204, "This can recover missed revisits. Adjacent windows share 15 of 16 frames.", 17, INK, True)
    txt(c, 53, 174, "Report independent locations/route pairs, not window count as sample size.", 17)
    txt(c, 53, 124, "Research precedent: short video sequences for place recognition; sliding temporal", 14, GRAY)
    txt(c, 53, 105, "windows appear in the literature. A one-frame hop is our evaluation choice, not", 14, GRAY)
    txt(c, 53, 86, "a published optimum or evidence of biological cells.", 14, GRAY)
    txt(c, 53, 56, "Garg & Milford, SeqNet (2021); Zhao et al., RA-L (2024). Sources on final page.", 12, GRAY)
    c.showPage()

    base(c, 6, "Evidence and next measurement")
    lines(c, 53, 402, ["The 9.57-block result is a world-average error, not one example match.",
                        "This JSON did not save the model-selected route-0 clip for each query.",
                        "The two real views on slides 1-2 show an oracle pair, never a model prediction."], 18, 31)
    txt(c, 53, 271, "For a real predicted-vs-actual map figure, the next report must save:", 19, INK, True)
    lines(c, 70, 232, ["query clip ID and XYZ; selected reference clip ID and XYZ; similarity;",
                        "oracle reference ID; heading difference; 3D distance for each query."], 17, 27)
    txt(c, 53, 142, "Research", 16, INK, True)
    txt(c, 53, 119, "Garg & Milford (2021), SeqNet: arxiv.org/abs/2102.11603", 13, GRAY)
    txt(c, 53, 97, "Zhao et al. (2024), Sequence Descriptor: arxiv.org/abs/2305.11467", 13, GRAY)
    txt(c, 53, 75, "V-JEPA 2 official 16-frame evaluation config: github.com/facebookresearch/vjepa2", 13, GRAY)
    c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    make()
