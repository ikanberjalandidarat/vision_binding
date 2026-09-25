"""Flat diagnostic drawings, explicitly NOT a Minecraft simulator."""
from PIL import Image, ImageDraw
from .base import Capture, WorldBackend

PALETTE = {"red": "#ca3838", "blue": "#3669c9", "green": "#389454", "yellow": "#d3bd35"}


class FixtureBackend(WorldBackend):
    def reset(self, arena_spec):
        self.objects = []
        self.pose = {}

    def build(self, objects, distractors):
        if distractors:
            raise ValueError("Fixture has no clutter/occlusion implementation")
        self.objects = objects

    def set_camera(self, pose):
        if pose != {"position": [0, 6, 0], "yaw": 0, "pitch": 0}:
            raise ValueError("Fixture does not implement camera projection")
        self.pose = pose

    def capture(self):
        im = Image.new("RGB", (448, 280), "#dddddd")
        draw = ImageDraw.Draw(im)
        rois = {}
        for j, obj in enumerate(self.objects):
            x, y, unit = 60 + j*224, 196, 20
            cells = set((p[0]-obj['bounds'][0][0], p[1]-obj['bounds'][0][1]) for p in obj['blocks'])
            for dx, dy in cells:
                draw.rectangle((x+dx*unit, y-(dy+1)*unit, x+(dx+1)*unit-1, y-dy*unit-1), fill=PALETTE[obj['color']], outline="#333333")
            rois[obj['object_id']] = [x, y-80, x+(max(p[0] for p in cells)+1)*unit, y]
        return Capture(im, self.pose, {"backend": "fixture", "is_minecraft": False, "roi_method": "fixture_drawing_bounds", "rois": rois, "visibility": "fully_visible_fixture", "projection": None})

    def close(self):
        pass
