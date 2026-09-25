"""Deterministic world specifications; screen-side labels belong to annotations."""
import random

COLORS = ("red", "blue", "green", "yellow")
TYPES = ("tower", "arch", "stairs", "pillar")


def blocks(kind, origin):
    if kind == "tower":
        offsets = [(x, y, z) for x in range(2) for y in range(4) for z in range(2)]
    elif kind == "arch":
        offsets = [(x, y, 0) for x in range(4) for y in range(4) if x in (0, 3) or y == 3]
    elif kind == "stairs":
        offsets = [(x, y, z) for x in range(4) for y in range(x + 1) for z in range(2)]
    elif kind == "pillar":
        offsets = [(0, y, 0) for y in range(4)]
    else:
        raise ValueError(kind)
    return [[a + b for a, b in zip(origin, p)] for p in offsets]


def family(index, seed, split):
    rng = random.Random(seed + index)
    colors = list(COLORS)
    kinds = list(TYPES)
    # Recipient palette/type assignment is independent of placement.
    recipient_colors = rng.sample(colors[:2], 2)
    recipient_types = rng.sample(kinds[:2], 2)
    donor_colors = rng.sample(colors[2:], 2)
    donor_types = rng.sample(kinds[2:], 2)
    fid = f"{split}-{seed:08d}-{index:06d}"
    contexts = {}
    for name, cs, ts in [("recipient", recipient_colors, recipient_types), ("disjoint", donor_colors, donor_types)]:
        objects = []
        for j in range(2):
            xyz = blocks(ts[j], [-6 + 10*j, 4, 12])
            bounds = [[min(p[k] for p in xyz) for k in range(3)], [max(p[k] for p in xyz)+1 for k in range(3)]]
            objects.append({"object_id": f"{fid}-{name}-{j}", "color": cs[j], "type": ts[j], "blocks": xyz, "bounds": bounds})
        contexts[name] = {"objects": objects, "camera": {"position": [0, 6, 0], "yaw": 0, "pitch": 0}, "arena": {"seed": seed+index, "time": 6000, "weather": "clear"}}
    return {"schema_version": "1.0", "family_id": fid, "split": split, "seed": seed+index, "target_side": ("left", "right")[index % 2], "contexts": contexts}


def prompt(side, task="pair"):
    if side not in ("left", "right") or task not in ("pair", "color", "type"):
        raise ValueError("Unsupported prompt")
    attribute = {"pair": "color and type", "color": "color", "type": "type"}[task]
    return f"What {attribute} is the {side}most structure? Answer with {attribute} only."
