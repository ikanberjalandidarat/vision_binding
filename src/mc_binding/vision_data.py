"""Fixed-camera visual counterfactuals and pre-merger spatial token indexing."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .capture_pairs import pilot_contexts, color_mask
from .backends.minestudio_smoke import verify_pose
from .io import file_hash


def replication_contexts(index, seed):
    """24 prespecified conditions: 3 palettes × 2 shape pairs × 2 poses × 2 assignments.

    Poses here mean object placement; the observer camera stays fixed.
    These share an arena and are not independent Minecraft worlds.
    """
    from .scenes import blocks
    if not 0 <= index < 24:
        raise ValueError('replication_v1 has exactly 24 configurations')
    assignment = index % 2
    placement = (index // 2) % 2
    shape_pair = (index // 4) % 2
    palette = index // 8
    colors = list((('yellow', 'blue'), ('red', 'yellow'), ('red', 'blue'))[palette])
    kinds = (('stairs', 'pillar'), ('arch', 'tower'))[shape_pair]
    if assignment:
        colors.reverse()
    origins = ((-7, 200, 13), (5, 200, 13)) if placement == 0 else ((-5, 200, 15), (6, 200, 14))
    rows = pilot_contexts(index, seed)[:3]
    objects = []
    for slot, (color, kind, origin) in enumerate(zip(colors, kinds, origins)):
        xyz = blocks(kind, origin)
        objects.append({'object_id': f'f{index:04d}-recipient-{slot}', 'color': color,
                        'type': kind, 'blocks': xyz,
                        'bounds': [[min(p[k] for p in xyz) for k in range(3)],
                                   [max(p[k] for p in xyz)+1 for k in range(3)]]})
    for row, selected in zip(rows, (objects, objects[:1], objects[1:])):
        arena = [s for s in row['commands'] if not s.startswith(('/setblock ', '/tp '))]
        row['objects'] = copy.deepcopy(selected)
        row['commands'] = arena + [f"/setblock {x} {y} {z} minecraft:{o['color']}_wool"
                                   for o in selected for x, y, z in o['blocks']] + ['/tp @p 0.5 200 0.5 0 0']
        row['scene_set'] = 'replication_v1'
        row['factors'] = dict(palette=palette, shape_pair=shape_pair, placement=placement, assignment=assignment)
    return rows


def swap_contexts(index, seed, scene_set="original"):
    if scene_set not in ("original", "depth_spacing_v1", "replication_v1"):
        raise ValueError("Unknown scene set")
    originals = replication_contexts(index, seed) if scene_set == "replication_v1" else pilot_contexts(index, seed)[:3]
    if scene_set == 'depth_spacing_v1':
        # Held-out geometry relative to original z=12 spacing; camera stays fixed.
        dz = 1 + index
        def move(x,y,z):
            return [x + (1 if x>0 else -1), y, z+dz]
        for row in originals:
            row['scene_set'] = scene_set
            for obj in row['objects']:
                obj['blocks'] = [move(*p) for p in obj['blocks']]
                obj['bounds'] = [[min(p[k] for p in obj['blocks']) for k in range(3)],
                                 [max(p[k] for p in obj['blocks'])+1 for k in range(3)]]
            commands=[]
            for command in row['commands']:
                parts=command.split()
                if parts[0]=='/setblock':
                    xyz=move(*map(int,parts[1:4]))
                    command='/setblock '+' '.join(map(str,xyz))+' '+' '.join(parts[4:])
                commands.append(command)
            row['commands']=commands
    colors = [o['color'] for o in originals[0]['objects']]
    swap = dict(zip(colors, reversed(colors)))
    donor = copy.deepcopy(originals)
    for row in donor:
        row['context'] = 'color_swap'
        for obj in row['objects']:
            obj['color'] = swap[obj['color']]
        row['commands'] = [' '.join(s.split()[:-1] + ['minecraft:'+swap[s.split()[-1].split(':')[-1][:-5]]+'_wool'])
                           if s.startswith('/setblock ') else s for s in row['commands']]
    return originals + donor


def object_mask(image, obj):
    """Color evidence inside the reviewed box; never treat the whole box as object."""
    mask = color_mask(image, obj['color'])
    x0, y0, x1, y1 = obj['bbox']
    keep = np.zeros(mask.shape, dtype=bool)
    keep[y0:y1, x0:x1] = True
    mask &= keep
    if mask.sum() < 30:
        raise ValueError('Insufficient object color evidence')
    return mask


def token_cells(h, w, merge):
    """Qwen2-VL pre-merger order: merge-block row/col, then inner row/col."""
    if h <= 0 or w <= 0 or merge <= 0 or h % merge or w % merge:
        raise ValueError('Unsupported vision grid')
    return [(br+dr, bc+dc) for br in range(0, h, merge) for bc in range(0, w, merge)
            for dr in range(merge) for dc in range(merge)]


def mask_coverage(mask, h, w, merge):
    # BOX gives area coverage in the resized patch grid; selection threshold is frozen.
    coverage = np.asarray(Image.fromarray(mask.astype('float32'), mode='F').resize((w, h), Image.Resampling.BOX))
    return [float(coverage[r, c]) for r, c in token_cells(h, w, merge)]


def regions(image, record, h, w, merge, threshold=.5):
    if not 0 < threshold <= 1:
        raise ValueError('Mask threshold must be in (0, 1]')
    cover = [mask_coverage(object_mask(image, o), h, w, merge) for o in record['objects']]
    selected = [[i for i, v in enumerate(c) if v >= threshold] for c in cover]
    if any(not s for s in selected) or sum(map(len, selected)) != len(set().union(*map(set,selected))):
        raise ValueError('Empty or overlapping object token sets')
    # Background excludes even partial overlap with either object's bounding box,
    # plus the fixed crosshair-cover rectangle. Avoids holes in color masks.
    excluded = np.zeros((image.height, image.width), dtype=bool)
    for obj in record['objects']:
        x0, y0, x1, y1 = obj['bbox']
        excluded[max(0,y0-4):y1+4, max(0,x0-4):x1+4] = True
    excluded[90:110, 214:233] = True
    bg = [i for i,v in enumerate(mask_coverage(excluded, h, w, merge)) if v == 0]
    return selected, bg


def overlay(image, positions, h, w, merge, output):
    out = image.convert('RGBA')
    layer = Image.new('RGBA', out.size)
    draw = ImageDraw.Draw(layer)
    cells = token_cells(h, w, merge)
    for i in positions:
        r, c = cells[i]
        box = (c*image.width/w, r*image.height/h, (c+1)*image.width/w, (r+1)*image.height/h)
        draw.rectangle(box, fill=(255,140,0,70), outline=(255,140,0,255))
    Image.alpha_composite(out, layer).convert('RGB').save(output)


def load_swaps(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema_version') != 'vision_swap_v1' or manifest.get('state') != 'captured_needs_visual_review' or not manifest.get('is_minecraft'):
        raise ValueError('Expected completed capture-vision dataset')
    groups, ids = {}, set()
    for r in manifest['records']:
        if r['record_id'] in ids:
            raise ValueError('Duplicate capture')
        ids.add(r['record_id'])
        for key, sha in [('image','image_sha256'), ('raw_image','raw_sha256')]:
            path = (root/r[key]).resolve()
            if root.resolve() not in path.parents or file_hash(path) != r[sha]:
                raise ValueError('Capture path/hash mismatch')
        verify_pose({'player_pos': r['actual_pose']}, [.5,200,.5], 0)
        with Image.open(root/r['image']) as im:
            if im.size != (448,155):
                raise ValueError('Unexpected capture size')
        groups.setdefault(r['scene_family_id'], []).append(r)
    if not groups:
        raise ValueError('Empty capture dataset')
    for rows in groups.values():
        if len(rows) != 6:
            raise ValueError('Incomplete capture group')
        pairs = {}
        for context in ('recipient', 'color_swap'):
            p = [r for r in rows if r['context']==context and r['kind']=='pair']
            iso = [r for r in rows if r['context']==context and r['kind']=='isolated']
            if len(p)!=1 or len(p[0]['objects'])!=2 or len(iso)!=2 or any(len(r['objects'])!=1 for r in iso):
                raise ValueError('Invalid pair/isolated captures')
            if {r['objects'][0]['object_id'] for r in iso} != {o['object_id'] for o in p[0]['objects']}:
                raise ValueError('Isolated identities mismatch')
            pairs[context] = p[0]
        palette = [o['color'] for o in pairs['recipient']['objects']]
        if len(set(palette)) != 2 or not set(palette) <= {'red','blue','yellow','green'}:
            raise ValueError('Expected two distinct supported colors')
        for a,b in zip(pairs['recipient']['objects'], pairs['color_swap']['objects']):
            if any(a[k]!=b[k] for k in ('object_id','type','blocks','bounds')) or {a['color'],b['color']}!=set(palette):
                raise ValueError('Donor must preserve geometry/identity and swap colors')
            if max(abs(x-y) for x,y in zip(a['bbox'],b['bbox'])) > 2:
                raise ValueError('Recipient/donor image alignment failed')
    return manifest, groups
