"""Small MineStudio recognition pilot; candidate labels need visual review."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .backends.minestudio_smoke import smoke_scene, json_value, verify_pose
from .io import atomic_json, environment, file_hash, source_hash
from .scenes import blocks

# Frozen for the V3 pose/resolution. No adaptive crops or generative inpainting.
CROP = (0, 40, 448, 195)
CROSSHAIR = (218, 134, 229, 146)
CLEANUP = {'version': 'static_v1', 'raw_size': [448, 280], 'crop': list(CROP),
           'crosshair_box_raw': list(CROSSHAIR), 'crosshair_fill_rgb': [128, 128, 128]}


def color_mask(frame, color):
    r, g, b = np.asarray(frame, dtype=float).transpose(2, 0, 1)
    if color == 'red':
        return (r > 45) & (r > 1.45*g) & (r > 1.45*b)
    if color == 'blue':
        return (b > 45) & (b > 1.3*r) & (b > 1.2*g)
    if color == 'green':
        return (g > 35) & (g > 1.2*r) & (g > 1.15*b)
    if color == 'yellow':
        return (r > 65) & (g > 55) & (r > 1.5*b) & (g > 1.5*b) & (r < 1.8*g)
    raise ValueError(color)


def clean_frame(frame, objects):
    frame = np.asarray(frame)
    if frame.shape != (280, 448, 3) or frame.dtype != np.uint8:
        raise ValueError('Cleanup requires uint8 RGB at 448x280')
    annotated = copy.deepcopy(objects)
    for obj in annotated:
        mask = color_mask(frame, obj['color'])
        # Command text/tutorial pixels outside the retained viewport are not objects.
        mask[:CROP[1]] = False
        mask[CROP[3]:] = False
        ys, xs = np.where(mask)
        if len(xs) < 30:
            raise ValueError(f"Too few {obj['color']} pixels; inspect raw capture")
        box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
        x0, y0, x1, y1 = box
        if not (CROP[0] < x0 < x1 < CROP[2] and CROP[1] < y0 < y1 < CROP[3]):
            raise ValueError('Candidate structure touches/outside fixed crop; inspect raw capture')
        a, b, c, d = CROSSHAIR
        if x0 < c and x1 > a and y0 < d and y1 > b:
            raise ValueError('Crosshair cover overlaps candidate structure')
        obj.update(bbox_raw=box, bbox=[x0, y0-CROP[1], x1, y1-CROP[1]],
                   roi_method='color_threshold_candidate_not_segmentation', detected_pixels=len(xs))
    annotated.sort(key=lambda o: o['bbox'][0]+o['bbox'][2])
    if len(annotated) == 2 and annotated[0]['bbox'][2] >= annotated[1]['bbox'][0]:
        raise ValueError('Candidate structures overlap in screen X')
    im = Image.fromarray(frame.copy())
    a, b, c, d = CROSSHAIR
    ImageDraw.Draw(im).rectangle((a, b, c-1, d-1), fill=tuple(CLEANUP['crosshair_fill_rgb']))
    return im.crop(CROP), annotated


def pilot_contexts(index, seed):
    base, arena_commands = smoke_scene(seed)
    arena_commands = [s for s in arena_commands if not s.startswith(('/setblock ', '/tp '))]
    contexts = []
    for name, colors, kinds in [('recipient', ['red', 'blue'], ['tower', 'arch']),
                                ('disjoint', ['green', 'yellow'], ['stairs', 'pillar'])]:
        if index % 2:
            colors.reverse()
        if (index//2) % 2:
            kinds.reverse()
        objects = []
        for slot, (color, kind) in enumerate(zip(colors, kinds)):
            xyz = blocks(kind, [-6+10*slot, 200, 12])
            objects.append({'object_id': f'f{index:04d}-{name}-{slot}', 'color': color, 'type': kind,
                            'blocks': xyz, 'bounds': [[min(p[k] for p in xyz) for k in range(3)],
                                                     [max(p[k] for p in xyz)+1 for k in range(3)]]})
        for isolated in (None, 0, 1):
            chosen = objects if isolated is None else [objects[isolated]]
            commands = list(arena_commands)
            commands += [f"/setblock {x} {y} {z} minecraft:{o['color']}_wool" for o in chosen for x, y, z in o['blocks']]
            commands.append('/tp @p 0.5 200 0.5 0 0')
            contexts.append({'context': name, 'kind': 'pair' if isolated is None else 'isolated',
                             'objects': copy.deepcopy(chosen), 'commands': commands,
                             'requested_camera': copy.deepcopy(base['camera'])})
    return contexts


def capture_pairs(output, n=4, seed=731):
    if not 1 <= n <= 4:
        raise ValueError('Recognition pilot supports 1–4 balanced configurations, not independent test worlds')
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'raw').mkdir()
    (root/'frames').mkdir()
    manifest = {'schema_version': 'recognition_pilot_v1', 'is_minecraft': True,
                'state': 'running', 'labels_validated': False, 'environment': environment(),
                'source_hash': source_hash(), 'cleanup': CLEANUP, 'seed': seed,
                'scope': 'fixed-pose recognition calibration; not independent test families', 'records': []}
    sim = None
    try:
        sim = MinecraftSim(action_type='env', obs_size=(448, 280), render_size=(448, 280), seed=seed)
        obs, info = sim.reset()
        bootstrap = ['/gamemode creative', '/forceload add -20 -5 20 25', '/tp @p 0.5 200 0.5 0 0']
        obs, info = CommandsCallback(commands=bootstrap).after_reset(sim, obs, info)
        manifest['bootstrap_commands'] = bootstrap
        def settle(ticks):
            for _ in range(ticks):
                obs, _, done, truncated, info = sim.step(sim.noop_action())
                if done or truncated:
                    raise RuntimeError('Simulator terminated during capture')
            return obs, info
        obs, info = settle(200)
        for i in range(n):
            for context in pilot_contexts(i, seed):
                obs, info = CommandsCallback(commands=context['commands']).after_reset(sim, obs, info)
                obs, info = settle(200)
                ident = f'{len(manifest["records"]):06d}'
                raw = root/'raw'/f'{ident}.png'
                Image.fromarray(np.asarray(obs['image'])).save(raw)
                verify_pose(info, [0.5, 200, 0.5], 0)
                cleaned, objects = clean_frame(obs['image'], context['objects'])
                path = root/'frames'/f'{ident}.png'
                cleaned.save(path)
                manifest['records'].append({**context, 'objects': objects, 'record_id': ident,
                    'scene_family_id': f'f{i:04d}', 'image': str(path.relative_to(root)),
                    'image_sha256': file_hash(path), 'raw_image': str(raw.relative_to(root)),
                    'raw_sha256': file_hash(raw), 'actual_pose': json_value(info['player_pos']),
                    'location_stats': json_value(info.get('location_stats'))})
                atomic_json(root/'manifest.json', manifest)
                print(f'Captured {ident}: configuration {i}, {context["context"]}, {context["kind"]}', flush=True)
        manifest['state'] = 'captured_needs_visual_review'
        contact_sheet(root, manifest)
    except BaseException as exc:
        manifest.update(state='error', error=str(exc))
        raise
    finally:
        atomic_json(root/'manifest.json', manifest)
        if sim is not None:
            sim.close()


def contact_sheet(root, manifest):
    records = manifest['records']
    sheet = Image.new('RGB', (896, ((len(records)+1)//2)*185), 'white')
    for i, row in enumerate(records):
        with Image.open(root/row['image']) as image:
            tile = image.convert('RGB')
        draw = ImageDraw.Draw(tile)
        for obj in row['objects']:
            draw.rectangle(obj['bbox'], outline='white', width=1)
        x, y = (i%2)*448, (i//2)*185
        sheet.paste(tile, (x, y+30))
        label = row['record_id']+' '+row['kind']+' | '+', '.join(o['color']+' '+o['type'] for o in row['objects'])
        ImageDraw.Draw(sheet).text((x+4, y+5), label, fill='black')
    sheet.save(root/'contact_sheet.png')


def load_pilot(root):
    root = Path(root)
    data = json.loads((root/'manifest.json').read_text())
    if data.get('schema_version') != 'recognition_pilot_v1' or data.get('state') != 'captured_needs_visual_review' or not data.get('is_minecraft'):
        raise ValueError('Expected a complete Minecraft recognition pilot')
    ids = set()
    for row in data['records']:
        if row['record_id'] in ids or not row['record_id'].isdigit():
            raise ValueError('Invalid/repeated record ID')
        ids.add(row['record_id'])
        for field, hash_field in [('image', 'image_sha256'), ('raw_image', 'raw_sha256')]:
            path = (root/row[field]).resolve()
            if root.resolve() not in path.parents or file_hash(path) != row[hash_field]:
                raise ValueError('Capture hash/path mismatch')
        if len(row['objects']) != (2 if row['kind'] == 'pair' else 1):
            raise ValueError('Object count mismatch')
    if not ids:
        raise ValueError('Empty pilot')
    groups = {}
    for row in data['records']:
        groups.setdefault(row['scene_family_id'], []).append(row)
        verify_pose({'player_pos': row['actual_pose']}, [0.5, 200, 0.5], 0)
        with Image.open(root/row['image']) as image:
            if image.size != (448, 155):
                raise ValueError('Unexpected cleaned-image dimensions')
    for rows in groups.values():
        for context in ('recipient', 'disjoint'):
            selected = [r for r in rows if r['context'] == context]
            pairs = [r for r in selected if r['kind'] == 'pair']
            isolated = [r for r in selected if r['kind'] == 'isolated']
            if len(pairs) != 1 or len(isolated) != 2:
                raise ValueError('Incomplete paired/isolated capture group')
            if {o['object_id'] for o in pairs[0]['objects']} != {r['objects'][0]['object_id'] for r in isolated}:
                raise ValueError('Isolated objects do not match paired scene')
    return data
