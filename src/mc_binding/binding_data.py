"""Four-object conjunction tasks. No rendering or model imports at module load."""
import copy
import itertools
import json
import re
from pathlib import Path
from PIL import Image
from .capture_pairs import pilot_contexts
from .scenes import blocks
from .io import file_hash
from .backends.minestudio_smoke import verify_pose

FEATURES = [('red','pillar'), ('blue','pillar'), ('red','arch'), ('blue','arch')]


def binding_contexts(index, seed):
    if not 0 <= index < 24:
        raise ValueError('binding_v1 has 24 balanced permutations, not independent worlds')
    features = list(itertools.permutations(FEATURES))[index]
    base = pilot_contexts(0, seed)[0]
    arena = [s for s in base['commands'] if not s.startswith(('/setblock ', '/tp '))]
    objects = []
    for slot, ((color, shape), x) in enumerate(zip(features, (-13,-7,3,9))):
        xyz = blocks(shape, (x,200,20))
        objects.append(dict(object_id=f'f{index:04d}-slot{slot}', color=color, type=shape,
                            blocks=xyz, bounds=[[min(p[k] for p in xyz) for k in range(3)],
                                               [max(p[k] for p in xyz)+1 for k in range(3)]]))
    rows = []
    for context in ('recipient','color_swap'):
        objs = copy.deepcopy(objects)
        if context == 'color_swap':
            for obj in objs:
                obj['color'] = {'red':'blue','blue':'red'}[obj['color']]
        # Isolated images establish separate ROIs for repeated colors; no global-color bbox.
        scenes = [('isolated',[o]) for o in objs] + [('full',objs), ('absent',[o for j,o in enumerate(objs) if j != index%4])]
        for kind, selected in scenes:
            commands = arena + [f"/setblock {x} {y} {z} minecraft:{o['color']}_wool"
                                for o in selected for x,y,z in o['blocks']] + ['/tp @p 0.5 200 0.5 0 0']
            rows.append(dict(context=context, kind=kind, objects=copy.deepcopy(selected), commands=commands,
                             requested_camera=copy.deepcopy(base['requested_camera']), scene_set='binding_v1',
                             permutation=index, split='development', removed_slot=index%4 if kind=='absent' else None))
    return rows


def queries(record):
    objects = record['objects']  # strictly screen-left-to-right, assigned after rendering
    result = []
    for color, shape in FEATURES:
        matches = [i+1 for i,o in enumerate(objects) if (o['color'],o['type'])==(color,shape)]
        if len(matches)>1:
            raise ValueError('Ambiguous conjunction target')
        prompt = (f"Find the {color} {shape}. Count structures from left to right starting at 1. "
                  f"Which numbered structure is it? Answer with one digit from 1 to {len(objects)}, "
                  "or 0 if no structure has BOTH that color and shape.")
        result.append(dict(query_id=f'{color}-{shape}', task='binding', prompt=prompt,
                           expected=str(matches[0] if matches else 0), goal_color=color, goal_shape=shape))
    for i,obj in enumerate(objects,1):
        for task, field in [('color','color'),('shape','type')]:
            vocabulary = 'red or blue' if task=='color' else 'arch or pillar'
            result.append(dict(query_id=f'{task}-{i}', task=task, expected=obj[field],
                               prompt=f'Count structures from left to right starting at 1. What {task} is structure {i}? Answer {vocabulary} only.'))
    return result


def parse_answer(raw, task):
    value = raw.strip().lower().rstrip('.!')
    allowed = {'binding': {'0','1','2','3','4'}, 'color': {'red','blue'}, 'shape': {'arch','pillar'}}[task]
    return value if value in allowed else None


def load_binding(root):
    root = Path(root).resolve()
    data = json.loads((root/'manifest.json').read_text())
    if data.get('schema_version')!='binding_v1' or not data.get('is_minecraft') or data.get('state')!='captured_needs_visual_review':
        raise ValueError('Expected completed real Minecraft binding captures')
    groups, ids = {}, set()
    for r in data['records']:
        if not re.fullmatch(r'\d{6}', r['record_id']) or r['record_id'] in ids:
            raise ValueError('Invalid or duplicate capture ID')
        ids.add(r['record_id'])
        for name,sha in [('image','image_sha256'),('raw_image','raw_sha256')]:
            p=(root/r[name]).resolve()
            if root not in p.parents or file_hash(p)!=r[sha]:
                raise ValueError('Binding capture hash/path mismatch')
        verify_pose({'player_pos':r['actual_pose']}, [.5,200,.5],0)
        with Image.open(root/r['image']) as im:
            if im.size!=(448,155): raise ValueError('Unexpected binding image size')
        objects=r['objects']
        if r['kind'] not in ('full','isolated','absent') or len(objects)!={'full':4,'isolated':1,'absent':3}[r['kind']]:
            raise ValueError('Incorrect binding object count')
        for obj in objects:
            x0,y0,x1,y1=obj['bbox']
            if not (0<=x0<x1<=448 and 0<=y0<y1<=155): raise ValueError('Invalid bounding box')
        if any(a['bbox'][2]>=b['bbox'][0] for a,b in zip(objects,objects[1:])):
            raise ValueError('Binding objects must be screen ordered and separated')
        groups.setdefault(r['scene_family_id'],[]).append(r)
    if not groups: raise ValueError('Empty binding dataset')
    for records in groups.values():
        if len(records)!=12: raise ValueError('Incomplete binding family')
        full={}
        for ctx in ('recipient','color_swap'):
            selected=[r for r in records if r['context']==ctx]
            fs=[r for r in selected if r['kind']=='full']
            missing=[r for r in selected if r['kind']=='absent']
            isolated=[r for r in selected if r['kind']=='isolated']
            if len(fs)!=1 or len(missing)!=1 or len(isolated)!=4: raise ValueError('Incomplete binding contexts')
            full[ctx]=fs[0]
            objs=fs[0]['objects']
            if sorted((o['color'],o['type']) for o in objs)!=sorted(FEATURES): raise ValueError('Incorrect feature conjunctions')
            byid={o['object_id']:o for o in objs}
            if len(byid)!=4 or {r['objects'][0]['object_id'] for r in isolated}!=set(byid): raise ValueError('Invalid isolated identities')
            for r in isolated+missing:
                for o in r['objects']:
                    a=byid.get(o['object_id'])
                    if a is None or any(a[k]!=o[k] for k in ('color','type','blocks','bounds')): raise ValueError('Context object mismatch')
                    if max(abs(x-y) for x,y in zip(a['bbox'],o['bbox']))>2: raise ValueError('Isolation alignment mismatch')
            if len({o['object_id'] for o in missing[0]['objects']})!=3: raise ValueError('Invalid absent-target scene')
        for a,b in zip(full['recipient']['objects'],full['color_swap']['objects']):
            if any(a[k]!=b[k] for k in ('object_id','type','blocks','bounds')) or a['color']==b['color']:
                raise ValueError('Donor must change colors and preserve geometry')
            if max(abs(x-y) for x,y in zip(a['bbox'],b['bbox']))>2: raise ValueError('Donor alignment failed')
    return data, groups
