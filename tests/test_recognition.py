import copy
import json
import numpy as np
import pytest
from PIL import Image
from mc_binding.capture_pairs import clean_frame, pilot_contexts, load_pilot
from mc_binding.io import atomic_json, file_hash
from mc_binding.recognition import recognize


def frame_for(objects):
    frame = np.full((280, 448, 3), 100, dtype=np.uint8)
    palette = {'red': [160, 20, 20], 'blue': [20, 20, 160], 'green': [20, 140, 20], 'yellow': [160, 150, 20]}
    for obj in objects:
        # Positive world X is screen-left at the fixed Minecraft yaw, unlike fixtures.
        x = 90 if obj['bounds'][0][0] > 0 else 290
        frame[100:165, x:x+40] = palette[obj['color']]
    frame[230:235, :100] = [200, 0, 0]  # red command text must not affect ROIs
    return frame


def make_dataset(root):
    root.mkdir()
    (root/'raw').mkdir()
    (root/'frames').mkdir()
    records = []
    for i, ctx in enumerate(pilot_contexts(0, 731)):
        raw = frame_for(ctx['objects'])
        clean, objects = clean_frame(raw, ctx['objects'])
        raw_path, path = root/'raw'/f'{i:06d}.png', root/'frames'/f'{i:06d}.png'
        Image.fromarray(raw).save(raw_path)
        clean.save(path)
        records.append({**ctx, 'objects': objects, 'record_id': f'{i:06d}', 'scene_family_id': 'f0000',
            'image': str(path.relative_to(root)), 'image_sha256': file_hash(path),
            'raw_image': str(raw_path.relative_to(root)), 'raw_sha256': file_hash(raw_path),
            'actual_pose': dict(x=.5, y=200, z=.5, yaw=0, pitch=0)})
    atomic_json(root/'manifest.json', {'schema_version': 'recognition_pilot_v1',
        'is_minecraft': True, 'state': 'captured_needs_visual_review', 'records': records})


def test_fixed_cleanup_and_screen_order():
    ctx = pilot_contexts(0, 731)[0]
    raw = frame_for(ctx['objects'])
    before = raw.copy()
    clean, objects = clean_frame(raw, ctx['objects'])
    assert clean.size == (448, 155)
    assert np.array_equal(raw, before)
    assert objects[0]['color'] == 'blue'
    assert objects[1]['color'] == 'red'
    assert np.array_equal(np.array(clean)[60:125, 90:130], raw[100:165, 90:130])
    assert np.all(np.array(clean)[94:106, 218:229] == 128)
    raw[100:160, 218:229] = [160, 20, 20]
    with pytest.raises(ValueError, match='overlaps'):
        clean_frame(raw, ctx['objects'])


def test_counterbalance_and_disjoint_descriptions():
    assignments = set()
    for i in range(4):
        contexts = pilot_contexts(i, 731)
        assert len(contexts) == 6
        r, d = contexts[0], contexts[3]
        assignments.add(tuple((o['color'], o['type']) for o in r['objects']))
        assert {o['color'] for o in r['objects']}.isdisjoint({o['color'] for o in d['objects']})
        assert r['requested_camera'] == d['requested_camera']
    assert len(assignments) == 4


class Fake:
    def __init__(self):
        self.calls = 0
    def manifest(self):
        return {'test_double': True}
    def inputs(self, image, prompt):
        assert isinstance(image, Image.Image)
        assert all(word not in prompt for word in ['red', 'blue', 'green', 'yellow', 'object_id'])
        return prompt
    def answer(self, prompt):
        self.calls += 1
        if 'color and type' in prompt:
            return 'red tower'
        return 'red' if 'color' in prompt else 'tower'


def test_recognition_resume_and_integrity(tmp_path):
    root = tmp_path/'data'
    make_dataset(root)
    model = Fake()
    with pytest.raises(ValueError, match='Inspect'):
        recognize(root, tmp_path/'run', {}, model=model)
    store = recognize(root, tmp_path/'run', {}, reviewed=True, model=model)
    assert len(store.export()) == 24  # 12 pair probes + 12 isolated probes
    assert model.calls == 24
    recognize(root, tmp_path/'run', {}, reviewed=True, model=model)
    assert model.calls == 24
    summary = json.loads((tmp_path/'run'/'summary.json').read_text())
    assert len(summary['rates']) == 12
    (root/'frames'/'000000.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        load_pilot(root)
