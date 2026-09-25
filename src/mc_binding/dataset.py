import json
from pathlib import Path
from PIL import Image, ImageDraw
from .backends.fixture import FixtureBackend
from .io import atomic_json, digest, file_hash
from .scenes import family


def generate(root, n=20, seed=731, split="debug"):
    root = Path(root)
    if n < 1:
        raise ValueError("n must be positive")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Dataset directory must be empty; datasets are immutable")
    records = []
    backend = FixtureBackend()
    try:
        for i in range(n):
            spec = family(i, seed, split)
            for j, context in enumerate(spec["contexts"].values()):
                backend.reset(context["arena"])
                backend.build(context["objects"], [])
                backend.set_camera(context["camera"])
                capture = backend.capture()
                path = Path("frames") / f"{i:06d}-{j}.png"
                (root / path).parent.mkdir(exist_ok=True)
                capture.rgb.save(root / path)
                context.update(image=str(path), image_sha256=file_hash(root/path), width=capture.rgb.width, height=capture.rgb.height, actual_pose=capture.actual_pose, render=capture.metadata)
                for obj in context["objects"]:
                    obj["bbox"] = capture.metadata['rois'][obj['object_id']]
                context["objects"].sort(key=lambda o: o['bbox'][0]+o['bbox'][2])
            records.append(spec)
        atomic_json(root / "manifest.json", {"schema_version": "1.0", "backend": "fixture", "is_minecraft": False, "generation": {"n": n, "seed": seed, "split": split}, "families": records})
    finally:
        backend.close()
    validate(root)
    contact_sheet(root)
    return root / "manifest.json"


def validate(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    from .schema import check
    check("dataset", manifest)
    if manifest["schema_version"] != "1.0":
        raise ValueError("Unsupported dataset schema")
    ids = set()
    for spec in manifest['families']:
        if spec['family_id'] in ids:
            raise ValueError("Repeated scene family")
        ids.add(spec['family_id'])
        for context in spec['contexts'].values():
            path = (root / context['image']).resolve()
            if root.resolve() not in path.parents or file_hash(path) != context['image_sha256']:
                raise ValueError("Invalid image path or hash")
            with Image.open(path) as im:
                if im.size != (context['width'], context['height']):
                    raise ValueError("Image dimensions changed")
            if len(context['objects']) != 2:
                raise ValueError("Static pilot requires two objects")
            centers = []
            for obj in context['objects']:
                x0, y0, x1, y1 = obj['bbox']
                if not (0 <= x0 < x1 <= context['width'] and 0 <= y0 < y1 <= context['height']):
                    raise ValueError("ROI outside image")
                centers.append((x0+x1)/2)
            if centers[0] >= centers[1]:
                raise ValueError("Object table must use validated screen order")
            if context['render']['is_minecraft'] != manifest['is_minecraft']:
                raise ValueError("Mixed fixture and Minecraft provenance")
            if manifest['is_minecraft'] and not context['render'].get('labels_validated', False):
                raise ValueError("Real frames require validated camera/ROI/visibility labels")
        rec, donor = [spec['contexts'][k]['objects'] for k in ('recipient', 'disjoint')]
        if {(o['color'], o['type']) for o in rec} & {(o['color'], o['type']) for o in donor}:
            raise ValueError("Disjoint donor repeats recipient description")
    if not ids:
        raise ValueError("Empty dataset")
    return manifest


def contact_sheet(root):
    root = Path(root)
    manifest = validate(root)
    selected = manifest['families'][:20]
    sheet = Image.new('RGB', (896, len(selected)*310), 'white')
    for row, spec in enumerate(selected):
        for col, ctx in enumerate(spec['contexts'].values()):
            with Image.open(root/ctx['image']) as source:
                im = source.convert('RGB')
            draw = ImageDraw.Draw(im)
            for obj in ctx['objects']:
                draw.rectangle(obj['bbox'], outline='black', width=2)
                draw.text((obj['bbox'][0], obj['bbox'][1]-15), obj['color']+' '+obj['type'], fill='black')
            sheet.paste(im.resize((448, 280)), (col*448, row*310+30))
        ImageDraw.Draw(sheet).text((4, row*310+5), ('FIXTURE / NOT MINECRAFT ' if not manifest['is_minecraft'] else '')+spec['family_id'], fill='black')
    sheet.save(root/'contact_sheet.png')
    return digest(manifest)
