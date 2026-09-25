import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def environment():
    versions = {}
    for package in ("mc-binding", "torch", "transformers", "Pillow", "numpy", "bitsandbytes", "minestudio", "minedojo"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


class RunStore:
    """One writer per output directory; complete families are atomic checkpoints."""
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        path = self.root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("Resume refused: configuration, environment or dataset changed")
        else:
            atomic_json(path, manifest)

    def completed(self, family):
        return (self.root / "families" / (family + ".json")).exists()

    def save(self, family, rows):
        keys = [r["trial_key"] for r in rows]
        if len(keys) != len(set(keys)) or self.completed(family):
            raise ValueError("Duplicate scene-condition keys or completed family")
        atomic_json(self.root / "families" / (family + ".json"), rows)

    def export(self):
        rows = []
        for path in sorted((self.root / "families").glob("*.json")):
            rows.extend(json.loads(path.read_text()))
        path = self.root / "results.jsonl"
        temporary = self.root / ".results.pending"
        temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        temporary.replace(path)
        return rows


def source_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*')) if p.suffix in ('.py', '.json')})
