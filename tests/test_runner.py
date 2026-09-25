import json
import pytest
import torch
from mc_binding.dataset import generate
from mc_binding.runner import run
from mc_binding.analysis import analyze
from mc_binding.models.qwen import Qwen


class FakeModel:
    """Plumbing test double: fixed text, never a model evaluation."""
    def manifest(self):
        return {'test_double': True}

    def inputs(self, image, prompt):
        return {'input_ids': torch.zeros((1, 10), dtype=torch.long)}

    def answer(self, inp, specs=()):
        return 'red tower'

    def roi(self, inp, context, index):
        return [1, 2], (2, 2)

    def capture(self, inp, requests):
        return {key: torch.zeros(10, 4) for key in requests}

    def channels(self, layer, kind):
        return [0, 1, 2, 3]


def test_full_matrix_resume_and_analysis(tmp_path):
    data, output = tmp_path/'data', tmp_path/'run'
    generate(data, 2)
    cfg = {'layers': [0], 'task': 'pair', 'aliases': {}}
    with pytest.raises(ValueError, match='Fixture inference'):
        run(data, output, cfg, True, model=FakeModel())
    store = run(data, output, cfg, True, True, FakeModel())
    rows = store.export()
    assert len(rows) == 2*(1+3+4*5)
    assert len({r['trial_key'] for r in rows}) == len(rows)
    assert run(data, output, cfg, True, True, FakeModel()).export() == rows
    result = analyze(output)
    assert result['is_minecraft'] is False
    primary = [r for r in result['comparisons'] if r['primary'] and r['subset'] == 'all']
    assert len(primary) == 2
    assert all(r['difference'] == 0 for r in primary)


def test_resized_roi_mapping():
    from types import SimpleNamespace
    adapter = object.__new__(Qwen)
    adapter.model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    adapter.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    inp = {'input_ids': torch.tensor([[1]+[99]*8+[2]]), 'image_grid_thw': torch.tensor([[1, 4, 8]])}
    context = {'width': 800, 'height': 400, 'objects': [{'bbox': [0, 0, 400, 400]}]}
    positions, shape = adapter.roi(inp, context, 0)
    assert shape == (2, 4)
    assert positions == [1, 2, 5, 6]
