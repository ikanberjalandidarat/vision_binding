import json
from pathlib import Path
import pytest
import torch
from test_recognition import make_dataset
from mc_binding.q_pilot import q_pilot, make_specs

CONFIG = {'load_in_4bit': False, 'dtype': 'bfloat16', 'use_fast': True,
          'layers': [0], 'seed': 731, 'query_heads': [0]}


class FakeQ:
    """Answers from synthetic pixel colors, not annotations; no ROI API."""
    def __init__(self, corrupt_self=False, wrong_clean=False):
        self.patches = []
        self.corrupt_self, self.wrong_clean = corrupt_self, wrong_clean
    def manifest(self):
        return {'test_double': True}
    def inputs(self, image, prompt):
        # Left/right object slots from the synthetic fixture used in these tests.
        options = []
        palette = {(160, 20, 20): 'red', (20, 20, 160): 'blue',
                   (20, 140, 20): 'green', (160, 150, 20): 'yellow'}
        for x in (100, 300):
            pixel = image.getpixel((x, 80))
            if pixel in palette:
                options.append(palette[pixel])
        chosen = options[-1] if 'rightmost' in prompt else options[0]
        return {'input_ids': torch.zeros((1, 10), dtype=torch.long), 'chosen': chosen}
    def answer(self, inp, specs=()):
        if specs:
            self.patches.append(specs)
            if self.corrupt_self:
                return 'yellow'
        return 'nonsense' if self.wrong_clean else inp['chosen']
    def capture(self, inp, requests):
        number = {'red': 1., 'blue': 2., 'green': 3., 'yellow': 4.}[inp['chosen']]
        return {key: torch.full((10, 4), number) for key in requests}
    def channels(self, layer, kind):
        assert kind == 'q'
        return [0, 1]


def test_q_matrix_gates_resume_and_no_roi_dependency(tmp_path):
    data = tmp_path/'data'
    make_dataset(data)
    model = FakeQ()
    store = q_pilot(data, tmp_path/'run', CONFIG, True, model)
    rows = store.export()
    assert len(rows) == 26
    assert len({r['trial_key'] for r in rows}) == 26
    assert sum(r['condition'] == 'clean' for r in rows) == 8
    assert sum(r['condition'] == 'self_q' for r in rows) == 2
    assert len(model.patches) == 18
    for specs in model.patches:
        assert all(s['kind'] == 'q' and s['positions'] == [9] and s['channels'] == [0, 1] for s in specs)
    q_pilot(data, tmp_path/'run', CONFIG, True, model)
    assert len(model.patches) == 18
    summary = json.loads((tmp_path/'run'/'summary.json').read_text())
    assert all(r['n_probes'] == 2 for r in summary['primary'])
    assert all(r['q_minus_random_q'] == 0 for r in summary['paired_differences'])


@pytest.mark.parametrize('mode', ['clean', 'self'])
def test_failed_gates_stop_before_donor_runs(tmp_path, mode):
    data = tmp_path/'data'
    make_dataset(data)
    model = FakeQ(corrupt_self=mode == 'self', wrong_clean=mode == 'clean')
    with pytest.raises(RuntimeError, match='gate failed|self-Q patch failed'):
        q_pilot(data, tmp_path/'run', CONFIG, True, model)
    assert len(model.patches) == (1 if mode == 'self' else 0)
    assert json.loads((tmp_path/'run'/'status.json').read_text())['state'] == 'error'
    assert not list((tmp_path/'run'/'families').glob('*.json'))
    assert (tmp_path/'run'/'diagnostics'/'f0000.json').exists()


def test_norm_control_selected_channels_and_nonfinite_rejection():
    model = FakeQ()
    base = {(0, 'q'): torch.zeros(10, 4)}
    donor = {(0, 'q'): torch.full((10, 4), 2.)}
    specs, metadata = make_specs(model, [0], base, donor, 9, 9, 13)
    assert specs[0]['value'][0, 2:].tolist() == [0., 0.]
    assert metadata[0]['applied_delta_norm_float32'] == pytest.approx(8**.5)
    donor[0, 'q'][9, 0] = float('nan')
    with pytest.raises(RuntimeError, match='non-finite'):
        make_specs(model, [0], base, donor, 9, 9)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_generation_numerical_failure_is_not_scored(value):
    from types import SimpleNamespace
    from mc_binding.models.qwen import Qwen
    adapter = object.__new__(Qwen)
    adapter.config = {'max_new_tokens': 1, 'check_finite_scores': True}
    adapter.model = SimpleNamespace(generate=lambda **kwargs: SimpleNamespace(
        sequences=torch.zeros((1, 3), dtype=torch.long), scores=[torch.full((1, 4), value)]))
    with pytest.raises(RuntimeError, match='Non-finite generation scores'):
        adapter.answer({'input_ids': torch.zeros((1, 2), dtype=torch.long)})
