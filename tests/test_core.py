import json
import pytest
from mc_binding.dataset import generate, validate
from mc_binding.io import RunStore
from mc_binding.scoring import parse, score
from mc_binding.scenes import family
from mc_binding.analysis import wilson


def test_dataset_reproducible_and_hash_checked(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    generate(a, 4)
    generate(b, 4)
    assert validate(a) == validate(b)
    data = validate(a)
    assert [s['target_side'] for s in data['families']] == ['left', 'right']*2
    path = a/data['families'][0]['contexts']['recipient']['image']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        validate(a)


def test_parser_ambiguity_aliases_and_recombination():
    assert parse('Red tower.')['parsed'] == {'color': 'red', 'type': 'tower'}
    for raw in ('red tower or blue arch', 'not red tower', 'red red tower', 'pink tower'):
        assert parse(raw)['status'] == 'invalid'
    assert parse('scarlet tower', {'scarlet': 'red'})['status'] == 'valid'
    rec = [{'color': 'red', 'type': 'tower'}, {'color': 'blue', 'type': 'arch'}]
    donor = [{'color': 'green', 'type': 'stairs'}, {'color': 'yellow', 'type': 'pillar'}]
    assert score('blue arch', rec, donor, 0, 1)['outcome'] == 'recipient_at_donor_address'
    assert score('red arch', rec, donor, 0, 1)['outcome'] == 'recipient_recombination'
    assert parse('blue', task='color')['parsed'] == {'color': 'blue'}


def test_resume_and_duplicate_protection(tmp_path):
    store = RunStore(tmp_path, {'data_hash': 'a'})
    store.save('f', [{'trial_key': 'f:q'}])
    resumed = RunStore(tmp_path, {'data_hash': 'a'})
    assert resumed.completed('f')
    assert len(resumed.export()) == 1
    with pytest.raises(ValueError):
        resumed.save('f', [])
    with pytest.raises(ValueError):
        RunStore(tmp_path, {'data_hash': 'b'})
    with pytest.raises(ValueError):
        resumed.save('g', [{'trial_key': 'g:q'}]*2)


def test_split_and_counterfactuals():
    a, b = family(0, 731, 'calibration'), family(10000, 731, 'test')
    assert a['seed'] != b['seed']
    for spec in [a, b]:
        r, d = [spec['contexts'][k] for k in ('recipient', 'disjoint')]
        assert r['camera'] == d['camera']
        assert {o['color'] for o in r['objects']}.isdisjoint({o['color'] for o in d['objects']})


def test_wilson():
    assert wilson(0, 0) == [None, None]
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(.40383, abs=1e-4)
    assert hi == pytest.approx(.59617, abs=1e-4)
