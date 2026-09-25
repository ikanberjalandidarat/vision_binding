import pytest
import torch
from mc_binding.interventions import patch_hooks, channels, random_replacement


def test_head_isolation_once_and_cleanup():
    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(4))
    value = torch.full((1, 4), 9.)
    spec = {'layer': 0, 'kind': 'q', 'positions': [2], 'channels': channels(4, 2, [1]), 'value': value}
    x = torch.zeros(1, 3, 4)
    with patch_hooks(lambda l, k: module, [spec], 3):
        actual = module(x)
        assert actual[0, 2].tolist() == [0., 0., 9., 9.]
        assert torch.equal(module(x[:, :1]), x[:, :1])
    assert not module._forward_hooks
    assert torch.equal(value, torch.full((1, 4), 9.))
    with pytest.raises(RuntimeError, match='boom'):
        with patch_hooks(lambda l, k: module, [spec], 3):
            raise RuntimeError('boom')
    assert not module._forward_hooks
    with pytest.raises(RuntimeError, match='did not fire'):
        with patch_hooks(lambda l, k: module, [spec], 3):
            pass
    assert not module._forward_hooks


def test_random_control_matches_selected_norm():
    a, b = torch.zeros(2, 8), torch.arange(16.).reshape(2, 8)
    ch = channels(8, 4, [1, 3])
    result = random_replacement(a, b, ch, 123)
    assert torch.allclose((result[:, ch]-a[:, ch]).norm(), (b[:, ch]-a[:, ch]).norm())
    assert torch.equal(result[:, [0, 1, 4, 5]], a[:, [0, 1, 4, 5]])
    assert torch.equal(result, random_replacement(a, b, ch, 123))


def test_o_projection_input_hook():
    module = torch.nn.Linear(4, 4, bias=False)
    x = torch.randn(1, 3, 4)
    spec = {'layer': 0, 'kind': 'o', 'positions': [2], 'channels': [0, 1, 2, 3], 'value': x[0, 2:3].clone()}
    clean = module(x)
    with patch_hooks(lambda l, k: module, [spec], 3):
        assert torch.equal(module(x), clean)
    assert not module._forward_pre_hooks
