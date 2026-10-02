"""Vision-block output interventions: 2D pre-merger tokens, all channels."""
from contextlib import contextmanager
import torch


def vision_backbone(model):
    for name in ('model.visual', 'visual'):
        obj = model
        try:
            for part in name.split('.'):
                obj = getattr(obj, part)
            if hasattr(obj, 'blocks') and hasattr(obj, 'merger'):
                return obj, name
        except AttributeError:
            pass
    raise RuntimeError('Unsupported Qwen2-VL vision backbone')


@contextmanager
def capture_blocks(blocks, layers, values, kind="residual"):
    handles = []
    try:
        for index in layers:
            def save(module, args, out, index=index):
                if out.ndim != 2 or not torch.isfinite(out).all() or index in values:
                    raise RuntimeError('Expected one finite 2D vision-block output')
                values[index] = out[:, projection_slice(out.shape[1], kind)].detach().float().cpu().clone()
            handles.append(blocks[index].register_forward_hook(save))
        yield
        if set(values) != set(layers):
            raise RuntimeError('Vision capture hook did not fire')
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_block(block, positions, value, expected_tokens, kind="residual", *, heads=None, num_heads=None, alpha=1.):
    fired = 0
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    if heads is not None and (kind == 'residual' or not num_heads or not heads or
            any(type(h) is not int or h < 0 or h >= num_heads for h in heads) or len(set(heads)) != len(heads)):
        raise ValueError('Invalid projection heads')
    if not positions or len(set(positions)) != len(positions) or min(positions)<0 or max(positions)>=expected_tokens:
        raise ValueError('Invalid vision token positions')
    def edit(module, args, out):
        nonlocal fired
        fired += 1
        if fired != 1 or out.ndim != 2 or out.shape[0] != expected_tokens:
            raise RuntimeError('Expected exactly one full-image vision forward')
        selected = projection_slice(out.shape[1], kind)
        width = out[:, selected].shape[1]
        if value.shape != (len(positions), width) or not torch.isfinite(value).all():
            raise ValueError('Invalid replacement tensor')
        result = out.clone()
        cast = value.to(device=out.device, dtype=out.dtype)
        if not torch.isfinite(cast).all():
            raise RuntimeError('Replacement overflow after cast')
        if heads is None:
            columns = torch.arange(selected.start, selected.stop, device=out.device)
        else:
            if width % num_heads:
                raise ValueError('Projection width is not divisible by head count')
            d = width // num_heads
            columns = torch.tensor([selected.start+h*d+c for h in heads for c in range(d)], device=out.device)
            cast = cast[:, columns-selected.start]
        rows = torch.tensor(positions, device=out.device)[:, None]
        # Interpolate against the LIVE recipient output, including earlier patches.
        original = out[rows, columns]
        changed = original if alpha == 0 else cast if alpha == 1 else (1-alpha)*original+alpha*cast
        result[rows, columns] = changed
        return result
    handle = block.register_forward_hook(edit)
    try:
        yield
        if fired != 1:
            raise RuntimeError('Vision patch did not fire')
    finally:
        handle.remove()


def replacement(recipient, donor, positions, condition, seed):
    original, value = recipient[positions].clone(), donor[positions].clone()
    if condition == 'self':
        value = original.clone()
    elif condition == 'random':
        noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(seed))
        value = original + noise * ((value-original).norm()/noise.norm().clamp_min(1e-12))
    return value, float((value-original).norm())


def projection_slice(width, kind):
    if kind == 'residual':
        return slice(0, width)
    if kind not in ('q','k','v') or width % 3:
        raise ValueError('Expected fused vision QKV width divisible by three')
    d = width // 3
    start = {'q':0,'k':1,'v':2}[kind]*d
    return slice(start, start+d)
