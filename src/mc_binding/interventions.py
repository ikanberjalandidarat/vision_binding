"""Ported from thoughtful_emoji cell 41; single-example, prefill-only hooks."""
from contextlib import contextmanager
import torch


def channels(width, n_heads, heads=None):
    if width % n_heads:
        raise ValueError("Projection width incompatible with head count")
    if heads is None:
        return list(range(width))
    if not heads or len(set(heads)) != len(heads) or any(h < 0 or h >= n_heads for h in heads):
        raise ValueError("Invalid head selection")
    dim = width // n_heads
    return [h*dim+i for h in heads for i in range(dim)]


def random_replacement(original, donor, selected_channels, seed):
    """Norm matches the donor delta ONLY over channels actually patched."""
    original, donor = original.float().cpu(), donor.float().cpu()
    result = original.clone()
    delta = donor[:, selected_channels]-original[:, selected_channels]
    noise = torch.randn(delta.shape, generator=torch.Generator().manual_seed(seed))
    noise *= delta.norm()/noise.norm().clamp_min(1e-12)
    result[:, selected_channels] += noise
    return result


@contextmanager
def patch_hooks(module_for, specs, prefill_length):
    handles, fired = [], [False]*len(specs)
    keys = [(s['layer'], s['kind']) for s in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("Combine edits to the same module into one spec")
    try:
        for i, spec in enumerate(specs):
            def edit(x, s=spec, index=i):
                if fired[index]:
                    return x
                if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != prefill_length:
                    raise ValueError("Expected single-example full prefill before cached decoding")
                pos, ch = s['positions'], s['channels']
                if not pos or min(pos) < 0 or max(pos) >= x.shape[1] or not ch or min(ch) < 0 or max(ch) >= x.shape[2]:
                    raise ValueError("Patch index outside projection")
                val = s['value'].to(x.device, x.dtype)
                if tuple(val.shape) != (len(pos), x.shape[-1]):
                    raise ValueError("Replacement shape mismatch")
                y = x.clone()
                for j, p in enumerate(pos):
                    y[0, p, ch] = val[j, ch]
                fired[index] = True
                return y
            module = module_for(spec['layer'], spec['kind'])
            if spec['kind'] == 'o':
                def pre(m, args, edit=edit):
                    return (edit(args[0]),)+args[1:]
                handles.append(module.register_forward_pre_hook(pre))
            else:
                def post(m, args, out, edit=edit):
                    return edit(out)
                handles.append(module.register_forward_hook(post))
        yield fired
        if not all(fired):
            raise RuntimeError("Intervention did not fire")
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_hooks(module_for, requests, values):
    handles = []
    try:
        for layer, kind in requests:
            def save(x, key=(layer, kind)):
                values[key] = x[0].detach().float().cpu().clone()
            if kind == 'o':
                def pre(m, args, save=save):
                    save(args[0])
                handles.append(module_for(layer, kind).register_forward_pre_hook(pre))
            else:
                def post(m, args, out, save=save):
                    save(out)
                handles.append(module_for(layer, kind).register_forward_hook(post))
        yield
        if set(values) != set(requests):
            raise RuntimeError("Capture did not fire")
    finally:
        for handle in handles:
            handle.remove()
