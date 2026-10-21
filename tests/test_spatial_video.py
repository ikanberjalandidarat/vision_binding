"""NON-MINECRAFT synthetic fixtures. No renderer/pretrained-model validation."""
import numpy as np
import pytest
import torch

from mc_binding.spatial_video import clips, capture_blocks, partition
from mc_binding.spatial_video_probe import ridge_predict, occupancy_map, circular_labels


def entry():
    # Boolean exercises the production validator; data remain synthetic.
    return dict(episode="NON-MINECRAFT-fixture", geometry="fixture", route=2,
                split="train", is_minecraft=True, route_complete=True,
                frames=[dict(elapsed_ticks=t, pose=dict(x=t, z=0, yaw=0))
                        for t in range(0, 256, 8)])


def test_clip_clock_split_and_no_overlap():
    e = entry()
    e["frames"].insert(1, dict(elapsed_ticks=2, pose=dict(x=-999,z=0,yaw=0)))
    windows = list(clips([e]))
    assert len(windows) == 2
    assert windows[0]["position"] == [120, 0]
    assert windows[0]["split"] == "heldout_route"
    assert not set(windows[0]["ticks"]) & set(windows[1]["ticks"])
    e["frames"] = [f for f in e["frames"] if f["elapsed_ticks"] != 8]
    assert len(list(clips([e]))) == 1
    with pytest.raises(ValueError):
        list(clips([e, e]))
    e["route_complete"] = False
    with pytest.raises(ValueError):
        list(clips([e]))


def test_hooks_and_cleanup():
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4,4) for _ in range(3)])
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x
    net = Fixture().eval()
    values = capture_blocks(net, [0,2], torch.ones(1,6,4))
    assert values[2].shape == (1,6,4)
    assert not values[2].requires_grad
    assert all(not b._forward_hooks for b in net.blocks)


def test_probe_train_only_scaling_and_sparse_maps():
    x = np.arange(20.).reshape(10,2)
    y = x[:, :1] * 3 + 5
    prediction = ridge_predict(x, y, x + 20, .00001)
    np.testing.assert_allclose(prediction, (x[:, :1]+20)*3+5, atol=.001)
    m, n = occupancy_map(np.array([[.2,.2],[.2,.2],[.2,.2]]),
                         np.array([-1.,0.,1.]), ([0,1,2],[0,1,2]))
    assert m[0,0] == 0 and n[0,0] == 3 and np.isnan(m[1,1])
    ids = np.array(["a"]*5+["b"]*5)
    shifted = circular_labels(y, ids, 731)
    assert set(shifted[:5,0]) == set(y[:5,0])
    assert set(shifted[5:,0]) == set(y[5:,0])


def test_analysis_fixture_end_to_end(tmp_path):
    """Entire cache-to-review flow, explicitly NON-MINECRAFT data."""
    import json
    from mc_binding.spatial_video_probe import analyze
    rows = []
    for part, geom in [("train","a"), ("heldout_route","a"), ("validation","b"), ("test","c")]:
        for i in range(8):
            row = dict(episode="NON-MINECRAFT-"+part, split=part, geometry=geom,
                       ticks=[i*128], heading=i*30, position=[i*.5, i%3],
                       cache=f"{part}-{i}.npz")
            x = np.array([i*.5, i%3, np.sin(i), np.cos(i)])
            np.savez(tmp_path / row["cache"], video_layer3=x, repeat_last_frame_layer3=x)
            rows.append(row)
    (tmp_path/"status.json").write_text('{"state":"complete"}')
    (tmp_path/"clips.json").write_text(json.dumps(rows))
    analyze(tmp_path)
    result = json.loads((tmp_path/"analysis/review.json").read_text())
    assert len(result["results"]) == 18
    assert all(np.isfinite(r["mean_error_blocks"]) for r in result["results"])


def test_offset_clock_clips_are_recovered_without_resampling():
    e = entry()
    for f in e['frames']:
        f['elapsed_ticks'] += 6
    windows = list(clips([e]))
    assert len(windows) == 2
    assert windows[0]['ticks'] == list(range(6, 134, 8))
    assert windows[1]['ticks'][0] > windows[0]['ticks'][-1]
