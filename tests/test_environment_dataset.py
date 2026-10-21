"""Specification tests only. No real Minecraft rendering is claimed."""
from mc_binding.environment_dataset import specification, route, layout_map
from mc_binding.io import digest


def test_eight_distinct_safe_layouts():
    specs=[specification(i) for i in range(8)]
    assert len({digest(s['landmarks']) for s in specs})==8
    for s in specs:
        assert s['commands'][-1]=='/tp @p 0.5 200 0.5 0 0'
        assert all(not(-7<=b['x']<=7 and 1<=b['z']<=15) for b in s['landmarks'])
        assert all(200<=b['y']<208 for b in s['landmarks'])
    assert route(True)==list(reversed(route(False)))


def test_topdown_specification_not_minecraft_fixture(tmp_path):
    from PIL import Image
    path=tmp_path/'spec.png';layout_map(specification(0),[],path)
    with Image.open(path) as image:assert image.size==(760,850)
