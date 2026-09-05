from pathlib import Path

import numpy as np
import pytest

from chromaflow.analysis import Analyzer
from chromaflow.audio import synth_demo
from chromaflow.canvas import Canvas
from chromaflow.engine import Visualizer
from chromaflow.presets import by_name
from chromaflow.scene import load_scene

SCENES = sorted((Path(__file__).resolve().parent.parent / "scenes").glob("*.py"))


def test_scene_directory_is_not_empty():
    assert SCENES, "expected example scenes to ship with the package"


@pytest.mark.parametrize("path", SCENES, ids=lambda p: p.stem)
def test_example_scene_draws(path):
    track = synth_demo(seconds=2.0)
    an = Analyzer(44100)
    vis = Visualizer(by_name("Nebula"), 128, 72, particles=500)
    vis.scene = load_scene(path, vis.canvas)
    img = None
    for i in range(20):
        t = i / 30.0
        img = vis.step(an.analyse(an.window_for(track, t), t, 1 / 30.0), 1 / 30.0)
    assert np.isfinite(vis.canvas.plane).all()
    assert vis.canvas.plane.max() > 0.0, f"{path.name} drew nothing"
    assert img.max() > 8


def test_scene_without_setdrawclass_is_reported(tmp_path):
    bad = tmp_path / "bad_scene.py"
    bad.write_text("class Nope:\n    pass\n")
    with pytest.raises(ImportError, match="SetDrawClass"):
        load_scene(bad, Canvas(32, 32))


def test_scene_reloads_when_the_file_changes(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(
        "from chromaflow.canvas import SetDrawClass\n"
        "class S:\n"
        "    tag = 'one'\n"
        "    def __init__(self, canvas): self.canvas = canvas\n"
        "    def Draw(self, fft, time): pass\n"
        "SetDrawClass(S)\n")
    scene = load_scene(src, Canvas(32, 32))
    assert scene.instance.tag == "one"
    src.write_text(src.read_text().replace("'one'", "'two'"))
    import os
    os.utime(src, (0, 0))
    assert scene.poll() is True
    assert scene.instance.tag == "two"
