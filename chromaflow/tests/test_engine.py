import numpy as np
import pytest

from chromaflow.analysis import Analyzer
from chromaflow.audio import synth_demo
from chromaflow.canvas import Canvas, VertexList
from chromaflow.engine import Visualizer
from chromaflow.palette import PALETTES, build_lut, hue_rotate
from chromaflow.presets import PRESETS, by_name, load_config


@pytest.fixture(scope="module")
def track():
    return synth_demo(seconds=4.0, sample_rate=44100)


def render(preset, track, frames=40, w=160, h=90, particles=2000, **kw):
    an = Analyzer(44100)
    vis = Visualizer(preset, w, h, particles=particles, **kw)
    img = None
    for i in range(frames):
        t = i / 30.0
        img = vis.step(an.analyse(an.window_for(track, t), t, 1 / 30.0), 1 / 30.0)
    return vis, img


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_every_preset_renders_something_finite_and_not_blown_out(preset, track):
    vis, img = render(preset, track)
    assert img.shape == (90, 160, 3) and img.dtype == np.uint8
    assert np.isfinite(vis.canvas.plane).all()
    lit = float((img.max(axis=2) > 8).mean())
    assert 0.02 < lit < 0.995, f"{preset.name}: {lit:.1%} of the frame is lit"


def test_feedback_plane_stays_bounded(track):
    # the flow field can amplify; the engine must not run away over a long take
    vis, _ = render(by_name("Hyperdrive"), track, frames=200)
    assert vis.canvas.plane.max() <= 6.0 + 1e-3


def test_the_audio_is_what_drives_the_picture(track):
    # Same preset, same seed, same frame count: only the sound differs, so any
    # difference in the output is the analysis actually reaching the render.
    silence = np.zeros(44100 * 3, dtype=np.float32)
    _, quiet = render(by_name("Nebula"), silence, frames=40)
    _, loud = render(by_name("Nebula"), track, frames=40)
    delta = np.abs(quiet.astype(np.int16) - loud.astype(np.int16)).mean()
    assert delta > 8, f"audio barely moved the image (mean delta {delta:.1f})"


def test_silence_produces_a_still_picture(track):
    silence = np.zeros(44100 * 3, dtype=np.float32)
    an = Analyzer(44100)
    vis = Visualizer(by_name("Skyline"), 160, 90, particles=1000)
    for i in range(60):
        t = i / 30.0
        prev = vis.step(an.analyse(an.window_for(silence, t), t, 1 / 30.0), 1 / 30.0).copy()
    last = vis.step(an.analyse(an.window_for(silence, 2.0), 2.0, 1 / 30.0), 1 / 30.0)
    # no onsets, so nothing should kick; the residual drift is just the flow field
    assert np.abs(last.astype(np.int16) - prev.astype(np.int16)).mean() < 12


def test_deterministic_for_a_given_seed(track):
    _, a = render(by_name("Vortex"), track, frames=20, seed=11)
    _, b = render(by_name("Vortex"), track, frames=20, seed=11)
    # grain is the only stochastic term and it comes from the same seeded rng
    assert np.array_equal(a, b)


def test_overrides_reach_the_render(track):
    _, dark = render(by_name("Nebula"), track, frames=30, overrides={"exposure": 0.2})
    _, bright = render(by_name("Nebula"), track, frames=30, overrides={"exposure": 3.0})
    assert bright.mean() > dark.mean() * 2


def test_palette_lut_is_linear_light_and_starts_black():
    lut = build_lut("Ion")
    assert lut.shape == (256, 3)
    assert lut[0].max() < 0.01, "entry 0 must be black or the background glows"
    assert lut[-1].min() > 0.8
    assert np.all(np.diff(lut.sum(axis=1)) >= -1e-3), "ramp should be monotone in brightness"


def test_hue_rotation_preserves_greys():
    grey = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    assert hue_rotate(grey, 0.3) == pytest.approx(grey, abs=1e-5)


def test_every_palette_parses():
    for name in PALETTES:
        assert build_lut(name).shape == (256, 3)


def test_canvas_draws_where_it_is_told():
    c = Canvas(101, 101)
    c.SetColor(1.0)
    c.DrawPoints(VertexList([[0.0, 0.0]]))
    ys, xs = np.nonzero(c.plane > 0)
    assert xs.mean() == pytest.approx(50, abs=1.5)
    assert ys.mean() == pytest.approx(50, abs=1.5)


def test_line_ink_scales_with_length_not_vertex_count():
    def ink(n):
        c = Canvas(200, 200)
        c.SetColor(1.0)
        xs = np.linspace(-0.9, 0.9, n)
        c.DrawPolyline(VertexList(np.stack([xs, np.zeros(n)], 1)))
        return float(c.plane.sum())
    assert ink(200) == pytest.approx(ink(20), rel=0.08)


def test_transform_stack_round_trips():
    c = Canvas(64, 64)
    before = c._m.copy()
    c.PushMatrix()
    c.Translate(0.5, -0.25)
    c.Rotate(1.1)
    c.Scale(2.0)
    c.PopMatrix()
    assert np.allclose(c._m, before)


def test_config_file_round_trip(tmp_path):
    cfg = tmp_path / "x.cfg"
    cfg.write_text(
        "name = Test\n"
        "palette = Ember\n"
        "flow.dx = -y*0.5 \\\n"
        "          - x*0.1\n"
        "flow.dy = x*0.5\n"
        "wave.x = cos(s*tau)*0.5\n"
        "wave.y = sin(s*tau)*0.5\n"
        "wave.ink = 0.5 + treb\n"
        "decay = 0.95   # comment\n"
    )
    p = load_config(cfg)
    assert p.name == "Test" and p.pal == "Ember"
    assert p.fx == "-y*0.5 - x*0.1"
    assert p.settings()["decay"] == pytest.approx(0.95)


def test_config_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "bad.cfg"
    cfg.write_text("wobble = 3\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(cfg)
