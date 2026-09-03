import numpy as np
import pytest

from chromaflow.analysis import Analyzer
from chromaflow.audio import synth_demo


@pytest.fixture(scope="module")
def track():
    return synth_demo(seconds=6.0, sample_rate=44100)


def run(track, fps=60, seconds=6.0, **kw):
    an = Analyzer(44100, **kw)
    out = []
    dt = 1.0 / fps
    for i in range(int(seconds * fps)):
        t = i * dt
        out.append(an.analyse(an.window_for(track, t), t, dt))
    return out


def test_shapes_and_ranges(track):
    frames = run(track, seconds=2.0)
    f = frames[-1]
    assert f.spec.shape == (512,) and f.wave.shape == (1024,)
    assert np.isfinite(f.spec).all() and np.isfinite(f.wave).all()
    assert 0.0 <= f.beat <= 1.0
    assert 0.0 <= f.hue < 1.0
    assert f.key in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def test_silence_stays_quiet():
    silence = np.zeros(44100 * 2, dtype=np.float32)
    f = run(silence, seconds=1.5)[-1]
    assert f.level < 1e-3
    assert f.mag < 1e-2
    assert f.beat == 0.0


def test_finds_onsets_in_a_track_with_a_beat(track):
    frames = run(track, seconds=6.0)
    onsets = [i for i, f in enumerate(frames) if f.beat > 0.9]
    assert len(onsets) > 8, "a 122 BPM loop should produce plenty of onsets"
    gaps = np.diff(onsets) / 60.0
    assert gaps.min() >= 0.15, "refractory window should stop double triggers"


def test_bass_beats_treble_on_a_low_sine():
    sr = 44100
    t = np.arange(sr * 2) / sr
    low = np.sin(2 * np.pi * 70.0 * t).astype(np.float32)
    f = run(low, seconds=1.8)[-1]
    assert f.bass > f.treb * 2.0


def test_treble_beats_bass_on_a_high_sine():
    sr = 44100
    t = np.arange(sr * 2) / sr
    high = np.sin(2 * np.pi * 7000.0 * t).astype(np.float32)
    f = run(high, seconds=1.8)[-1]
    assert f.treb > f.bass * 2.0


def test_chroma_locks_onto_a_pitch():
    sr = 44100
    t = np.arange(sr * 3) / sr
    a440 = (np.sin(2 * np.pi * 440.0 * t) * 0.7).astype(np.float32)
    f = run(a440, seconds=2.5)[-1]
    assert f.key == "A"


def test_gain_scales_the_spectrum(track):
    quiet = run(track, seconds=1.5, gain=0.5)[-1]
    loud = run(track, seconds=1.5, gain=2.0)[-1]
    assert loud.spec.mean() > quiet.spec.mean() * 2.5
