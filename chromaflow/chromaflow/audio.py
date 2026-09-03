"""Getting samples in.

Everything downstream wants the same thing: one mono ``float32`` track and a
sample rate.  WAV files are decoded with the standard library; anything else is
handed to ffmpeg if one can be found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = ["load_audio", "find_ffmpeg", "synth_demo"]

_FFMPEG_HINTS = (
    "ffmpeg",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
)


def find_ffmpeg() -> str | None:
    """Return a usable ffmpeg path, or ``None``.

    ``CHROMAFLOW_FFMPEG`` wins if it is set, then ``PATH``, then a few places
    ffmpeg habitually installs itself.
    """
    env = os.environ.get("CHROMAFLOW_FFMPEG")
    if env and Path(env).exists():
        return env
    for hint in _FFMPEG_HINTS:
        found = shutil.which(hint) if os.sep not in hint else (hint if Path(hint).exists() else None)
        if found:
            return found
    return None


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 1:                       # 8-bit WAV is unsigned
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def load_audio(path: str | os.PathLike, sample_rate: int | None = None) -> Tuple[np.ndarray, int]:
    """Load ``path`` as mono float32 in ``[-1, 1]``, optionally resampled.

    Raises ``FileNotFoundError`` if the file is missing and ``RuntimeError`` if
    it is not a WAV and no ffmpeg is available to convert it.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".wav":
        data, sr = _read_wav(p)
    else:
        ff = find_ffmpeg()
        if ff is None:
            raise RuntimeError(
                f"{p.name} is not a WAV and no ffmpeg was found to decode it. "
                "Install ffmpeg, set CHROMAFLOW_FFMPEG, or convert to WAV first."
            )
        target = sample_rate or 44100
        out = subprocess.run(
            [ff, "-v", "error", "-i", str(p), "-f", "f32le", "-ac", "1",
             "-ar", str(target), "-"],
            capture_output=True, check=True,
        )
        data = np.frombuffer(out.stdout, dtype="<f4").copy()
        sr = target
    if sample_rate and sr != sample_rate:
        n = int(round(len(data) * sample_rate / sr))
        data = np.interp(
            np.linspace(0.0, len(data) - 1.0, n, dtype=np.float64),
            np.arange(len(data), dtype=np.float64), data,
        ).astype(np.float32)
        sr = sample_rate
    return data, sr


def synth_demo(seconds: float = 20.0, sample_rate: int = 44100, bpm: float = 122.0) -> np.ndarray:
    """A small generated loop, so the renderer has something to chew on.

    Kick, bass, chord pad and hats on a sixteenth grid — deliberately plain, but
    it has transients and harmony, which is all the analyser needs.
    """
    n = int(seconds * sample_rate)
    out = np.zeros(n, dtype=np.float32)
    spb = 60.0 / bpm
    step = spb / 4.0
    roots = [0, 0, 5, 7, 3, 5, 10, 7]
    rng = np.random.default_rng(7)

    def add(start: float, sig: np.ndarray) -> None:
        i = int(start * sample_rate)
        if i >= n:
            return
        m = min(len(sig), n - i)
        out[i:i + m] += sig[:m]

    def env(length: int, attack: float, decay: float) -> np.ndarray:
        k = np.arange(length, dtype=np.float32) / sample_rate
        a = np.clip(k / max(attack, 1e-4), 0.0, 1.0)
        return (a * np.exp(-k / decay)).astype(np.float32)

    total_steps = int(seconds / step)
    for i in range(total_steps):
        tt = i * step
        s16 = i % 16
        root = roots[(i // 16) % len(roots)]
        if s16 % 4 == 0:                                          # kick
            ln = int(0.42 * sample_rate)
            k = np.arange(ln, dtype=np.float32) / sample_rate
            f = 150.0 * np.exp(-k * 26.0) + 42.0
            add(tt, np.sin(2 * np.pi * np.cumsum(f) / sample_rate) * env(ln, 0.002, 0.09) * 0.9)
        if s16 in (4, 12):                                        # snare
            ln = int(0.22 * sample_rate)
            add(tt, rng.standard_normal(ln).astype(np.float32) * env(ln, 0.001, 0.05) * 0.35)
        if s16 % 2 == 0:                                          # hat
            ln = int(0.07 * sample_rate)
            h = rng.standard_normal(ln).astype(np.float32)
            h = np.diff(h, prepend=np.float32(0.0))               # crude high-pass
            add(tt, h * env(ln, 0.001, 0.012) * 0.16)
        if s16 % 2 == 0:                                          # bass
            ln = int(step * 1.7 * sample_rate)
            hz = 55.0 * 2 ** ((root + 12) / 12.0)
            k = np.arange(ln, dtype=np.float32) / sample_rate
            add(tt, (2.0 * np.mod(hz * k, 1.0) - 1.0) * env(ln, 0.005, 0.16) * 0.30)
        if s16 == 0:                                              # pad
            ln = int(spb * 3.6 * sample_rate)
            k = np.arange(ln, dtype=np.float32) / sample_rate
            chord = np.zeros(ln, dtype=np.float32)
            for iv in (0, 3, 7, 10, 14):
                hz = 55.0 * 2 ** ((root + iv + 24) / 12.0)
                chord += np.sin(2 * np.pi * hz * k).astype(np.float32)
            shape = np.clip(k / (spb * 1.2), 0, 1) * np.exp(-k / (spb * 2.4))
            add(tt, chord * shape.astype(np.float32) * 0.045)

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.85).astype(np.float32)
