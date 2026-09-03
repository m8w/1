"""What the music is doing, frame by frame.

G-Force fed its expressions a single smoothed magnitude, ``mag``.  Thirty years
on we can afford more: log-spaced bands, an onset detector with an adaptive
threshold, a tempo estimate, and a chroma vector that tracks the key — all in
NumPy, all cheap enough to run per video frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["Features", "Analyzer", "NOTE_NAMES"]

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class Features:
    """One frame's worth of analysis."""

    spec: np.ndarray          #: log-spaced magnitudes, 0..1-ish
    wave: np.ndarray          #: decimated waveform, -1..1
    bass: float = 0.0
    mid: float = 0.0
    treb: float = 0.0
    level: float = 0.0
    mag: float = 0.0
    beat: float = 0.0         #: 1.0 on an onset, decaying
    beat_phase: float = 0.0
    bpm: float = 0.0
    hue: float = 0.0          #: key colour, 0..1 around the circle of fifths
    key: str = "—"
    chroma: np.ndarray = field(default_factory=lambda: np.zeros(12))


class Analyzer:
    """Streaming analyser.  Feed it a window per frame, get :class:`Features`.

    The state that has to persist between frames — smoothed bands, the spectral
    flux history the onset detector thresholds against, the tempo estimate —
    lives here, so calling :meth:`analyse` in order is what makes beats work.
    """

    def __init__(self, sample_rate: int, fft_size: int = 4096, spec_bins: int = 512,
                 wave_bins: int = 1024, gain: float = 1.6, attack: float = 0.55,
                 release: float = 0.10, beat_threshold: float = 1.35):
        self.sr = sample_rate
        self.fft_size = fft_size
        self.spec_bins = spec_bins
        self.wave_bins = wave_bins
        self.gain = gain
        self.attack = attack
        self.release = release
        self.beat_threshold = beat_threshold

        self._window = np.hanning(fft_size).astype(np.float32)
        self._spec_sm = np.zeros(spec_bins, dtype=np.float32)
        self._prev = np.zeros(spec_bins, dtype=np.float32)
        self._flux_hist: list[float] = []
        self._last_beat = -9.0
        self._period = 0.5
        self._beat = 0.0
        self._chroma_sm = np.zeros(12, dtype=np.float32)
        self._smooth = {"bass": 0.0, "mid": 0.0, "treb": 0.0, "level": 0.0, "mag": 0.0}
        self._hue = 0.0

        nyq = sample_rate / 2.0
        bin_hz = nyq / (fft_size // 2)
        f_min, f_max = 26.0, min(17000.0, nyq)
        edges = np.geomspace(f_min, f_max, spec_bins + 1)
        # Contiguous, non-decreasing bin edges so one reduceat covers every band.
        idx = np.clip(np.round(edges / bin_hz).astype(int), 0, fft_size // 2 - 1)
        self._seg = np.maximum.accumulate(idx)[:-1]
        centres = np.sqrt(edges[:-1] * edges[1:])
        self._is_bass = centres < 170.0
        self._is_mid = (centres >= 170.0) & (centres < 2200.0)
        self._is_treb = centres >= 2200.0
        # pitch class of every FFT bin, for chroma
        freqs = np.arange(fft_size // 2) * bin_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            pc = np.round(12 * np.log2(np.where(freqs > 0, freqs, 1.0) / 440.0))
        self._pitch_class = ((pc.astype(int) % 12) + 21) % 12
        self._chroma_mask = (freqs >= 55.0) & (freqs <= 2100.0)

    # ------------------------------------------------------------------
    def analyse(self, window: np.ndarray, t: float, dt: float) -> Features:
        """Analyse one window of samples ending at time ``t``."""
        buf = np.zeros(self.fft_size, dtype=np.float32)
        take = min(len(window), self.fft_size)
        if take:
            buf[-take:] = window[-take:]

        mags = np.abs(np.fft.rfft(buf * self._window)[: self.fft_size // 2])
        db = 20.0 * np.log10(np.maximum(mags * (2.0 / self.fft_size), 1e-9))

        # log-spaced bands: peak-hold inside each band, then dB -> 0..1.
        # Where a band is narrower than an FFT bin the segment is empty and
        # reduceat returns that bin's own value, which is what we want.
        peaks = np.maximum.reduceat(db, self._seg)
        v = np.clip((peaks + 96.0) / 84.0, 0.0, 1.4).astype(np.float32) * self.gain

        rising = v > self._spec_sm
        k = np.where(rising, self.attack, self.release).astype(np.float32)
        self._spec_sm += (v - self._spec_sm) * k
        spec = self._spec_sm.copy()

        flux = float(np.sum(np.maximum(v - self._prev, 0.0)) / self.spec_bins)
        self._prev = v

        wave = np.interp(
            np.linspace(0, self.fft_size - 1, self.wave_bins),
            np.arange(self.fft_size), buf,
        ).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(buf))))

        def band(mask):
            return float(v[mask].mean()) if mask.any() else 0.0

        tgt = {
            "bass": band(self._is_bass), "mid": band(self._is_mid),
            "treb": band(self._is_treb), "level": rms * 2.6 * self.gain,
        }
        for name, target in tgt.items():
            rate = min(1.0, dt * (16.0 if name == "treb" else 14.0))
            self._smooth[name] += (target - self._smooth[name]) * rate
        mag_t = min(1.6, (self._smooth["bass"] * 0.55 + self._smooth["mid"] * 0.30
                          + self._smooth["treb"] * 0.15) * 1.5)
        self._smooth["mag"] += (mag_t - self._smooth["mag"]) * min(1.0, dt * 18.0)

        # onset: flux above a running mean + k*sd, with a refractory window
        self._flux_hist.append(flux)
        if len(self._flux_hist) > 240:
            del self._flux_hist[0]
        hist = np.asarray(self._flux_hist[-120:], dtype=np.float64)
        mean, sd = (float(hist.mean()), float(hist.std())) if len(hist) > 8 else (flux, 1.0)
        if (flux > mean + sd * self.beat_threshold and t - self._last_beat > 0.19
                and flux > 1.5e-3):
            gap = t - self._last_beat
            if 0.25 < gap < 2.0:
                self._period = self._period * 0.82 + gap * 0.18
            self._last_beat = t
            self._beat = 1.0
        self._beat = max(0.0, self._beat - dt * 3.4)

        # chroma -> key -> hue, moving around the circle of fifths
        lin = np.maximum((db + 96.0) / 84.0, 0.0)
        chroma = np.bincount(
            self._pitch_class[self._chroma_mask],
            weights=np.square(lin[self._chroma_mask]), minlength=12,
        ).astype(np.float32)
        self._chroma_sm = self._chroma_sm * 0.90 + chroma * 0.10
        ci = int(np.argmax(self._chroma_sm))
        target_hue = ((ci * 7) % 12) / 12.0
        dh = target_hue - self._hue
        dh -= round(dh)
        self._hue = (self._hue + dh * min(1.0, dt * 1.4)) % 1.0

        return Features(
            spec=spec, wave=wave,
            bass=self._smooth["bass"], mid=self._smooth["mid"], treb=self._smooth["treb"],
            level=self._smooth["level"], mag=self._smooth["mag"],
            beat=self._beat,
            beat_phase=min(1.0, (t - self._last_beat) / max(0.12, self._period)),
            bpm=60.0 / max(0.2, self._period),
            hue=self._hue, key=NOTE_NAMES[ci], chroma=self._chroma_sm.copy(),
        )

    def window_for(self, samples: np.ndarray, t: float) -> np.ndarray:
        """The ``fft_size`` samples ending at time ``t`` in ``samples``."""
        end = int(t * self.sr)
        start = max(0, end - self.fft_size)
        return samples[start:max(start, end)]
