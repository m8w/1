"""Color maps.

A G-Force ColorMap was a list of 256 RGB entries; intensity indexed straight
into it.  Same idea here, only the tables are generated from a handful of stops
and the pipeline is linear-light, so entry 0 really is black.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["PALETTES", "build_lut", "hue_rotate", "srgb_to_linear", "linear_to_srgb"]

Stop = Tuple[float, str]

#: The same tables the browser build ships, kept in step by hand.
PALETTES: Dict[str, List[Stop]] = {
    "Ion":    [(0, "#000006"), (.18, "#101a4b"), (.42, "#0f6fb8"), (.66, "#39e0d6"), (.86, "#b6ffe9"), (1, "#ffffff")],
    "Ember":  [(0, "#050002"), (.20, "#38040e"), (.45, "#a81616"), (.68, "#ff7a18"), (.87, "#ffd76a"), (1, "#fffdf0")],
    "Ultra":  [(0, "#04000a"), (.20, "#2a0a52"), (.44, "#8a1fb0"), (.64, "#ff45a3"), (.83, "#66e8ff"), (1, "#ffffff")],
    "Chloro": [(0, "#000603"), (.22, "#04321f"), (.48, "#0f8a3c"), (.70, "#8fe03a"), (.88, "#e8ffa8"), (1, "#ffffff")],
    "Aurora": [(0, "#000308"), (.18, "#04263a"), (.40, "#00b1a0"), (.60, "#5cf07a"), (.78, "#ff77c8"), (1, "#f4f0ff")],
    "Solar":  [(0, "#02000b"), (.16, "#26034d"), (.38, "#8d0b52"), (.60, "#e6431f"), (.80, "#ffb020"), (1, "#fff6d8")],
    "Bone":   [(0, "#010104"), (.25, "#1b2136"), (.52, "#5a6a91"), (.75, "#b8c4e0"), (1, "#ffffff")],
    "Vapor":  [(0, "#05010c"), (.20, "#3d1150"), (.42, "#f0399b"), (.62, "#ffa9d8"), (.80, "#7de8ff"), (1, "#f0ffff")],
    "Rust":   [(0, "#040202"), (.24, "#2e1206"), (.50, "#8f3d10"), (.72, "#d98f3a"), (.90, "#f2dcae"), (1, "#ffffff")],
    "Mono":   [(0, "#000000"), (.55, "#8a8a94"), (1, "#ffffff")],
}


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.power(np.clip(c, 0.0, 1.0), 2.2)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.power(np.clip(c, 0.0, 1.0), 1.0 / 2.2)


def _hex(h: str) -> np.ndarray:
    return np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)], dtype=np.float64) / 255.0


def build_lut(name_or_stops="Ion", size: int = 256) -> np.ndarray:
    """Return a ``(size, 3)`` float32 LUT in **linear** light."""
    stops = PALETTES[name_or_stops] if isinstance(name_or_stops, str) else list(name_or_stops)
    pos = np.array([p for p, _ in stops], dtype=np.float64)
    cols = np.stack([_hex(c) for _, c in stops])
    f = np.linspace(0.0, 1.0, size)
    out = np.stack([np.interp(f, pos, cols[:, i]) for i in range(3)], axis=1)
    return srgb_to_linear(out).astype(np.float32)


def hue_rotate(rgb: np.ndarray, turns: float) -> np.ndarray:
    """Rotate hue about the grey axis, leaving black and white untouched."""
    if not turns:
        return rgb
    k = np.float32(0.57735026919)
    theta = float(turns) * 2.0 * np.pi
    ca, sa = np.cos(theta), np.sin(theta)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dot = (r + g + b) * k
    cross = np.stack([g - b, b - r, r - g], axis=-1) * k
    return rgb * ca + cross * sa + (dot[..., None] * k) * (1.0 - ca)
