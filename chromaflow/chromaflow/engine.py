"""The renderer.

Three passes, in the order G-Force ran them:

1. **Flow field** — the whole plane is advected.  The expressions give a
   velocity at each destination pixel; the plane is resampled from
   ``p - v*dt``, which is a semi-Lagrangian step and is why the picture
   smears into itself instead of jittering.
2. **Wave shape** — the audio is drawn as a parametric curve through
   :class:`~chromaflow.canvas.Canvas`, adding ink on top.
3. **Color map** — accumulated ink indexes a 256-entry palette, then bloom,
   tone mapping and the small dirt (grain, vignette, mirror) that makes it
   look like something rather than a plot.

Particles ride the same field between 1 and 2.  Everything is NumPy, so this
runs anywhere Python does; the browser build is the same design on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .analysis import Features
from .canvas import Canvas, VertexList
from .expr import Expr, compile_expr
from .palette import build_lut, hue_rotate, linear_to_srgb
from .presets import DEFAULTS, Preset

__all__ = ["Visualizer"]

_WAVE_INK = 0.018        # ink laid per pixel of wave-shape length, per frame
_PART_PIXEL_INK = 0.004  # ink laid per screen pixel by the whole particle set


def _aces(x: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _bilinear(src: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``src`` at fractional pixel coordinates, zero outside."""
    h, w = src.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = (xs - x0).astype(np.float32)
    fy = (ys - y0).astype(np.float32)
    inside = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    xc = np.clip(x0, 0, w - 2)
    yc = np.clip(y0, 0, h - 2)
    a = src[yc, xc]
    b = src[yc, xc + 1]
    c = src[yc + 1, xc]
    d = src[yc + 1, xc + 1]
    out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    return np.where(inside, out, 0.0).astype(np.float32)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via cumulative sums — O(n) and good enough for bloom."""
    if radius < 1:
        return img
    out = img
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        p = np.pad(out, pad, mode="edge")
        cs = np.cumsum(p, axis=axis, dtype=np.float32)
        lo = np.take(cs, np.arange(0, out.shape[axis]), axis=axis)
        hi = np.take(cs, np.arange(2 * radius + 1, out.shape[axis] + 2 * radius + 1), axis=axis)
        out = (hi - lo) / np.float32(2 * radius + 1)
    return out


class Visualizer:
    """Render a preset, frame by frame, driven by :class:`~chromaflow.analysis.Features`."""

    def __init__(self, preset: Preset, width: int = 960, height: int = 540,
                 particles: int = 40000, seed: int = 7,
                 overrides: Optional[Dict[str, float]] = None,
                 scene: object | None = None):
        self.preset = preset
        #: An optional PythonCanvas-style scene drawn in place of the wave shape.
        self.scene = scene
        self.p: Dict[str, float] = preset.settings()
        if overrides:
            self.p.update(overrides)
        self.width, self.height = int(width), int(height)
        self.aspect = self.width / self.height
        self.canvas = Canvas(self.width, self.height)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.pal_shift = 0.0
        self.lut = build_lut(str(self.p["pal"]))

        self.fx: Expr = compile_expr(preset.fx)
        self.fy: Expr = compile_expr(preset.fy)
        self.wx: Expr = compile_expr(preset.wx)
        self.wy: Expr = compile_expr(preset.wy)
        self.wi: Expr = compile_expr(preset.wi)

        ys, xs = np.mgrid[0:self.height, 0:self.width]
        self.X = ((xs / (self.width - 1)) * 2.0 - 1.0).astype(np.float32) * np.float32(self.aspect)
        self.Y = (1.0 - (ys / (self.height - 1)) * 2.0).astype(np.float32)
        self.R = np.hypot(self.X, self.Y).astype(np.float32)
        self.TH = np.arctan2(self.Y, self.X).astype(np.float32)

        self.n_particles = int(particles)
        ang = self.rng.uniform(0, 2 * np.pi, self.n_particles)
        rad = np.sqrt(self.rng.uniform(0, 1, self.n_particles)) * 0.9
        self.pp = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1).astype(np.float32)
        self.pv = np.stack([-np.sin(ang), np.cos(ang)], 1).astype(np.float32) * 0.2
        self.page = self.rng.uniform(0.2, 3.0, self.n_particles).astype(np.float32)

    # ------------------------------------------------------------------
    def _env(self, f: Features, **extra) -> Dict[str, object]:
        n_s, n_w = len(f.spec), len(f.wave)
        env: Dict[str, object] = {
            "t": np.float32(self.t), "mag": f.mag, "bass": f.bass, "mid": f.mid,
            "treb": f.treb, "level": f.level, "beat": f.beat, "asp": self.aspect,
            "uBeatPhase": f.beat_phase, "uBPM": f.bpm, "uHue": f.hue,
            "phase": f.beat_phase, "bpm": f.bpm, "hue": f.hue,
            "spec": lambda q: np.interp(np.clip(q, 0, 1) * (n_s - 1), np.arange(n_s), f.spec),
            "wave": lambda q: np.interp(np.clip(q, 0, 1) * (n_w - 1), np.arange(n_w), f.wave),
        }
        for i in range(8):
            env[f"a{i}"] = 0.0
        env.update(extra)
        return env

    def _advect(self, f: Features, dt: float) -> None:
        p = self.p
        env = self._env(f, x=self.X, y=self.Y, r=self.R, th=self.TH,
                        s=np.float32(0.0), w=np.float32(0.0))
        # An expression with no positional term evaluates to a scalar; broadcast
        # so downstream code never has to care which it got.
        vx = np.broadcast_to(np.asarray(self.fx(env), dtype=np.float32), self.X.shape)
        vy = np.broadcast_to(np.asarray(self.fy(env), dtype=np.float32), self.Y.shape)
        flow = float(p["flow"]) * (1.0 + f.beat * float(p["kick"])) * dt
        sx = self.X - vx * flow
        sy = self.Y - vy * flow
        spin = float(p["spin"]) * dt
        cs, sn = np.cos(spin), np.sin(spin)
        zoom = 1.0 - float(p["zoom"]) * dt
        rx = (cs * sx - sn * sy) * zoom
        ry = (sn * sx + cs * sy) * zoom
        px = (rx / self.aspect * 0.5 + 0.5) * (self.width - 1)
        py = (0.5 - ry * 0.5) * (self.height - 1)
        plane = _bilinear(self.canvas.plane, px, py)
        # Conserve ink across the warp: where the field spreads the picture out,
        # the same ink now covers more pixels, so the gathered value comes down
        # by the Jacobian of the map.  Without this an outward flow multiplies
        # brightness every frame and the feedback saturates to white.
        step_x = 2.0 * self.aspect / max(1, self.width - 1)
        step_y = 2.0 / max(1, self.height - 1)
        div = (np.gradient(vx, step_x, axis=1) - np.gradient(vy, step_y, axis=0))
        jac = np.clip(1.0 - div * flow - float(p["zoom"]) * dt * 2.0, 0.0, 2.0).astype(np.float32)
        self.canvas.plane = plane * jac * np.float32(float(p["decay"]) ** (dt * 60.0))

    def _draw_wave(self, f: Features, dt: float) -> None:
        p = self.p
        steps = int(max(8, p["steps"]))
        s = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        w = np.interp(s * (len(f.wave) - 1), np.arange(len(f.wave)), f.wave).astype(np.float32)
        w = w * np.float32(p["wave"])
        th = s * np.float32(2.0 * np.pi)
        env = self._env(f, s=s, w=w, r=s, th=th, x=np.cos(th), y=np.sin(th))
        xs = np.broadcast_to(np.asarray(self.wx(env), dtype=np.float32), s.shape).copy()
        ys = np.broadcast_to(np.asarray(self.wy(env), dtype=np.float32), s.shape).copy()
        env_i = self._env(f, s=s, w=w, x=xs, y=ys, r=np.hypot(xs, ys), th=np.arctan2(ys, xs))
        ink = np.maximum(np.asarray(self.wi(env_i), dtype=np.float32), 0.0)
        vl = VertexList(np.stack([xs, ys], 1), weight=ink)
        self.canvas.SetLineWidth(float(p["lw"]))
        self.canvas.SetColor(_WAVE_INK * float(p["ink"]) * dt * 60.0)
        self.canvas.DrawPolyline(vl)

    def _step_particles(self, f: Features, dt: float) -> None:
        p = self.p
        if self.n_particles <= 0 or p["pcount"] <= 0:
            return
        px, py = self.pp[:, 0], self.pp[:, 1]
        r = np.hypot(px, py)
        env = self._env(f, x=px, y=py, r=r, th=np.arctan2(py, px),
                        s=self.page, w=np.float32(0.0))
        ax = np.broadcast_to(np.asarray(self.fx(env), dtype=np.float32), px.shape)
        ay = np.broadcast_to(np.asarray(self.fy(env), dtype=np.float32), py.shape)
        flow = float(p["flow"]) * 0.8
        acc = np.stack([ax, ay], 1) * np.float32(flow)
        # curl of a value-noise field, which is divergence free and so looks fluid
        eps = np.float32(0.08)
        from .expr import _value_noise
        n_up = _value_noise(px * 1.6, py * 1.6 + eps + self.t * 0.09)
        n_dn = _value_noise(px * 1.6, py * 1.6 - eps + self.t * 0.09)
        n_rt = _value_noise(px * 1.6 + eps, py * 1.6 + self.t * 0.09)
        n_lf = _value_noise(px * 1.6 - eps, py * 1.6 + self.t * 0.09)
        curl = np.stack([(n_up - n_dn), (n_lf - n_rt)], 1) / (2.0 * eps)
        acc += curl.astype(np.float32) * np.float32(p["curl"] * (0.35 + f.mid * 2.2))
        acc += np.stack([-py, px], 1) * np.float32(p["spin"] * 0.4)
        wall = np.clip((r - 0.75) / 0.85, 0.0, 1.0) ** 2 * (3.0 - 2.0 * np.clip((r - 0.75) / 0.85, 0, 1))
        acc -= self.pp * (np.float32(0.30 + f.bass * 0.6) * wall.astype(np.float32))[:, None]
        acc *= np.float32(p["pforce"])

        self.pv += acc * np.float32(dt)
        self.pv *= np.float32(float(p["drag"]) ** (dt * 60.0))
        speed = np.hypot(self.pv[:, 0], self.pv[:, 1])
        over = speed > 6.0
        if over.any():
            self.pv[over] *= (6.0 / speed[over])[:, None]
        self.pp += self.pv * np.float32(dt)

        self.page -= dt * (0.13 + 0.5 * np.abs(np.sin(np.arange(self.n_particles) * 3.7)))
        dead = (self.page <= 0) | (np.abs(self.pp[:, 0]) > self.aspect * 1.6) | (np.abs(self.pp[:, 1]) > 1.6)
        n_dead = int(dead.sum())
        if n_dead:
            a = self.rng.uniform(0, 2 * np.pi, n_dead)
            rad = float(p["spawn"]) * (0.05 + 1.15 * self.rng.uniform(0, 1, n_dead) ** 0.6)
            rad *= 0.55 + f.beat * 0.8
            self.pp[dead] = np.stack([np.cos(a) * rad, np.sin(a) * rad], 1).astype(np.float32)
            self.pv[dead] = (np.stack([-np.sin(a), np.cos(a)], 1)
                             * 0.25 * (0.4 + f.level * 2.0)).astype(np.float32)
            self.page[dead] = self.rng.uniform(0.7, 3.3, n_dead).astype(np.float32)

        vl = VertexList(self.pp)
        # A splat deposits its colour once, however wide it is, so ink per
        # particle has to track pixels-per-particle if brightness is to stay put
        # as either the resolution or the particle count changes.
        per_particle = (self.width * self.height) / max(1, self.n_particles)
        self.canvas.SetColor(_PART_PIXEL_INK * float(p["ink"]) * dt * 60.0 * per_particle)
        self.canvas.DrawPoints(vl, size=float(p["psize"]))

    def _mirror_map(self, segments: int) -> tuple:
        """Cached kaleidoscope gather indices for ``segments`` wedges."""
        if getattr(self, "_mirror_cache", (None,))[0] == segments:
            return self._mirror_cache[1]
        seg = 2.0 * np.pi / segments
        a = np.abs(np.mod(self.TH, seg) - seg * 0.5)
        mx = np.cos(a) * self.R
        my = np.sin(a) * self.R
        px = np.clip((mx / self.aspect * 0.5 + 0.5) * (self.width - 1), 0, self.width - 1)
        py = np.clip((0.5 - my * 0.5) * (self.height - 1), 0, self.height - 1)
        idx = (py.astype(np.int32), px.astype(np.int32))
        self._mirror_cache = (segments, idx)
        return idx

    def _colorize(self, f: Features, dt: float) -> np.ndarray:
        p = self.p
        self.pal_shift = (self.pal_shift + dt * float(p["palSpin"])) % 1.0
        shift = (self.pal_shift + float(p["chroma"]) * f.hue) % 1.0
        plane = self.canvas.plane
        segments = int(p["mirror"])
        if segments >= 3:
            plane = plane[self._mirror_map(segments)]
        idx = np.clip(plane * np.float32(p["palScale"]), 0.0, 1.0)
        col = self.lut[(idx * 255.0).astype(np.int32)]
        col = hue_rotate(col, shift).astype(np.float32)
        col *= np.float32(p["exposure"])

        if p["bloom"] > 0.001:
            lum = col.mean(axis=2)
            thresh = np.maximum(lum - 0.55, 0.0)
            small = thresh[::4, ::4]
            glow = _box_blur(_box_blur(small, 3), 7)
            glow = np.repeat(np.repeat(glow, 4, axis=0), 4, axis=1)
            glow = glow[: self.height, : self.width]
            if glow.shape != lum.shape:
                pad = ((0, self.height - glow.shape[0]), (0, self.width - glow.shape[1]))
                glow = np.pad(glow, pad, mode="edge")
            glow = _box_blur(glow, 3)      # hide the nearest-neighbour upsample
            col = col + glow[..., None] * np.float32(p["bloom"] * 0.7)

        col = col * np.float32(1.0 + f.beat * float(p["kick"]) * 0.5)
        col = _aces(col)
        if p["vig"] > 0:
            d = np.clip((1.25 - self.R) / 1.0, 0.0, 1.0)
            v = d * d * (3.0 - 2.0 * d)
            col *= (1.0 - float(p["vig"]) + float(p["vig"]) * v)[..., None]
        out = linear_to_srgb(col)
        if p["grain"] > 0:
            out = out + (self.rng.random(out.shape[:2], dtype=np.float32)[..., None] - 0.5) * np.float32(
                p["grain"] * 0.05)
        return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def step(self, f: Features, dt: float) -> np.ndarray:
        """Advance one frame and return an ``(H, W, 3)`` uint8 image."""
        dt = float(max(1e-4, min(0.1, dt)))
        self.t += dt
        self._advect(f, dt)
        self._step_particles(f, dt)
        if self.scene is not None:
            self.canvas.LoadIdentity()
            self.canvas.SetColor(1.0)
            self.canvas.SetLineWidth(float(self.p["lw"]))
            self.scene.draw(f, self.t)
        else:
            self._draw_wave(f, dt)
        # A pathological config can otherwise run the plane away to infinity.
        np.clip(self.canvas.plane, 0.0, 6.0, out=self.canvas.plane)
        return self._colorize(f, dt)
