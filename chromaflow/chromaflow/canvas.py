"""A drawing surface in the shape of Aeon's PythonCanvas.

SoundSpectrum's PythonCanvas exposed a C++ renderer to an embedded interpreter:
you built a ``VertexList``, handed it to ``DrawLines`` or ``DrawTriangles`` with
a start index, a group size and a stride, and the engine did the rest.  The
contract was designed so that a scene could push a lot of geometry per frame
without paying interpreter cost per vertex.

This is the same contract over NumPy.  A ``VertexList`` is an ``(n, 2)`` array,
the draw calls are whole-array operations, and the surface underneath is a
single intensity plane — an "ink" buffer — which a ColorMap later turns into
light, exactly as G-Force did.  Nothing here needs a GPU, which is what makes
it usable for offline rendering at any resolution.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["VertexList", "Canvas", "BLEND_ADD", "BLEND_MAX", "SetDrawClass", "GetDrawClass"]

BLEND_ADD = "add"
BLEND_MAX = "max"

_registered_draw_class = None


def SetDrawClass(cls):
    """Register the scene class, as a PythonCanvas scene's last line does."""
    global _registered_draw_class
    _registered_draw_class = cls
    return cls


def GetDrawClass():
    """The class most recently passed to :func:`SetDrawClass`."""
    return _registered_draw_class


class VertexList:
    """An ordered list of 2-D vertices, optionally carrying a per-vertex weight.

    Coordinates are in *field space*: y runs -1..1 bottom to top and x runs
    -aspect..aspect, so a scene does not have to know the output resolution.
    """

    __slots__ = ("xy", "weight")

    def __init__(self, n_or_xy, weight: Optional[np.ndarray] = None):
        if isinstance(n_or_xy, int):
            self.xy = np.zeros((n_or_xy, 2), dtype=np.float32)
        else:
            self.xy = np.ascontiguousarray(n_or_xy, dtype=np.float32).reshape(-1, 2)
        if weight is None:
            self.weight = None
        else:
            # An expression with no per-vertex term evaluates to a scalar; carry
            # it as a full-length array so the draw calls need no special case.
            self.weight = np.broadcast_to(
                np.asarray(weight, dtype=np.float32), (len(self.xy),)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.xy)

    @property
    def x(self) -> np.ndarray:
        return self.xy[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xy[:, 1]

    def set(self, x, y, weight=None) -> "VertexList":
        """Fill from two coordinate arrays; returns self so calls can chain."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = max(len(x), len(y))
        if len(self.xy) != n:
            self.xy = np.zeros((n, 2), dtype=np.float32)
        self.xy[:, 0] = x
        self.xy[:, 1] = y
        if weight is not None:
            self.weight = np.broadcast_to(np.asarray(weight, dtype=np.float32), (n,)).copy()
        return self


class Canvas:
    """An additive intensity plane plus the draw calls a scene needs."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.aspect = self.width / self.height
        self.plane = np.zeros((self.height, self.width), dtype=np.float32)
        self.color = 1.0
        self.line_width = 1.0
        self.blend = BLEND_ADD
        self._stack: list = []
        self._m = np.eye(3, dtype=np.float32)

    # -- state ------------------------------------------------------------
    def SetColor(self, intensity: float) -> None:
        """Ink strength for subsequent draws (the ColorMap supplies the hue)."""
        self.color = float(intensity)

    def SetLineWidth(self, w: float) -> None:
        self.line_width = max(0.1, float(w))

    def SetBlend(self, mode: str) -> None:
        self.blend = mode

    def Clear(self, value: float = 0.0) -> None:
        self.plane.fill(value)

    def PushMatrix(self) -> None:
        self._stack.append(self._m.copy())

    def PopMatrix(self) -> None:
        if self._stack:
            self._m = self._stack.pop()

    def LoadIdentity(self) -> None:
        self._m = np.eye(3, dtype=np.float32)

    def Translate(self, dx: float, dy: float) -> None:
        t = np.eye(3, dtype=np.float32)
        t[0, 2], t[1, 2] = dx, dy
        self._m = self._m @ t

    def Scale(self, sx: float, sy: Optional[float] = None) -> None:
        s = np.eye(3, dtype=np.float32)
        s[0, 0], s[1, 1] = sx, sx if sy is None else sy
        self._m = self._m @ s

    def Rotate(self, radians: float) -> None:
        c, s = np.cos(radians), np.sin(radians)
        r = np.eye(3, dtype=np.float32)
        r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
        self._m = self._m @ r

    # -- geometry ---------------------------------------------------------
    def _to_pixels(self, xy: np.ndarray) -> np.ndarray:
        pts = np.empty_like(xy, dtype=np.float32)
        m = self._m
        pts[:, 0] = m[0, 0] * xy[:, 0] + m[0, 1] * xy[:, 1] + m[0, 2]
        pts[:, 1] = m[1, 0] * xy[:, 0] + m[1, 1] * xy[:, 1] + m[1, 2]
        out = np.empty_like(pts)
        out[:, 0] = (pts[:, 0] / self.aspect * 0.5 + 0.5) * (self.width - 1)
        out[:, 1] = (0.5 - pts[:, 1] * 0.5) * (self.height - 1)
        return out

    def _splat(self, px: np.ndarray, weight: np.ndarray, width: float) -> None:
        """Bilinear point splat, widened by a small disc kernel.

        Every contribution is reduced with a single ``bincount`` rather than
        ``np.add.at``; for the hundreds of thousands of splats a frame of
        particles costs, that is the difference between usable and not.
        """
        if len(px) == 0:
            return
        keep = np.isfinite(px).all(axis=1)
        px, weight = px[keep], weight[keep]
        if len(px) == 0:
            return
        radius = int(max(0, np.ceil(width * 0.5 - 0.5)))
        offs = np.array(
            [(dx, dy) for dy in range(-radius, radius + 1)
             for dx in range(-radius, radius + 1)
             if np.hypot(dx, dy) <= width * 0.5 + 0.5], dtype=np.int32)
        if len(offs) == 0:
            offs = np.zeros((1, 2), dtype=np.int32)
        norm = np.float32(1.0 / len(offs))

        x0 = np.floor(px[:, 0]).astype(np.int32)
        y0 = np.floor(px[:, 1]).astype(np.int32)
        fx = (px[:, 0] - x0).astype(np.float32)
        fy = (px[:, 1] - y0).astype(np.float32)
        corners = np.array([(0, 0), (1, 0), (0, 1), (1, 1)], dtype=np.int32)
        cw = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                       (1 - fx) * fy, fx * fy], axis=0)          # (4, n)

        # (4, len(offs), n) -> flat pixel index and weight
        ix = x0[None, None, :] + corners[:, None, 0, None] + offs[None, :, 0, None]
        iy = y0[None, None, :] + corners[:, None, 1, None] + offs[None, :, 1, None]
        wt = (cw[:, None, :] * (weight * norm)[None, None, :])
        ok = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        if not ok.any():
            return
        flat = (iy[ok] * self.width + ix[ok]).astype(np.int64)
        acc = np.bincount(flat, weights=np.broadcast_to(wt, ok.shape)[ok].astype(np.float64),
                          minlength=self.width * self.height)
        self.plane += acc.reshape(self.height, self.width).astype(np.float32)

    def DrawPoints(self, vl: VertexList, size: Optional[float] = None) -> None:
        """Splat every vertex."""
        w = vl.weight if vl.weight is not None else 1.0
        self._splat(self._to_pixels(vl.xy),
                    np.broadcast_to(np.float32(self.color) * w, (len(vl),)).astype(np.float32),
                    self.line_width if size is None else size)

    def DrawLines(self, vl: VertexList, start: int = 0, group: int = 2, stride: int = 2,
                  closed: bool = False) -> None:
        """Draw line segments.

        Mirrors the PythonCanvas call shape: take ``group`` vertices starting at
        ``start``, advance by ``stride``, repeat.  ``group=2, stride=2`` gives
        independent segments; ``group=2, stride=1`` gives a polyline.

        Segments are sampled at roughly one point per pixel of their own length,
        so ink laid down is proportional to length — a stroke does not get
        brighter just because it was cut into more pieces.
        """
        n = len(vl)
        if n < 2:
            return
        idx = np.arange(start, n - group + 1, stride)
        if len(idx) == 0:
            return
        ia, ib = idx, np.minimum(idx + group - 1, n - 1)
        if closed:
            ia = np.append(ia, n - 1)
            ib = np.append(ib, 0)
        pa, pb = self._to_pixels(vl.xy[ia]), self._to_pixels(vl.xy[ib])
        length = np.hypot(pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1])
        length = np.where(np.isfinite(length), length, 0.0)
        counts = np.clip(np.ceil(length), 1, 4096).astype(np.int64)
        total = int(counts.sum())
        if total <= 0:
            return
        budget = 4_000_000
        if total > budget:                       # keep a pathological scene bounded
            counts = np.maximum(1, (counts * (budget / total)).astype(np.int64))
            total = int(counts.sum())
        seg = np.repeat(np.arange(len(counts)), counts)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        u = ((np.arange(total) - starts[seg]) / counts[seg]).astype(np.float32)[:, None]
        pts = pa[seg] * (1.0 - u) + pb[seg] * u
        if vl.weight is not None:
            wa, wb = vl.weight[ia], vl.weight[ib]
            wts = (wa[seg] * (1.0 - u[:, 0]) + wb[seg] * u[:, 0]).astype(np.float32)
        else:
            wts = np.ones(total, dtype=np.float32)
        self._splat(pts, wts * np.float32(self.color), self.line_width)

    def DrawPolyline(self, vl: VertexList, closed: bool = False) -> None:
        """Convenience: ``DrawLines`` with ``group=2, stride=1``."""
        self.DrawLines(vl, 0, 2, 1, closed=closed)

    def DrawTriangles(self, vl: VertexList, start: int = 0, group: int = 3, stride: int = 3) -> None:
        """Fill triangles, scanline-free: barycentric splat over the bounding box.

        Kept simple on purpose — the wave shapes this engine draws are curves,
        and filled geometry is the rarer case.
        """
        n = len(vl)
        idx = np.arange(start, n - group + 1, stride)
        for i in idx:
            tri = self._to_pixels(vl.xy[i:i + 3])
            x0, y0 = np.floor(tri.min(axis=0)).astype(int)
            x1, y1 = np.ceil(tri.max(axis=0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, self.width - 1), min(y1, self.height - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            (ax, ay), (bx, by), (cx, cy) = tri
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-9:
                continue
            l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
            l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
            l3 = 1.0 - l1 - l2
            inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if inside.any():
                self.plane[y0:y1 + 1, x0:x1 + 1][inside] += np.float32(self.color)
