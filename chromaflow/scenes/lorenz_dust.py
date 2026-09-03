"""A scene that is not a wave shape at all: a Lorenz attractor, dusted in.

Shows the other half of the PythonCanvas idea — the canvas does not care where
the vertices came from, so a scene can carry its own state between frames.
"""

import numpy as np

from chromaflow.canvas import SetDrawClass, VertexList


class LorenzDust:
    N = 6000

    def __init__(self, canvas):
        self.canvas = canvas
        rng = np.random.default_rng(3)
        self.p = rng.normal(0.0, 6.0, (self.N, 3)).astype(np.float32)
        self.p[:, 2] += 25.0
        self.vl = VertexList(self.N)

    def Draw(self, fft, time):
        sigma = 10.0
        rho = 24.0 + fft.bass * 14.0
        beta = 8.0 / 3.0
        x, y, z = self.p[:, 0], self.p[:, 1], self.p[:, 2]
        h = 0.0055 * (1.0 + fft.level)
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        self.p[:, 0] = x + dx * h
        self.p[:, 1] = y + dy * h
        self.p[:, 2] = z + dz * h
        np.clip(self.p, -80.0, 80.0, out=self.p)

        spin = time * 0.25
        cs, sn = np.cos(spin), np.sin(spin)
        px = (self.p[:, 0] * cs - self.p[:, 1] * sn) / 26.0
        py = (self.p[:, 2] - 25.0) / 26.0
        self.canvas.SetColor(0.020 + fft.treb * 0.02)
        self.vl.set(px, py)
        self.canvas.DrawPoints(self.vl, size=1.8)


SetDrawClass(LorenzDust)
