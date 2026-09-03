"""A PythonCanvas-shaped scene: concentric rings that breathe with the bands.

Run it with:

    python -m chromaflow still --scene scenes/rings.py -o rings.png
"""

import numpy as np

from chromaflow.canvas import SetDrawClass, VertexList


class Rings:
    """Aeon's contract: __init__ sets the scene up, Draw runs once per frame."""

    STEPS = 512
    COUNT = 7

    def __init__(self, canvas):
        self.canvas = canvas
        self.s = np.linspace(0.0, 1.0, self.STEPS, dtype=np.float32)
        self.th = self.s * np.float32(2.0 * np.pi)
        self.vl = VertexList(self.STEPS)

    def Draw(self, fft, time):
        c = self.canvas
        c.SetLineWidth(1.6 + fft.beat * 2.5)
        for i in range(self.COUNT):
            k = i / (self.COUNT - 1.0)
            # each ring reads a different slice of the spectrum
            band = float(np.interp(k, [0, 1], [0.05, 0.75]))
            lo = int(band * (len(fft.spec) - 1))
            energy = float(fft.spec[lo:lo + 24].mean())
            radius = 0.16 + k * 0.72 + energy * 0.22
            wob = np.sin(self.th * (3 + i) + time * (0.4 + 0.25 * i)) * (0.02 + energy * 0.09)
            r = radius + wob
            self.vl.set(np.cos(self.th) * r, np.sin(self.th) * r,
                        weight=0.25 + energy * 1.9)
            c.SetColor(0.030)
            c.DrawPolyline(self.vl, closed=True)


SetDrawClass(Rings)
