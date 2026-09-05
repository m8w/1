"""Chromaflow — a music visualizer in the tradition of SoundSpectrum's
G-Force, WhiteCap and Aeon.

Three ideas, borrowed openly and rebuilt from scratch:

* a **flow field** that advects the whole picture every frame,
* a **wave shape** that draws the sound as a parametric curve, and
* a **color map** that turns accumulated intensity into light.

The browser build (``web/index.html``) runs them on the GPU in real time.
This package runs the same design in NumPy, which is slower but renders at any
resolution, offline, and can be scripted — including with scenes written
against a PythonCanvas-shaped drawing API.

No SoundSpectrum code or assets are used.  See CREDITS.md.
"""

__version__ = "1.0.0"

from .analysis import Analyzer, Features        # noqa: F401
from .canvas import Canvas, VertexList, SetDrawClass, GetDrawClass   # noqa: F401
from .engine import Visualizer                  # noqa: F401
from .expr import Expr, ExprError, compile_expr  # noqa: F401
from .presets import PRESETS, Preset, by_name   # noqa: F401

__all__ = [
    "Analyzer", "Features", "Canvas", "VertexList", "SetDrawClass", "GetDrawClass",
    "Visualizer", "Expr", "ExprError", "compile_expr", "PRESETS", "Preset", "by_name",
    "__version__",
]
