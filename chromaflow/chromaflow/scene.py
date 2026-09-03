"""Scenes: Python files that draw themselves, as PythonCanvas scenes did.

A scene is a module with a class that has ``__init__`` and ``Draw``, registered
on the last line::

    from chromaflow.canvas import *

    class Rings:
        def __init__(self, canvas):
            self.canvas = canvas
            self.vl = VertexList(720)

        def Draw(self, fft, time):
            ...

    SetDrawClass(Rings)

Aeon called ``Draw(self, fft, time)``.  So does this, with ``fft`` carrying far
more than a spectrum — see :class:`~chromaflow.analysis.Features` — and the
canvas handed to ``__init__`` so a scene can size its buffers once.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

from .canvas import Canvas, GetDrawClass, SetDrawClass

__all__ = ["load_scene", "Scene"]


class Scene:
    """A loaded scene file, reloadable in place."""

    def __init__(self, path: str | Path, canvas: Canvas):
        self.path = Path(path)
        self.canvas = canvas
        self.instance: Any = None
        self._mtime = 0.0
        self.reload()

    def reload(self) -> None:
        """(Re)execute the scene file and instantiate its draw class."""
        spec = importlib.util.spec_from_file_location(
            f"chromaflow_scene_{self.path.stem}", self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load scene {self.path}")
        module = importlib.util.module_from_spec(spec)
        SetDrawClass(None)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = GetDrawClass()
        if cls is None:
            raise ImportError(
                f"{self.path.name} never called SetDrawClass(...) — a scene has to "
                "register its draw class on the last line")
        try:
            self.instance = cls(self.canvas)
        except TypeError:
            self.instance = cls()              # PythonCanvas-style zero-arg __init__
            if not hasattr(self.instance, "canvas"):
                self.instance.canvas = self.canvas
        self._mtime = self.path.stat().st_mtime

    def poll(self) -> bool:
        """Reload if the file changed on disk.  Returns True if it did."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if mtime != self._mtime:
            self.reload()
            return True
        return False

    def draw(self, features, time: float) -> None:
        self.instance.Draw(features, time)


def load_scene(path: str | Path, canvas: Optional[Canvas] = None,
               width: int = 960, height: int = 540) -> Scene:
    """Load ``path`` as a scene, creating a canvas if one is not supplied."""
    return Scene(path, canvas or Canvas(width, height))
