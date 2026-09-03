"""The stock looks, mirroring the browser build's presets one for one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["Preset", "PRESETS", "DEFAULTS", "by_name"]

#: Every knob, with the value it takes unless a preset says otherwise.
DEFAULTS: Dict[str, float | str] = {
    "flow": 1.0, "decay": 0.965, "spin": 0.15, "zoom": 0.02,
    "wave": 1.0, "steps": 1536, "lw": 1.8, "ink": 0.9,
    "pcount": 4, "pforce": 1.0, "psize": 2.2, "curl": 1.0, "drag": 0.965, "spawn": 1.0,
    "pal": "Ion", "palSpin": 0.04, "palScale": 1.35, "chroma": 0.35, "colorMix": 0.9,
    "exposure": 1.45, "bloom": 0.95, "aberr": 0.35, "grain": 0.22, "vig": 0.55, "mirror": 0,
    "gain": 1.6, "kick": 0.7,
}


@dataclass
class Preset:
    """A flow field, a wave shape, an ink rule, a palette and some knobs."""

    name: str
    fx: str
    fy: str
    wx: str
    wy: str
    wi: str
    pal: str = "Ion"
    params: Dict[str, float] = field(default_factory=dict)

    def settings(self) -> Dict[str, float | str]:
        out = dict(DEFAULTS)
        out["pal"] = self.pal
        out.update(self.params)
        return out


PRESETS: List[Preset] = [
    Preset("Nebula", pal="Ion",
           fx="-y*0.30 + noise(x*1.1 + t*0.07, y*1.1)*0.95*(0.35 + mag)",
           fy=" x*0.30 + noise(y*1.1 - t*0.06, x*1.1)*0.95*(0.35 + mag)",
           wx="cos(s*tau)*(0.52 + w*0.55 + 0.10*sin(s*tau*5 - t*1.1))",
           wy="sin(s*tau)*(0.52 + w*0.55 + 0.10*cos(s*tau*7 + t*0.9))",
           wi="0.45 + treb*1.1",
           params={"spin": 0.18, "decay": 0.968, "bloom": 0.95}),
    Preset("Vortex", pal="Ultra",
           fx="-y*(1.05 + bass*1.5) - x*0.22",
           fy=" x*(1.05 + bass*1.5) - y*0.22",
           wx="cos(s*tau*3 + t*0.55)*(0.12 + s*0.82)*(1.0 + w*0.7)",
           wy="sin(s*tau*3 + t*0.55)*(0.12 + s*0.82)*(1.0 + w*0.7)",
           wi="0.35 + mag*1.2",
           params={"decay": 0.972, "zoom": -0.05, "psize": 1.8, "aberr": 0.5}),
    Preset("Lissajous", pal="Vapor",
           fx="sin(y*3.1 + t*0.45)*0.55",
           fy="sin(x*2.7 - t*0.38)*0.55",
           wx="sin(s*tau*3 + t*0.33)*(0.78 + w*0.45)",
           wy="sin(s*tau*4 + t*0.49 + 1.2)*(0.78 + w*0.45)",
           wi="0.5 + level*1.4",
           params={"spin": 0.0, "decay": 0.958, "lw": 2.4, "pcount": 3}),
    Preset("Spectrum ring", pal="Aurora",
           fx="x/max(r,0.03)*sin(r*8.0 - t*2.0)*0.55*(0.25 + bass*1.6)",
           fy="y/max(r,0.03)*sin(r*8.0 - t*2.0)*0.55*(0.25 + bass*1.6)",
           wx="cos(s*tau)*(0.32 + spec(s)*0.85)",
           wy="sin(s*tau)*(0.32 + spec(s)*0.85)",
           wi="0.30 + spec(s)*1.6",
           params={"spin": 0.05, "decay": 0.955, "lw": 2.6, "steps": 1024, "pcount": 3}),
    Preset("Skyline", pal="Ember",
           fx="sin(y*2.0 + t*0.3)*0.10",
           fy="0.45 + bass*0.9",
           wx="s*2.0 - 1.0",
           wy="-0.80 + spec(s)*1.35",
           wi="0.35 + spec(s)*1.8",
           params={"spin": 0.0, "zoom": 0.0, "decay": 0.952, "lw": 2.2, "steps": 512,
                   "pcount": 2, "vig": 0.7}),
    Preset("Mandala", pal="Solar",
           fx="-y*0.55 + x*(0.12 - r*0.35) + noise(x*2.0, y*2.0 + t*0.2)*0.4",
           fy=" x*0.55 + y*(0.12 - r*0.35) + noise(y*2.0, x*2.0 - t*0.2)*0.4",
           wx="cos(s*tau*6 + t*0.2)*(0.25 + 0.55*abs(sin(s*tau*3)))*(1.0 + w*0.6)",
           wy="sin(s*tau*6 + t*0.2)*(0.25 + 0.55*abs(sin(s*tau*3)))*(1.0 + w*0.6)",
           wi="0.4 + mid*1.3",
           params={"mirror": 8, "decay": 0.970, "spin": 0.1, "bloom": 1.1}),
    Preset("Aurora", pal="Chloro",
           fx="noise(x*0.65, y*0.65 + t*0.14)*0.85",
           fy="0.55 + noise(x*1.3 + t*0.10, y*0.45)*0.7 + bass*0.5",
           wx="s*2.0 - 1.0",
           wy="noise(s*3.5, t*0.35)*0.55 + w*0.65",
           wi="0.4 + treb*1.5",
           params={"spin": 0.0, "zoom": 0.0, "decay": 0.962, "lw": 3.0, "vig": 0.65,
                   "grain": 0.3}),
    Preset("Hyperdrive", pal="Bone",
           fx="x*(0.60 + bass*0.7) + y*0.25",
           fy="y*(0.60 + bass*0.7) - x*0.25",
           wx="cos(s*tau + t*0.3)*(0.04 + w*0.35)",
           wy="sin(s*tau + t*0.3)*(0.04 + w*0.35)",
           wi="0.8 + mag*1.5",
           params={"decay": 0.975, "zoom": 0.0, "spin": 0.05, "pcount": 5, "psize": 1.6,
                   "pforce": 1.5, "bloom": 1.3, "aberr": 0.6, "ink": 3.0,
                   "palScale": 1.15}),
]


def by_name(name: str) -> Preset:
    """Look a preset up by name, case-insensitively."""
    key = name.strip().lower()
    for p in PRESETS:
        if p.name.lower() == key:
            return p
    raise KeyError(f"no preset named {name!r}; have: "
                   + ", ".join(p.name for p in PRESETS))


def load_config(path) -> Preset:
    """Read a plain-text config — the direct descendant of a G-Force config file.

    One ``key = value`` per line, ``#`` starts a comment, and a trailing ``\\``
    continues a long expression onto the next line::

        name     = Slow spiral
        palette  = Aurora
        flow.dx  = -y*0.9 - x*0.2
        flow.dy  =  x*0.9 - y*0.2
        wave.x   = cos(s*tau)*(0.5 + w*0.5)
        wave.y   = sin(s*tau)*(0.5 + w*0.5)
        wave.ink = 0.4 + treb*1.2
        decay    = 0.97
        spin     = 0.1

    Unknown keys are an error rather than a silent typo.
    """
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    fields: Dict[str, str] = {}
    pending_key = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            pending_key = None
            continue
        if pending_key is not None:
            cont, more = (line[:-1], True) if line.endswith("\\") else (line, False)
            fields[pending_key] += " " + cont.strip()
            if not more:
                pending_key = None
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: expected key = value")
        key, value = line.split("=", 1)
        key, value = key.strip().lower(), value.strip()
        if value.endswith("\\"):
            fields[key] = value[:-1].strip()
            pending_key = key
        else:
            fields[key] = value

    aliases = {"flow.dx": "fx", "flow.dy": "fy", "wave.x": "wx", "wave.y": "wy",
               "wave.ink": "wi", "palette": "pal", "name": "name"}
    core = {"name": "Config", "fx": "0", "fy": "0", "wx": "cos(s*tau)*0.6",
            "wy": "sin(s*tau)*0.6", "wi": "0.5", "pal": "Ion"}
    params: Dict[str, float] = {}
    for key, value in fields.items():
        if key in aliases:
            core[aliases[key]] = value
        elif key in DEFAULTS:
            params[key] = value if key == "pal" else float(value)
        else:
            raise ValueError(
                f"{path}: unknown key {key!r}; expected one of "
                + ", ".join(sorted(list(aliases) + list(DEFAULTS))))
    name = core.pop("name")
    pal = core.pop("pal")
    return Preset(name=name, pal=pal, params=params, **core)
