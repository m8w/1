# Chromaflow

A music visualizer built on three old, good ideas — a **flow field** that drags
the whole picture somewhere new each frame, a **wave shape** that draws the sound
as a parametric curve, and a **color map** that turns raw intensity into light —
rebuilt on the GPU with a million particles, live-editable expressions, and
modern beat, tempo and key detection.

Those three ideas come from SoundSpectrum's **G-Force**, **WhiteCap** and
**Aeon**, whose config scripting and PythonCanvas API this is an homage to.
No SoundSpectrum code, art or presets are used. See [CREDITS.md](CREDITS.md).

Public domain (Unlicense). Use it for anything.

---

## Two engines, one design

| | `web/index.html` | the `chromaflow` Python package |
|---|---|---|
| Runs | WebGL2, real time | NumPy, offline |
| Good for | listening, VJing, playing | rendering video, scripting, batch work |
| Needs | a browser | Python 3.9+ and NumPy |
| Speed | 60–144 fps at 1080p+ | ~10–40 fps at 480p, resolution-independent |

Both implement the same pipeline, share the same expression language, the same
presets and the same palettes.

## The browser build

Open `web/index.html`. That is the whole install — one self-contained file, no
build step, no dependencies, nothing uploaded. Or serve it:

```bash
python -m chromaflow web --serve      # http://127.0.0.1:8765/
```

**Sound in.** Microphone, another browser tab or your system output (Chromium
desktop — tick *Share tab audio*), a dropped audio file, or the built-in
generative demo track so the page is never silent.

**Keys.** `Space` next preset · `←/→` previous/next · `1`–`8` jump ·
`A` auto-VJ · `R` randomise a fresh look · `C` console · `H` cinema mode ·
`F` fullscreen · `M` cycle source · `[` `]` particle count · `P` pause ·
`?` help and credits.

**Pointer, and everything else.** Drag to push the flow field, click to
detonate, scroll to zoom. Plug in a MIDI controller and CC 1–8 become the
variables `a0`–`a7`, live in every expression. A gamepad's right stick steers
spin and zoom. On a phone, enable tilt.

**Sharing.** *Copy share link* in the Code tab packs the whole look — every
expression and every changed knob — into the URL fragment. Nothing is uploaded;
the link is the preset.

## Writing a look

The Code tab edits five expressions and recompiles the shader as you type. This
is the descendant of a G-Force config, and the same text runs in the Python
engine:

```
flow.dx  = -y*(0.75 + bass*0.9) - x*0.18     # where the picture flows to
flow.dy  =  x*(0.75 + bass*0.9) - y*0.18
wave.x   = cos(s*tau*2 + t*0.25) * (0.18 + s*0.62) * (1.0 + w*0.8)
wave.y   = sin(s*tau*2 + t*0.25) * (0.18 + s*0.62) * (1.0 + w*0.8)
wave.ink = 0.35 + treb*1.4                   # pen brightness along the curve
```

**Variables.** `x y r th` (position, in a frame where y runs −1…1 and x runs
−aspect…aspect) · `s` (0…1 along the wave shape) · `w` (the audio sample at that
step) · `t dt` · `mag bass mid treb level beat phase bpm hue` · `asp` ·
`a0`–`a7` (MIDI) · `mx my mdown` (pointer) · `pi tau e`.

**Functions.** `sin cos tan asin acos atan atan2 sinh cosh tanh abs floor ceil
fract sign exp log sqrt pow min max mod step clamp mix smoothstep hypot wrap tri
saw pulse sq rnd noise` plus `spec(f)` and `wave(f)` to read the spectrum and
the waveform at any position.

The language is total and closed: only these names resolve, every function is
defined over its whole domain (`sqrt` clamps, `log` guards zero, division never
produces NaN), and there is no way to reach anything else. A config from a
stranger's link cannot do anything but make a picture.

## The Python package

```bash
pip install -e .          # or: pip install numpy, and run from this directory
```

```bash
# render a track to video (needs ffmpeg on PATH for mp4; --frames writes PNGs)
python -m chromaflow render -a track.wav -p Mandala -s 1080p --fps 60 -o out.mp4

# one settled frame, no ffmpeg needed
python -m chromaflow still -p "Spectrum ring" -s 1440p -o poster.png

# your own config file, and a knob or two overridden
python -m chromaflow render -c configs/slow-spiral.cfg --set bloom=1.4 --set mirror=6

# what is available
python -m chromaflow presets
```

With no `-a`, it generates its own demo track, so every command above works
with nothing but the repository checked out.

### As a library

```python
from chromaflow import Analyzer, Visualizer, by_name
from chromaflow.audio import load_audio

samples, sr = load_audio("track.wav", sample_rate=44100)
an  = Analyzer(sr)
vis = Visualizer(by_name("Vortex"), 1920, 1080, particles=200_000)

for i in range(60 * 30):
    t = i / 60.0
    frame = vis.step(an.analyse(an.window_for(samples, t), t, 1 / 60.0), 1 / 60.0)
    ...   # frame is (1080, 1920, 3) uint8
```

### Scenes, PythonCanvas style

A scene is a class with `__init__` and `Draw(self, fft, time)`, registered on the
last line — the contract Aeon used:

```python
from chromaflow.canvas import SetDrawClass, VertexList
import numpy as np

class Rings:
    def __init__(self, canvas):
        self.canvas = canvas
        self.th = np.linspace(0, 2*np.pi, 512, dtype=np.float32)
        self.vl = VertexList(512)

    def Draw(self, fft, time):
        r = 0.5 + fft.bass * 0.3
        self.vl.set(np.cos(self.th) * r, np.sin(self.th) * r)
        self.canvas.SetColor(0.03)
        self.canvas.DrawPolyline(self.vl, closed=True)

SetDrawClass(Rings)
```

```bash
python -m chromaflow still --scene scenes/rings.py -o rings.png
```

`fft` is a [`Features`](chromaflow/analysis.py) record — spectrum, waveform,
bands, onset, tempo, chroma and key — not just a spectrum. The canvas offers
`SetColor`, `SetLineWidth`, `Clear`, a matrix stack (`PushMatrix`, `Translate`,
`Rotate`, `Scale`), and `DrawPoints`, `DrawLines`, `DrawPolyline` and
`DrawTriangles` with PythonCanvas's start/group/stride arguments. Scenes hot-
reload: `Scene.poll()` re-executes the file when it changes on disk.

## How it works

Every frame, in this order:

1. **Advect.** The flow-field expressions give a velocity at each pixel; the
   plane is resampled from `p − v·dt`. That is a semi-Lagrangian step, and it is
   why the picture smears into itself instead of jittering. The gathered value
   is scaled by the Jacobian of the map, so ink is conserved where the field
   spreads out — without that, any outward flow multiplies brightness every
   frame and saturates to white.
2. **Particles.** Integrated against the same field plus curl noise, an audio
   term and pointer forces, then splatted into the plane so they leave trails
   through step 1. On the GPU their state lives in ping-pong float textures.
3. **Wave shape.** The parametric curve, drawn as an antialiased ribbon whose
   width tracks the beat.
4. **Color map.** Accumulated ink indexes a 256-entry palette. Palette rotation
   is a hue spin about the grey axis, so entry 0 stays black and the ramp stays
   monotone in intensity.
5. **Post.** Bloom, ACES tone mapping, chromatic aberration, kaleidoscope
   mirror, vignette, grain.

Ink is deposited *per unit of time*, not per frame, so the equilibrium
brightness of a look is the same at 30 fps and at 144 fps.

## Tests

```bash
python -m pytest              # 54 tests, ~7 seconds
```

They cover the expression language (including that it rejects everything off the
whitelist and never returns NaN), the analyser (silence stays silent, a 70 Hz
sine is bass, A440 reads as A, onsets land at musical intervals), the renderer
(every preset produces a finite, unsaturated frame; the feedback plane stays
bounded over a long take; audio is demonstrably what drives the picture), the
canvas (ink scales with stroke length, not vertex count) and the scene loader.

For deterministic capture of the browser build — recording, regression shots —
set `window.chromaflowFixedDt = 1/60` and every frame advances the simulation by
the same slice regardless of how long it actually took to draw.

## Layout

```
web/index.html        the real-time WebGL2 build — one file, no dependencies
chromaflow/expr.py    the expression language: parse once, run as NumPy or GLSL
chromaflow/analysis.py  bands, onsets, tempo, chroma
chromaflow/canvas.py  the PythonCanvas-shaped drawing API
chromaflow/engine.py  advect, particles, wave, colour, post
chromaflow/presets.py the eight stock looks, and the config-file reader
chromaflow/scene.py   loading and hot-reloading scene files
chromaflow/cli.py     render / still / presets / web
configs/              plain-text configs
scenes/               example scenes
```
