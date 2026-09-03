# Credit where it is due

Chromaflow is an independent, from-scratch homage to the music visualizers made
by **[SoundSpectrum](https://www.soundspectrum.com/)** — **G-Force**,
**WhiteCap**, **SoftSkies** and **Aeon** — software Drew O'Meara has been
building and actively maintaining since 1998; SoundSpectrum's own documentation
describes G-Force plainly as "the art of Drew O'Meara." These are not legacy
programs: the G-Force changelog runs current, with a release dated August 2026
adding new sound generators and audio-processing improvements. If you use and
enjoy this project, do the original the favor of trying G-Force, WhiteCap or
Aeon for yourself — [soundspectrum.com](https://www.soundspectrum.com/).

Two pieces of SoundSpectrum's own documentation shaped this project directly:

- **G-Force's config-programming reference** — the idea that a visualizer is a
  handful of plain-text *configs* full of arithmetic. A **WaveShape** turns a
  short sound snippet into line segments and dots. A **FlowField** is two
  equations — quoting the doc directly — that "express a 2D source coordinate
  as a function of a destination 2D coordinate," used to warp the previous
  frame into the next, over and over. A **ColorMap** is a table of 256 RGB
  colors that an 8-bit intensity value indexes into. Documentation for this
  reference was contributed by G-Force's own user community, submitted back to
  SoundSpectrum the same way this repository invites contributions.

  G-Force's own summary of its per-frame loop is five steps: *use the current
  FlowField on the previous frame to generate a new frame; sample some audio;
  draw the current WaveShape and any running particles on the new frame;
  colorize the new frame using the current ColorMap; copy the new frame to the
  screen.* Chromaflow's render loop — `_advect` → `_step_particles` →
  `_draw_wave` → `_colorize` → post — is that same sequence, unchanged in
  spirit twenty-plus years on (`chromaflow/engine.py`; the same five passes
  run in `web/index.html`'s WebGL2 pipeline).

  A handful of the Mix tab's sliders come straight from G-Force's own
  preference names: *Smoothing* is `Audio.FFT.Smooth`/`Audio.PCM.Smooth`,
  *Adaptive* resolution is the same idea as `WaveShape.AutoLineScale`, and
  *Line width* answers to `WaveShape.LineWidth.Scale`/`.Offset`.

- **Aeon's PythonCanvas API documentation** (dated January 2011) — SoundSpectrum's
  description of exposing "high-performance C++ based graphics engine to an
  embedded python interpreter." A scene is a class with `__init__` and
  `Draw(self, fft, time)`, registered with a single `SetDrawClass(MyDrawClass)`
  call at the bottom of the file — `chromaflow/scene.py` and
  `chromaflow/canvas.py` keep that contract on purpose, template included
  (`scenes/rings.py`, `scenes/lorenz_dust.py`). Its `VertexList` object — build
  it once, draw it many times with `DrawLines(start, group, stride, ...)` and
  matrix transforms instead of rebuilding per-vertex data every frame — is the
  same reason `chromaflow/canvas.py`'s `VertexList` is a NumPy array under the
  hood: batch the work, keep the interpreter out of the per-vertex loop.

**No SoundSpectrum code, shaders, presets, artwork or configs are used here** —
none of G-Force's config-file syntax (its 1–4 character key/value pairs like
`X0=`, `Y0=`, `Pen=`, `Aspc=`), none of Aeon's actual Python drawing engine, and
none of their function names verbatim (`mag(x)`/`fft(x)` as audio-access
functions and a single `BASS` scalar in G-Force, versus Chromaflow's own
`bass`/`mid`/`treb`/`beat`/`spec()`/`wave()` vocabulary, computed by a
from-scratch analyser). Everything in this repository, including its
expression language, its config file format, and every preset and palette, was
written independently. The debt is one of architecture and of ideas, not of
text, and it is still a large debt.

This project is not affiliated with, endorsed by, or sponsored by SoundSpectrum.
"G-Force," "WhiteCap," "SoftSkies" and "Aeon" are SoundSpectrum's names for
their own products.

## What is new here

The ideas above are theirs; the following are not, and are what "bringing it
up to date" turned out to mean:

- **The whole pipeline on the GPU.** The plane is advected in a fragment
  shader, particles integrate in a ping-pong float texture with MRT, and the
  wave shape is expanded from a parametric curve into an antialiased ribbon in
  the vertex shader. Nothing round-trips to the CPU.
- **Ink-conserving advection.** The gathered plane value is scaled by the
  Jacobian of the flow map, computed from screen-space derivatives. Without it,
  any outward-flowing field multiplies brightness every frame and saturates —
  which is why zoom-heavy configs have historically been so fiddly to tune.
- **Frame-rate-independent accumulation.** Ink is deposited per unit of time,
  not per frame, so a look holds still between 30 and 144 fps.
- **Linear-light rendering** with an ACES tone map, so palettes keep their
  colour instead of clipping to white.
- **Modern analysis:** log-spaced bands with asymmetric attack/release,
  spectral-flux onset detection against an adaptive threshold, a tempo
  estimate, and a chroma vector that tracks the key and steers the palette —
  well beyond G-Force's single auto-normalized `BASS` scalar.
- **A safe, total expression language.** Only whitelisted names and functions
  resolve, every function is defined over its whole domain, and the same
  parsed tree compiles to both GLSL and vectorised NumPy — so a config in a
  shared link can never smuggle shader code into someone's driver.
- **Live editing**, preset sharing via URL, MIDI, gamepad and multi-pointer
  input, auto-VJ, and adaptive resolution.

Released into the public domain (Unlicense), like the rest of this repository.
