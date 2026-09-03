# Credit where it is due

Chromaflow is an independent, from-scratch homage to the music visualizers made
by **[SoundSpectrum](https://www.soundspectrum.com/)** — **G-Force**,
**WhiteCap**, **SoftSkies** and **Aeon** — and to two pieces of their
documentation in particular:

- **[G-Force config programming](https://www.soundspectrum.com/g-force/Documentation/config-programming.html)**
  — the idea that a visualizer is a handful of plain-text *configs* full of
  arithmetic. WaveShapes turn a snippet of sound into line segments arranged by
  an expression. FlowFields are two equations giving a source coordinate as a
  function of a destination coordinate. ColorMaps are 256 RGB entries that turn
  intensity into light. That decomposition — *shape the sound, move the picture,
  colour the result* — is the whole architecture of this project.

- **[Aeon's PythonCanvas API](https://www.soundspectrum.com/aeon/Documentation/PythonCanvas.html)**
  — a 3-D drawing API exposed to an embedded Python interpreter, designed so a
  scene could push a lot of geometry per frame without paying interpreter cost
  per vertex: build a `VertexList`, hand it to a draw call with a start index, a
  group size and a stride, and let the engine do the work. A scene is a class
  with `__init__` and `Draw(self, fft, time)`, registered with `SetDrawClass`.
  `chromaflow/canvas.py` keeps that contract deliberately.

**No SoundSpectrum code, shaders, presets, artwork or configs are used here.**
Everything in this repository was written from scratch. The debt is one of
ideas, and it is a large one: G-Force has been making music look like something
since 1998, and nothing since has improved on the basic recipe.

SoundSpectrum's products are their own, and their names are theirs.
This project is not affiliated with, endorsed by, or derived from SoundSpectrum.

## What is new here

The ideas are theirs; the following are not, and are what "bringing it up to
date" turned out to mean:

- **The whole pipeline on the GPU.** The plane is advected in a fragment shader,
  particles integrate in a ping-pong float texture with MRT, and the wave shape
  is expanded from a parametric curve into an antialiased ribbon in the vertex
  shader. Nothing round-trips to the CPU.
- **Ink-conserving advection.** The gathered plane value is scaled by the
  Jacobian of the flow map, computed from screen-space derivatives. Without it,
  any outward-flowing field multiplies brightness every frame and saturates —
  which is why zoom-heavy configs have historically been so fiddly to tune.
- **Frame-rate-independent accumulation.** Ink is deposited per unit of time,
  not per frame, so a look holds still between 30 and 144 fps.
- **Linear-light rendering** with an ACES tone map, so palettes keep their
  colour instead of clipping to white.
- **Modern analysis:** log-spaced bands with asymmetric attack/release, spectral
  flux onset detection against an adaptive threshold, a tempo estimate, and a
  chroma vector that tracks the key and steers the palette.
- **A safe, total expression language.** Only whitelisted names and functions
  resolve, every function is defined over its whole domain, and the same parsed
  tree compiles to both GLSL and vectorised NumPy — so a config in a shared link
  can never smuggle shader code into someone's driver.
- **Live editing**, preset sharing via URL, MIDI, gamepad and multi-pointer
  input, auto-VJ, and adaptive resolution.

Released into the public domain (Unlicense), like the rest of this repository.
