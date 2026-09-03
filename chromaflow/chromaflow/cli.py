"""Command line: render a track to video, or a still, or list what is available."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .analysis import Analyzer
from .audio import find_ffmpeg, load_audio, synth_demo
from .engine import Visualizer
from .presets import PRESETS, by_name, load_config

SIZES = {"360p": (640, 360), "480p": (854, 480), "720p": (1280, 720),
         "1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}


def _resolve_size(spec: str) -> tuple[int, int]:
    if spec in SIZES:
        return SIZES[spec]
    if "x" in spec:
        w, h = spec.lower().split("x", 1)
        return int(w), int(h)
    raise argparse.ArgumentTypeError(
        f"unknown size {spec!r}; use WxH or one of: {', '.join(SIZES)}")


def _load_track(args) -> tuple[np.ndarray, int]:
    if args.audio:
        return load_audio(args.audio, sample_rate=44100)
    return synth_demo(seconds=args.seconds, sample_rate=44100), 44100


def _make_visualizer(args, w: int, h: int) -> Visualizer:
    preset = load_config(args.config) if getattr(args, "config", None) else by_name(args.preset)
    overrides = {}
    for kv in args.set or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip() if k.strip() == "pal" else float(v)
    vis = Visualizer(preset, w, h, particles=args.particles, seed=args.seed,
                     overrides=overrides)
    if getattr(args, "scene", None):
        from .scene import load_scene
        vis.scene = load_scene(args.scene, vis.canvas)
    return vis


def cmd_presets(args) -> int:
    for i, p in enumerate(PRESETS):
        print(f"{i + 1}. {p.name}  [{p.pal}]")
        print(f"     flow  dx = {p.fx}")
        print(f"           dy = {p.fy}")
        print(f"     wave   x = {p.wx}")
        print(f"           y = {p.wy}")
    return 0


def cmd_still(args) -> int:
    w, h = _resolve_size(args.size)
    samples, sr = _load_track(args)
    an = Analyzer(sr, gain=args.gain)
    vis = _make_visualizer(args, w, h)
    dt = 1.0 / args.fps
    frames = max(1, int(args.at * args.fps))
    img = None
    for i in range(frames):
        t = i * dt
        img = vis.step(an.analyse(an.window_for(samples, t), t, dt), dt)
    _write_png(Path(args.out), img)
    print(f"wrote {args.out}  ({w}×{h}, settled over {frames} frames)")
    return 0


def _write_png(path: Path, img: np.ndarray) -> None:
    """Write RGB uint8 as PNG using only the standard library."""
    import struct
    import zlib

    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def cmd_render(args) -> int:
    w, h = _resolve_size(args.size)
    samples, sr = _load_track(args)
    duration = len(samples) / sr if args.duration is None else min(args.duration, len(samples) / sr)
    an = Analyzer(sr, gain=args.gain)
    vis = _make_visualizer(args, w, h)
    dt = 1.0 / args.fps
    n_frames = max(1, int(duration * args.fps))
    out = Path(args.out)

    ff = None if args.frames else find_ffmpeg()
    proc = None
    if ff is None:
        out.mkdir(parents=True, exist_ok=True)
        if not args.frames:
            print("no ffmpeg found — writing a PNG sequence instead", file=sys.stderr)
        print(f"writing {n_frames} PNGs into {out}/", file=sys.stderr)
    else:
        cmd = [ff, "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(args.fps), "-i", "-"]
        if args.audio:
            cmd += ["-i", str(args.audio), "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    t0 = time.time()
    for i in range(n_frames):
        t = i * dt
        img = vis.step(an.analyse(an.window_for(samples, t), t, dt), dt)
        if proc is None:
            _write_png(out / f"frame_{i:06d}.png", img)
        else:
            try:
                proc.stdin.write(img.tobytes())
            except BrokenPipeError:
                # ffmpeg died; its own message is far more use than ours
                proc.stdin.close()
                err = proc.stderr.read().decode(errors="replace").strip()
                proc.wait()
                print(file=sys.stderr)
                print("ffmpeg stopped after "
                      f"{i} frames:\n{err or '(no message)'}", file=sys.stderr)
                print("Re-run with --frames to write a PNG sequence instead.",
                      file=sys.stderr)
                return 1
        if i % 25 == 0 or i == n_frames - 1:
            done = i + 1
            rate = done / max(1e-6, time.time() - t0)
            eta = (n_frames - done) / max(1e-6, rate)
            print(f"\r  {done}/{n_frames} frames · {rate:5.1f} fps · eta {eta:5.1f}s",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if proc is not None:
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            print(proc.stderr.read().decode(errors="replace"), file=sys.stderr)
            return rc
    print(f"wrote {out}  ({w}×{h} @ {args.fps}fps, {n_frames} frames, "
          f"{time.time() - t0:.1f}s)")
    return 0


def cmd_web(args) -> int:
    page = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if not page.exists():
        print(f"cannot find {page}", file=sys.stderr)
        return 1
    if args.serve:
        import http.server
        import functools
        import socketserver
        handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                    directory=str(page.parent))
        with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
            print(f"Chromaflow at http://127.0.0.1:{args.port}/  (ctrl-c to stop)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print()
        return 0
    print(page)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="chromaflow",
        description="A music visualizer in the tradition of SoundSpectrum's "
                    "G-Force, WhiteCap and Aeon.")
    ap.add_argument("--version", action="version", version=f"chromaflow {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_out_default):
        p.add_argument("-a", "--audio", help="audio file (WAV always; anything else needs ffmpeg)")
        p.add_argument("-p", "--preset", default="Nebula", help="preset name (see `presets`)")
        p.add_argument("-c", "--config", help="a plain-text config file, instead of --preset")
        p.add_argument("--scene", help="a PythonCanvas-style scene .py, drawn instead of the wave shape")
        p.add_argument("-s", "--size", default="720p",
                       help="WxH, or " + "/".join(SIZES))
        p.add_argument("--fps", type=float, default=30.0)
        p.add_argument("--particles", type=int, default=40000)
        p.add_argument("--gain", type=float, default=1.6, help="analysis sensitivity")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--seconds", type=float, default=20.0,
                       help="length of the built-in demo track when --audio is absent")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override a preset knob, repeatable (e.g. --set bloom=1.4)")
        p.add_argument("-o", "--out", default=need_out_default)

    r = sub.add_parser("render", help="render a track to video")
    common(r, "chromaflow.mp4")
    r.add_argument("--duration", type=float, default=None, help="seconds to render")
    r.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better")
    r.add_argument("--frames", action="store_true",
                   help="write a PNG sequence into --out instead of calling ffmpeg")
    r.set_defaults(func=cmd_render)

    s = sub.add_parser("still", help="render one settled frame to PNG")
    common(s, "chromaflow.png")
    s.add_argument("--at", type=float, default=3.0,
                   help="seconds of simulation to settle before capturing")
    s.set_defaults(func=cmd_still)

    p = sub.add_parser("presets", help="list the built-in presets and their expressions")
    p.set_defaults(func=cmd_presets)

    wv = sub.add_parser("web", help="print, or serve, the real-time browser build")
    wv.add_argument("--serve", action="store_true", help="serve it on localhost")
    wv.add_argument("--port", type=int, default=8765)
    wv.set_defaults(func=cmd_web)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
