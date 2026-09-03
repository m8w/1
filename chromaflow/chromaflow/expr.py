"""The Chromaflow expression language.

G-Force and WhiteCap let you write a visualizer as plain text: a handful of
arithmetic expressions in ``t``, ``s`` and ``mag`` that say where the pen goes
and where the picture flows to next.  This module is that idea, rebuilt.

A source string is parsed once into a small tree and can then be

* evaluated over NumPy arrays, so a 4096-step wave shape costs one pass, not
  4096 interpreter trips, and
* emitted as GLSL, so exactly the same text can be compiled into a shader.

The language is deliberately total and closed: only the names in :data:`VARS`
and the functions in :data:`FUNCS` resolve, every function is defined over its
whole domain (``sqrt`` clamps, ``log`` guards zero), and there is no way to
reach the host.  That makes a config safe to accept from a file, a URL or a
stranger.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

__all__ = ["Expr", "ExprError", "compile_expr", "VARS", "FUNCS"]


class ExprError(ValueError):
    """Raised for anything the parser will not accept."""


#: Names an expression may read, mapped to their GLSL spelling.
VARS: Dict[str, str] = {
    "x": "x", "y": "y", "r": "r", "th": "th", "theta": "th",
    "s": "s", "w": "w", "t": "t", "time": "t", "dt": "uDt",
    "mag": "mag", "bass": "bass", "mid": "mid", "treb": "treb",
    "level": "level", "beat": "beat", "phase": "uBeatPhase",
    "bpm": "uBPM", "hue": "uHue", "asp": "asp", "aspect": "asp",
    "pi": "3.14159265359", "tau": "6.28318530718", "e": "2.71828182846",
    "a0": "uA0", "a1": "uA1", "a2": "uA2", "a3": "uA3",
    "a4": "uA4", "a5": "uA5", "a6": "uA6", "a7": "uA7",
    "mx": "uPtr.x", "my": "uPtr.y", "mdown": "uPtr.z",
}

CONSTS = {"pi": math.pi, "tau": math.tau, "e": math.e}


def _safe_sqrt(v):
    return np.sqrt(np.maximum(v, 0.0))


def _safe_log(v):
    return np.log(np.maximum(np.abs(v), 1e-6))


def _safe_pow(a, b):
    """``a ** b`` extended to negative bases as an odd function, never NaN."""
    return np.power(np.abs(a), b) * np.where(np.asarray(a) < 0.0, -1.0, 1.0)


def _wrap(v, lo, hi):
    span = hi - lo
    return lo + np.mod(np.mod(v - lo, span) + span, span)


def _tri(v):
    return np.abs(np.mod(v, 1.0) * 2.0 - 1.0) * 2.0 - 1.0


def _saw(v):
    return np.mod(v, 1.0) * 2.0 - 1.0


def _pulse(v, duty):
    return np.where(np.mod(v, 1.0) < duty, 1.0, -1.0)


def _rnd(seed):
    """Deterministic hash noise in ``[-1, 1]`` — the same one the shader uses."""
    p = np.mod(np.asarray(seed, dtype=np.float64) * 17.13 + 0.7, 1.0)
    p = np.mod(p * 0.1031, 1.0)
    p = p * (p + 33.33)
    p = p * (p + p)
    return np.mod(p, 1.0) * 2.0 - 1.0


def _value_noise(a, b):
    """Smooth 2-D value noise in roughly ``[-1, 1]``.

    Not the shader's simplex noise — matching that exactly is not worth the
    cost — but the same character: unit frequency, zero mean.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ia, ib = np.floor(a), np.floor(b)
    fa, fb = a - ia, b - ib
    ua = fa * fa * (3.0 - 2.0 * fa)
    ub = fb * fb * (3.0 - 2.0 * fb)

    def h(i, j):
        n = np.sin(i * 127.1 + j * 311.7) * 43758.5453
        return np.mod(n, 1.0) * 2.0 - 1.0

    n00, n10 = h(ia, ib), h(ia + 1.0, ib)
    n01, n11 = h(ia, ib + 1.0), h(ia + 1.0, ib + 1.0)
    return (n00 * (1 - ua) + n10 * ua) * (1 - ub) + (n01 * (1 - ua) + n11 * ua) * ub


#: ``name -> (python callable, glsl name, arity)``
FUNCS: Dict[str, tuple] = {
    "sin": (np.sin, "sin", 1), "cos": (np.cos, "cos", 1), "tan": (np.tan, "tan", 1),
    "asin": (lambda v: np.arcsin(np.clip(v, -1, 1)), "asin", 1),
    "acos": (lambda v: np.arccos(np.clip(v, -1, 1)), "acos", 1),
    "atan": (np.arctan, "atan", 1), "atan2": (np.arctan2, "atan", 2),
    "sinh": (np.sinh, "sinh", 1), "cosh": (np.cosh, "cosh", 1), "tanh": (np.tanh, "tanh", 1),
    "abs": (np.abs, "abs", 1), "floor": (np.floor, "floor", 1), "ceil": (np.ceil, "ceil", 1),
    "fract": (lambda v: v - np.floor(v), "fract", 1), "sign": (np.sign, "sign", 1),
    "exp": (lambda v: np.exp(np.clip(v, -60, 60)), "exp", 1),
    "log": (_safe_log, "safelog", 1), "sqrt": (_safe_sqrt, "safesqrt", 1),
    "pow": (_safe_pow, "safepow", 2),
    "min": (np.minimum, "min", 2), "max": (np.maximum, "max", 2),
    "mod": (lambda a, b: np.mod(a, np.where(np.asarray(b) == 0, 1e-6, b)), "mod", 2),
    "step": (lambda e, v: np.where(v < e, 0.0, 1.0), "step", 2),
    "clamp": (lambda v, lo, hi: np.clip(v, lo, hi), "clamp", 3),
    "mix": (lambda a, b, k: a + (b - a) * k, "mix", 3),
    "smoothstep": (
        lambda e0, e1, v: (lambda k: k * k * (3.0 - 2.0 * k))(
            np.clip((v - e0) / np.where(np.asarray(e1 - e0) == 0, 1e-6, e1 - e0), 0.0, 1.0)),
        "smoothstep", 3),
    "hypot": (np.hypot, "hypotf", 2),
    "wrap": (_wrap, "wrapf", 3), "tri": (_tri, "trif", 1), "saw": (_saw, "sawf", 1),
    "pulse": (_pulse, "pulsef", 2), "sq": (lambda v: v * v, "sqf", 1),
    "rnd": (_rnd, "rndf", 1), "noise": (_value_noise, "noisef", 2),
    "spec": (None, "specAt", 1), "wave": (None, "waveAt", 1),
}

_TOKEN = re.compile(r"""
    \s+                                  |
    \#[^\n]*                             |
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?) |
    (?P<id>[A-Za-z_][A-Za-z_0-9]*)       |
    (?P<op><=|>=|==|!=|&&|\|\||[-+*/%^(),<>?:])
""", re.VERBOSE)


def _tokenize(src: str) -> List[tuple]:
    out: List[tuple] = []
    i = 0
    while i < len(src):
        m = _TOKEN.match(src, i)
        if m is None or m.end() == i:
            raise ExprError(f'unexpected character {src[i]!r}')
        i = m.end()
        for kind in ("num", "id", "op"):
            v = m.group(kind)
            if v is not None:
                out.append((kind, v.lower() if kind == "id" else v))
                break
    out.append(("end", ""))
    return out


# --- AST -------------------------------------------------------------------

@dataclass(frozen=True)
class Num:
    v: float


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple


@dataclass(frozen=True)
class Bin:
    op: str
    a: Any
    b: Any


@dataclass(frozen=True)
class Cond:
    c: Any
    a: Any
    b: Any


class _Parser:
    def __init__(self, src: str):
        self.tk = _tokenize(src)
        self.i = 0

    def peek(self):
        return self.tk[self.i]

    def eat(self, v) -> bool:
        if self.tk[self.i][1] == v and self.tk[self.i][0] == "op":
            self.i += 1
            return True
        return False

    def need(self, v):
        if not self.eat(v):
            raise ExprError(f'expected {v!r} near {self.peek()[1] or "end"!r}')

    def primary(self):
        kind, v = self.peek()
        if kind == "num":
            self.i += 1
            return Num(float(v))
        if kind == "op" and v in "+-":
            self.i += 1
            inner = self.primary()
            return inner if v == "+" else Bin("neg", inner, None)
        if kind == "op" and v == "(":
            self.i += 1
            e = self.ternary()
            self.need(")")
            return e
        if kind == "id":
            self.i += 1
            if self.peek() == ("op", "("):
                if v not in FUNCS:
                    raise ExprError(f'unknown function {v!r}')
                self.i += 1
                args: List[Any] = []
                if not self.eat(")"):
                    while True:
                        args.append(self.ternary())
                        if not self.eat(","):
                            break
                    self.need(")")
                want = FUNCS[v][2]
                if len(args) != want:
                    raise ExprError(f'{v}() takes {want} argument{"s" if want > 1 else ""}')
                return Call(v, tuple(args))
            if v in CONSTS:
                return Num(CONSTS[v])
            if v not in VARS:
                raise ExprError(f'unknown name {v!r}')
            return Var(v)
        raise ExprError(f'unexpected {v or "end of expression"!r}')

    def power(self):
        a = self.primary()
        if self.eat("^"):
            return Call("pow", (a, self.power()))
        return a

    def unary(self):
        if self.eat("-"):
            return Bin("neg", self.unary(), None)
        if self.eat("+"):
            return self.unary()
        return self.power()

    def mul(self):
        a = self.unary()
        while True:
            if self.eat("*"):
                a = Bin("*", a, self.unary())
            elif self.eat("/"):
                a = Bin("/", a, self.unary())
            elif self.eat("%"):
                a = Call("mod", (a, self.unary()))
            else:
                return a

    def add(self):
        a = self.mul()
        while True:
            if self.eat("+"):
                a = Bin("+", a, self.mul())
            elif self.eat("-"):
                a = Bin("-", a, self.mul())
            else:
                return a

    def cmp(self):
        a = self.add()
        while self.peek()[0] == "op" and self.peek()[1] in ("<", ">", "<=", ">=", "==", "!="):
            op = self.peek()[1]
            self.i += 1
            a = Bin(op, a, self.add())
        return a

    def and_(self):
        a = self.cmp()
        while self.eat("&&"):
            a = Bin("&&", a, self.cmp())
        return a

    def or_(self):
        a = self.and_()
        while self.eat("||"):
            a = Bin("||", a, self.and_())
        return a

    def ternary(self):
        c = self.or_()
        if self.eat("?"):
            a = self.ternary()
            self.need(":")
            return Cond(c, a, self.ternary())
        return c

    def parse(self):
        e = self.ternary()
        if self.peek()[0] != "end":
            raise ExprError(f'trailing {self.peek()[1]!r}')
        return e


_CMP = {
    "<": np.less, ">": np.greater, "<=": np.less_equal, ">=": np.greater_equal,
    "==": np.equal, "!=": np.not_equal,
}


class Expr:
    """A parsed expression, evaluable over NumPy arrays or emittable as GLSL."""

    __slots__ = ("src", "root")

    def __init__(self, src: str):
        self.src = src
        parser = _Parser(src)
        # blank, or nothing but comments, is a legitimate "contribute nothing"
        self.root = Num(0.0) if parser.peek()[0] == "end" else parser.parse()

    def __repr__(self) -> str:
        return f"Expr({self.src!r})"

    # -- evaluation --------------------------------------------------------
    def __call__(self, env: Dict[str, Any]) -> np.ndarray:
        return self._ev(self.root, env)

    def _ev(self, n, env):
        if isinstance(n, Num):
            return n.v
        if isinstance(n, Var):
            key = VARS[n.name]
            if key in ("3.14159265359", "6.28318530718", "2.71828182846"):
                return float(key)
            v = env.get(n.name)
            if v is None:
                v = env.get({"theta": "th", "time": "t", "aspect": "asp"}.get(n.name, n.name))
            return 0.0 if v is None else v
        if isinstance(n, Call):
            fn = FUNCS[n.name][0]
            args = [self._ev(a, env) for a in n.args]
            if fn is None:                       # spec()/wave() are host lookups
                host = env.get(n.name)
                if host is None:
                    return 0.0
                return host(args[0])
            return fn(*args)
        if isinstance(n, Cond):
            c = self._ev(n.c, env)
            a, b = self._ev(n.a, env), self._ev(n.b, env)
            return np.where(np.asarray(c) > 0.5, a, b)
        op, a = n.op, self._ev(n.a, env)
        if op == "neg":
            return -a
        b = self._ev(n.b, env)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / np.where(np.asarray(b) == 0, 1e-9, b)
        if op in _CMP:
            return _CMP[op](a, b).astype(np.float64)
        if op == "&&":
            return ((np.asarray(a) > 0.5) & (np.asarray(b) > 0.5)).astype(np.float64)
        if op == "||":
            return ((np.asarray(a) > 0.5) | (np.asarray(b) > 0.5)).astype(np.float64)
        raise ExprError(f"bad operator {op}")

    # -- GLSL --------------------------------------------------------------
    def to_glsl(self) -> str:
        """Emit a GLSL float expression using the same names the web build uses."""
        return self._gl(self.root)

    def _gl(self, n) -> str:
        if isinstance(n, Num):
            return f"({n.v!r})" if "." in repr(n.v) or "e" in repr(n.v) else f"({n.v}.0)"
        if isinstance(n, Var):
            return f"({VARS[n.name]})"
        if isinstance(n, Call):
            return FUNCS[n.name][1] + "(" + ",".join(self._gl(a) for a in n.args) + ")"
        if isinstance(n, Cond):
            return f"(({self._gl(n.c)}>0.5)?{self._gl(n.a)}:{self._gl(n.b)})"
        if n.op == "neg":
            return f"(-{self._gl(n.a)})"
        if n.op in _CMP:
            return f"(({self._gl(n.a)}{n.op}{self._gl(n.b)})?1.0:0.0)"
        if n.op in ("&&", "||"):
            return f"(({self._gl(n.a)}>0.5{n.op}{self._gl(n.b)}>0.5)?1.0:0.0)"
        return f"({self._gl(n.a)}{n.op}{self._gl(n.b)})"


def compile_expr(src: str) -> Expr:
    """Parse ``src``; raises :class:`ExprError` with a readable message."""
    return Expr(src)
