import numpy as np
import pytest

from chromaflow.expr import ExprError, compile_expr


def test_scalar_arithmetic():
    assert compile_expr("2 + 3 * 4")({}) == pytest.approx(14.0)
    assert compile_expr("(2 + 3) * 4")({}) == pytest.approx(20.0)
    assert compile_expr("2^3^2")({}) == pytest.approx(512.0)      # right associative
    assert compile_expr("-x^2")({"x": 3.0}) == pytest.approx(-9.0)  # unary binds loosest


def test_vectorises_over_arrays():
    s = np.linspace(0.0, 1.0, 5)
    out = np.asarray(compile_expr("sin(s*tau)")({"s": s}))
    assert out.shape == s.shape
    assert out == pytest.approx(np.sin(s * 2 * np.pi), abs=1e-6)


def test_constants_and_aliases():
    assert compile_expr("pi")({}) == pytest.approx(np.pi)
    assert compile_expr("theta")({"th": 0.5}) == pytest.approx(0.5)
    assert compile_expr("time")({"t": 2.0}) == pytest.approx(2.0)


def test_comparison_and_ternary_are_floats():
    assert compile_expr("(r < 0.5) ? 1 : 2")({"r": 0.25}) == pytest.approx(1.0)
    assert compile_expr("(r < 0.5) ? 1 : 2")({"r": 0.75}) == pytest.approx(2.0)
    got = np.asarray(compile_expr("x > 0")({"x": np.array([-1.0, 1.0])}))
    assert got.tolist() == [0.0, 1.0]


def test_host_lookups():
    env = {"s": np.array([0.0, 1.0]), "spec": lambda q: np.asarray(q) * 10.0}
    assert np.asarray(compile_expr("spec(s)")(env)).tolist() == [0.0, 10.0]
    # an unbound host function is 0, not a crash
    assert compile_expr("wave(0.5)")({}) == pytest.approx(0.0)


@pytest.mark.parametrize("src, arg", [
    ("sqrt(x)", -4.0), ("log(x)", 0.0), ("pow(x, 0.5)", -2.0), ("x / y", 0.0),
])
def test_total_over_awkward_input(src, arg):
    env = {"x": arg, "y": 0.0}
    out = np.asarray(compile_expr(src)(env), dtype=float)
    assert np.isfinite(out).all(), f"{src} produced {out}"


def test_empty_source_is_zero():
    assert compile_expr("")({}) == pytest.approx(0.0)
    assert compile_expr("   # just a comment")({}) == pytest.approx(0.0)


@pytest.mark.parametrize("src", [
    "sin(", "foo(1)", "1 +", "x)", "sin(1, 2)", "x $ y", "gl_FragColor",
    "__import__('os')", "open('/etc/passwd')",
])
def test_rejects_junk_and_anything_off_the_whitelist(src):
    with pytest.raises(ExprError):
        compile_expr(src)


def test_glsl_emission_uses_shader_names():
    glsl = compile_expr("sqrt(mag) + a0 * mx").to_glsl()
    assert "safesqrt" in glsl and "uA0" in glsl and "uPtr.x" in glsl
    assert "sqrt(" not in glsl.replace("safesqrt(", "")


def test_glsl_and_numpy_agree_on_structure():
    # the same source parses once and serves both back ends
    e = compile_expr("clamp(x, 0, 1) * 2")
    assert e({"x": 0.25}) == pytest.approx(0.5)
    assert e.to_glsl().startswith("(clamp(")
