"""
Verify golden test vectors against an implementation.

Loads JSON files from test_vectors/ and runs a chosen implementation
on each input, comparing the result with the stored expected output
within the test's tolerance.

The stored oracles are computed in float64 by Prest's falcon.py
(falcon_ref/), so this checks that the fxp port *agrees with the float64
reference* within tolerance — a consistency / non-regression check, not a
measure of absolute precision (how fxp compares to float64 depends on the
precision p and the operation; see the precision benchmarks).

Layers dispatch to functions imported from the falcon_ref/ and fxp/ trees.
As functions are progressively ported to fixed-point, this script exercises
the ported versions against the locked JSON oracle without any further change.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # <repo>/tests/
ROOT = HERE.parent  # repo root
VECTORS_DIR = HERE / "test_vectors"

# Make both Prest's falcon.py reference (falcon_ref/*.py) and the
# fixed-point primitives (fxp/*.py) importable. As functions are
# progressively ported, the checker exercises whichever implementation
# is imported below for each layer.
sys.path.insert(0, str(ROOT / "falcon_ref"))
sys.path.insert(0, str(ROOT / "fxp"))
from fft import (  # noqa: E402
    add as impl_add,
    sub as impl_sub,
    mul as impl_mul,
    adj as impl_adj,
    add_fft as impl_add_fft,
    sub_fft as impl_sub_fft,
    mul_fft as impl_mul_fft,
    adj_fft as impl_adj_fft,
    split_fft as impl_split_fft,
    merge_fft as impl_merge_fft,
)
from ffsampling import (  # noqa: E402
    gram as impl_gram,
    ldl_fft as impl_ldl_fft,
    ffldl_fft as impl_ffldl_fft,
)
from fft_fxp import fft_fxp, ifft_fxp, div_fft_fxp  # noqa: E402
from fxtypes import FxR, FxC  # noqa: E402


def _smallest_m(max_abs) -> int:
    """Smallest integer m >= 1 such that max_abs < 2^m."""
    m = 1
    while max_abs >= (1 << m):
        m += 1
    return m


def fft_fxp_wrapper(f):
    """Adapter: int list -> fft_fxp -> complex list."""
    max_abs = max((abs(int(x)) for x in f), default=0)
    m_in = _smallest_m(max_abs)
    p = 63
    f_fxr = [FxR.from_int(int(x), m_in, p) for x in f]
    return [z.to_complex() for z in fft_fxp(f_fxr)]


def ifft_fxp_wrapper(f_fft):
    """Adapter: complex list -> ifft_fxp -> float list.

    Under complex-modulus convention for FxC, m must bound |z|, not max(|Re|,|Im|).
    """
    max_abs = max(abs(z) for z in f_fft)
    m_in = _smallest_m(max_abs)
    p = 63
    f_fxc = [FxC.from_complex(z, m_in, p) for z in f_fft]
    return [r.to_float() for r in ifft_fxp(f_fxc)]


def div_fft_fxp_wrapper(f_fft, g_fft, m_out):
    """Adapter: complex f / real g (the vector's divisor is real) -> div_fft_fxp."""
    m_f = _smallest_m(max(abs(z) for z in f_fft))
    m_g = _smallest_m(max(abs(z) for z in g_fft))
    p = 63
    f_fxc = [FxC.from_complex(z, m_f, p) for z in f_fft]
    g_fxr = [FxR.from_float(z.real, m_g, p) for z in g_fft]   # PolyR divisor
    return [z.to_complex() for z in div_fft_fxp(f_fxc, g_fxr, m_out)]


def _p2c(pair) -> complex:
    return complex(pair[0], pair[1])


def _max_abs_diff_complex(a, b) -> float:
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    return max(abs(ca - cb) for ca, cb in zip(a, b))


def _max_abs_diff_real(a, b) -> float:
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    return max(abs(float(ca) - float(cb)) for ca, cb in zip(a, b))


# --------------------------------------------------------------------- #
# FFT layer
# --------------------------------------------------------------------- #


def check_fft(test: dict, fft_fn, ifft_fn) -> tuple[bool, float]:
    if "f" in test["input"]:
        got = fft_fn(test["input"]["f"])
        expected = [_p2c(p) for p in test["expected"]["f_fft"]]
        err = _max_abs_diff_complex(got, expected)
    else:
        f_fft_in = [_p2c(p) for p in test["input"]["f_fft"]]
        got = ifft_fn(f_fft_in)
        err = _max_abs_diff_real(got, test["expected"]["f"])
    return err <= test["tol"], err


def _poly_pairs_to_complex(pairs):
    return [_p2c(p) for p in pairs]


# --------------------------------------------------------------------- #
# Poly layer
# --------------------------------------------------------------------- #


def check_poly(test: dict, add_fn, sub_fn, mul_fn, adj_fn,
               add_fft_fn, sub_fft_fn, mul_fft_fn, adj_fft_fn,
               split_fft_fn, merge_fft_fn, div_fft_fn) -> tuple[bool, float]:
    op = test["op"]
    inp = test["input"]
    exp = test["expected"]

    # Coefficient-domain ops: inputs and outputs are lists of real numbers.
    if op == "add":
        got = add_fn(inp["f"], inp["g"])
        err = _max_abs_diff_real(got, exp["result"])
    elif op == "sub":
        got = sub_fn(inp["f"], inp["g"])
        err = _max_abs_diff_real(got, exp["result"])
    elif op == "mul":
        got = mul_fn(inp["f"], inp["g"])
        err = _max_abs_diff_real(got, exp["result"])
    elif op == "adj":
        got = adj_fn(inp["f"])
        err = _max_abs_diff_real(got, exp["result"])

    # FFT-domain ops: inputs and outputs are lists of complex numbers.
    elif op == "add_fft":
        got = add_fft_fn(_poly_pairs_to_complex(inp["f_fft"]),
                         _poly_pairs_to_complex(inp["g_fft"]))
        err = _max_abs_diff_complex(got, _poly_pairs_to_complex(exp["result"]))
    elif op == "sub_fft":
        got = sub_fft_fn(_poly_pairs_to_complex(inp["f_fft"]),
                         _poly_pairs_to_complex(inp["g_fft"]))
        err = _max_abs_diff_complex(got, _poly_pairs_to_complex(exp["result"]))
    elif op == "mul_fft":
        got = mul_fft_fn(_poly_pairs_to_complex(inp["f_fft"]),
                         _poly_pairs_to_complex(inp["g_fft"]))
        err = _max_abs_diff_complex(got, _poly_pairs_to_complex(exp["result"]))
    elif op == "adj_fft":
        got = adj_fft_fn(_poly_pairs_to_complex(inp["f_fft"]))
        err = _max_abs_diff_complex(got, _poly_pairs_to_complex(exp["result"]))
    elif op == "split_fft":
        f0_got, f1_got = split_fft_fn(_poly_pairs_to_complex(inp["f_fft"]))
        err = max(
            _max_abs_diff_complex(f0_got, _poly_pairs_to_complex(exp["f0_fft"])),
            _max_abs_diff_complex(f1_got, _poly_pairs_to_complex(exp["f1_fft"])),
        )
    elif op == "merge_fft":
        got = merge_fft_fn([_poly_pairs_to_complex(inp["f0_fft"]),
                            _poly_pairs_to_complex(inp["f1_fft"])])
        err = _max_abs_diff_complex(got, _poly_pairs_to_complex(exp["result"]))
    elif op == "div_fft":
        got = div_fft_fn(_poly_pairs_to_complex(inp["f_fft"]),
                         _poly_pairs_to_complex(inp["g_fft"]),
                         inp["m_out"])
        err = _max_abs_diff_complex(got, _poly_pairs_to_complex(exp["result"]))
    else:
        raise ValueError(f"unknown poly op: {op}")

    return err <= test["tol"], err


# --------------------------------------------------------------------- #
# LDL layer
# --------------------------------------------------------------------- #


def _matrix_fft_from_json(M_json):
    return [[_poly_pairs_to_complex(M_json[i][j]) for j in range(2)] for i in range(2)]


def _max_abs_diff_tree(got, expected_json) -> float:
    """Recursively compare an ffldl_fft tree against its JSON serialization."""
    # Leaf node: [l10, d00, d11], all polynomials of length 2.
    # Internal node: [l10, subtree0, subtree1].
    err = _max_abs_diff_complex(got[0], _poly_pairs_to_complex(expected_json[0]))
    if len(got[0]) == 2:  # At leaf level (length-2 polys).
        err = max(err, _max_abs_diff_complex(got[1], _poly_pairs_to_complex(expected_json[1])))
        err = max(err, _max_abs_diff_complex(got[2], _poly_pairs_to_complex(expected_json[2])))
    else:
        err = max(err, _max_abs_diff_tree(got[1], expected_json[1]))
        err = max(err, _max_abs_diff_tree(got[2], expected_json[2]))
    return err


def check_ldl(test: dict, gram_fn, ldl_fft_fn, ffldl_fft_fn) -> tuple[bool, float]:
    op = test["op"]
    inp = test["input"]
    exp = test["expected"]

    if op == "gram":
        B = inp["B"]
        G = gram_fn(B)
        expected_G = exp["G"]
        # G is a 2x2 matrix of real polynomials.
        err = 0.0
        for i in range(2):
            for j in range(2):
                err = max(err, _max_abs_diff_real(G[i][j], expected_G[i][j]))

    elif op == "ldl_fft":
        G_fft = _matrix_fft_from_json(inp["G_fft"])
        L, D = ldl_fft_fn(G_fft)
        err = max(
            _max_abs_diff_complex(L[1][0], _poly_pairs_to_complex(exp["L10_fft"])),
            _max_abs_diff_complex(D[0][0], _poly_pairs_to_complex(exp["D00_fft"])),
            _max_abs_diff_complex(D[1][1], _poly_pairs_to_complex(exp["D11_fft"])),
        )

    elif op == "ffldl_fft":
        G_fft = _matrix_fft_from_json(inp["G_fft"])
        T = ffldl_fft_fn(G_fft)
        err = _max_abs_diff_tree(T, exp["tree"])

    else:
        raise ValueError(f"unknown ldl op: {op}")

    return err <= test["tol"], err


# --------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------- #


# Per layer: the checker function and the implementation functions to use.
# To check the fxp version of a layer, swap the functions here.
LAYERS = {
    "fft": {
        "check": check_fft,
        "fns": {"fft_fn": fft_fxp_wrapper, "ifft_fn": ifft_fxp_wrapper},
    },
    "poly": {
        "check": check_poly,
        "fns": {
            "add_fn": impl_add, "sub_fn": impl_sub, "mul_fn": impl_mul, "adj_fn": impl_adj,
            "add_fft_fn": impl_add_fft, "sub_fft_fn": impl_sub_fft,
            "mul_fft_fn": impl_mul_fft, "adj_fft_fn": impl_adj_fft,
            "split_fft_fn": impl_split_fft, "merge_fft_fn": impl_merge_fft,
            "div_fft_fn": div_fft_fxp_wrapper,
        },
    },
    "ldl": {
        "check": check_ldl,
        "fns": {
            "gram_fn": impl_gram, "ldl_fft_fn": impl_ldl_fft, "ffldl_fft_fn": impl_ffldl_fft,
        },
    },
    # TODO: "samplerz":    {"check": check_samplerz,   "fns": {...}},
    # TODO: "ffsampling":  {"check": check_ffsampling, "fns": {...}},
    # TODO: "sign_verify": {"check": check_sign_verify, "fns": {...}},
}


def check_file(path: Path) -> int:
    with path.open() as fh:
        data = json.load(fh)
    layer = data["layer"]
    if layer not in LAYERS:
        print(f"[skip] {path.name}: layer '{layer}' has no checker yet")
        return 0
    check_fn = LAYERS[layer]["check"]
    fns = LAYERS[layer]["fns"]
    failures = 0
    for t in data["tests"]:
        ok, err = check_fn(t, **fns)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {t['name']:<20} err={err:8.2e}   tol={t['tol']:.0e}")
        if not ok:
            failures += 1
    return failures


def main() -> int:
    total = 0
    for p in sorted(VECTORS_DIR.glob("*.json")):
        print(f"# {p.name}")
        total += check_file(p)
    print()
    if total:
        print(f"{total} test(s) failed.")
        return 1
    print("All vectors verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
