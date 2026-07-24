"""
Generate golden test vectors for the fixed-point transition.

Imports from the local falcon_ref/ tree and writes JSON files under
tests/test_vectors/. The oracles are the float64 outputs of the
falcon_ref/ reference — run this only when that reference is in a
known-good state (regenerating re-bases the oracle). Otherwise the
committed JSON files in test_vectors/ are the source of truth used by
check_test_vectors.py.

Each test vector file follows the shape:

    {
      "layer":     "<algorithm layer name>",
      "reference": "<path to the reference source file>",
      "version":   <int>,
      "tests": [
        {
          "name":        "<unique identifier>",
          "description": "<short human-readable note>",
          "n":           <polynomial dimension>,
          "tol":         <absolute tolerance, float>,
          "input":       { ... },
          "expected":    { ... }
        },
        ...
      ]
    }

Complex numbers are stored as [real, imag] pairs.
"""

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # tests/
ROOT = HERE.parent  # repo root
VECTORS_DIR = HERE / "test_vectors"

# The generator imports Prest's falcon.py reference (falcon_ref/).
# Run this only when the reference is in a known-good state (typically:
# before starting to port a function to fixed-point).
sys.path.insert(0, str(ROOT / "falcon_ref"))
from fft import (  # noqa: E402
    fft,
    ifft,
    add,
    sub,
    mul,
    adj,
    add_fft,
    sub_fft,
    mul_fft,
    adj_fft,
    split_fft,
    merge_fft,
)
from ffsampling import gram, ldl_fft, ffldl_fft  # noqa: E402


def _c2p(z: complex) -> list:
    """Complex -> [real, imag] pair."""
    return [z.real, z.imag]


def _poly_c2p(poly) -> list:
    """List of complex -> list of [real, imag] pairs."""
    return [_c2p(z) for z in poly]


def _small_coefs(n: int, seed: int, lo: int = -8, hi: int = 8) -> list:
    """Deterministic small-magnitude integer polynomial."""
    r = random.Random(seed)
    return [r.randint(lo, hi) for _ in range(n)]


# --------------------------------------------------------------------- #
# FFT layer
# --------------------------------------------------------------------- #


def gen_fft_tests() -> dict:
    tests = []

    # Hand-picked small dimensions (eyeball-friendly).
    hand_picked = [
        ("fft_n8_tiny", [1, -2, 3, 0, -1, 4, 2, -3]),
        ("fft_n8_zero", [0] * 8),
        ("fft_n8_delta", [1] + [0] * 7),
    ]
    for name, f in hand_picked:
        tests.append(
            {
                "name": name,
                "description": f"Forward FFT of hand-picked polynomial ({name})",
                "n": len(f),
                "tol": 1e-12,
                "input": {"f": list(f)},
                "expected": {"f_fft": [_c2p(z) for z in fft(f)]},
            }
        )

    # Seeded larger dimensions.
    for n, seed in [(64, 1), (512, 2), (1024, 3)]:
        f = _small_coefs(n, seed)
        tests.append(
            {
                "name": f"fft_n{n}_seed{seed}",
                "description": f"Forward FFT of a random small-coef polynomial (n={n}, seed={seed})",
                "n": n,
                "tol": 1e-10,
                "input": {"f": f},
                "expected": {"f_fft": [_c2p(z) for z in fft(f)]},
            }
        )

    # Inverse FFT roundtrip.
    for n, seed in [(8, 10), (64, 11), (512, 12)]:
        f = _small_coefs(n, seed)
        f_fft = fft(f)
        tests.append(
            {
                "name": f"ifft_n{n}_seed{seed}",
                "description": f"Inverse FFT recovers the original polynomial (n={n})",
                "n": n,
                "tol": 1e-10,
                "input": {"f_fft": [_c2p(z) for z in f_fft]},
                "expected": {"f": [float(x) for x in ifft(f_fft)]},
            }
        )

    return {
        "layer": "fft",
        "reference": "falcon_ref/fft.py",
        "version": 1,
        "tests": tests,
    }


# --------------------------------------------------------------------- #
# Poly layer
# --------------------------------------------------------------------- #


def _poly_test(op: str, name: str, n: int, tol: float, inputs: dict, expected: dict) -> dict:
    return {
        "op": op,
        "name": name,
        "description": f"{op} on polynomials of dim {n}",
        "n": n,
        "tol": tol,
        "input": inputs,
        "expected": expected,
    }


def gen_poly_tests() -> dict:
    tests = []

    # A few (f, g) pairs covering small, medium, and realistic dimensions.
    pairs = [
        (8, 20),
        (64, 21),
        (512, 22),
        (1024, 23),
    ]

    for n, seed in pairs:
        f = _small_coefs(n, seed)
        g = _small_coefs(n, seed + 100)
        f_fft = fft(f)
        g_fft = fft(g)

        # --- coefficient domain ---
        tests.append(
            _poly_test("add", f"add_n{n}", n, 1e-12,
                       {"f": f, "g": g},
                       {"result": add(f, g)})
        )
        tests.append(
            _poly_test("sub", f"sub_n{n}", n, 1e-12,
                       {"f": f, "g": g},
                       {"result": sub(f, g)})
        )
        tests.append(
            _poly_test("mul", f"mul_n{n}", n, 1e-9,
                       {"f": f, "g": g},
                       {"result": [float(x) for x in mul(f, g)]})
        )
        tests.append(
            _poly_test("adj", f"adj_n{n}", n, 1e-10,
                       {"f": f},
                       {"result": [float(x) for x in adj(f)]})
        )

        # --- FFT domain ---
        tests.append(
            _poly_test("add_fft", f"add_fft_n{n}", n, 1e-12,
                       {"f_fft": _poly_c2p(f_fft), "g_fft": _poly_c2p(g_fft)},
                       {"result": _poly_c2p(add_fft(f_fft, g_fft))})
        )
        tests.append(
            _poly_test("sub_fft", f"sub_fft_n{n}", n, 1e-12,
                       {"f_fft": _poly_c2p(f_fft), "g_fft": _poly_c2p(g_fft)},
                       {"result": _poly_c2p(sub_fft(f_fft, g_fft))})
        )
        tests.append(
            _poly_test("mul_fft", f"mul_fft_n{n}", n, 1e-10,
                       {"f_fft": _poly_c2p(f_fft), "g_fft": _poly_c2p(g_fft)},
                       {"result": _poly_c2p(mul_fft(f_fft, g_fft))})
        )
        tests.append(
            _poly_test("adj_fft", f"adj_fft_n{n}", n, 1e-12,
                       {"f_fft": _poly_c2p(f_fft)},
                       {"result": _poly_c2p(adj_fft(f_fft))})
        )

        # --- split/merge (FFT domain) ---
        if n >= 4:
            f0_fft, f1_fft = split_fft(f_fft)
            tests.append(
                _poly_test("split_fft", f"split_fft_n{n}", n, 1e-12,
                           {"f_fft": _poly_c2p(f_fft)},
                           {"f0_fft": _poly_c2p(f0_fft),
                            "f1_fft": _poly_c2p(f1_fft)})
            )
            tests.append(
                _poly_test("merge_fft", f"merge_fft_n{n}", n, 1e-12,
                           {"f0_fft": _poly_c2p(f0_fft),
                            "f1_fft": _poly_c2p(f1_fft)},
                           {"result": _poly_c2p(merge_fft([f0_fft, f1_fft]))})
            )

        # --- div_fft: the divisor is a REAL Gram diagonal in [q/16, 16q]
        # (production divides by D_ii). Use Re(fft([q, 4000, 0, ..., 0])) ∈
        # [q-4000, q+4000] ≈ [8289, 16289] ⊂ [q/16, 16q] — the domain
        # nr_reciprocal actually inverts. |quotient| < 1, so m_out = 1.
        g_re = [complex(z.real, 0.0) for z in fft([12289, 4000] + [0] * (n - 2))]
        max_q = max(abs(a / b) for a, b in zip(f_fft, g_re))
        m_out_div = max(1, int(max_q).bit_length() + 1)
        tests.append(
            _poly_test(
                "div_fft",
                f"div_fft_n{n}_m{m_out_div}",
                n,
                1e-12,
                {
                    "f_fft": _poly_c2p(f_fft),
                    "g_fft": _poly_c2p(g_re),
                    "m_out": m_out_div,
                },
                {"result": _poly_c2p([a / b for a, b in zip(f_fft, g_re)])},
            )
        )

    return {
        "layer": "poly",
        "reference": "falcon_ref/fft.py",
        "version": 1,
        "tests": tests,
    }


# --------------------------------------------------------------------- #
# LDL layer
# --------------------------------------------------------------------- #


def _matrix_to_json(M) -> list:
    """Serialize a 2x2 matrix of coefficient-domain polynomials."""
    return [[list(M[i][j]) for j in range(2)] for i in range(2)]


def _matrix_fft_to_json(M_fft) -> list:
    """Serialize a 2x2 matrix of FFT-domain polynomials."""
    return [[_poly_c2p(M_fft[i][j]) for j in range(2)] for i in range(2)]


def _tree_to_json(tree) -> list:
    """Serialize an ffldl_fft tree. At each node the first element is a
    polynomial (l10 or d at a leaf) and subsequent elements are either
    subtrees (internal node) or length-2 polynomials (leaf)."""
    # A leaf is reached when all elements are polynomials of length 2 (no
    # further recursion). Detect by looking at the second element:
    # polynomials have complex numbers at their top level, subtrees have
    # lists (polys).
    if len(tree) == 3 and isinstance(tree[1][0], (int, float, complex)):
        # Leaf node: [l10, d00, d11], all polynomials.
        return [_poly_c2p(tree[0]), _poly_c2p(tree[1]), _poly_c2p(tree[2])]
    # Internal node: [l10, subtree0, subtree1].
    return [_poly_c2p(tree[0]), _tree_to_json(tree[1]), _tree_to_json(tree[2])]


def _build_gram_input(n: int, seed: int):
    """Generate a 2x2 basis B of small-coef polynomials. Returns (B, B_fft)."""
    r = random.Random(seed)
    B = [
        [[r.randint(-4, 4) for _ in range(n)] for _ in range(2)]
        for _ in range(2)
    ]
    B_fft = [[fft(B[i][j]) for j in range(2)] for i in range(2)]
    return B, B_fft


def gen_ldl_tests() -> dict:
    tests = []

    # Small and medium dimensions. n=1024 makes ffldl tree huge; stop at 64.
    for n, seed in [(8, 30), (64, 31)]:
        B, B_fft = _build_gram_input(n, seed)
        G = gram(B)

        # gram in coefficient domain
        tests.append({
            "op": "gram",
            "name": f"gram_n{n}",
            "description": f"Gram matrix of a 2x2 basis (coefficient domain, n={n})",
            "n": n,
            "tol": 1e-8,
            "input": {"B": _matrix_to_json(B)},
            "expected": {"G": [[[float(x) for x in G[i][j]] for j in range(2)]
                               for i in range(2)]},
        })

        # ldl_fft: takes a FFT-domain Gram matrix, returns L10, D00, D11
        G_fft = [[fft(G[i][j]) for j in range(2)] for i in range(2)]
        L, D = ldl_fft(G_fft)
        tests.append({
            "op": "ldl_fft",
            "name": f"ldl_fft_n{n}",
            "description": f"LDL decomposition of a 2x2 Gram matrix in FFT domain (n={n})",
            "n": n,
            "tol": 1e-10,
            "input": {"G_fft": _matrix_fft_to_json(G_fft)},
            "expected": {
                "L10_fft": _poly_c2p(L[1][0]),
                "D00_fft": _poly_c2p(D[0][0]),
                "D11_fft": _poly_c2p(D[1][1]),
            },
        })

        # ffldl_fft: recursive LDL tree
        # Rebuild G_fft since ldl_fft may mutate it
        G_fft = [[fft(G[i][j]) for j in range(2)] for i in range(2)]
        T = ffldl_fft(G_fft)
        tests.append({
            "op": "ffldl_fft",
            "name": f"ffldl_fft_n{n}",
            "description": f"Full ffLDL tree of a 2x2 Gram matrix (n={n})",
            "n": n,
            "tol": 1e-9,
            "input": {"G_fft": _matrix_fft_to_json(
                [[fft(G[i][j]) for j in range(2)] for i in range(2)]
            )},
            "expected": {"tree": _tree_to_json(T)},
        })

    return {
        "layer": "ldl",
        "reference": "falcon_ref/ffsampling.py",
        "version": 1,
        "tests": tests,
    }


# --------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------- #


GENERATORS = {
    "fft.json": gen_fft_tests,
    "poly.json": gen_poly_tests,
    "ldl.json": gen_ldl_tests,
    # TODO: "samplerz.json":    gen_samplerz_tests,
    # TODO: "ffsampling.json":  gen_ffsampling_tests,
    # TODO: "sign_verify.json": gen_sign_verify_tests,
}


# Collapse two-element float arrays ([re, im] pairs) onto a single line.
_NUM = r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
_PAIR_RE = re.compile(rf"\[\n\s+({_NUM}),\n\s+({_NUM})\n\s+\]")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2)
    text = _PAIR_RE.sub(r"[\1, \2]", text)
    with path.open("w") as fh:
        fh.write(text)
        fh.write("\n")
    print(f"wrote {path}")


def main() -> int:
    for filename, generator in GENERATORS.items():
        write_json(VECTORS_DIR / filename, generator())
    return 0


if __name__ == "__main__":
    sys.exit(main())
