"""
Unit tests for the scalar Newton-Raphson primitives in `nr_fxp`:
`nr_reciprocal` (1/b) and `rsqrt` (1/√x).

High-precision references use the stdlib only — `fractions.Fraction` for the
exact reciprocal (b is a dyadic rational, so 1/b is exact) and `decimal.Decimal`
for 1/√x — so the core test suite stays free of the mpmath (experiments) dep.

These cover the primitives directly: previously they were only exercised
indirectly (via div_fft_fxp KATs and the e2e signature tests), so a Newton-
Raphson bug invisible on Gram-diagonal inputs could have slipped through.
"""

import math
import random
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fxp"))
from fxtypes import FxR             # noqa: E402
from nr_fxp import rsqrt, nr_reciprocal  # noqa: E402

getcontext().prec = 80              # ~265 bits, ample for p=127
Q = 12289

# Relative-error budgets: generous vs the true floor (~2^-60 at p=63,
# ~2^-124 at p=127) so the tests never flake, yet a broken NR (~2^-1) fails loudly.
_BOUND = {63: 2.0 ** -50, 127: 2.0 ** -110}


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _value(a: FxR) -> Fraction:
    """Exact rational value of an FxR: x · 2^(m-p)."""
    e = a.m - a.p
    return Fraction(a.x) * (Fraction(1, 1 << -e) if e < 0 else Fraction(1 << e))


def _dec(v: Fraction) -> Decimal:
    return Decimal(v.numerator) / Decimal(v.denominator)


def _build(val: float, p: int) -> FxR:
    """FxR at a tight m for `val` (m = floor(log2|val|)+1)."""
    m = int(math.floor(math.log2(abs(val)))) + 1
    return FxR.from_float(val, m=m, p=p)


def _rel_recip(b: FxR) -> float:
    true = 1 / _value(b)                       # exact (Fraction)
    got = _value(nr_reciprocal(b))
    return abs(float((got - true) / true))


def _rel_rsqrt(x: FxR) -> float:
    true = Decimal(1) / _dec(_value(x)).sqrt()
    got = _dec(_value(rsqrt(x)))
    return abs(float((got - true) / true))


# --------------------------------------------------------------------- #
# nr_reciprocal
# --------------------------------------------------------------------- #

def test_nr_reciprocal_gram_domain():
    """1/b over the ffLDL divisor domain [q/16, 16q], both signs, both p."""
    lo, hi = Q / 16.0, 16.0 * Q
    for p in (63, 127):
        for i in range(60):
            val = lo * (hi / lo) ** (i / 59)
            for s in (1.0, -1.0):
                err = _rel_recip(_build(s * val, p))
                assert err < _BOUND[p], f"p={p} val={s*val:.1f} err={err:.2e}"


def test_nr_reciprocal_extremes():
    for p in (63, 127):
        for val in (Q / 16.0, 16.0 * Q, float(Q)):
            assert _rel_recip(_build(val, p)) < _BOUND[p]


def test_nr_reciprocal_out_of_domain():
    """Divisors outside [q/16, 16q] (the Lemma-9 domain) must raise."""
    for val in (16.0, float(Q) // 32, 64.0 * Q, 1e9):
        try:
            nr_reciprocal(_build(val, 63))
        except AssertionError:
            continue
        raise AssertionError(f"expected domain AssertionError for val={val}")


def test_nr_reciprocal_randomized():
    random.seed(7)
    for p in (63, 127):
        for _ in range(2000):
            val = random.uniform(Q / 16.0, 16.0 * Q) * random.choice((1.0, -1.0))
            assert _rel_recip(_build(val, p)) < _BOUND[p]


def test_nr_reciprocal_div_by_zero():
    try:
        nr_reciprocal(FxR(x=0, m=1, p=63))
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for nr_reciprocal(0)")


# --------------------------------------------------------------------- #
# rsqrt
# --------------------------------------------------------------------- #

def test_rsqrt_domain_grid():
    """1/√x over the rsqrt domain [q/1.17², 1.17²·q] (a strict-inside grid)."""
    lo, hi = Q / 1.3689, 1.3689 * Q              # α² = 1.17² = 1.3689
    for p in (63, 127):
        for i in range(60):
            val = lo * 1.0005 + (hi * 0.9995 - lo * 1.0005) * i / 59
            assert _rel_rsqrt(_build(val, p)) < _BOUND[p], f"p={p} val={val:.1f}"


def test_rsqrt_at_q():
    for p in (63, 127):
        assert _rel_rsqrt(_build(float(Q), p)) < _BOUND[p]


def test_rsqrt_domain_assert():
    """A value outside [q/1.17², 1.17²·q] must raise (convergence guard)."""
    for val in (4.0 * Q, Q / 8.0):              # 4q ≫ 1.37q ; q/8 ≪ q/1.37
        try:
            rsqrt(_build(val, 63))
        except AssertionError:
            continue
        raise AssertionError(f"expected domain AssertionError for val={val}")


# --------------------------------------------------------------------- #
# Standalone runner (mirrors test_fxtypes.py style)
# --------------------------------------------------------------------- #

ALL_TESTS = [
    test_nr_reciprocal_gram_domain,
    test_nr_reciprocal_extremes,
    test_nr_reciprocal_out_of_domain,
    test_nr_reciprocal_randomized,
    test_nr_reciprocal_div_by_zero,
    test_rsqrt_domain_grid,
    test_rsqrt_at_q,
    test_rsqrt_domain_assert,
]


def main() -> int:
    for t in ALL_TESTS:
        t()
        print(f"{t.__name__}: PASS")
    print(f"\nAll {len(ALL_TESTS)} nr_fxp tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
