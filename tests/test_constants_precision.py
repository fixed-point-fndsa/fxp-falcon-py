"""
Guard test: every COMPUTED fxp constant must be precise to ~p bits, at BOTH
p=63 and p=127.

Why this exists: the reciprocal constants 1/q and 1/σ were once generated in
float64 (`from_float_mantissa(1.0/q)`), which caps them at ~2^-53 relative —
invisible at p=63 (near the format floor) but a HARD ceiling at p=127, silently
defeating the whole point of the 128-bit variant wherever the constant is a
direct multiplicand (the target's ·1/q, the leaf's ·1/σ). A float64-derived
constant must never ship again. (See scripts/generate_constants_fxp.sage:
INV_Q / INV_SIGMA use `hp_recip_mantissa`, INV_SQRT_Q the integer isqrt recipe.)

The check is EXACT (stdlib `fractions`, no mpmath) so it runs in `make test`:
for each constant we verify its exact rational value satisfies the defining
identity (INV_Q·q=1, INV_SQRT_Q²·q=1, INV_SIGMA·SIGMA=1) to within 2^-(p-4).
A float64-capped constant (rel err ~2^-55) fails this at both p.
"""

import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fxp"))

from fxp_constants_p63 import (  # noqa: E402
    INV_Q_FXC as _IQ63, INV_SQRT_Q_FXR as _ISQ63,
    SIGMA_FXR_BY_N as _SG63, INV_SIGMA_FXR_BY_N as _IS63)
from fxp_constants_p127 import (  # noqa: E402
    INV_Q_FXC as _IQ127, INV_SQRT_Q_FXR as _ISQ127,
    SIGMA_FXR_BY_N as _SG127, INV_SIGMA_FXR_BY_N as _IS127)

Q = 12289
_MARGIN = 4   # threshold 2^-(p-4): passes p-precise consts, fails float64 (~2^-55)

_BY_P = {
    63:  (_IQ63, _ISQ63, _SG63, _IS63),
    127: (_IQ127, _ISQ127, _SG127, _IS127),
}


def _val(a) -> Fraction:
    """Exact rational value of an FxR: x · 2^(m-p)."""
    e = a.p - a.m                       # > 0 for every constant here
    return Fraction(a.x, 1 << e)


def _check(name, identity_value, p):
    """identity_value should equal 1 exactly; assert |·−1| < 2^-(p-MARGIN)."""
    rel = abs(identity_value - 1)
    bound = Fraction(1, 1 << (p - _MARGIN))
    bits = math.log2(float(rel)) if rel > 0 else float("-inf")
    assert rel < bound, (
        f"{name}: |identity−1| = 2^{bits:.1f} >= 2^-{p - _MARGIN} "
        f"— constant is NOT p-precise (float64-capped?)"
    )


def test_inv_q_precise():
    """INV_Q · q == 1 to p bits."""
    for p in (63, 127):
        iq = _BY_P[p][0]
        _check(f"INV_Q p={p}", _val(iq.re) * Q, p)
        assert iq.im.x == 0, f"INV_Q.im must be 0 (p={p})"


def test_inv_sqrt_q_precise():
    """INV_SQRT_Q² · q == 1 to p bits."""
    for p in (63, 127):
        isq = _BY_P[p][1]
        _check(f"INV_SQRT_Q p={p}", _val(isq) ** 2 * Q, p)


def test_inv_sigma_precise():
    """INV_SIGMA[n] · SIGMA[n] == 1 to p bits (SIGMA is the spec float64 σ,
    embedded exactly; INV_SIGMA must be its high-precision reciprocal)."""
    for p in (63, 127):
        _, _, sg, isig = _BY_P[p]
        for n in sg:
            _check(f"INV_SIGMA[{n}] p={p}", _val(isig[n]) * _val(sg[n]), p)


ALL_TESTS = [test_inv_q_precise, test_inv_sqrt_q_precise, test_inv_sigma_precise]


def main() -> int:
    for t in ALL_TESTS:
        t()
        print(f"{t.__name__}: PASS")
    print(f"\nAll {len(ALL_TESTS)} constant-precision tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
