"""
Fixed-point types FxR and FxC for Falcon.

Notation (matches the paper):

    FX_{m,p}  = { 2^{m-p} * x : x in Z, |x| < 2^p }
    cFX_{m,p} = { z = a + i * b : a, b in FX_{m,p} }

`p` is the precision (bits of the signed mantissa `x`), `m` is the
magnitude bound (every value satisfies |v| < 2^m). Both are configurable
per value; the scale is 2^{m-p} and can be of either sign.

Addition and subtraction keep the operands' (m, p) exactly. Multiplication
widens m to m_a + m_b and preserves p via a round-to-nearest-even shift.
Division is not a primitive here: the pipeline divides via the Newton-Raphson
reciprocal `nr_fxp.nr_reciprocal` (it needs the 1/q seed constant).
Invariants (0 < p, m <= p, |x| < 2^p) are enforced via assertions.

FxC uses the **complex-modulus** convention: `m` bounds `|z|` (the modulus),
not each of `re`/`im` individually. This keeps FFT merge to +1 bit of `m`
per level (rather than +2). See `fxp/README.md` for the full m-arithmetic
table and the rationale.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from math import ldexp
from typing import NamedTuple
from beartype import beartype


# Opt-in FxC modulus check. FxC promises |z| < 2^m, but verifying it costs a
# big-int square + compare per construction — too much for the FFT inner loop,
# so it is OFF by default (the FxR component check |x| < 2^p backstops the
# catastrophic case; pipeline magnitude margins absorb the ½-bit soft case).
# Enable it in tests to turn the promise into a tested invariant. The check is
# EXACT: |z| < 2^m ⟺ |z|² < 2^{2m} ⟺ re.x² + im.x² < 2^{2p} (m cancels).
_CHECK_MODULUS = os.environ.get("FXP_CHECK_MODULUS") == "1"


@contextmanager
def check_modulus(enabled: bool = True):
    """Enable (or disable) the FxC modulus assert within this block, for
    tests / debugging. Restores the previous setting on exit."""
    global _CHECK_MODULUS
    old, _CHECK_MODULUS = _CHECK_MODULUS, enabled
    try:
        yield
    finally:
        _CHECK_MODULUS = old


def _bankers_shift(x: int, n: int) -> int:
    """Round x * 2^{-n} to nearest integer, ties-to-even. Requires n >= 1.

    Note of TP: the tie-to-even branch is awkward to mask. A masked
    implementation might prefer naive right-shift (bit-dropping) despite
    the bias that accumulates over long chains of operations.
    """
    q, r = divmod(x, 1 << n)
    half = 1 << (n - 1)
    if r > half or (r == half and q & 1):
        q += 1
    return q


@beartype
@dataclass(frozen=True)
class FxR:
    """
    Real fixed-point number in FX_{m,p}.

    value = x * 2^{m - p},  with x in Z and |x| < 2^p.
    """

    x: int
    m: int
    p: int

    def __post_init__(self) -> None:
        assert self.p > 0, f"p must be positive, got p={self.p}"
        # m > p would give sub-unit resolution and break small-integer mul.
        assert self.m <= self.p, f"m must not exceed p, got m={self.m}, p={self.p}"
        assert abs(self.x) < (1 << self.p), f"|x|={abs(self.x)} ≥ 2^{self.p}"

    # ---- constructors -----------------------------------------------

    @classmethod
    def from_int(cls, a: int, m: int, p: int) -> FxR:
        """Exact embedding of an integer a with |a| < 2^m."""
        return cls(x=a << (p - m), m=m, p=p)

    @classmethod
    def from_float(cls, a: float, m: int, p: int) -> FxR:
        """Round-to-nearest-even of a into FX_{m,p}."""
        return cls(x=round(ldexp(a, p - m)), m=m, p=p)

    # ---- value recovery ---------------------------------------------

    def to_int(self) -> int:
        """Return the integer value. Asserts the value is exactly an integer."""
        q, r = divmod(self.x, 1 << (self.p - self.m))
        assert r == 0, f"FxR not integer: x={self.x}, m={self.m}, p={self.p}"
        return q

    def to_float(self) -> float:
        """Return the best-effort float value. Exact when p <= 53."""
        return ldexp(float(self.x), self.m - self.p)

    # ---- arithmetic -------------------------------------------------

    def __neg__(self) -> FxR:
        """Exact negation (|x| < 2^p implies |-x| < 2^p)."""
        return FxR(x=-self.x, m=self.m, p=self.p)

    def __add__(self, other: FxR) -> FxR:
        """Exact sum in the common (m, p). Overflow raises via __post_init__."""
        assert (self.m, self.p) == (other.m, other.p), f"FxR +: (m,p) {(self.m, self.p)} != {(other.m, other.p)}"
        return FxR(x=self.x + other.x, m=self.m, p=self.p)

    def __sub__(self, other: FxR) -> FxR:
        """Exact difference in the common (m, p)."""
        return self + (-other)

    def __mul__(self, other: FxR) -> FxR:
        """Round-to-nearest-even of (self * other) in FX_{m_a+m_b, p}.

        Inputs must share p. Output m is the tight bound m_a+m_b; p is
        preserved via a round-to-nearest-even shift of p bits on the
        exact integer product. Rounding error at most 2^{m_a+m_b-p-1}.
        """
        assert self.p == other.p, f"FxR *: p {self.p} != {other.p}"
        x = _bankers_shift(self.x * other.x, self.p)
        return FxR(x=x, m=self.m + other.m, p=self.p)

    # ---- debug ------------------------------------------------------

    def __repr__(self) -> str:
        return f"FxR(x={self.x}, m={self.m}, p={self.p}) ~ {self.to_float()}"


@beartype
@dataclass(frozen=True)
class FxC:
    """Complex fixed-point number in cFX_{m,p}: two FxRs with same (m,p).

    Convention: `m` bounds the complex modulus, i.e. |z|^2 = Re^2 + Im^2
    < 2^{2m}. This is STRICTLY TIGHTER than bounding each component
    separately (|Re|, |Im| < 2^m) and is what makes FFT operations grow m
    by only +1 per merge level (rather than +2). The component bound
    |Re|, |Im| < 2^m is implied since |Re|, |Im| <= |z|.

    By default __post_init__ verifies only each component's FxR invariant
    (|Re|, |Im| < 2^p), not the modulus bound — checking |z| < 2^m costs a
    big-int square per construction, too much for the FFT inner loop. Enable
    the exact check (re.x² + im.x² < 2^{2p}) via `check_modulus()` / the
    FXP_CHECK_MODULUS env var; `tests/` runs the deployed pipeline under it,
    turning the modulus promise into a tested invariant.
    """

    re: FxR
    im: FxR

    def __post_init__(self) -> None:
        assert (self.re.m, self.re.p) == (self.im.m, self.im.p), f"FxC re/im (m,p) {(self.re.m, self.re.p)} != {(self.im.m, self.im.p)}"
        if _CHECK_MODULUS:
            p = self.re.p
            assert self.re.x ** 2 + self.im.x ** 2 < (1 << (2 * p)), \
                f"FxC |z| ≥ 2^m (m={self.re.m}): re.x²+im.x² ≥ 2^{2 * p}"

    # ---- accessors --------------------------------------------------

    @property
    def m(self) -> int:
        return self.re.m

    @property
    def p(self) -> int:
        return self.re.p

    def to_complex(self) -> complex:
        return complex(self.re.to_float(), self.im.to_float())

    def conjugate(self) -> FxC:
        """Exact complex conjugate: (re, im) -> (re, -im)."""
        return FxC(re=self.re, im=-self.im)

    # ---- constructors -----------------------------------------------

    @classmethod
    def from_complex(cls, z: complex, m: int, p: int) -> FxC:
        return cls(
            re=FxR.from_float(z.real, m, p),
            im=FxR.from_float(z.imag, m, p),
        )

    @classmethod
    def from_int(cls, a: int, m: int, p: int) -> FxC:
        """Exact embedding of an integer a as FxC (zero imaginary part)."""
        return cls(re=FxR.from_int(a, m, p), im=FxR(x=0, m=m, p=p))

    # ---- arithmetic -------------------------------------------------

    def __neg__(self) -> FxC:
        """Exact component-wise negation."""
        return FxC(re=-self.re, im=-self.im)

    def __add__(self, other: FxC) -> FxC:
        """Exact component-wise sum in the common (m, p)."""
        return FxC(re=self.re + other.re, im=self.im + other.im)

    def __sub__(self, other: FxC) -> FxC:
        """Exact component-wise difference in the common (m, p)."""
        return FxC(re=self.re - other.re, im=self.im - other.im)

    def __mul__(self, other: FxC) -> FxC:
        """Complex multiplication: (a+ib)(c+id) = (ac-bd) + i(ad+bc), at the
        tight output bound m_1 + m_2 (|z_1 z_2| = |z_1| |z_2| < 2^{m_1+m_2}).

        Delegates to `mul_to` at m_out = m_1 + m_2 (the shift is then exactly
        p bits): one banker's shift per component on the exact integer
        expression, rounding error at most 2^{m_1+m_2-p-1} per component.
        """
        return self.mul_to(other, self.m + other.m)

    def mul_to(self, other: FxC, m_out: int) -> FxC:
        """Complex multiply emitting directly at the caller-chosen budget m_out,
        with a SINGLE round.

        Same value as `retag_fxc(self * other, m_out)` but the exact
        integer product is shifted straight to m_out (one banker's shift),
        instead of rounding at p to the natural m_self+m_other and then
        re-rounding to m_out. Dropping the intermediate round makes it slightly
        more accurate — so it is NOT bit-identical to the two-step form.

        The |x| < 2^p invariant is checked by FxR's __post_init__ (loud on
        overflow). At m_out = m_1 + m_2 it holds for the EXACT product by
        Lagrange: (ac - bd)² + (ad + bc)² = (a² + b²)(c² + d²); the banker's
        shift then adds up to half an ULP per component, so an input within
        ~2^-p (relative) of the modulus bound 2^{m1+m2} can round a component
        exactly to 2^p and trip the assert (loud, not silent). Pipeline
        values keep multi-bit modulus margins.
        """
        assert self.p == other.p, f"FxC.mul_to: p {self.p} != {other.p}"
        p = self.p
        x_a, x_b = self.re.x, self.im.x
        x_c, x_d = other.re.x, other.im.x
        e_re = x_a * x_c - x_b * x_d
        e_im = x_a * x_d + x_b * x_c
        s = p + m_out - self.m - other.m          # total shift to land at m_out
        if s > 0:
            x_re, x_im = _bankers_shift(e_re, s), _bankers_shift(e_im, s)
        elif s == 0:
            x_re, x_im = e_re, e_im
        else:
            x_re, x_im = e_re << -s, e_im << -s
        return FxC(re=FxR(x=x_re, m=m_out, p=p), im=FxR(x=x_im, m=m_out, p=p))

    # ---- debug ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FxC(re={self.re.x}, im={self.im.x}; m={self.m}, p={self.p}) "
            f"~ {self.to_complex()}"
        )


# --------------------------------------------------------------------- #
# Type aliases re-exported across fxp modules.
# beartype sample-checks element types at call boundaries (~µs overhead),
# which catches the common mistakes (real vs complex polys, etc.).
# `FFLDLTree` is loose (just `list`) since the recursive shape is awkward
# to express; per-call `assert isinstance(tree[i], FxR | list)` does that
# job at the leaves.
# --------------------------------------------------------------------- #

PolyR = list[FxR]              # coefficient or FFT-domain real polynomial
PolyC = list[FxC]              # coefficient or FFT-domain complex polynomial


# Hermitian 2x2 Gram, stored by its two independent entries. Gram and RootGram
# share this shape but stay distinct classes so `ldl_fft_fxp` and
# `ldl_fft_fxp_ntru_root` can't be confused.
class Gram(NamedTuple):
    g00: PolyR     # diagonal (real); g11 == g00 for ffLDL child Grams
    g10: PolyC     # off-diagonal (G_01 = adj(g10) implied)


class RootGram(NamedTuple):
    """NTRU root Gram. No g11: the root LDL recovers D_11 = q²/D_00 via
    symplecticity (the true G_11 = c·c̄ + d·d̄ sits ~6 bits looser)."""
    g00: PolyR     # diagonal (real)
    g10: PolyC     # off-diagonal (G_01 = adj(g10) implied)


FFLDLTree = list               # [L10, sub_L, sub_R] (internal) or [L10, D00, D11] (pre-leaf)


# --------------------------------------------------------------------- #
#   retag_fxr / retag_fxc (a/z, m_new)
#     Value-preserving: x is shifted to keep `value` invariant.
#       m_new > a.m: banker's-shift right by (m_new − a.m) [precision loss].
#       m_new < a.m: left-shift by (a.m − m_new). Caller must ensure no
#                    overflow (|x| · 2^{a.m − m_new} < 2^p).
#     Used everywhere "renormalize-to-tight-m" or "widen-m" is needed
#     (ffLDL, samplerz, ffsampling, signature reconstruction).
# --------------------------------------------------------------------- #


def retag_fxr(a: FxR, m_new: int) -> FxR:
    """Value-preserving retag. x is shifted to keep value(a) invariant."""
    if m_new == a.m:
        return a
    elif m_new > a.m:
        return FxR(x=_bankers_shift(a.x, m_new - a.m), m=m_new, p=a.p)
    else:
        return FxR(x=a.x << (a.m - m_new), m=m_new, p=a.p)


def retag_fxc(z: FxC, m_new: int) -> FxC:
    """Value-preserving retag on FxC, applied componentwise."""
    return FxC(re=retag_fxr(z.re, m_new), im=retag_fxr(z.im, m_new))
