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
reciprocal `fft_fxp.nr_reciprocal` (it needs the 1/q seed constant).
Invariants (0 < p, m <= p, |x| < 2^p) are enforced via assertions.

FxC uses the **complex-modulus** convention: `m` bounds `|z|` (the modulus),
not each of `re`/`im` individually. This keeps FFT merge to +1 bit of `m`
per level (rather than +2). See `fxp/README.md` for the full m-arithmetic
table and the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ldexp
from typing import NamedTuple
from beartype import beartype


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

    The construction does not verify the complex bound at runtime (only
    each component's FxR invariant via __post_init__); correctness is
    verified by the test suite. Callers building FxC values directly must
    ensure the complex invariant holds.
    """

    re: FxR
    im: FxR

    def __post_init__(self) -> None:
        assert (self.re.m, self.re.p) == (self.im.m, self.im.p), f"FxC re/im (m,p) {(self.re.m, self.re.p)} != {(self.im.m, self.im.p)}"

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
        """Complex multiplication: (a+ib)(c+id) = (ac-bd) + i(ad+bc).

        Under the complex-modulus convention (|z| < 2^m): output m =
        m_1 + m_2, because |z_1 z_2| = |z_1| |z_2| < 2^{m_1 + m_2}.
        Each component is computed via one banker's shift by p bits on
        the exact integer expression. Rounding error at most 2^{m_1+m_2-p-1}
        per component.

        The |x_out| < 2^p invariant holds for the EXACT product by Lagrange:
        (ac - bd)^2 + (ad + bc)^2 = (a^2 + b^2)(c^2 + d^2). The banker's
        shift then adds up to half an ULP per component, so an input within
        ~2^-p (relative) of the modulus bound 2^{m1+m2} can round a
        component exactly to 2^p and trip the __post_init__ assert (loud,
        not silent). Pipeline values keep multi-bit modulus margins.
        """
        assert self.p == other.p, f"FxC *: p {self.p} != {other.p}"
        p = self.p
        x_a, x_b = self.re.x, self.im.x
        x_c, x_d = other.re.x, other.im.x
        m_out = self.m + other.m
        x_re = _bankers_shift(x_a * x_c - x_b * x_d, p)
        x_im = _bankers_shift(x_a * x_d + x_b * x_c, p)
        return FxC(re=FxR(x=x_re, m=m_out, p=p), im=FxR(x=x_im, m=m_out, p=p))

    def mul_to(self, other: FxC, m_out: int) -> FxC:
        """Complex multiply emitting directly at the caller-chosen budget m_out,
        with a SINGLE round.

        Same value as `retag_fxc(self * other, m_out)` but the exact
        integer product is shifted straight to m_out (one banker's shift),
        instead of rounding at p to the natural m_self+m_other and then
        re-rounding to m_out. Dropping the intermediate round makes it slightly
        more accurate — so it is NOT bit-identical to the two-step form. The
        |x| < 2^p invariant is checked by FxR's __post_init__ (loud on overflow).
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
