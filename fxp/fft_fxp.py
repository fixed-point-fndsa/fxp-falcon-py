"""
Fixed-point FFT over R[x]/(x^n+1), on FxR/FxC values.

Twiddle constants live at (m=1, p), selected by input's p (63 or 127).
Under the complex-modulus convention, multiplying by a unit twiddle
widens m by +1 (from the m_w=1 label), and each merge level grows the
output m by +1 (structurally, from the butterfly sum |a ± w·b| ≤ 2·max).
Full FFT of a length-n input: m_out = m_in + log₂(n) − 1. Caveat: the n=2
base case keeps the label m_in although its output MODULUS can reach
√2·2^{m_in} (components stay < 2^{m_in}), so the m_out contract carries a
half-bit deficit for inputs that saturate m_in — such adversarial inputs
trip a butterfly's |x| < 2^p assert (loud, not silent). Pipeline inputs
keep multi-bit modulus margins that absorb it.

Magnitude changes use `retag_value_fxc(z, m_new)` (value-preserving; x is
banker's-shifted): to widen m before the butterfly sums and to bring split's
f1 back to m. The split's exact ÷2 is a label-only change (keep x, drop the
exponent) done inline in `split_complex_fxp` (see `_halve`).
"""

from beartype import beartype

from fxtypes import FxR, FxC, PolyR, PolyC, retag_value_fxr, retag_value_fxc
from fxp_constants_p63 import roots_dict_fxp as _roots_p63
from fxp_constants_p127 import roots_dict_fxp as _roots_p127
from nr_fxp import nr_reciprocal


_ROOTS_BY_P = {63: _roots_p63, 127: _roots_p127}


def _roots_for(p: int):
    """Select the precomputed twiddle table matching precision p."""
    try:
        return _ROOTS_BY_P[p]
    except KeyError:
        raise ValueError(
            f"no FFT constants for p={p}. Available: {sorted(_ROOTS_BY_P)}. "
            f"Regenerate via: sage scripts/generate_constants_fxp.sage {p} 1"
        )


# --------------------------------------------------------------------- #
# Forward FFT
# --------------------------------------------------------------------- #


@beartype
def fft_fxp(f: PolyR) -> PolyC:
    """FFT of a real polynomial in R[x]/(x^n+1).

    Input:  n FxR values sharing (m_in, p).
    Output: n FxC values at (m_in + log₂n − 1, p) for n > 2, (m_in, p) at n=2.
    """
    n = len(f)
    assert n >= 2 and (n & (n - 1)) == 0, "n must be a power of 2"
    if n == 2:
        # f_fft = [f[0] + i·f[1], f[0] − i·f[1]]. Components unchanged, but
        # the MODULUS can reach √2·2^{m_in}: the kept label m_in is √2-loose
        # vs the complex-modulus convention (see module docstring caveat).
        return [FxC(re=f[0], im=f[1]), FxC(re=f[0], im=-f[1])]
    return merge_fft_fxp([fft_fxp(f[0::2]), fft_fxp(f[1::2])])


@beartype
def merge_fft_fxp(f_list_fft: list[PolyC]) -> PolyC:
    """Combine two length-n/2 FFTs into one length-n (inverse of split_complex_fxp).

    Both halves must share (m_k, p). Output is (m_k + 1, p). The +1 growth
    is structural (butterfly |a ± w·b| ≤ 2·max), and naturally produced
    by the w·f1 mul (w at m=1) — we only widen f0 to match.
    """
    f0_fft, f1_fft = f_list_fft
    n = 2 * len(f0_fft)
    w = _roots_for(f0_fft[0].p)[n]
    out = [None] * n
    for i in range(n // 2):
        w_f1 = w[2 * i] * f1_fft[i]                              # (m_k + 1, p)
        f0_wide = retag_value_fxc(f0_fft[i], f0_fft[i].re.m + 1)  # widen m by 1
        out[2 * i] = f0_wide + w_f1
        out[2 * i + 1] = f0_wide - w_f1
    return out


# --------------------------------------------------------------------- #
# Inverse FFT
# --------------------------------------------------------------------- #


@beartype
def ifft_fxp(f_fft: PolyC) -> PolyR:
    """Inverse FFT, returning a list of FxR values. For a real-input FFT,
    f_fft = [a + i·b, a − i·b] at n=2, so (a, b) = (Re, Im) of f_fft[0]."""
    n = len(f_fft)
    assert n >= 2 and (n & (n - 1)) == 0, "n must be a power of 2"
    if n == 2:
        return [f_fft[0].re, f_fft[0].im]
    f0_fft, f1_fft = split_complex_fxp(f_fft)
    f0 = ifft_fxp(f0_fft)
    f1 = ifft_fxp(f1_fft)
    # `split_complex_fxp` retags f1 back to m, so f0 and f1 share m at every
    # recursion depth → straight interleave, no alignment retag.
    out = [None] * n
    for i in range(n // 2):
        out[2 * i] = f0[i]
        out[2 * i + 1] = f1[i]
    return out


# --------------------------------------------------------------------- #
# Element-wise polynomial helpers (FFT domain)
# --------------------------------------------------------------------- #


# Coefficient-wise FFT-domain ops (operands share p; add/sub also share m).

@beartype
def add_fft_fxp(f: PolyC, g: PolyC) -> PolyC:
    return [a + b for a, b in zip(f, g)]

@beartype
def sub_fft_fxp(f: PolyC, g: PolyC) -> PolyC:
    return [a - b for a, b in zip(f, g)]

@beartype
def mul_fft_fxp(f: PolyC, g: PolyC) -> PolyC:
    return [a * b for a, b in zip(f, g)]

@beartype
def adj_fft_fxp(f: PolyC) -> PolyC:
    """Complex conjugate (= FFT-domain adjoint of a real poly)."""
    return [z.conjugate() for z in f]


@beartype
def retag_poly_fxc(poly: PolyC, m_new: int) -> PolyC:
    """Value-preserving retag of every element in a PolyC to m_new.
    No-op (returns the same list) if already at m_new."""
    if not poly or poly[0].re.m == m_new:
        return poly
    return [retag_value_fxc(z, m_new) for z in poly]


@beartype
def retag_poly_fxr(poly: PolyR, m_new: int) -> PolyR:
    """Value-preserving retag of every FxR in a PolyR to m_new (the real-poly
    counterpart of `retag_poly_fxc`). No-op if already at m_new."""
    if not poly or poly[0].m == m_new:
        return poly
    return [retag_value_fxr(z, m_new) for z in poly]


@beartype
def div_fft_fxp(f: PolyC, g: PolyR, m_out: int) -> PolyC:
    """Pointwise FxC ÷ real division: each f[i] divided by g[i] via the
    Newton-Raphson reciprocal (`nr_reciprocal`) then a multiply. The divisor
    is a Gram diagonal — real in FFT domain, enforced by the `PolyR` type.
    m_out depends on a lower bound for |g[i]|."""
    assert len(f) == len(g)
    out = []
    for a, b in zip(f, g):
        r = nr_reciprocal(b)                         # 1/g[i]
        out.append(FxC(re=retag_value_fxr(a.re * r, m_out),
                       im=retag_value_fxr(a.im * r, m_out)))
    return out


@beartype
def split_complex_fxp(f_fft: PolyC) -> tuple[PolyC, PolyC]:
    """Split a length-n **complex** FFT into halves (sign / ifft):
        f0[i] = 0.5·(f[2i] + f[2i+1])
        f1[i] = 0.5·(f[2i] − f[2i+1])·conj(w[2i])
    Both halves are returned at (m, p). The mul by conj(w) widens to m+1
    transiently; we banker-shift f1 back to m so downstream sees uniform m.
    """
    n = len(f_fft)
    w = _roots_for(f_fft[0].p)[n]
    m, p = f_fft[0].m, f_fft[0].p
    f0 = [None] * (n // 2)
    f1 = [None] * (n // 2)

    def _halve(z):
        # Exact ÷2: keep the mantissa, drop the exponent (m+1) → m.
        return FxC(re=FxR(x=z.re.x, m=m, p=p), im=FxR(x=z.im.x, m=m, p=p))

    for i in range(n // 2):
        a = retag_value_fxc(f_fft[2 * i], m + 1)
        b = retag_value_fxc(f_fft[2 * i + 1], m + 1)
        f0[i] = _halve(a + b)
        f1_wide = _halve(a - b) * w[2 * i].conjugate()  # mul widens to m+1
        f1[i] = retag_value_fxc(f1_wide, m)             # lossless left-shift to m
    return f0, f1


@beartype
def split_real_fxp(f_fft: PolyR) -> tuple[PolyR, PolyC]:
    """Split a length-n **real** FFT (a Hermitian Gram diagonal, keygen path):
        f0[i] = 0.5·(f[2i] + f[2i+1])               → real  (PolyR)
        f1[i] = 0.5·(f[2i] − f[2i+1])·conj(w[2i])   → complex (PolyC)
    f0 stays real (the child diagonal); f1 picks up the twiddle and is complex.

    Implemented as `split_complex_fxp` on the FxC embedding (im = 0): the two
    are bit-identical (f0's imaginary part is exactly 0, so `.re` is lossless),
    while this entry keeps the typed PolyR → (PolyR, PolyC) contract.
    """
    zero = FxR(x=0, m=f_fft[0].m, p=f_fft[0].p)
    f0, f1 = split_complex_fxp([FxC(re=x, im=zero) for x in f_fft])
    return [z.re for z in f0], f1


