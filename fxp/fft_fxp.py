"""
Fixed-point FFT over R[x]/(x^n+1), on FxR/FxC values.

Twiddle constants live at (m=1, p), selected by input's p (63 or 127).

The forward FFT runs at a SINGLE tag m, resolved once at entry and fed
unchanged to every level (no per-level growth). Correctness rests on the
averaging bound: a size-N partial transform of f is a value of some sub-FFT,
so ‖·‖_∞ ≤ ‖FFT(f)‖_∞ — hence if the OUTPUT fits in 2^m, EVERY intermediate
does too, and no butterfly overflows. Two ways to fix m (see `fft_fxp`):
  - m given  : the certified output tag (e.g. B0 rows at their γ tags) —
               tight and retag-free.
  - m = None : m(f) + log₂n, the always-valid output bound (‖FFT‖ ≤
               (n/√2)·2^{m(f)} < 2^{m(f)+log₂n}), for uncertified inputs.
Since the halves already sit at m, the butterfly is one twiddle mul (emit at
m, |w|=1 preserves modulus) plus an EXACT add — one rounding per butterfly,
no f0 widen. See `merge_fft_fxp`.

The inverse FFT is separate: `split_complex_fxp` preserves m (the ÷2 offsets
the twiddle mul's +1), so `ifft_fxp` is m-preserving throughout.

Magnitude changes use `retag_fxc` (value-preserving): the base-case load
retag, and bring split's f1 back to m. Split's ÷2 is a label-only retag.
"""

from beartype import beartype

from fxtypes import FxR, FxC, PolyR, PolyC, retag_fxr, retag_fxc
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
def fft_fxp(f: PolyR, m: int | None = None) -> PolyC:
    """FFT of a real polynomial in R[x]/(x^n+1), run at a SINGLE tag m.

    m is resolved once at entry and fed unchanged to every level:
      - m given : run the whole transform at m. Valid iff ‖FFT(f)‖_∞ < 2^m
        (by ‖sub-FFT‖ ≤ ‖FFT‖, every intermediate is < 2^m — no overflow).
        Pass a CERTIFIED output tag: the B0 rows use their γ tags M_B_FG /
        M_B_FG_UP, giving the tight, retag-free transform.
      - m None  : m = m(f) + log₂n, the always-valid output bound (‖FFT‖ ≤
        (n/√2)·2^{m(f)} < 2^{m(f)+log₂n}). For uncertified inputs (c/q, qt);
        set-and-forget, robust to a missing m.

    Integer coefficients embed exactly at any m; a fractional input tagged
    below m (c/q) loses its sub-ULP bits at the base-case load retag, far
    below the pipeline's needs. One rounding per butterfly — see `merge_fft_fxp`.
    """
    n = len(f)
    assert n >= 2 and (n & (n - 1)) == 0, "n must be a power of 2"
    if m is None:
        m = f[0].m + (n.bit_length() - 1)            # m(f) + log₂n
    if n == 2:
        # Retag the two real components to m FIRST, then pair them: the FxC is
        # born at m, never as a transient at the (tighter) load m where its
        # modulus √2·max could exceed 2^{load m}. Bit-identical to retagging
        # the FxC (banker's shift is sign-symmetric).
        r0, r1 = retag_fxr(f[0], m), retag_fxr(f[1], m)
        return [FxC(re=r0, im=r1), FxC(re=r0, im=-r1)]
    return merge_fft_fxp([fft_fxp(f[0::2], m), fft_fxp(f[1::2], m)], m)


@beartype
def merge_fft_fxp(f_list_fft: list[PolyC], m: int) -> PolyC:
    """Combine two length-n/2 FFTs (both already at m) into one length-n at m.

    Inverse of `split_complex_fxp`. |w|=1 so the twiddle mul keeps the
    modulus; the butterfly output is a value of the (sub-)FFT, ≤ ‖FFT‖ < 2^m
    by the averaging bound, so `f0 ± w_f1` fits at m with no widen. One
    rounding per butterfly (the mul emitting at m), the add is exact.
    """
    f0_fft, f1_fft = f_list_fft
    n = 2 * len(f0_fft)
    w = _roots_for(f0_fft[0].p)[n]
    out = [None] * n
    for i in range(n // 2):
        w_f1 = w[2 * i].mul_to(f1_fft[i], m)         # emit at m (|w|=1)
        f0 = f0_fft[i]                                        # already at m
        out[2 * i] = f0 + w_f1
        out[2 * i + 1] = f0 - w_f1
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
def mul_fft_to(f: PolyC, g: PolyC, m_out: int) -> PolyC:
    """Pointwise FxC multiply emitting directly at the budget m_out (fused
    multiply-and-retag, single round; see `FxC.mul_to`) — one rounding rather
    than a separate multiply then retag."""
    return [a.mul_to(b, m_out) for a, b in zip(f, g)]

@beartype
def adj_fft_fxp(f: PolyC) -> PolyC:
    """Complex conjugate (= FFT-domain adjoint of a real poly)."""
    return [z.conjugate() for z in f]


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
        out.append(FxC(re=retag_fxr(a.re * r, m_out),
                       im=retag_fxr(a.im * r, m_out)))
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
        a = retag_fxc(f_fft[2 * i], m + 1)
        b = retag_fxc(f_fft[2 * i + 1], m + 1)
        f0[i] = _halve(a + b)
        f1_wide = _halve(a - b) * w[2 * i].conjugate()  # mul widens to m+1
        f1[i] = retag_fxc(f1_wide, m)             # lossless left-shift to m
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


