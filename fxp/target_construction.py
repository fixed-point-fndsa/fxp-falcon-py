"""
Per-signature target construction: t = (−c·F/q, c·f/q) given c = hash(msg).

Two target builders, selected by `use_tweak`:

  USE_TWEAK_STD (0) : standard target t_std            (`_build_t_standard`).
  USE_TWEAK_NTT (1) : Section-5.1 NTT-exact tweak t_frac (`_build_t_tweaked`).

Each has a float64 reference builder and an fxp counterpart (`_*_fxp`).
"""

import _path_setup  # noqa: F401  (prepends falcon_ref/ + fxp/ to sys.path)

from fft import fft  # noqa: E402
from ntt import mul_zq  # noqa: E402
from common import q as FALCON_Q  # noqa: E402

from fxtypes import FxR, retag_fxr  # noqa: E402
from fft_fxp import fft_fxp, mul_fft_to  # noqa: E402
from fxp_constants_p63 import INV_Q_FXC  # noqa: E402  (m=-13, |1/q| ≈ 2^-13.586 < 2^-13)
from m_budgets import (  # noqa: E402
    M_POINT_COEF, M_CQ_COEF, M_QT_COEF, M_B_FG, M_B_FG_UP,
)


# Tweak variant labels (single source of truth). Passed as `use_tweak` to
# `sign_tweak.sample_preimage` / `sign`.
USE_TWEAK_STD = 0          # no tweak (standard target)
USE_TWEAK_NTT = 1          # Section-5.1 NTT-exact tweak


# Coefficient-domain m for the growing FFTs (M_CQ_COEF, M_QT_COEF; plain
# tight bounds — see m_budgets.py): fft_fxp widens by log₂n = 9 (n=512), so
# fft(c/q) lands at 9 and qt at 22. The B0 rows instead use the FIXED-m FFT
# at their γ tags M_B_FG / M_B_FG_UP (see `_b0_fft_fxp`).


def _center_signed(poly, modulus):
    """Center a poly with coefs in [0, q) to signed [−q/2, q/2]."""
    half = modulus >> 1
    return [(c - modulus) if c > half else c for c in poly]


def _build_qt(sk, point):
    """NTT-exact (q·t0, q·t1) = (−c·F, c·f) mod^± q, centered to [−q/2, q/2].
    Pure integer arithmetic, shared by the float and fxp Section-5.1 builders."""
    q = FALCON_Q
    F_zq = [c % q for c in sk.F]
    f_zq = [c % q for c in sk.f]
    c_zq = list(point)
    qt0 = _center_signed(mul_zq([(-ci) % q for ci in c_zq], F_zq), q)
    qt1 = _center_signed(mul_zq(c_zq, f_zq), q)
    return qt0, qt1


# --------------------------------------------------------------------- #
# Float reference builders.
# --------------------------------------------------------------------- #


def _build_t_standard(sk, point):
    """Standard target t = (−c·F/q, c·f/q) in FFT domain, computed in
    float64 via the precomputed `sk.B0_fft`. Returns (t_fft, None)."""
    q = FALCON_Q
    [[_, b], [_c, d]] = sk.B0_fft
    c_fft = fft(point)
    t0_fft = [(c_fft[i] * d[i]) / q for i in range(sk.n)]
    t1_fft = [(-c_fft[i] * b[i]) / q for i in range(sk.n)]
    return [t0_fft, t1_fft], None


def _build_t_tweaked(sk, point):
    """Section-5.1 NTT-exact tweak in float64: q·t_frac = (−c·F, c·f) mod^± q
    via NTT (exact integers), then t_frac = FFT(q·t_frac) / q. Returns
    (t_frac_fft, [qt0, qt1]); the qt integer polys drive the equivalent
    z_std = z_tweaked + t_int relation. Lemma 10 (lem:sign) gives
    distributional equivalence with the standard target."""
    q = FALCON_Q
    qt0, qt1 = _build_qt(sk, point)
    t0_fft = [z / q for z in fft([float(c) for c in qt0])]
    t1_fft = [z / q for z in fft([float(c) for c in qt1])]
    return [t0_fft, t1_fft], [qt0, qt1]


# --------------------------------------------------------------------- #
# fxp builders.
#
# The float64 builders above leak ~2^-15 of round-off into t (intermediate
# magnitudes ~2^36 before div by q), which is enough to flip floor(mu) at
# integer boundaries ~1/1000–1/8000 of the time. The fxp builders below
# run the same math at p=63: round-off ~2^-45, making the std vs tweak
# KAT exact (1000/1000 in our experiments).
# --------------------------------------------------------------------- #


def _b0_fft_fxp(sk, p=63):
    """fxp FFT of B0 = [[g, −f], [G, −F]] at precision p (pure, uncached).

    Each row is loaded at its certified γ tag (M_B_FG for f,g via γ_fg /
    Check 1b, M_B_FG_UP for F,G via γ_FG / Check 3) — integers embed exactly
    at any tag — and transformed with `certified=True`, so the whole FFT runs
    at that single tight m: no growth, no retag, one rounding per butterfly.
    The checks are what buy this: without them the structural bound would put
    the rows at 14 / 16, coarsening every internal rounding by 6 / 4 bits.
    """
    a, b = (fft_fxp([FxR.from_int(co, m=M_B_FG, p=p) for co in poly], certified=True)
            for poly in (sk.g, [-c for c in sk.f]))          # fft(g), fft(−f) — γ_fg
    c, d = (fft_fxp([FxR.from_int(co, m=M_B_FG_UP, p=p) for co in poly], certified=True)
            for poly in (sk.G, [-c for c in sk.F]))          # fft(G), fft(−F) — γ_FG
    return [[a, b], [c, d]]


def _build_B0_fft_fxp_cache(sk, p=63):
    """Lazily build & cache `_b0_fft_fxp(sk, p)` on sk (once per key)."""
    if sk._B0_fft_fxp is None:
        sk._B0_fft_fxp = _b0_fft_fxp(sk, p)
    return sk._B0_fft_fxp


def _fft_int_poly_fxp(coefs, m_in, p=63):
    """fxp FFT of an integer polynomial (|c| < 2^{m_in})."""
    return fft_fxp([FxR.from_int(c, m=m_in, p=p) for c in coefs])


def _build_t_standard_fxp(sk, point, m_sign):
    """Standard target c·d/q built directly in fxp (no float64 detour).

    The division by q happens in COEFFICIENT domain: c_i/q < 1 is retagged
    to its tight bound M_CQ_COEF = 0 (exact left shift from the natural
    from_int(c)·INV_Q tag m = 1), the FFT runs at tags 0→9, and each
    pointwise product (c/q)̂·B̂ is emitted straight at m_sign by `mul_fft_to`
    (single rounding). Same values as the former (ĉ·d̂)·INV_Q route — the
    FFT's relative error is scale-invariant — but the runtime magnitudes
    collapse: held ĉ 2^22 → 2^9, transient products 2^34/2^30 → 2^21.
    Returns (t_fxc_pair, None).
    """
    [_, b_fxc], [_, d_fxc] = _build_B0_fft_fxp_cache(sk)  # fft(−f) @M_B_FG, fft(−F) @M_B_FG_UP
    inv_q = INV_Q_FXC.re
    cq = [retag_fxr(FxR.from_int(ci, m=M_POINT_COEF, p=inv_q.p) * inv_q,
                    M_CQ_COEF)                            # 1 → 0: exact shift
          for ci in point]
    cq_fft = fft_fxp(cq)                                  # m = M_CQ_COEF + 9 = 9
    t0 = mul_fft_to(cq_fft, d_fxc, m_sign)
    t1 = [-z for z in mul_fft_to(cq_fft, b_fxc, m_sign)]  # negation exact
    return [t0, t1], None


def _build_t_tweaked_fxp(sk, point, m_sign):
    """Section-5.1 target in fxp: NTT-exact qt then fft_fxp + ·INV_Q.

    fxp counterpart of `_build_t_tweaked`. qt is exact in Z/qZ; the only
    roundings are the banker's-shifts in fft_fxp plus ONE per coefficient in
    the fused ·INV_Q emit (`mul_to` straight at m_sign, ULP 2^{m_sign−63} —
    the former ·INV_Q-then-retag pair rounded twice). Output at (m_sign, 63).
    Returns (t_fxc_pair, [qt0, qt1]).
    """
    qt0, qt1 = _build_qt(sk, point)
    # |qt| ≤ q/2 < 2^13; FFT → m = 13 + 9 = 22; fused ·INV_Q → m_sign.
    t0 = [z.mul_to(INV_Q_FXC, m_sign) for z in _fft_int_poly_fxp(qt0, M_QT_COEF)]
    t1 = [z.mul_to(INV_Q_FXC, m_sign) for z in _fft_int_poly_fxp(qt1, M_QT_COEF)]
    return [t0, t1], [qt0, qt1]
