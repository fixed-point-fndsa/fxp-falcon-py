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

from fxtypes import FxR, FxC, retag_fxc  # noqa: E402
from fft_fxp import fft_fxp, retag_poly_fxc  # noqa: E402
from fxp_constants_p63 import INV_Q_FXC  # noqa: E402  (m=-13, |1/q| ≈ 2^-13.586 < 2^-13)
from m_budgets import M_POINT_COEF, M_B0_COEF, M_B_FG, M_B_FG_UP  # noqa: E402


def _div_by_q_fxc(z: FxC, m_out: int) -> FxC:
    """z / q at (m_out, 63). `z * INV_Q_FXC` lands at m = z.m − 13; the
    value-preserving retag aligns to m_out. Bit-output differs from an
    exact `/ q` by at most 2^-34 absolute (well below the samplerz
    boundary sensitivity ~2^-15)."""
    return retag_fxc(z * INV_Q_FXC, m_out)


# Tweak variant labels (single source of truth). Passed as `use_tweak` to
# `sign_tweak.sample_preimage` / `sign`.
USE_TWEAK_STD = 0          # no tweak (standard target)
USE_TWEAK_NTT = 1          # Section-5.1 NTT-exact tweak


# Coefficient-domain m for fxp inputs (M_POINT_COEF, M_B0_COEF; see
# m_budgets.py): fft_fxp widens by log₂n − 1 = 8 (n=512), so FFT-domain polys
# land at m = 22 (c_fft) and m = 21 (b_fft, d_fft).


def _center_signed(poly, modulus):
    """Center a poly with coefs in [0, q) to signed [−q/2, q/2]."""
    half = modulus >> 1
    return [(c - modulus) if c > half else c for c in poly]


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
    z_std = z_tweaked + t_int relation. Lemma 14 gives distributional
    equivalence with the standard target."""
    q = FALCON_Q
    F_zq = [c % q for c in sk.F]
    f_zq = [c % q for c in sk.f]
    c_zq = list(point)
    qt0 = _center_signed(mul_zq([(-ci) % q for ci in c_zq], F_zq), q)
    qt1 = _center_signed(mul_zq(c_zq, f_zq), q)
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


def _build_B0_fft_fxp_cache(sk, p=63):
    """Lazily build & cache the fxp FFT of B0 = [[g, −f], [G, −F]] on sk, each
    row retagged once to its tight NTRUGen bound (fft(g), fft(−f) → M_B_FG;
    fft(G), fft(−F) → M_B_FG_UP). Baking the retag here (an exact left-shift from
    the loose post-FFT m=21) lets every consumer see the tight m directly —
    bit-identical to the old per-consumer retags, but run once per key.
    """
    if sk._B0_fft_fxp is not None:
        return sk._B0_fft_fxp
    rows = [[sk.g, [-c for c in sk.f]], [sk.G, [-c for c in sk.F]]]
    [a, b], [c, d] = [[fft_fxp([FxR.from_int(co, m=M_B0_COEF, p=p) for co in poly])
                       for poly in row] for row in rows]
    sk._B0_fft_fxp = [
        [retag_poly_fxc(a, M_B_FG), retag_poly_fxc(b, M_B_FG)],        # fft(g), fft(−f) — γ_fg
        [retag_poly_fxc(c, M_B_FG_UP), retag_poly_fxc(d, M_B_FG_UP)],  # fft(G), fft(−F) — γ_FG
    ]
    return sk._B0_fft_fxp


def _fft_int_poly_fxp(coefs, m_in, p=63):
    """fxp FFT of an integer polynomial (|c| < 2^{m_in})."""
    return fft_fxp([FxR.from_int(c, m=m_in, p=p) for c in coefs])


def _build_t_standard_fxp(sk, point, m_sign):
    """Standard target c·d/q built directly in fxp (no float64 detour).

    fxp counterpart of `_build_t_standard`. The B0 rows arrive pre-retagged to
    their tight γ bounds from `_build_B0_fft_fxp_cache`, so the products keep
    precision through the ·INV_Q and retag to (m_sign, 63). `c = fft(point)` is
    left at its worst-case m=22 (no tighter bound for a random hashed point).
    Returns (t_fxc_pair, None).
    """
    [_, b_fxc], [_, d_fxc] = _build_B0_fft_fxp_cache(sk)  # fft(−f) @M_B_FG, fft(−F) @M_B_FG_UP
    c_fxc = _fft_int_poly_fxp(point, M_POINT_COEF)
    t0 = [_div_by_q_fxc(ci * di, m_sign) for ci, di in zip(c_fxc, d_fxc)]
    t1 = [_div_by_q_fxc(-(ci * bi), m_sign) for ci, bi in zip(c_fxc, b_fxc)]
    return [t0, t1], None


def _build_t_tweaked_fxp(sk, point, m_sign):
    """Section-5.1 target in fxp: NTT-exact qt then fft_fxp + ·INV_Q.

    fxp counterpart of `_build_t_tweaked`. qt is exact in Z/qZ; the only
    rounding is the banker's-shift in fft_fxp + ·INV_Q + retag. Output at
    (m_sign, 63). Returns (t_fxc_pair, [qt0, qt1]).
    """
    q = FALCON_Q
    F_zq, f_zq, c_zq = [c % q for c in sk.F], [c % q for c in sk.f], list(point)
    qt0 = _center_signed(mul_zq([(-ci) % q for ci in c_zq], F_zq), q)
    qt1 = _center_signed(mul_zq(c_zq, f_zq), q)
    # |qt| < q/2 < 2^13; FFT widens to m=21 (n=512); mul·INV_Q gives m_sign.
    t0 = [_div_by_q_fxc(z, m_sign) for z in _fft_int_poly_fxp(qt0, M_B0_COEF)]
    t1 = [_div_by_q_fxc(z, m_sign) for z in _fft_int_poly_fxp(qt1, M_B0_COEF)]
    return [t0, t1], [qt0, qt1]
