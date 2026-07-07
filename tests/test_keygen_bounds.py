"""Pin the two hard coefficient-domain bounds behind the fxp B0 loads
(m_budgets M_B0_COEF_FG = 5 / M_B0_COEF_FG_UP = 8): gen_poly's CDT support
(n=512: |c| <= 17, the gauss_512 table length) and the ||F,G||_inf <= 127
encoding filter — both checked against the sqrt2 FFT-load rule
sqrt2*max <= 2^m (see m_budgets). A regression (e.g. an unbounded
sum-of-samplerz gen_poly, or a load tag tightened below the rule) would
silently invalidate the budgets and trip |x| < 2^p on a tail key."""

import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fxp"))
sys.path.insert(0, str(_ROOT / "falcon_ref"))

from ntrugen import gen_poly, _GAUSS_512          # noqa: E402  (falcon_ref/)
from ntrugen_filters import FG_COEF_LIMIT         # noqa: E402  (falcon_ref/)
from m_budgets import M_B0_COEF_FG, M_B0_COEF_FG_UP  # noqa: E402  (fxp/)


def _fft_load_covered(max_coef: int, m: int) -> bool:
    """sqrt2 FFT-load rule, in exact integer arithmetic:
    sqrt2*max <= 2^m  <=>  2*max^2 <= 4^m."""
    return 2 * max_coef * max_coef <= 4 ** m


def test_gen_poly_512_hard_support_and_shape():
    kmax = len(_GAUSS_512) // 2
    assert kmax == 17, "gauss_512 table length changed - re-derive M_B0_COEF_FG"
    coefs = [c for _ in range(10) for c in gen_poly(512)]
    assert max(abs(c) for c in coefs) <= kmax
    # Distribution sanity: sigma_fg = 1.17*sqrt(q/2n) = 4.05 for n=512.
    # 5120 samples -> std-of-std ~ 0.04; the window is ~8 sigma_est wide.
    assert 3.7 < statistics.pstdev(coefs) < 4.4


def test_coefficient_bounds_license_m_budgets():
    # CDT support fits the f,g FFT-load tag (sqrt2 rule): sqrt2*17 < 2^5.
    assert _fft_load_covered(len(_GAUSS_512) // 2, M_B0_COEF_FG)
    # The encoding filter fits the F,G FFT-load tag: sqrt2*127 < 2^8
    # (m=7 fails this: 2*127^2 = 32258 > 4^7 = 16384).
    assert _fft_load_covered(FG_COEF_LIMIT, M_B0_COEF_FG_UP)
