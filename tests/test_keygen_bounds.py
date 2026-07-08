"""Pin the two hard coefficient-domain bounds that make the B0 fixed-m FFT
loads exact: gen_poly's CDT support (n=512: |c| <= 17, the gauss_512 table
length) fits M_B_FG, and the ||F,G||_inf <= 127 encoding filter fits
M_B_FG_UP. The B0 rows load and run the FFT at their certified γ tags
(M_B_FG / M_B_FG_UP), so the only requirement is that the coefficients fit
that tag for the integer load to be exact. A regression (e.g. an unbounded
sum-of-samplerz gen_poly, or a coefficient exceeding the γ tag) would trip
|x| < 2^p on a tail key."""

import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fxp"))
sys.path.insert(0, str(_ROOT / "falcon_ref"))

from ntrugen import gen_poly, _GAUSS_512          # noqa: E402  (falcon_ref/)
from ntrugen_filters import FG_COEF_LIMIT         # noqa: E402  (falcon_ref/)
from m_budgets import M_B_FG, M_B_FG_UP           # noqa: E402  (fxp/)


def test_gen_poly_512_hard_support_and_shape():
    kmax = len(_GAUSS_512) // 2
    assert kmax == 17, "gauss_512 table length changed - re-check M_B_FG fits"
    coefs = [c for _ in range(10) for c in gen_poly(512)]
    assert max(abs(c) for c in coefs) <= kmax
    # Distribution sanity: sigma_fg = 1.17*sqrt(q/2n) = 4.05 for n=512.
    # 5120 samples -> std-of-std ~ 0.04; the window is ~8 sigma_est wide.
    assert 3.7 < statistics.pstdev(coefs) < 4.4


def test_coefficient_bounds_fit_fft_load_tags():
    # f,g coefficients fit the fixed-m FFT tag M_B_FG (exact integer load).
    assert len(_GAUSS_512) // 2 < 2 ** M_B_FG
    # F,G coefficients fit the fixed-m FFT tag M_B_FG_UP.
    assert FG_COEF_LIMIT < 2 ** M_B_FG_UP
