"""
Scalar fixed-point Newton-Raphson primitives: `rsqrt` (1/√x) and
`nr_reciprocal` (1/b).

Both are FxR → FxR, purely multiplicative (no division), and each needs a seed
constant from the generated table (1/√q for rsqrt, 1/q for the reciprocal).
That dependency on the constants is why they live here — one layer above
`fxtypes` (precision-agnostic types + constant-free ops) and the generated
constant tables, and below `fft_fxp` / `ffldl_fxp` (the algorithms that consume
them). Putting them in `fxtypes` would need the seed constants there, creating a
`fxtypes → constants → fxtypes` import cycle.
"""

from beartype import beartype

from fxtypes import FxR, retag_value_fxr
from fxp_constants_p63 import INV_SQRT_Q_FXR as _Y0_P63, INV_Q_FXC as _INV_Q_P63
from fxp_constants_p127 import INV_SQRT_Q_FXR as _Y0_P127, INV_Q_FXC as _INV_Q_P127


# --------------------------------------------------------------------- #
# rsqrt — 1/√x (used by ffLDL leaf normalization)
# --------------------------------------------------------------------- #

# rsqrt internals: Falcon q = 12289 and the domain assert α² = 1.17²
# (NTRUGen `gs_norm ≤ 1.17²·q` filter on D_ii leaves). Encoded as an
# integer ratio to avoid any float at runtime.
_RSQRT_Q = 12289
_RSQRT_ALPHA_SQ_NUM = 13689   # 1.17² · 10000
_RSQRT_ALPHA_SQ_DEN = 10000

# Anchor y0 = 1/√q for the Newton-Raphson seed, keyed by p. m=-6 is the
# natural bound on 1/√x over the domain (also the output m of rsqrt).
_Y0_M = _Y0_P63.m  # = -6 (consistent across p; see generate_constants_fxp.sage)
assert _Y0_P63.m == _Y0_P127.m == -6
_Y0_INV_SQRT_Q_BY_P = {63: _Y0_P63, 127: _Y0_P127}


@beartype
def rsqrt(x: FxR, iters: int = 6) -> FxR:
    """1/sqrt(x) via Newton-Raphson: y ← 0.5·y·(3 − x·y²). Pure fxp,
    no float anywhere. Returns FxR at (m=-6, p=x.p).

    Domain (asserted): `x.value ∈ [q/1.17², 1.17²·q]` ≈ [8970, 16836] for
    Falcon q = 12289. This range is guaranteed pointwise on every ffLDL
    leaf D_ii under Falcon's NTRUGen `gs_norm ≤ 1.17²·q` filter.

    With y0 = 1/√q the worst-case `|ε_0| ≤ |α − 1| ≈ 0.17` (α = 1.17). Each
    NR step satisfies `|ε_{k+1}| ≤ (3/2)·ε_k²·(1 + ε_k/3)`, radius
    `|ε_0| < 2/3`. Iterating the strict bound from ε_0 = 0.17:
        ε_1 ≤ 4.6e-2,  ε_2 ≤ 3.2e-3,  ε_3 ≤ 1.5e-5,
        ε_4 ≤ 3.4e-10, ε_5 ≤ 1.7e-19, ε_6 ≤ 4.3e-38.
    `iters=5` is borderline (ε_5 ≈ 1.7e-19 vs 2^-63 ≈ 1.08e-19, ~½ bit
    over). **`iters=6` is the safe default**, reaching |ε| ≪ 2^-63 with
    margin to spare and absorbing the ~1 ULP fxp rounding noise per step.
    For p=127, `iters=6` is also adequate.
    """
    p = x.p

    # Domain assert: q/α² ≤ x.value ≤ α²·q with α² = N/D = 13689/10000.
    # Rescale q to x's denominator (q_scaled = q · 2^(p-m), exact since
    # x.value ≈ q implies x.m ≪ x.p), then cross-multiply to clear N/D:
    #     D · q_scaled  ≤  N · x.x  ≤  (N²/D) · q_scaled.
    q_scaled = _RSQRT_Q << (x.p - x.m)
    N, D = _RSQRT_ALPHA_SQ_NUM, _RSQRT_ALPHA_SQ_DEN
    assert D * q_scaled <= N * x.x and D * x.x <= N * q_scaled, (
        f"rsqrt: x outside [q/1.17², 1.17²·q] for q={_RSQRT_Q} "
        f"(x.x={x.x}, m={x.m}, p={x.p})"
    )

    m_out = _Y0_M
    y = _Y0_INV_SQRT_Q_BY_P[p]  # already at m=_Y0_M=m_out

    # y ← 0.5·y·(3 − x·y²). The 0.5 is an exact retag (m_xy → m_xy−1).
    # `from_int(3, m=m_xy)` needs m_xy ≥ 2, i.e. x.m ≥ 14 — implied by the
    # domain assert (x.value ≥ q/α² > 2^13 forces m ≥ 14), so no extra check.
    m_xy = x.m + 2 * m_out
    three = FxR.from_int(3, m=m_xy, p=p)
    for _ in range(iters):
        diff = three - (x * (y * y))
        half = FxR(x=diff.x, m=m_xy - 1, p=p)
        y_new = y * half
        y = FxR(x=y_new.x << (y_new.m - m_out), m=m_out, p=p)
    return y


# --------------------------------------------------------------------- #
# nr_reciprocal — 1/b (used by div_fft_fxp and the ffLDL root q²/G_00)
# --------------------------------------------------------------------- #

# NR reciprocal seed y0 = 1/q (real part of INV_Q_FXC), m=-13.
_INV_Q_BY_P = {63: _INV_Q_P63.re, 127: _INV_Q_P127.re}
# Normalization target: q = 12289 ∈ [2^13, 2^14), so we shift the divisor (a
# Gram diagonal in [q/16, 16q]) into that binade and seed with ±1/q.
_RECIP_BINADE = 13                     # target binade exponent


@beartype
def nr_reciprocal(b: FxR) -> FxR:
    """1/b via Newton-Raphson (y ← y·(2 − b·y)), pure fxp, no division.

    `b` is an ffLDL divisor — a Gram diagonal D_ii in [q/16, 16q] by Lemma 9
    (γ_hybrid ≤ 4), asserted below. We normalize |b'| = |b|·2^-k into the binade
    [2^13, 2^14) (an exact label-only retag), seed y0 = sign(b)/q so
    |e0| = |1 − |b'|/q| ≤ 0.333 (⇒ e_k = e0^{2^k}, quadratic), then denormalize
    1/b = (1/b')·2^-k (exact). 6 iters reach |e| ≪ 2^-63 (e_6 ≈ 3e-31); p=127
    needs 7 (e_7 ≈ 8e-62). The 2^±k shifts are lossless, so the only error is
    ~1 ULP/step fxp rounding — input-limited (e.g. ~2^-57 for a divisor stored
    at m=18). Not constant-time (bit_length normalization) — division is
    keygen-only, so acceptable."""
    p = b.p
    assert b.x != 0, "nr_reciprocal: division by zero"
    # value(b) = b.x·2^{b.m-p}; e_b = floor(log2 |value|) (bit_length ignores sign).
    e_b = (b.x.bit_length() - 1) + (b.m - p)
    # DOMAIN GUARD — DO NOT REMOVE. Every divisor is a Gram diagonal in
    # [q/16, 16q] (Lemma 9, γ_hybrid ≤ 4). This assert IS the contract: it
    # catches an unfiltered key or an upstream bug, and keeps the routine honest
    # about what it inverts. Dropping it (e.g. to accept a generic divisor)
    # silently re-opens an over-general path that should not exist.
    assert 9 <= e_b <= 17, (
        f"nr_reciprocal: |b| outside [q/16, 16q] (floor log2 = {e_b}, expect [9, 17])"
    )
    k = e_b - _RECIP_BINADE
    b_norm = FxR(x=b.x, m=b.m - k, p=p)              # |value| ∈ [2^13, 2^14), exact

    seed = _INV_Q_BY_P[p]                            # 1/q (m ≈ -13)
    # 1/b' ∈ (2^-14, 2^-13]; its tight magnitude bound is seed.m+1 (the top,
    # 2^-13, is NOT < 2^{seed.m}). Running the loop at m_y also avoids x = 2^p
    # for the boundary b_norm = 2^13 (an exact-power-of-2 divisor — never a real
    # Gram diagonal, but cheap to keep correct).
    m_y = seed.m + 1
    y = retag_value_fxr(seed, m_y)                   # 1/q at m_y
    if b.x < 0:
        y = FxR(x=-y.x, m=m_y, p=p)                  # sign(b)/q
    two = FxR.from_int(2, m=2, p=p)
    for _ in range(7 if p > 63 else 6):
        by = retag_value_fxr(b_norm * y, 2)          # b'·y ≈ 1 (m=2)
        diff = two - by                              # 2 − b'·y ≈ 1 (m=2)
        y_new = y * diff                             # m = m_y + 2
        y = FxR(x=y_new.x << (y_new.m - m_y), m=m_y, p=p)   # back to m_y (exact)

    return FxR(x=y.x, m=y.m - k, p=p)                # 1/b = (1/b')·2^-k, exact
