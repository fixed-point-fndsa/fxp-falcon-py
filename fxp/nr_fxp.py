"""
Scalar fixed-point Newton-Raphson primitives: `rsqrt` (1/√x) and
`nr_reciprocal` (1/b). Both are FxR → FxR and division-free (purely
multiplicative). They live here, not in `fxtypes`, because each needs a seed
constant (1/√q, 1/q) from the generated tables — importing those into `fxtypes`
would create a cycle.
"""

from beartype import beartype

from fxtypes import FxR, retag_fxr
from fxp_constants_p63 import INV_SQRT_Q_FXR as _Y0_P63, INV_Q_FXC as _INV_Q_P63
from fxp_constants_p127 import INV_SQRT_Q_FXR as _Y0_P127, INV_Q_FXC as _INV_Q_P127


# --------------------------------------------------------------------- #
# rsqrt — 1/√x (used by ffLDL leaf normalization)
# --------------------------------------------------------------------- #

# Domain α² = 1.17² (NTRUGen gs_norm ≤ 1.17²·q filter on D_ii leaves), as an
# integer ratio to stay float-free at runtime.
_RSQRT_Q = 12289
_RSQRT_ALPHA_SQ_NUM = 13689   # 1.17² · 10000
_RSQRT_ALPHA_SQ_DEN = 10000

# NR seed y0 = 1/√q, keyed by p; m = -6 is the output bound of rsqrt.
_Y0_M = _Y0_P63.m  # = -6
assert _Y0_P63.m == _Y0_P127.m == -6
_Y0_INV_SQRT_Q_BY_P = {63: _Y0_P63, 127: _Y0_P127}


@beartype
def rsqrt(x: FxR, iters: int = 6) -> FxR:
    """1/√x via Newton-Raphson (y ← 0.5·y·(3 − x·y²)), pure fxp. Returns FxR at
    (m=-6, p=x.p).

    Domain (asserted): x.value ∈ [q/1.17², 1.17²·q], guaranteed on every ffLDL
    leaf D_ii by Falcon's NTRUGen gs_norm ≤ 1.17²·q filter.

    The quadratic NR converges from |ε_0| ≤ 0.17 (seed y0 = 1/√q); iters=6 reaches
    |ε| ≪ 2^-63 with margin (iters=5 is ~½ bit short). Adequate for p=127 too.
    """
    p = x.p

    # Cross-multiply the domain bound q/α² ≤ value ≤ α²·q by N/D = 13689/10000,
    # with q rescaled to x's denominator (exact), to stay in integer arithmetic.
    q_scaled = _RSQRT_Q << (x.p - x.m)
    N, D = _RSQRT_ALPHA_SQ_NUM, _RSQRT_ALPHA_SQ_DEN
    assert D * q_scaled <= N * x.x and D * x.x <= N * q_scaled, \
        f"rsqrt: x ∉ [q/1.17², 1.17²·q] (x.x={x.x}, m={x.m})"

    m_out = _Y0_M
    y = _Y0_INV_SQRT_Q_BY_P[p]  # already at m=_Y0_M=m_out

    # y ← 0.5·y·(3 − x·y²); the 0.5 is an exact retag (m_xy → m_xy−1). m_xy ≥ 2
    # (so from_int(3) is valid) follows from x.value > 2^13 ⇒ x.m ≥ 14.
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

# NR seed y0 = 1/q (m ≈ -13). We shift the divisor (a Gram diagonal in
# [q/16, 16q]) into the binade [2^13, 2^14) where q lives, and seed with ±1/q.
_INV_Q_BY_P = {63: _INV_Q_P63.re, 127: _INV_Q_P127.re}
_RECIP_BINADE = 13


@beartype
def nr_reciprocal(b: FxR) -> FxR:
    """1/b via Newton-Raphson (y ← y·(2 − b·y)), pure fxp, no division.

    `b` is an ffLDL divisor — a Gram diagonal in [q/16, 16q] (Lemma 9, γ_hybrid ≤
    4), asserted below. We normalize |b| into the binade [2^13, 2^14) (exact
    label-only shift by k), seed y0 = sign(b)/q so |e0| ≤ 0.333 (quadratic),
    iterate, then denormalize 1/b = (1/b')·2^-k (exact). 6 iters (7 for p=127)
    reach |e| ≪ 2^-63; residual ~2^-59 relative (1 ULP/step at the binade scale).
    Not constant-time (bit_length normalization) — division is keygen-only."""
    p = b.p
    assert b.x != 0, "nr_reciprocal: division by zero"
    # value(b) = b.x·2^{b.m-p}; e_b = floor(log2 |value|) (bit_length ignores sign).
    e_b = (b.x.bit_length() - 1) + (b.m - p)
    # DOMAIN GUARD — do not remove. This assert IS the contract (divisor ∈
    # [q/16, 16q], Lemma 9); it catches an unfiltered key or upstream bug rather
    # than silently inverting a generic divisor.
    assert 9 <= e_b <= 17, f"nr_reciprocal: |b| ∉ [q/16, 16q] (floor log2={e_b}, want [9,17])"
    k = e_b - _RECIP_BINADE
    # Exact label-only normalization, then tighten the tag to m = 14 (the
    # binade's bound; the retag is an exact left shift). A loose input tag would
    # otherwise cost ~1 bit of loop accuracy per binade of slack.
    b_norm = retag_fxr(FxR(x=b.x, m=b.m - k, p=p), _RECIP_BINADE + 1)

    seed = _INV_Q_BY_P[p]                            # 1/q (m ≈ -13)
    # 1/b' ∈ (2^-14, 2^-13] ⇒ tight bound m_y = seed.m+1; running the loop at m_y
    # also keeps it off x = 2^p at the b_norm = 2^13 boundary.
    m_y = seed.m + 1
    y = retag_fxr(seed, m_y)                   # 1/q at m_y
    if b.x < 0:
        y = FxR(x=-y.x, m=m_y, p=p)                  # sign(b)/q
    two = FxR.from_int(2, m=2, p=p)
    for _ in range(7 if p > 63 else 6):
        by = b_norm * y                        # b'·y ≈ 1 at m = 14 + m_y = 2 (the sub below asserts it)
        diff = two - by                              # 2 − b'·y ≈ 1 (m=2)
        y_new = y * diff                             # m = m_y + 2
        y = FxR(x=y_new.x << (y_new.m - m_y), m=m_y, p=p)   # back to m_y (exact)

    return FxR(x=y.x, m=y.m - k, p=p)                # 1/b = (1/b')·2^-k, exact
