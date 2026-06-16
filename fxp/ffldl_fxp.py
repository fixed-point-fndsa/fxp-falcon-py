"""
Fixed-point ffLDL* / LDL* on a 2x2 Gram matrix of FFT-domain FxC polys.

Mirrors the reference `ldl_fft` / `ffldl_fft` but operates on FxR/FxC.

Three tricks over the textbook LDL formulas:

1. D_11 = G_11 − adj(L_10)·G_10  instead of  G_11 − |L_10|²·G_00.
   Mathematically equivalent (via L_10·G_00 = G_10), but avoids squaring
   L_10, which would inflate intermediate m at the root where m_L10 is
   large.
2. At the NTRU root, det(G) = q² by symplecticity, so D_11 = q²/D_00
   directly — skips the catastrophic subtraction G_11 − |L_10|²·G_00
   where G_11 ~ γ_FG² ≫ D_11. See `ldl_fft_fxp_ntru_root`.
3. Renormalize D_ii to (m_D, p) between levels so m doesn't drift up
   across the recursion. Lemma 9 guarantees D_ii fits uniformly in m_D.

All magnitude budgets (M_D, M_L10_ROOT, M_L10_INNER, M_NORM_OUT) are fixed
constants defined with their derivations in `m_budgets`; the fxp pipeline
runs only on keys passing the full NTRUGen filter, so no fallback is needed.
"""

from beartype import beartype

from fxtypes import (FxR, PolyR, PolyC, Gram, RootGram, FFLDLTree,
                     retag_fxr)
from fft_fxp import (
    adj_fft_fxp,
    mul_fft_fxp,
    div_fft_fxp,
    split_real_fxp,
    retag_poly_fxr,
)
from nr_fxp import rsqrt, nr_reciprocal
from m_budgets import M_NORM_OUT, M_L10_ROOT, M_L10_INNER, M_D, M_D_LEAF


@beartype
def ldl_fft_fxp(G: Gram) -> tuple[PolyC, PolyR, PolyR]:
    """LDL* of a 2x2 Gram at inner levels. Returns (L_10, D_00, D_11): L_10
    complex at (M_L10_INNER, p), D_00/D_11 real at (M_D, p).

    D_11 = G_00 − adj(L_10)·G_10 (trick 1, with G_11 == G_00 — see `Gram`);
    the product adj(L_10)·G_10 = |G_10|²/G_00 is real, so D_11 is its `.re`.
    Asserted invariant: every G entry is at M_D, so the product lands at M_D
    and D_11 stays there.
    """
    G00, G10 = G.g00, G.g10
    ms = (G00[0].m, G10[0].m)
    assert all(m == M_D for m in ms), f"ldl_fft_fxp: G entries {ms} != M_D={M_D}"
    L10 = div_fft_fxp(G10, G00, m_out=M_L10_INNER)
    prod = mul_fft_fxp(adj_fft_fxp(L10), G10)        # = |G10|²/G00, real (im == 0)
    D11 = [g00 - pr.re for g00, pr in zip(G00, prod)]
    return L10, G00, D11


@beartype
def ldl_fft_fxp_ntru_root(G: RootGram, q: int) -> tuple[PolyC, PolyR, PolyR]:
    """LDL* at the NTRU root: by symplecticity det(G) = q², D_11 = q²/D_00
    directly (trick 2) — avoids the cancellation G_11 − |L_10|²·G_00 where
    G_11 ~ γ_FG² ≫ D_11. M_L10_ROOT (= 5) is pinned by NTRUGen Check 4
    (`norm_fft_k`); see `m_budgets`.

    TODO: use symplecticity at a global level.
    """
    G00, G10 = G.g00, G.g10   # G00 real (PolyR), G10 complex (PolyC)
    p = G00[0].p
    L10 = div_fft_fxp(G10, G00, m_out=M_L10_ROOT)
    D00 = retag_poly_fxr(G00, M_D)
    # det(G) = q² (symplectic) ⇒ D_11 = q²/G_00 componentwise; real. Use the
    # NR reciprocal (1/G_00) then multiply, matching div_fft_fxp.
    q_sq_fxr = FxR.from_int(q * q, m=(q * q).bit_length(), p=p)
    D11 = [retag_fxr(q_sq_fxr * nr_reciprocal(u), M_D) for u in G00]
    return L10, D00, D11


def _normalize_leaf_poly(poly, inv_sigma: FxR, sigmin: FxR, iters: int):
    """Replace a length-2 leaf D_ii poly (a real PolyR) by [dss_i, ccs_i] —
    the two samplerz constants, precomputed so samplerz is a pure consumer
    (no division, no square, no sigmin mult). 1/σ_i = √D_ii·inv_sigma
    (= √D_ii/σ), then dss_i = 1/(2σ_i²) = (1/σ_i)²/2 and ccs_i = σ_min/σ_i =
    sigmin·(1/σ_i). All multiplications (the ×½ is a label-only retag); the
    per-leaf reciprocal stays division-free via rsqrt.

    D_ii is Hermitian-positive, so D_ii(ζ_0) is real — `poly[0]` is the FxR."""
    assert len(poly) == 2
    # Tighten the leaf D_ii from the shared M_D=18 to M_D_LEAF=15 (value-
    # preserving): a final leaf satisfies D_ii ≤ 1.17²q < 2^15 (gs_norm), so this
    # sharpens rsqrt's intermediates (m_xy = 15−12 = 3) and σ_i by ~3 bits.
    a_re = retag_fxr(poly[0], M_D_LEAF)
    y = rsqrt(a_re, iters=iters)
    sqrt_D = a_re * y                                              # √D_ii
    inv_sigma_i = retag_fxr(sqrt_D * inv_sigma, M_NORM_OUT)  # 1/σ_i (m=0)
    inv_sq = inv_sigma_i * inv_sigma_i
    half = FxR(x=inv_sq.x, m=inv_sq.m - 1, p=inv_sq.p)            # ÷2 exact
    dss_i = retag_fxr(half, M_NORM_OUT)                      # 1/(2σ_i²)
    ccs_i = retag_fxr(sigmin * inv_sigma_i, M_NORM_OUT)      # σ_min/σ_i
    return [dss_i, ccs_i]


@beartype
def normalize_tree_fxp(tree: FFLDLTree, inv_sigma: FxR, sigmin: FxR,
                       iters: int = 6) -> FFLDLTree:
    """Walk the ffLDL tree and replace every leaf D_ii poly by [dss_i, ccs_i]
    (the two samplerz constants), emitted at m=0 (see `M_NORM_OUT`).
    `inv_sigma` is the 1/σ constant, `sigmin` the per-degree σ_min."""
    args = (inv_sigma, sigmin, iters)
    if len(tree[1]) == 3:  # internal node
        return [tree[0], normalize_tree_fxp(tree[1], *args),
                         normalize_tree_fxp(tree[2], *args)]
    L10, D00, D11 = tree  # pre-leaf: replace D_ii polys with [σ_i, 0]
    return [L10, _normalize_leaf_poly(D00, *args),
                 _normalize_leaf_poly(D11, *args)]


@beartype
def keygen_fxp(
    G_fft_fxp: RootGram,
    q: int,
    inv_sigma: FxR,
    sigmin: FxR,
    iters: int = 6,
) -> FFLDLTree:
    """ffLDL* + normalize_tree on the NTRU Gram, yielding the signing tree.

    The caller builds the Gram `G = B·B*` in FFT domain as a 2x2 matrix of
    FxC polys (in production via `sign_tweak._gram_fft_fxp`, which emits g00/g10
    at the fixed budgets M_G00 / M_G01).
    The m budgets are fixed Falcon-512 constants (M_L10_ROOT=5, M_L10_INNER=0,
    M_D=18 — see `m_budgets` and `ldl_fft_fxp_ntru_root`); the caller's Gram
    must come from a key passing the full NTRUGen filter. `inv_sigma` (= 1/σ,
    INV_SIGMA_FXR_BY_N) and `sigmin` (per-degree σ_min) let the leaves store
    the precomputed samplerz pair [dss_i, ccs_i] — so samplerz needs no
    division. `iters=6` is the rsqrt NR count (see `rsqrt`).
    """
    tree = ffldl_fft_fxp_ntru_root(G_fft_fxp, q=q)
    return normalize_tree_fxp(tree, inv_sigma=inv_sigma, sigmin=sigmin, iters=iters)


def _ffldl_recurse(L10: PolyC, D00: PolyR, D11: PolyR) -> FFLDLTree:
    """Common recursive tail of both ffLDL entry points. Given a node's LDL
    output (L10, D00, D11), return its tree: a leaf [L10, D00, D11] at n == 2
    (`len(D00) == 2`), else split the diagonals into two child Grams and
    recurse with the inner ldl_fft_fxp."""
    if len(D00) == 2:
        return [L10, D00, D11]
    # split each real diagonal into the child Gram [[d, d'], [adj(d'), d]]
    # (g00 = d, g10 = adj(d'); g11 == g00 implied — see `Gram`).
    d00, d01 = split_real_fxp(D00)
    d10, d11 = split_real_fxp(D11)
    G0 = Gram(g00=d00, g10=adj_fft_fxp(d01))
    G1 = Gram(g00=d10, g10=adj_fft_fxp(d11))
    return [L10, ffldl_fft_fxp(G0), ffldl_fft_fxp(G1)]


@beartype
def ffldl_fft_fxp_ntru_root(G: RootGram, q: int) -> FFLDLTree:
    """Full ffLDL* for an NTRU root Gram: symplectic LDL at the root, then
    the common recursive tail. All m budgets are fixed constants."""
    return _ffldl_recurse(*ldl_fft_fxp_ntru_root(G, q))


@beartype
def ffldl_fft_fxp(G: Gram) -> FFLDLTree:
    """Recursive ffLDL* at inner levels (m budgets fixed: M_L10_INNER, M_D):
    generic LDL then the common recursive tail."""
    return _ffldl_recurse(*ldl_fft_fxp(G))
