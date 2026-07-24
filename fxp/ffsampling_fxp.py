"""
Fixed-point port of Falcon's signing ffsampling (`ffsampling_fft` in the
reference). Mirrors the float version but uses FxR/FxC throughout and
calls samplerz_fxp at the leaves.

Format:
  - t, z FxC polys at (m_sign, 63); m_sign = M_SIGN_DEFAULT = 18 (tweak) or
    M_SIGN_STD = 21 (std) for Falcon-512, from Lemma 13 (lem:ffsampling)
    with Lemma 12 (lem:inverse-square-root) + γ_root = 24 (drift < 2^18).
  - Tree from `keygen_fxp` / `normalize_tree_fxp`:
      L_10 at root     : m = 5    (NTRUGen Check 4: ‖L_10_root‖_∞ ≤ γ_root = 24)
      L_10 non-root    : m = 0    (|L_10| < 1 strict by Cauchy-Schwarz)
      [dss_i, ccs_i] leaf : m = 0 (precomputed samplerz constants, both < 1)
  - `diff · L_10` is emitted straight at m_sign via `mul_fft_to` (fused
    multiply-and-retag, single rounding); the value fits well below
    2^{m_sign} even at the root where m_L10 = 5 (Lemma 13 drift).

Tree shape:
  internal : [L_10_poly, subtree_left, subtree_right]
  pre-leaf : [L_10_len2, D00_leaf, D11_leaf]         (leaf = [dss_i, ccs_i])
At the base case (len(t[0]) == 1), `tree` is the `[dss_i, ccs_i]` leaf.
"""

from typing import Callable

from beartype import beartype

from fxtypes import FxR, FxC, PolyC, FFLDLTree
from fft_fxp import (
    add_fft_fxp, sub_fft_fxp, mul_fft_to, split_complex_fxp, merge_fft_fxp,
)
from samplerz_fxp import samplerz_fxp
from m_budgets import M_SIGN_DEFAULT


@beartype
def ffsampling_fxp(t: list[PolyC], tree: FFLDLTree,
                   randombytes: Callable[[int], bytes],
                   m_sign: int = M_SIGN_DEFAULT) -> list[PolyC]:
    """Fixed-point signing ffsampling. `t` and the returned `z` are pairs
    of FxC polys at (m_sign, 63). Returns z with zero imaginary parts (z
    is integer-valued).
    """
    n = len(t[0])
    if n == 1:
        # Leaf: `tree` is [dss_i, ccs_i] (precomputed samplerz constants).
        dss_leaf, ccs_leaf = tree[0], tree[1]
        assert isinstance(dss_leaf, FxR), f"leaf must be FxR, got {type(dss_leaf).__name__}"
        z0 = samplerz_fxp(t[0][0].re, dss_leaf, ccs_leaf, randombytes=randombytes)
        z1 = samplerz_fxp(t[1][0].re, dss_leaf, ccs_leaf, randombytes=randombytes)
        # Defensive: |z| ≲ 2^17.6 by Lemma 13, ≪ 2^m_sign=2^18. Catches
        # drift if Lemma 13 premises ever break (e.g. n=1024, larger γ_root).
        assert abs(z0) < (1 << m_sign) and abs(z1) < (1 << m_sign), \
            f"|z|={max(abs(z0), abs(z1))} ≥ 2^{m_sign}"
        p = t[0][0].p                       # follow the target's precision
        return [[FxC.from_int(z0, m=m_sign, p=p)],
                [FxC.from_int(z1, m=m_sign, p=p)]]

    l10_fft, tree0, tree1 = tree

    def _recurse(t_in, subtree):
        """Split → recurse → merge, keeping everything at m_sign."""
        # split preserves the input m, and t is always at m_sign by construction
        # (the caller builds t at m_sign; t0p and every recursive t_split stay
        # there), so the split output needs no retag — assert the invariant.
        t_split = list(split_complex_fxp(t_in))
        assert t_split[0][0].re.m == m_sign, f"split m={t_split[0][0].re.m} != m_sign={m_sign}"
        z_sub = ffsampling_fxp(t_split, subtree, randombytes, m_sign)
        # merge directly at m_sign: z_sub already there, ‖ẑ‖ < 2^m_sign
        # (Lemma 13), so the fixed-m merge fits with no post-retag.
        return merge_fft_fxp(z_sub, m_sign)

    # Right: sample z_1 from t_1.
    z1_fft = _recurse(t[1], tree1)
    # Reduced target: t_0' = t_0 + (t_1 − z_1) · L_10.
    diff = sub_fft_fxp(t[1], z1_fft)
    prod = mul_fft_to(diff, l10_fft, m_sign)
    t0p = add_fft_fxp(t[0], prod)
    # Left: sample z_0 from t_0'.
    z0_fft = _recurse(t0p, tree0)
    return [z0_fft, z1_fft]
