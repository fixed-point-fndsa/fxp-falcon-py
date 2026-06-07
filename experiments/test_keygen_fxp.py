"""
Precision benchmark for `keygen_fxp` vs a 256-bit mpmath reference.

Pipeline:
  1. Build a valid NTRU basis (gamma_hybrid <= 4 so Lemma 9 holds).
  2. Compute the Gram G = B B* in FFT domain in mpmath (exact).
  3. Run three pipelines and compare to the mpmath reference:
       - FP:     ffldl_fft_float + normalize_tree  (float64)
       - FxP63:  keygen_fxp at p=63
       - FxP127: keygen_fxp at p=127
  4. Report max coefficient-wise error across:
       (a) L_10 polys at every tree node, and
       (b) sigma_i FxR values at every leaf (after normalization).

The first type of error stresses ffldl; the second stresses rsqrt/normalize.
"""

import copy
import random
from pathlib import Path

import mpmath

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from _outputs import save_fig  # noqa: E402

# Float reference.
from fft import neg  # noqa: E402
from ffsampling import ffldl_fft as ffldl_fft_float  # noqa: E402
from falcon import normalize_tree  # noqa: E402
from ntrugen import gen_poly, ntru_solve, gs_norm  # noqa: E402

# fxp.
from fxtypes import FxR, FxC, RootGram  # noqa: E402
from ffldl_fxp import keygen_fxp  # noqa: E402

# NTRUGen filters.
from ntrugen_filters import (  # noqa: E402
    GAMMA_FG_512,
    GAMMA_HYBRID,
    GAMMA_ROOT,
    alpha_hybrid_squared,
    norm_fft_fg,
    norm_fft_k,
)

# Reuse the mpmath FFT / ffldl machinery from the ffldl bench.
from bench_ffldl_precision import (  # noqa: E402
    _mp_split_fft,
    _mp_adj,
    _gram_from_B_mp,
    _fxr_to_mp,
    _fxc_poly_to_mp,
    _float_poly_to_mp,
)

mpmath.mp.prec = 256

Q = 12289
SIGMA_FALCON = 165.7366171829776  # Params[1024]["sigma"]; any sigma works here.


# --------------------------------------------------------------------- #
# Mpmath-domain ldl/ffldl with symplecticity + normalize, matching the fxp
# pipeline as closely as possible (so the reference is semantically the
# same algorithm, not a different LDL variant).
# --------------------------------------------------------------------- #


def _mp_ldl_fft_ntru_root(G, q_mp):
    """Root LDL in mpmath: L_10 = G_10 / G_00, D_00 = G_00,
    D_11 = q^2 / G_00 (symplecticity — same trick as the fxp root)."""
    G00, G10 = G[0][0], G[1][0]
    n = len(G00)
    L10 = [G10[i] / G00[i] for i in range(n)]
    D00 = list(G00)
    q_sq = q_mp * q_mp
    D11 = [q_sq / G00[i] for i in range(n)]
    return L10, D00, D11


def _mp_ldl_fft(G):
    """Inner LDL in mpmath: uses the same reformulation as the fxp code,
    D_11 = G_11 - conj(L_10) * G_10 (no squaring of L_10)."""
    G00, G10, G11 = G[0][0], G[1][0], G[1][1]
    n = len(G00)
    L10 = [G10[i] / G00[i] for i in range(n)]
    D00 = list(G00)
    D11 = [G11[i] - mpmath.conj(L10[i]) * G10[i] for i in range(n)]
    return L10, D00, D11


def _mp_ffldl_inner(G):
    n = len(G[0][0])
    L10, D00, D11 = _mp_ldl_fft(G)
    if n > 2:
        d00, d01 = _mp_split_fft(D00)
        d10, d11 = _mp_split_fft(D11)
        G0 = [[d00, d01], [_mp_adj(d01), d00]]
        G1 = [[d10, d11], [_mp_adj(d11), d10]]
        return [L10, _mp_ffldl_inner(G0), _mp_ffldl_inner(G1)]
    return [L10, D00, D11]


def _mp_ffldl_ntru_root(G, q_mp):
    n = len(G[0][0])
    L10, D00, D11 = _mp_ldl_fft_ntru_root(G, q_mp)
    if n > 2:
        d00, d01 = _mp_split_fft(D00)
        d10, d11 = _mp_split_fft(D11)
        G0 = [[d00, d01], [_mp_adj(d01), d00]]
        G1 = [[d10, d11], [_mp_adj(d11), d10]]
        return [L10, _mp_ffldl_inner(G0), _mp_ffldl_inner(G1)]
    return [L10, D00, D11]


def _mp_normalize_tree(tree, sigma_mp):
    """Walk the tree (mpmath). At each leaf D_ii (length-2 poly of scalars),
    replace it with [sigma / sqrt(D_ii[0].real), 0]. Matches normalize_tree_fxp.

    Tree shapes:
      - Internal: [L_10_poly, subtree, subtree]  (subtree[0] is itself a list)
      - Leaf:     [L_10_poly, D_00_poly, D_11_poly]  (D_ii[0] is a scalar)
    """
    # tree[1] is either a subtree (list whose first element is a list == L_10)
    # or a D_00 leaf poly (list whose first element is a scalar).
    is_internal = isinstance(tree[1][0], list)
    if is_internal:
        _mp_normalize_tree(tree[1], sigma_mp)
        _mp_normalize_tree(tree[2], sigma_mp)
    else:
        for idx in (1, 2):
            poly = tree[idx]
            val = sigma_mp / mpmath.sqrt(poly[0].real)
            tree[idx] = [val, mpmath.mpc(0)]


# --------------------------------------------------------------------- #
# Sampling + float->fxp conversion (same as bench_ffldl_precision).
# --------------------------------------------------------------------- #


def sample_ntru_basis(n: int, seed: int):
    """Sample (f, g, F, G) matching Falcon's actual NTRUGen filters:
      - γ_fg     (‖fft(f), fft(g)‖_∞ ≤ 256)        — paper Check 1
      - γ_hybrid (α_hybrid² ≤ 16)                   — paper Check 2
      - gs_norm  (‖B‖_GS² ≤ 1.17²·q)                — Falcon spec line 9 of
                                                      Algorithm 5; this is
                                                      what guarantees every
                                                      ffLDL leaf D_ii lands
                                                      in [q/1.37, 1.37·q]
                                                      (the rsqrt assert
                                                      domain).
      - γ_root   (‖fft(L_10_root)‖_∞ ≤ 24)          — paper Check 4; pins
                                                      M_L10_ROOT = 5.
    Without the gs_norm filter, sampled keys can produce D_ii outside the
    rsqrt convergence radius, manifesting either as an outright assert
    failure or as a precision spike where ε_0 ≫ 0.17. The full filter
    (Check 4 included) matches the deployed fxp pipeline, so keygen_fxp's
    fixed budgets hold with no overflow.
    """
    random.seed(seed)
    gs_bound = (1.17 ** 2) * Q
    while True:
        f = gen_poly(n)
        g = gen_poly(n)
        if norm_fft_fg(f, g) > GAMMA_FG_512:
            continue
        if alpha_hybrid_squared(f, g) > GAMMA_HYBRID**2:
            continue
        if gs_norm(f, g, Q) > gs_bound:
            continue
        try:
            F, G = ntru_solve(f, g)
        except (ValueError, AssertionError):
            continue
        F = [int(c) for c in F]
        G = [int(c) for c in G]
        if norm_fft_k(f, g, F, G) > GAMMA_ROOT:   # Check 4 → M_L10_ROOT = 5
            continue
        return f, g, F, G


def _tight_m_from_mp(max_abs: "mpmath.mpf") -> int:
    return max(1, int(mpmath.ceil(mpmath.log(max_abs, 2))) + 1)


def gram_mp_to_fxp(G_mp, p: int):
    """Convert a 2x2 mpmath Gram to a RootGram (g00, g10): real diagonal G_00
    (PolyR), complex off-diagonal g10; G_11 recovered by the root LDL."""
    def _fxr_from_mp(v_mp, m_, p_):
        return FxR(x=int(mpmath.nint(v_mp * mpmath.mpf(2) ** (p_ - m_))), m=m_, p=p_)

    def _fxc_from_mp(z_mp, m_, p_):
        return FxC(re=_fxr_from_mp(z_mp.real, m_, p_), im=_fxr_from_mp(z_mp.imag, m_, p_))

    g00_p, g10_p = G_mp[0][0], G_mp[1][0]
    g00 = [_fxr_from_mp(z.real, _tight_m_from_mp(max(abs(z) for z in g00_p)), p) for z in g00_p]
    g10 = [_fxc_from_mp(z, _tight_m_from_mp(max(abs(z) for z in g10_p)), p) for z in g10_p]
    return RootGram(g00=g00, g10=g10)


def gram_mp_to_float(G_mp):
    """Convert a 2x2 mpmath Gram matrix to float64 complex."""
    return [
        [[complex(float(z.real), float(z.imag)) for z in G_mp[i][j]] for j in range(2)]
        for i in range(2)
    ]


# --------------------------------------------------------------------- #
# Comparisons: walk reference mpmath tree alongside the candidate tree,
# compare L_10 polys at every node + normalized leaves.
# --------------------------------------------------------------------- #


def _abs_diff_mp(a_mp, b_mp):
    """|a - b| as a float, where a_mp, b_mp are mpmath scalars (complex or real)."""
    d = a_mp - b_mp
    if isinstance(d, mpmath.mpc):
        return float(mpmath.sqrt(d.real * d.real + d.imag * d.imag))
    return float(abs(d))


def _poly_max_abs_diff_mp(got_mp, ref_mp):
    return max(_abs_diff_mp(g, r) for g, r in zip(got_mp, ref_mp))


def tree_max_err(tree_ref_mp, tree_to_check, convert_poly, convert_leaf_scalar):
    """Walk the trees in lockstep; compare L_10 polys at internal nodes,
    and the normalized leaf scalars at leaves."""
    # L_10 comparison at this node.
    ref_L10 = tree_ref_mp[0]
    got_L10 = convert_poly(tree_to_check[0])
    err = _poly_max_abs_diff_mp(got_L10, ref_L10)

    # Are children leaves or subtrees? In the mpmath reference,
    # tree_ref_mp[1] is either a subtree (list of length 3) whose [0] is a
    # polynomial (list), or a normalized leaf [scalar, 0].
    child = tree_ref_mp[1]
    is_leaf = not isinstance(child[0], list)
    if is_leaf:
        for idx in (1, 2):
            ref_scalar = tree_ref_mp[idx][0]  # normalized sigma_i (mpmath)
            got_scalar_mp = convert_leaf_scalar(tree_to_check[idx][0])
            err = max(err, _abs_diff_mp(got_scalar_mp, ref_scalar))
    else:
        for idx in (1, 2):
            err = max(
                err,
                tree_max_err(
                    tree_ref_mp[idx], tree_to_check[idx],
                    convert_poly, convert_leaf_scalar,
                ),
            )
    return err


# --------------------------------------------------------------------- #
# Runners: one sample, all three precisions.
# --------------------------------------------------------------------- #


def run_fp(G_mp, sigma):
    """Float reference: go through gram_mp_to_float + float ffldl + normalize_tree."""
    G_float = gram_mp_to_float(G_mp)
    # The reference ffldl_fft mutates; deepcopy to be safe.
    tree = ffldl_fft_float(copy.deepcopy(G_float))
    normalize_tree(tree, sigma)
    return tree


def run_fxp(G_mp, p: int, sigma: float):
    G_fxp = gram_mp_to_fxp(G_mp, p=p)
    # 1/σ at full p-bit precision (mpmath reciprocal, NOT float64 1/σ) so the
    # benchmark measures the fxp arithmetic, not the constant's float64 limit.
    inv_sigma_mp = 1 / mpmath.mpf(sigma)
    inv_sigma_x = int(mpmath.nint(inv_sigma_mp * mpmath.mpf(2) ** (p - (-7))))
    inv_sigma_fxr = FxR(x=inv_sigma_x, m=-7, p=p)
    # σ_min only feeds ccs_i (leaf slot 1, not compared here); any valid value.
    sigmin_fxr = FxR.from_float(1.2778336969128337, m=1, p=p)
    return keygen_fxp(
        G_fxp,
        q=Q,
        inv_sigma=inv_sigma_fxr,
        sigmin=sigmin_fxr,
        iters=6,  # safe default for both p=63 and p=127 (see ffldl_fxp.rsqrt)
    )


def _float_leaf_to_mp(x):
    # Float leaf scalar: complex σ_i (with zero imag after normalize_tree).
    return mpmath.mpc(float(x.real), float(x.imag))


def _fxp_leaf_to_mp(x):
    # Fxp leaf slot 0 is now dss_i = 1/(2σ_i²); recover σ_i = 1/√(2·dss_i) to
    # measure the error against the σ_i references (float + mpmath), unchanged.
    dss = mpmath.mpf(_fxr_to_mp(x))
    return 1 / mpmath.sqrt(2 * dss)


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


def bench(dims, trials_by_n=None, n_trials: int = 3, seed0: int = 300):
    """trials_by_n: optional dict {n: n_trials}. Falls back to n_trials if missing."""
    if trials_by_n is None:
        trials_by_n = {}
    results = {n: {"fp": [], "fxp63": [], "fxp127": []} for n in dims}
    q_mp = mpmath.mpf(Q)
    sigma_mp = mpmath.mpf(SIGMA_FALCON)
    import time as _time
    for n in dims:
        t0 = _time.time()
        trials = trials_by_n.get(n, n_trials)
        for trial in range(trials):
            f, g, F, G = sample_ntru_basis(n, seed0 + n * 100 + trial)
            B = [[g, neg(f)], [G, neg(F)]]
            G_mp = _gram_from_B_mp(B, n)

            # Reference tree (mpmath) with the SAME algorithm as fxp (root
            # uses symplecticity; inner uses D_11 = G_11 - conj(L_10)*G_10).
            tree_ref_mp = _mp_ffldl_ntru_root(G_mp, q_mp)
            _mp_normalize_tree(tree_ref_mp, sigma_mp)

            # FP.
            tree_fp = run_fp(G_mp, SIGMA_FALCON)
            err_fp = tree_max_err(
                tree_ref_mp, tree_fp, _float_poly_to_mp, _float_leaf_to_mp
            )
            results[n]["fp"].append(err_fp)

            # FxP-63, FxP-127.
            for p, key in [(63, "fxp63"), (127, "fxp127")]:
                tree_fxp = run_fxp(G_mp, p=p, sigma=SIGMA_FALCON)
                err_fxp = tree_max_err(
                    tree_ref_mp, tree_fxp, _fxc_poly_to_mp, _fxp_leaf_to_mp
                )
                results[n][key].append(err_fxp)
        print(f"  [n={n}] {trials} trial(s) in {_time.time()-t0:.0f}s")
    return results


def print_table(results):
    print(f"{'n':>5} | {'FP (float64)':>14} | {'FxP p=63':>14} | {'FxP p=127':>14}")
    print("-" * 60)
    for n in sorted(results):
        med = {k: sorted(results[n][k])[len(results[n][k]) // 2] for k in results[n]}
        print(
            f"{n:>5} | {med['fp']:>14.3e} | {med['fxp63']:>14.3e} | {med['fxp127']:>14.3e}"
        )


def plot(results, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    dims = sorted(results)
    fp = [sorted(results[n]["fp"])[len(results[n]["fp"]) // 2] for n in dims]
    fxp63 = [sorted(results[n]["fxp63"])[len(results[n]["fxp63"]) // 2] for n in dims]
    fxp127 = [sorted(results[n]["fxp127"])[len(results[n]["fxp127"]) // 2] for n in dims]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dims, fp,     "o-", label="FP (float64, p=53)", color="C0", linewidth=2)
    ax.plot(dims, fxp63,  "s-", label="FxP, p=63",          color="C1", linewidth=2)
    ax.plot(dims, fxp127, "^-", label="FxP, p=127",         color="C2", linewidth=2)

    # Precision floors (theoretical lower bounds for each representation).
    for level, label, color in [(2 ** -53,  "$2^{-53}$ (float64 ULP)", "C0"),
                                 (2 ** -63,  "$2^{-63}$ (FxP-63 ULP)",  "C1"),
                                 (2 ** -127, "$2^{-127}$ (FxP-127 ULP)", "C2")]:
        ax.axhline(level, color=color, linestyle=":", linewidth=0.9, alpha=0.55)

    # Base-2 axes with mathtext labels.
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=2))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=2))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=2, numticks=18))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=2))

    ax.set_xlabel(r"$n$ (root polynomial dimension)")
    ax.set_ylabel(r"max error (L$_{10}$ nodes + normalized $\sigma_i$ leaves)")
    ax.set_title(
        "keygen_fxp precision: FP vs FxP-63 vs FxP-127\n"
        "ffLDL + normalize_tree, error measured against 256-bit mpmath reference"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def main():
    dims = [4, 8, 16, 32, 64, 128, 256, 512]
    # More samples around the n=16 region (cheap to run), few at the top
    # (ntru_solve and mpmath FFT slow down quadratically).
    trials_by_n = {4: 10, 8: 20, 16: 20, 32: 20, 64: 10,
                   128: 5, 256: 2, 512: 1}
    print(f"Running {len(dims)} dimensions...")
    results = bench(dims, trials_by_n=trials_by_n, n_trials=3)
    print()
    print_table(results)
    print()

    # Also report per-trial spread at n=16 to diagnose any spike.
    print("Per-trial spread at n=16 (sanity check for spikes):")
    for k in ("fp", "fxp63", "fxp127"):
        vals = sorted(results[16][k])
        print(f"  {k:>6}: min={vals[0]:.2e}  med={vals[len(vals)//2]:.2e}  "
              f"max={vals[-1]:.2e}  (n={len(vals)})")

    save_fig(plot(results, None), "keygen_precision", HERE)


if __name__ == "__main__":
    main()
