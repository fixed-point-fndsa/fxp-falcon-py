"""
Measure precision loss of ffLDL* (and its LDL subroutine) in three modes:
floating-point (double), FxP-63, and FxP-127. Compare each against a
high-precision mpmath reference.

Uses Gram matrices G = B·B* built from real NTRU bases B = [[g, -f], [G, -F]]
that pass the NTRUGen filters (γ_fg, γ_hybrid ≤ 4), so the fxp ffLDL runs
with m_L10 = 0 — see `_build_sample`.
"""

import random
from pathlib import Path

import mpmath
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # experiments/

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from _outputs import save_fig, write_csv  # noqa: E402

from fft import fft as fft_float  # noqa: E402
from ffsampling import gram as gram_float, ffldl_fft as ffldl_fft_float  # noqa: E402
from fft import neg  # noqa: E402
from ntrugen import gen_poly, ntru_solve  # noqa: E402

# Full NTRUGen filter: γ_fg (Check 1b), γ_hybrid (Check 2), γ_root (Check 4).
from ntrugen_filters import (  # noqa: E402
    GAMMA_FG_512,
    GAMMA_HYBRID,
    GAMMA_ROOT,
    alpha_hybrid_squared,
    norm_fft_fg,
    norm_fft_k,
)

from fxtypes import FxR, FxC, RootGram  # noqa: E402
from ffldl_fxp import ffldl_fft_fxp_ntru_root  # noqa: E402

# NTRU modulus (for the symplectic relation det(G_root) = q^2).
Q_NTRU = 12289

mpmath.mp.prec = 256


# --------------------------------------------------------------------- #
# mpmath-domain FFT and ffLDL (for the high-precision reference)
# --------------------------------------------------------------------- #


def _mp_roots(n):
    if n == 2:
        return [mpmath.mpc(0, 1), mpmath.mpc(0, -1)]
    prev = _mp_roots(n // 2)
    result = []
    for z in prev:
        s = mpmath.sqrt(z)
        result.append(s)
        result.append(-s)
    return result


def _mp_merge_fft(f0_fft, f1_fft):
    n = 2 * len(f0_fft)
    w = _mp_roots(n)
    out = [mpmath.mpc(0)] * n
    for i in range(n // 2):
        wf1 = w[2 * i] * f1_fft[i]
        out[2 * i] = f0_fft[i] + wf1
        out[2 * i + 1] = f0_fft[i] - wf1
    return out


def _mp_fft(coeffs):
    n = len(coeffs)
    if n == 2:
        a = mpmath.mpc(coeffs[0])
        b = mpmath.mpc(coeffs[1])
        i_unit = mpmath.mpc(0, 1)
        return [a + i_unit * b, a - i_unit * b]
    f0 = coeffs[0::2]
    f1 = coeffs[1::2]
    return _mp_merge_fft(_mp_fft(f0), _mp_fft(f1))


def _mp_split_fft(f_fft):
    n = len(f_fft)
    w = _mp_roots(n)
    f0 = [mpmath.mpc(0)] * (n // 2)
    f1 = [mpmath.mpc(0)] * (n // 2)
    for i in range(n // 2):
        f0[i] = (f_fft[2 * i] + f_fft[2 * i + 1]) / 2
        f1[i] = (f_fft[2 * i] - f_fft[2 * i + 1]) / 2 * mpmath.conj(w[2 * i])
    return f0, f1


def _mp_adj(f_fft):
    return [mpmath.conj(z) for z in f_fft]


def _mp_ldl_fft(G):
    G00, _G01, G10, G11 = G[0][0], G[0][1], G[1][0], G[1][1]
    n = len(G00)
    L10 = [G10[i] / G00[i] for i in range(n)]
    L10_sq = [L10[i] * mpmath.conj(L10[i]) for i in range(n)]
    prod = [L10_sq[i] * G00[i] for i in range(n)]
    D00 = list(G00)
    D11 = [G11[i] - prod[i] for i in range(n)]
    return L10, D00, D11


def _mp_ffldl_fft(G):
    n = len(G[0][0])
    L10, D00, D11 = _mp_ldl_fft(G)
    if n > 2:
        d00, d01 = _mp_split_fft(D00)
        d10, d11 = _mp_split_fft(D11)
        G0 = [[d00, d01], [_mp_adj(d01), d00]]
        G1 = [[d10, d11], [_mp_adj(d11), d10]]
        return [L10, _mp_ffldl_fft(G0), _mp_ffldl_fft(G1)]
    return [L10, D00, D11]


# --------------------------------------------------------------------- #
# Conversions for error measurement
# --------------------------------------------------------------------- #


def _fxr_to_mp(a):
    return mpmath.mpf(a.x) * mpmath.mpf(2) ** (a.m - a.p)


def _fxc_to_mp(z):
    # Tree polys are mixed now: L_10 is FxC, the real diagonal D_00/D_11 is FxR.
    if hasattr(z, "re"):
        return mpmath.mpc(_fxr_to_mp(z.re), _fxr_to_mp(z.im))
    return mpmath.mpc(_fxr_to_mp(z), 0)


def _fxc_poly_to_mp(poly):
    return [_fxc_to_mp(z) for z in poly]


def _float_poly_to_mp(poly):
    return [mpmath.mpc(z.real, z.imag) for z in poly]


# --------------------------------------------------------------------- #
# Tree traversal: max coefficient-wise error at every node
# --------------------------------------------------------------------- #


def _tree_max_err(tree_ref_mp, tree_to_check, convert_poly):
    """Recursively walk the ffldl tree. tree_ref_mp is mpmath. tree_to_check
    is in whatever format; `convert_poly` maps each leaf/node polynomial
    to mpmath for comparison. Returns the maximum per-coefficient error."""
    # Every node has first element = L10 polynomial; check this.
    ref_L10 = tree_ref_mp[0]
    got_L10 = convert_poly(tree_to_check[0])
    err = _poly_max_abs_diff(got_L10, ref_L10)

    # At a leaf (L10 polynomial has length 2), children are D00 and D11 polys.
    if len(tree_ref_mp[0]) == 2:
        for i in (1, 2):
            ref_poly = tree_ref_mp[i]
            got_poly = convert_poly(tree_to_check[i])
            err = max(err, _poly_max_abs_diff(got_poly, ref_poly))
    else:
        for i in (1, 2):
            err = max(err, _tree_max_err(tree_ref_mp[i], tree_to_check[i], convert_poly))
    return err


def _poly_max_abs_diff(got_mp, ref_mp):
    return max(
        float(mpmath.sqrt((g.real - r.real) ** 2 + (g.imag - r.imag) ** 2))
        for g, r in zip(got_mp, ref_mp)
    )


# --------------------------------------------------------------------- #
# Sample generator: random bases B, build Gram in mpmath (exact)
# --------------------------------------------------------------------- #


def _build_sample(n, seed):
    """Sample a real NTRU basis B = [[g, -f], [G, -F]] passing the FULL
    NTRUGen filter — γ_fg (Check 1b), γ_hybrid (Check 2), and γ_root
    (Check 4: ‖fft(L_10_root)‖_∞ ≤ 24). This matches the deployed fxp
    pipeline exactly, so the fixed budgets (M_L10_ROOT=5, M_L10_INNER=0,
    M_D=18) hold with no overflow.

    Involves ntru_solve which is slow for large n; we keep n <= 256 in
    the benchmark."""
    random.seed(seed)
    while True:
        f = gen_poly(n)
        g = gen_poly(n)
        if norm_fft_fg(f, g) > GAMMA_FG_512:
            continue
        if alpha_hybrid_squared(f, g) > GAMMA_HYBRID**2:
            continue
        try:
            F, G = ntru_solve(f, g)
        except (ValueError, AssertionError):
            continue
        F = [int(c) for c in F]
        G = [int(c) for c in G]
        if norm_fft_k(f, g, F, G) > GAMMA_ROOT:   # Check 4 → M_L10_ROOT = 5
            continue
        return [[g, neg(f)], [G, neg(F)]]


def _fft_int_to_mp(f_int):
    """FFT of an integer polynomial, in mpmath complex."""
    return _mp_fft(f_int)


def _gram_from_B_mp(B, n):
    """G = B B* computed in mpmath."""
    B_fft_mp = [[_fft_int_to_mp(B[i][j]) for j in range(2)] for i in range(2)]
    G_mp = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            # G[i][j] = sum_k B[i][k] * adj(B[j][k]), in FFT domain coefficient-wise.
            poly = [mpmath.mpc(0)] * n
            for k in range(2):
                for ell in range(n):
                    poly[ell] += B_fft_mp[i][k][ell] * mpmath.conj(B_fft_mp[j][k][ell])
            G_mp[i][j] = poly
    return G_mp


# --------------------------------------------------------------------- #
# Running each implementation on one sample
# --------------------------------------------------------------------- #


def _run_float(B, n):
    G = gram_float(B)
    G_fft = [[fft_float(G[i][j]) for j in range(2)] for i in range(2)]
    return ffldl_fft_float(G_fft)


def _run_fxp(G_mp, n, p):
    """Convert the mpmath Gram to FxC at precision p, then run ffldl_fft_fxp.

    Use a PER-ENTRY m so that each Gram block gets a tight format (smaller
    m = more precision for intermediates derived from that block).
    """

    def _fxr_from_mp(v_mp, m_, p_):
        return FxR(x=int(mpmath.nint(v_mp * mpmath.mpf(2) ** (p_ - m_))), m=m_, p=p_)

    def _fxc_from_mp(z_mp, m_, p_):
        return FxC(re=_fxr_from_mp(z_mp.real, m_, p_), im=_fxr_from_mp(z_mp.imag, m_, p_))

    def _tight_m(poly):
        return max(1, int(mpmath.ceil(mpmath.log(max(abs(z) for z in poly), 2))) + 1)

    # RootGram (g00, g10 only): diagonal G_00 real (PolyR), off-diagonal
    # g10 complex; G_11 recovered by the NTRU-root LDL via q²/D_00.
    g00_p, g10_p = G_mp[0][0], G_mp[1][0]
    g00 = [_fxr_from_mp(z.real, _tight_m(g00_p), p) for z in g00_p]
    g10 = [_fxc_from_mp(z, _tight_m(g10_p), p) for z in g10_p]
    G_fxp = RootGram(g00=g00, g10=g10)
    # Budgets are fixed constants inside ffldl_fft_fxp_ntru_root (M_L10_ROOT=5,
    # M_L10_INNER=0, M_D=18), valid because _build_sample applies the full
    # NTRUGen filter (Check 4 included). The root LDL computes D_11 via
    # symplecticity (D_11 = q^2 / D_00), avoiding the catastrophic subtraction.
    return ffldl_fft_fxp_ntru_root(G_fxp, q=Q_NTRU)


# --------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------- #


def bench(dims, n_trials: int = 3, seed0: int = 200):
    results = {n: {"fp": [], "fxp63": [], "fxp127": []} for n in dims}

    for n in dims:
        for trial in range(n_trials):
            B = _build_sample(n, seed0 + n * 100 + trial)
            G_mp = _gram_from_B_mp(B, n)

            # Reference tree in mpmath.
            tree_ref_mp = _mp_ffldl_fft(G_mp)

            # Float reference.
            tree_fp = _run_float(B, n)
            err_fp = _tree_max_err(tree_ref_mp, tree_fp, _float_poly_to_mp)
            results[n]["fp"].append(err_fp)

            # FxP63 / FxP127.
            for p, key in [(63, "fxp63"), (127, "fxp127")]:
                tree_fxp = _run_fxp(G_mp, n, p)
                err = _tree_max_err(tree_ref_mp, tree_fxp, _fxc_poly_to_mp)
                results[n][key].append(err)

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
    dims = sorted(results)
    fp = [sorted(results[n]["fp"])[len(results[n]["fp"]) // 2] for n in dims]
    fxp63 = [sorted(results[n]["fxp63"])[len(results[n]["fxp63"]) // 2] for n in dims]
    fxp127 = [sorted(results[n]["fxp127"])[len(results[n]["fxp127"]) // 2] for n in dims]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(dims, fp, "o-", label="FP (float64, p=53)", color="C0", linewidth=2)
    ax.loglog(dims, fxp63, "s-", label="FxP, p=63", color="C1", linewidth=2)
    ax.loglog(dims, fxp127, "^-", label="FxP, p=127", color="C2", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("n (root polynomial dimension)")
    ax.set_ylabel(r"$\max$ coefficient-wise error across ffLDL tree")
    ax.set_title(
        "ffLDL* precision: FP vs FxP-63 vs FxP-127\n"
        "Random small-coef basis B, max error over all tree nodes (vs 256-bit mpmath)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def main():
    # ntru_solve is slow; cap at n=128 for the bench.
    dims = [4, 8, 16, 32, 64, 128]
    print(f"Running {len(dims)} dimensions, 3 trials each...")
    results = bench(dims, n_trials=3)
    print()
    print_table(results)
    print()
    rows = []
    for n in sorted(results):
        med = {k: sorted(results[n][k])[len(results[n][k]) // 2]
               for k in results[n]}
        rows.append([n, f"{med['fp']:.6e}", f"{med['fxp63']:.6e}",
                     f"{med['fxp127']:.6e}"])
    write_csv(HERE / "tables" / "ffldl_precision.csv",
              headers=["n", "fp", "fxp63", "fxp127"], rows=rows)
    save_fig(plot(results, None), "ffldl_precision", HERE)


if __name__ == "__main__":
    main()
