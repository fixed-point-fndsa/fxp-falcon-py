"""
Real-conditions ffLDL precision for the camera-ready: how the ABSOLUTE
precision of `ffldl_fft_fxp` evolves as we DESCEND the ffLDL tree, on real
Falcon-512 keys through the deployed code path.

Unlike a kernel characterisation on an idealised (tight-m, mpmath-built) Gram
swept over synthetic root dimensions, this runs the actual production chain on
real keys —

    real key ─▶ B0 ─fxp FFT─▶ _gram_fft_fxp ─▶ ffldl_fft_fxp_ntru_root

— and measures, at EACH tree level (root n=512, then 256, …, down to the n=2
leaves), the error of the L_10 block vs a 256-bit mpmath reference built from
the exact integer key. The x-axis is the sub-polynomial size at each level
(= depth), the meaningful "scaling" axis: it shows how rounding accumulates as
the recursion splits down to the leaves.

Metric: ABSOLUTE per-coefficient error |L10_fxp(i) − L10_mpmath(i)|, aggregated
as the MEAN SQUARED ERROR over all (coefficient × key) pairs at each level
(MSE = mean e²; RMSE = √MSE). MSE is the quantity that feeds the Rényi-
divergence argument (Prest'17), hence preferred over median/max.

float64 / fxp-63 (deployed) / fxp-127, vs the exact mpmath tree.
"""

import math
from pathlib import Path

import mpmath

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402

from _outputs import save_fig, write_csv  # noqa: E402

from _precision_ref import (  # noqa: E402
    _mp_ffldl_fft, _gram_from_B_mp,
    _fxc_poly_to_mp, _float_poly_to_mp, _run_float,
    _abs_errs, _mse, _log2,
)

from falcon import SecretKey  # noqa: E402
from fft import neg  # noqa: E402
from fxtypes import FxR, RootGram  # noqa: E402
from fft_fxp import fft_fxp, retag_poly_fxr, retag_poly_fxc  # noqa: E402
from ffldl_fxp import ffldl_fft_fxp_ntru_root  # noqa: E402
from m_budgets import M_G00, M_G01, M_B0_COEF, M_B_FG, M_B_FG_UP  # noqa: E402
from sign_tweak import _gram_fft_fxp  # noqa: E402

Q = 12289
mpmath.mp.prec = 256

# `_abs_errs` / `_mse` / `_log2` are shared from `_precision_ref`.


def _b0_fft_at_p(sk, p):
    """fxp FFT of B0 = [[g,−f],[G,−F]] at precision p, retagged to the tight γ
    bounds (M_B_FG / M_B_FG_UP) like the deployed `_build_B0_fft_fxp_cache` —
    `_gram_fft_fxp` asserts that contract (no cache here, so p-parametric)."""
    rows = [[sk.g, neg(sk.f)], [sk.G, neg(sk.F)]]
    [a, b], [c, d] = [[fft_fxp([FxR.from_int(co, m=M_B0_COEF, p=p) for co in poly])
                       for poly in row] for row in rows]
    return [[retag_poly_fxc(a, M_B_FG), retag_poly_fxc(b, M_B_FG)],
            [retag_poly_fxc(c, M_B_FG_UP), retag_poly_fxc(d, M_B_FG_UP)]]


def _prod_tree(sk, p):
    """Deployed keygen path at precision p: B0 → _gram_fft_fxp → ffLDL root."""
    gram = _gram_fft_fxp(_b0_fft_at_p(sk, p))
    G = RootGram(g00=retag_poly_fxr(gram.g00, M_G00),
                 g10=retag_poly_fxc(gram.g10, M_G01))
    return ffldl_fft_fxp_ntru_root(G, q=Q)


def _walk_levels(ref_mp, tree, conv, acc):
    """Collect every L_10 per-coefficient absolute error at each tree level,
    keyed by sub-poly size (512 at the root, halving each level)."""
    n_level = len(ref_mp[0])
    acc.setdefault(n_level, []).extend(_abs_errs(conv(tree[0]), ref_mp[0]))
    if n_level > 2:
        _walk_levels(ref_mp[1], tree[1], conv, acc)
        _walk_levels(ref_mp[2], tree[2], conv, acc)


def measure_key(sk):
    """Return {n_level: {mode: [abs errors]}} for one key."""
    B = [[sk.g, neg(sk.f)], [sk.G, neg(sk.F)]]
    ref_mp = _mp_ffldl_fft(_gram_from_B_mp(B, sk.n))

    per = {}
    for mode, tree, conv in (
        ("fp", _run_float(B, sk.n), _float_poly_to_mp),
        ("fxp63", _prod_tree(sk, 63), _fxc_poly_to_mp),
        ("fxp127", _prod_tree(sk, 127), _fxc_poly_to_mp),
    ):
        acc = {}
        _walk_levels(ref_mp, tree, conv, acc)
        for n_level, errs in acc.items():
            per.setdefault(n_level, {}).setdefault(mode, []).extend(errs)
    return per


def bench(n_keys):
    levels = {}
    for k in range(n_keys):
        sk = SecretKey(512)
        per = measure_key(sk)
        for n_level, modes in per.items():
            for mode, errs in modes.items():
                levels.setdefault(n_level, {}).setdefault(mode, []).extend(errs)
        print(f"  key {k} done")
    return levels


def report(levels):
    print(f"\n{'level n':>8} | {'float64 MSE':>14} | {'fxp-63 MSE':>14} | {'fxp-127 MSE':>14}"
          f"   (absolute; RMSE=√MSE in parens, log2)")
    print("-" * 80)
    rows = []
    for n in sorted(levels, reverse=True):
        mse = {m: _mse(levels[n].get(m, [])) for m in ("fp", "fxp63", "fxp127")}
        def cell(m):
            return f"{mse[m]:.2e} (2^{_log2(math.sqrt(mse[m])):.1f})" if mse[m] == mse[m] else "n/a"
        print(f"{n:>8} | {cell('fp'):>14} | {cell('fxp63'):>14} | {cell('fxp127'):>14}")
        rows.append([n,
                     f"{mse['fp']:.6e}", f"{math.sqrt(mse['fp']):.6e}",
                     f"{mse['fxp63']:.6e}", f"{math.sqrt(mse['fxp63']):.6e}",
                     f"{mse['fxp127']:.6e}", f"{math.sqrt(mse['fxp127']):.6e}"])
    return rows


def plot(levels):
    import matplotlib.pyplot as plt
    ns = sorted(levels, reverse=True)
    depths = list(range(len(ns)))
    rmse = {m: [math.sqrt(_mse(levels[n].get(m, []))) for n in ns]
            for m in ("fp", "fxp63", "fxp127")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(depths, rmse["fp"], "o-", label="float64", color="C0", lw=2)
    ax.plot(depths, rmse["fxp63"], "s-", label="fxp-63 (deployed)", color="C1", lw=2)
    ax.plot(depths, rmse["fxp127"], "^-", label="fxp-127", color="C2", lw=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([f"{n}\n(d={d})" for d, n in zip(depths, ns)])
    ax.set_xlabel("ffLDL tree level: sub-polynomial size n (depth d)")
    ax.set_ylabel(r"L$_{10}$ absolute error  RMSE $=\sqrt{\mathrm{MSE}}$")
    ax.set_title("ffLDL absolute precision descending the tree (real Falcon-512 keys, deployed path)\n"
                 "RMSE of L₁₀ per level vs 256-bit mpmath (MSE for Rényi)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def main(n_keys=10):
    print(f"Real-conditions ffLDL precision over {n_keys} Falcon-512 keys "
          f"(absolute L10 error, MSE per tree level)...")
    levels = bench(n_keys)
    rows = report(levels)
    write_csv(HERE / "tables" / "ffldl_realcond_precision.csv",
              headers=["level_n", "fp_mse", "fp_rmse", "fxp63_mse", "fxp63_rmse",
                       "fxp127_mse", "fxp127_rmse"], rows=rows)
    save_fig(plot(levels), "ffldl_realcond_precision", HERE)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
