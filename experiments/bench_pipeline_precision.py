"""
Production-path precision benchmark for the fxp Falcon-512 pipeline.

Unlike `bench_{fft,div,ffldl,ffsampling}_precision.py` — which characterise each
arithmetic *kernel* on idealised (mpmath/float-built) inputs — this benchmark
runs the ACTUAL deployed code path on real Falcon-512 keys and measures the
error of every stage against a 256-bit mpmath reference:

    B0 (int basis) ──fxp──▶ _gram_fft_fxp ──▶ keygen_fxp (ffLDL + normalize)
                                                       │  L_10 / D_ii / σ_i
    point (hashed) ──fxp──▶ _build_t_standard_fxp ─────┤
                                  └──▶ ffsampling_fxp + samplerz_fxp ──▶
                                            _reconstruct_s_fxp ──▶ s

So it reveals the precision the *implementation* actually attains end to end —
which the kernel benchmarks, fed idealised Grams, do not.

Metric: ABSOLUTE per-coefficient error |fxp − ref|, aggregated as the MEAN
SQUARED ERROR (MSE = mean e²; RMSE = √MSE) over all (coefficient × key) pairs
at each stage. MSE is the Rényi-relevant aggregate (Prest'17). Absolute (not
relative): each stage's error is reported in its own magnitude units — note
g10 ≈ 2^20 vs σ_i ≈ 1, so compare WITHIN a stage / vs float64, not across.

NOT covered (by design / by assumption): the fxp sampler's *decision* fidelity
— the main theorem assumes `samplerz` is an ideal sampler, so only the
arithmetic feeding it is graded here; z is taken as produced.

This script is both the benchmark (CSV + figure) AND a hard gate: `gate()`
asserts a per-stage fxp-63 RMSE ceiling and `sig_mism == 0`, so running it
exits non-zero on a deployed-path precision regression.
"""

import math
import random as _random
from pathlib import Path

import mpmath

HERE = Path(__file__).resolve().parent  # experiments/

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from _outputs import save_fig, write_csv  # noqa: E402

from _precision_ref import (  # noqa: E402
    _mp_fft, _mp_split_fft, _mp_ffldl_fft, _gram_from_B_mp,
    _fxr_to_mp, _fxc_poly_to_mp, _float_poly_to_mp, _run_float,
    _abs_errs, _mse, _log2,
)

from falcon import SecretKey  # noqa: E402  (its ntru_gen applies the full NTRUGen filter)
from fft import neg, fft as fft_float  # noqa: E402
from ffsampling import gram as gram_float  # noqa: E402
from fxtypes import RootGram  # noqa: E402
from fft_fxp import retag_poly_fxr, retag_poly_fxc  # noqa: E402
from ffldl_fxp import ffldl_fft_fxp_ntru_root, normalize_tree_fxp  # noqa: E402
from ffsampling_fxp import ffsampling_fxp  # noqa: E402
from m_budgets import M_G00, M_G01  # noqa: E402
from target_construction import (  # noqa: E402
    _build_B0_fft_fxp_cache, _build_t_standard, _build_t_standard_fxp,
)
from sign_tweak import (  # noqa: E402
    _gram_fft_fxp, _inv_sigma_fxp, _sigmin_fxp, _build_fxp_tree_cache,
    _reconstruct_s_fxp,
)

Q = 12289
mpmath.mp.prec = 256
_J = mpmath.mpc(0, 1)


def _mp_ifft(f_fft):
    n = len(f_fft)
    if n == 2:
        a = (f_fft[0] + f_fft[1]) / 2
        b = (f_fft[0] - f_fft[1]) / (2 * _J)
        return [a, b]
    f0, f1 = _mp_split_fft(f_fft)
    c0, c1 = _mp_ifft(f0), _mp_ifft(f1)
    out = [mpmath.mpc(0)] * n
    for i in range(n // 2):
        out[2 * i] = c0[i]
        out[2 * i + 1] = c1[i]
    return out


# --------------------------------------------------------------------- #
# Absolute per-coefficient errors (lists; aggregated to MSE by the driver).
# `_abs_errs` / `_mse` / `_log2` are shared from `_precision_ref`.
# --------------------------------------------------------------------- #


def _tree_abs_errs(ref_mp, got, conv, out):
    """Append every node's L10 (and every leaf's D_00/D_11) absolute error."""
    out.extend(_abs_errs(conv(got[0]), ref_mp[0]))
    if len(ref_mp[0]) == 2:
        for i in (1, 2):
            out.extend(_abs_errs(conv(got[i]), ref_mp[i]))
    else:
        for i in (1, 2):
            _tree_abs_errs(ref_mp[i], got[i], conv, out)


def _sigma_abs_errs(ref_mp, tree_norm, inv_sigma_mp, sigmin_mp, out):
    """Append leaf dss_i, ccs_i absolute errors vs the exact-D_ii reference."""
    if len(ref_mp[0]) > 2:
        _sigma_abs_errs(ref_mp[1], tree_norm[1], inv_sigma_mp, sigmin_mp, out)
        _sigma_abs_errs(ref_mp[2], tree_norm[2], inv_sigma_mp, sigmin_mp, out)
        return
    for ref_poly, leaf in ((ref_mp[1], tree_norm[1]), (ref_mp[2], tree_norm[2])):
        d_mp = abs(ref_poly[0])
        inv_si = mpmath.sqrt(d_mp) * inv_sigma_mp
        out.append(float(abs(_fxr_to_mp(leaf[0]) - inv_si * inv_si / 2)))   # dss
        out.append(float(abs(_fxr_to_mp(leaf[1]) - sigmin_mp * inv_si)))    # ccs


# --------------------------------------------------------------------- #
# One key: per-coefficient absolute errors per stage, fxp-63 and float64.
# --------------------------------------------------------------------- #

STAGES = ["gram", "ffldl(L10+D_ii)", "sigma_i(dss,ccs)", "target_t"]


def _rng(seed):
    r = _random.Random(seed)
    return lambda k: bytes(r.getrandbits(8) for _ in range(k))


def measure_key(sk, key_idx, m_sign=21):
    n = sk.n
    out = {s: {"fp": [], "fxp": []} for s in STAGES}

    B = [[sk.g, neg(sk.f)], [sk.G, neg(sk.F)]]
    G_mp = _gram_from_B_mp(B, n)
    tree_ref_mp = _mp_ffldl_fft(G_mp)

    # ---- Stage 1: Gram ----
    gram = _gram_fft_fxp(_build_B0_fft_fxp_cache(sk))
    out["gram"]["fxp"] = _abs_errs(_fxc_poly_to_mp(gram.g00), G_mp[0][0]) + \
        _abs_errs(_fxc_poly_to_mp(gram.g10), G_mp[1][0])
    Gfl = gram_float(B)
    out["gram"]["fp"] = _abs_errs(_float_poly_to_mp(fft_float(Gfl[0][0])), G_mp[0][0]) + \
        _abs_errs(_float_poly_to_mp(fft_float(Gfl[1][0])), G_mp[1][0])

    # ---- Stage 2: ffLDL tree (L10 every node + leaf D_ii) ----
    G_fxp = RootGram(g00=retag_poly_fxr(gram.g00, M_G00),
                     g10=retag_poly_fxc(gram.g10, M_G01))
    tree_unnorm = ffldl_fft_fxp_ntru_root(G_fxp, q=Q)
    _tree_abs_errs(tree_ref_mp, tree_unnorm, _fxc_poly_to_mp, out["ffldl(L10+D_ii)"]["fxp"])
    _tree_abs_errs(tree_ref_mp, _run_float(B, n), _float_poly_to_mp, out["ffldl(L10+D_ii)"]["fp"])

    # ---- Stage 3: σ_i (leaf dss_i, ccs_i) — fxp only (no fxp-leaf float ref) ----
    inv_sigma_mp = _fxr_to_mp(_inv_sigma_fxp(sk))
    sigmin_mp = _fxr_to_mp(_sigmin_fxp(sk))
    tree_norm = normalize_tree_fxp(tree_unnorm, _inv_sigma_fxp(sk), _sigmin_fxp(sk))
    _sigma_abs_errs(tree_ref_mp, tree_norm, inv_sigma_mp, sigmin_mp, out["sigma_i(dss,ccs)"]["fxp"])

    # ---- Stage 4: target t = (c·d/q, −c·b/q) ----
    point = list(sk.hash_to_point(b"bench-pipeline-precision", bytes(40)))
    c_mp, b_mp, d_mp = _mp_fft(point), _mp_fft(neg(sk.f)), _mp_fft(neg(sk.F))
    t0_ref = [c_mp[i] * d_mp[i] / Q for i in range(n)]
    t1_ref = [-c_mp[i] * b_mp[i] / Q for i in range(n)]
    t_fxp, _ = _build_t_standard_fxp(sk, point, m_sign)
    out["target_t"]["fxp"] = _abs_errs(_fxc_poly_to_mp(t_fxp[0]), t0_ref) + \
        _abs_errs(_fxc_poly_to_mp(t_fxp[1]), t1_ref)
    t_fl, _ = _build_t_standard(sk, point)
    out["target_t"]["fp"] = _abs_errs(_float_poly_to_mp(t_fl[0]), t0_ref) + \
        _abs_errs(_float_poly_to_mp(t_fl[1]), t1_ref)

    # ---- Stage 5: signature s = (t − z)·B0 (real production sampling) ----
    tree_fxp = _build_fxp_tree_cache(sk)
    z_fxc = ffsampling_fxp(t_fxp, tree_fxp, _rng(key_idx), m_sign=m_sign)
    s0, s1, z0, z1 = _reconstruct_s_fxp(sk, t_fxp, z_fxc, m_sign)
    z0f, z1f = _mp_fft(z0), _mp_fft(z1)
    diff0 = [t0_ref[i] - z0f[i] for i in range(n)]
    diff1 = [t1_ref[i] - z1f[i] for i in range(n)]
    a_mp, c_mp2 = _mp_fft(sk.g), _mp_fft(sk.G)   # b_mp/d_mp (= FFT of −f/−F) reused from Stage 4
    s0_mp = [v.real for v in _mp_ifft([diff0[i] * a_mp[i] + diff1[i] * c_mp2[i] for i in range(n)])]
    s1_mp = [v.real for v in _mp_ifft([diff0[i] * b_mp[i] + diff1[i] * d_mp[i] for i in range(n)])]
    mism = sum(int(s0[i] != int(mpmath.nint(s0_mp[i]))) for i in range(n)) + \
        sum(int(s1[i] != int(mpmath.nint(s1_mp[i]))) for i in range(n))
    sig_abs = max(max(abs(float(s0[i]) - float(s0_mp[i])) for i in range(n)),
                  max(abs(float(s1[i]) - float(s1_mp[i])) for i in range(n)))
    out["_sig"] = {"abs": sig_abs, "mismatch": mism}
    return out


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


def bench(n_keys):
    agg = {s: {"fp": [], "fxp": []} for s in STAGES}
    sig_abs, sig_mism = [], 0
    for k in range(n_keys):
        sk = SecretKey(512)   # ntru_gen applies the full NTRUGen filter (γ_fg/hybrid/gs/FG/root)
        r = measure_key(sk, k)
        for s in STAGES:
            agg[s]["fp"].extend(r[s]["fp"])
            agg[s]["fxp"].extend(r[s]["fxp"])
        sig_abs.append(r["_sig"]["abs"])
        sig_mism += r["_sig"]["mismatch"]
        print(f"  key {k} done")
    return agg, sig_abs, sig_mism


def report(agg, sig_abs, sig_mism, n_keys):
    print(f"\nProduction-path ABSOLUTE precision over {n_keys} real Falcon-512 keys "
          f"(MSE vs 256-bit mpmath; RMSE=√MSE in parens):\n")
    print(f"{'stage':>20} | {'float64 MSE':>16} | {'fxp-63 MSE':>16}")
    print("-" * 62)
    rows = []
    for s in STAGES:
        mf, mx = _mse(agg[s]["fp"]), _mse(agg[s]["fxp"])
        def cell(v):
            return "—" if v != v else f"{v:.2e} (2^{_log2(math.sqrt(v)):.0f})"
        print(f"{s:>20} | {cell(mf):>16} | {cell(mx):>16}")
        rows.append([s,
                     "nan" if mf != mf else f"{mf:.6e}",
                     "nan" if mf != mf else f"{math.sqrt(mf):.6e}",
                     f"{mx:.6e}", f"{math.sqrt(mx):.6e}"])
    print("\n  signature s = (t−z)·B0 is integer-valued (a lattice point); the deployed "
          "reconstruct reproduces it")
    print(f"    exactly — integer mismatches vs exact: {sig_mism} over "
          f"{n_keys}×2×512 coefficients "
          f"(residual |s_fxp−s_exact| ≈ {sorted(sig_abs)[len(sig_abs)//2]:.0e}, mpmath noise).")
    return rows


def plot(agg):
    import matplotlib.pyplot as plt
    labels = [s.split("(")[0] for s in STAGES]
    fp = [math.sqrt(_mse(agg[s]["fp"])) for s in STAGES]
    fx = [math.sqrt(_mse(agg[s]["fxp"])) for s in STAGES]
    x = range(len(STAGES))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([i - 0.2 for i in x], [0 if v != v else v for v in fp], 0.4,
           label="float64", color="C0")
    ax.bar([i + 0.2 for i in x], [0 if v != v else v for v in fx], 0.4,
           label="fxp-63 (deployed)", color="C1")
    ax.set_yscale("log", base=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"absolute error  RMSE $=\sqrt{\mathrm{MSE}}$")
    ax.set_title("Production-path absolute precision per stage (Falcon-512, vs 256-bit mpmath)\n"
                 "real keys through the deployed fxp code path (MSE is the Rényi aggregate)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# fxp-63 RMSE ceilings per stage — the pass/fail contract (the *verification*,
# vs the measurement above). Observed RMSE is ~2^-41 / -46 / -60 / -43 (10
# keys, tight B0 coefficient loads M_B0_COEF_FG=5 / M_B0_COEF_FG_UP=7); each
# ceiling is observed + ~3 bits, so a gross precision regression (a wrong
# budget, a broken reciprocal, an overflow → many bits lost) trips it, while
# normal key-to-key variation of the aggregate MSE stays well under.
_RMSE_CEIL = {
    "gram":             2 ** -38,
    "ffldl(L10+D_ii)":  2 ** -43,
    "sigma_i(dss,ccs)": 2 ** -57,
    "target_t":         2 ** -40,
}


def gate(agg, sig_mism, n_keys):
    """Hard pass/fail on the deployed-path precision. Raises AssertionError on
    a per-stage RMSE regression or any signature integer mismatch."""
    assert sig_mism == 0, \
        f"signature reconstruct: {sig_mism} integer mismatch(es) vs exact (must be 0)"
    for s in STAGES:
        rmse = math.sqrt(_mse(agg[s]["fxp"]))
        assert rmse <= _RMSE_CEIL[s], \
            f"{s}: fxp-63 RMSE 2^{_log2(rmse):.1f} exceeds ceiling 2^{_log2(_RMSE_CEIL[s]):.0f}"
    print(f"\n  GATE PASS: {len(STAGES)} stages within RMSE ceiling, sig_mism=0 ({n_keys} keys).")


def main(n_keys=10):
    print(f"Running production-path ABSOLUTE precision over {n_keys} real Falcon-512 keys...")
    agg, sig_abs, sig_mism = bench(n_keys)
    rows = report(agg, sig_abs, sig_mism, n_keys)
    write_csv(HERE / "tables" / "pipeline_precision.csv",
              headers=["stage", "float64_mse", "float64_rmse", "fxp63_mse", "fxp63_rmse"],
              rows=rows)
    save_fig(plot(agg), "pipeline_precision", HERE)
    gate(agg, sig_mism, n_keys)   # artifacts written first; then the hard pass/fail


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
