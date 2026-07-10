"""
Precision benchmark for ffsampling — self-consistent per format.

For each of {FP float64, FxP-63, FxP-127}, we do:
  - Build the Gram in its native precision: for fxp, the DEPLOYED path
    (fxp FFT of B0 → `_gram_fft_fxp`, retag-before-mul); float for FP;
    mpmath-256 for the ground truth. Likewise the target t uses the deployed
    `_build_t_standard_fxp` arithmetic (γ-retag before the products).
  - Build the ffLDL tree at that precision (`keygen_fxp`).
  - Run ffsampling at that precision and grade the reduced target t_0' per
    recursion level vs the mpmath ground truth.

Ground truth is a full mpmath-256 pipeline (Gram, ffLDL, ffsampling).

**Target convention**: this benchmark uses the *standard* (non-tweaked)
target `t = (-c·F/q, c·f/q)` computed in FFT domain by pointwise
multiplication of FFT(c) with FFT(±F), FFT(f) and division by q.
FFT-domain magnitudes reach `n·γ_FG ≈ 2^20.93`, so we use `m_sign = 21`
throughout. This is the path the Section-5.1 tweak deliberately
avoids; running it here lets us measure the precision impact of those
larger magnitudes (vs. `m_sign = 18` in the tweak path, where
`‖t̂‖_∞ ≤ n/2 = 2^8`).

Per level i of ffsampling, we record per-coefficient
`|t_0'(i) − t_0'_mpmath(i)|` and aggregate as both the **mean**
(typical case) and the **max** (observed worst case) across all
(trial × coefficient) pairs at that level.
"""

import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from _outputs import save_fig, write_csv  # noqa: E402

from falcon import SecretKey  # noqa: E402
from fft import neg  # noqa: E402
from ffsampling import gram as gram_float  # noqa: E402
from rng import ChaCha20  # noqa: E402
from common import q as FALCON_Q  # noqa: E402
from samplerz import samplerz as samplerz_ref  # noqa: E402

from fxtypes import FxR, FxC, retag_fxr  # noqa: E402
from fft_fxp import (  # noqa: E402
    add_fft_fxp, sub_fft_fxp, mul_fft_to, split_complex_fxp, merge_fft_fxp,
    fft_fxp,
)
from ffldl_fxp import keygen_fxp  # noqa: E402
from sign_tweak import _gram_fft_fxp  # noqa: E402  (deployed Gram, retag-before-mul)
from m_budgets import (  # noqa: E402
    M_B_FG, M_B_FG_UP, M_CQ_COEF, M_POINT_COEF,
)
# p-precise generated constants (NOT from_float — a float64 1/q or 1/σ caps the
# target / σ_i at ~2^-53 regardless of p, which would hide fxp-127's headroom).
from fxp_constants_p63 import (  # noqa: E402
    INV_Q_FXC as _INVQ_63, INV_SIGMA_FXR_BY_N as _INVSIG_63, SIGMIN_FXR_BY_N as _SIGMIN_63)
from fxp_constants_p127 import (  # noqa: E402
    INV_Q_FXC as _INVQ_127, INV_SIGMA_FXR_BY_N as _INVSIG_127, SIGMIN_FXR_BY_N as _SIGMIN_127)
_INVQ = {63: _INVQ_63, 127: _INVQ_127}
_INVSIG = {63: _INVSIG_63, 127: _INVSIG_127}
_SIGMIN = {63: _SIGMIN_63, 127: _SIGMIN_127}
# mpmath pipeline (reuse from ffldl bench).
from _precision_ref import (  # noqa: E402
    _mp_fft, _mp_split_fft, _mp_merge_fft, _mp_adj,
    _mp_ffldl_fft as _mp_ffldl_inner,
)

mpmath.mp.prec = 256


# --------------------------------------------------------------------- #
# Exact integer Gram + mpmath FFT.
# --------------------------------------------------------------------- #


def _gram_int_then_mp_fft(sk):
    """Compute the Gram at high precision: start from exact integer Gram in
    coefficient domain, then FFT in mpmath."""
    B0 = [[sk.g, neg(sk.f)], [sk.G, neg(sk.F)]]
    Gram_float = gram_float(B0)  # integer-valued floats
    Gram_int = [[[int(round(c)) for c in Gram_float[i][j]] for j in range(2)]
                for i in range(2)]
    G_mp = [[_mp_fft(Gram_int[i][j]) for j in range(2)] for i in range(2)]
    return G_mp


def _mp_to_fxc(poly_mp, m, p):
    """Quantize an mpmath complex poly to FxC at (m, p)."""
    scale = mpmath.mpf(2) ** (p - m)
    out = []
    for z in poly_mp:
        x_re = int(mpmath.nint(z.real * scale))
        x_im = int(mpmath.nint(z.imag * scale))
        out.append(FxC(re=FxR(x=x_re, m=m, p=p),
                       im=FxR(x=x_im, m=m, p=p)))
    return out


def _mp_to_complex(poly_mp):
    return [complex(float(z.real), float(z.imag)) for z in poly_mp]


def build_fxp_tree_at_p_selfconsistent(sk, p, m_sign):
    """Fully self-consistent fxp pipeline at precision p: integer basis,
    FFT_fxp at p, polynomial-matrix multiply at p, ffLDL at p."""
    B = [[sk.g, [-c for c in sk.f]],
         [sk.G, [-c for c in sk.F]]]

    # Fixed-m FFT per block at its NTRUGen γ tag — exactly what the deployed
    # `_b0_fft_fxp` runs (fft(g), fft(−f) at M_B_FG; fft(G), fft(−F) at
    # M_B_FG_UP; load + transform + output all at that single tag).
    m_row = (M_B_FG, M_B_FG, M_B_FG_UP, M_B_FG_UP)
    B_fft = [[fft_fxp([FxR.from_int(int(c), m=m_row[2 * i + j], p=p) for c in B[i][j]],
                      certified=True) for j in range(2)] for i in range(2)]
    G_fxp = _gram_fft_fxp(B_fft)

    inv_sigma_fxr = _INVSIG[p][sk.n]   # 1/σ (INV_SIGMA), m=-7, p-precise (not float64)
    sigmin_fxr = _SIGMIN[p][sk.n]      # σ_min, m=1, p-precise
    # Budgets fixed inside keygen_fxp (M_L10_ROOT=5, M_L10_INNER=0, M_D=18);
    # valid because `ntru_gen` (hence SecretKey) applies the full filter.
    return keygen_fxp(
        G_fxp, q=FALCON_Q, inv_sigma=inv_sigma_fxr, sigmin=sigmin_fxr,
        iters=6,  # safe default for both p=63 and p=127 (see ffldl_fxp.rsqrt)
    )


def build_fp_tree(sk):
    """Reference float tree (same as sk.T_fft)."""
    return sk.T_fft


def build_mp_tree(G_mp, sk, q):
    """Build the ffLDL tree in mpmath with the same structure as our fxp tree
    (ntru_root at the top, inner uses G_11 - adj(L_10)*G_10 + renorm)."""
    n = len(G_mp[0][0])
    G00, G10 = G_mp[0][0], G_mp[1][0]
    L10 = [G10[i] / G00[i] for i in range(n)]
    D00 = list(G00)
    q_sq = mpmath.mpf(q) ** 2
    D11 = [q_sq / G00[i] for i in range(n)]
    if n > 2:
        d00, d01 = _mp_split_fft(D00)
        d10, d11 = _mp_split_fft(D11)
        G0 = [[d00, d01], [_mp_adj(d01), d00]]
        G1 = [[d10, d11], [_mp_adj(d11), d10]]
        return [L10, _mp_ffldl_inner(G0), _mp_ffldl_inner(G1)]
    return [L10, D00, D11]


def mp_normalize_tree(tree, sigma_mp):
    """Normalize the mpmath ffLDL tree by replacing leaves with [sigma/sqrt(D_ii), 0]."""
    if isinstance(tree[1][0], list):
        mp_normalize_tree(tree[1], sigma_mp)
        mp_normalize_tree(tree[2], sigma_mp)
    else:
        for idx in (1, 2):
            poly = tree[idx]
            val = sigma_mp / mpmath.sqrt(poly[0].real)
            tree[idx] = [val, mpmath.mpc(0)]


# --------------------------------------------------------------------- #
# Input t construction at each precision.
# --------------------------------------------------------------------- #


def build_t_at_all_precisions(sk, c, p_fxp_list, m_sign):
    """Return (t_fp, t_mp, {p: t_fxp}) where t is the **standard** target

        t = (−c·F/q,  c·f/q)

    computed in FFT domain by pointwise multiplication of FFT(c) with
    FFT(±F), FFT(f) and division by q. This is the path the tweak
    deliberately avoids (because the FFT-domain magnitudes here reach
    `n·γ_FG ≈ 2^20.93`, vs `n/2 = 2^8` for `t_frac`); we run it here
    precisely to measure the precision impact of operating on those
    larger magnitudes.

    `m_sign` should be 21 in the standard path (vs 18 with the tweak).
    """
    q = FALCON_Q
    n = len(c)

    # ----- mpmath ground truth: integer FFTs, then pointwise mul/div. -----
    c_mp_fft = _mp_fft(list(c))
    F_mp_fft = _mp_fft(list(sk.F))
    f_mp_fft = _mp_fft(list(sk.f))
    q_mp = mpmath.mpf(q)
    t0_mp = [(-c_mp_fft[i] * F_mp_fft[i]) / q_mp for i in range(n)]
    t1_mp = [( c_mp_fft[i] * f_mp_fft[i]) / q_mp for i in range(n)]
    t_mp = [t0_mp, t1_mp]

    # ----- Float version: downcast from mpmath (= what the float pipeline
    # would compute, modulo the float-domain operation order; downcasting
    # the mpmath result is the cleanest way to feed the same starting
    # point to all three pipelines without picking up extra float64
    # round-off in `t` itself before the walker even starts). -----
    t_fp = [_mp_to_complex(t0_mp), _mp_to_complex(t1_mp)]

    # ----- Self-consistent FxP at each p: c/q in COEFFICIENT domain at
    # M_CQ_COEF=0 (exact retag from the natural ·INV_Q tag 1), fft at small
    # tags, fused mul_fft_to to m_sign. Mirrors
    # `target_construction._build_t_standard_fxp` but parametric on p. -----
    t_fxps = {}
    for p in p_fxp_list:
        inv_q = _INVQ[p].re  # p-precise generated 1/q (m=-13), as in production
        cq_fft = fft_fxp([retag_fxr(FxR.from_int(int(ci), m=M_POINT_COEF, p=p)
                                    * inv_q, M_CQ_COEF)   # 1 → 0: exact shift
                          for ci in c])
        # F, f via the fixed-m FFT at their γ tags (M_B_FG_UP, M_B_FG) —
        # matches the deployed `_build_t_standard_fxp` / `_b0_fft_fxp`.
        F_fxc = fft_fxp([FxR.from_int(int(F_i), m=M_B_FG_UP, p=p) for F_i in sk.F], certified=True)
        f_fxc = fft_fxp([FxR.from_int(int(f_i), m=M_B_FG, p=p) for f_i in sk.f], certified=True)
        t0 = [-z for z in mul_fft_to(cq_fft, F_fxc, m_sign)]
        t1 = mul_fft_to(cq_fft, f_fxc, m_sign)
        t_fxps[p] = [t0, t1]
    return t_fp, t_mp, t_fxps


# --------------------------------------------------------------------- #
# Float ffsampling helpers (reference primitives).
# --------------------------------------------------------------------- #


def _split_float(f):
    from fft import split_fft
    return list(split_fft(f))


def _merge_float(f_list):
    from fft import merge_fft
    return merge_fft(f_list)


# --------------------------------------------------------------------- #
# Parallel walker: FP, FxP (arbitrary p), mpmath; records error per level.
# --------------------------------------------------------------------- #


def _retag_m(a_fxr, m_new):
    from fxtypes import _bankers_shift
    if m_new == a_fxr.m:
        return a_fxr
    if m_new > a_fxr.m:
        return FxR(x=_bankers_shift(a_fxr.x, m_new - a_fxr.m), m=m_new, p=a_fxr.p)
    return FxR(x=a_fxr.x << (a_fxr.m - m_new), m=m_new, p=a_fxr.p)


def _walker(t_fp, t_fxp, t_mp, tree_fp, tree_fxp, tree_mp,
            sigmin_fp, rng_fp, rng_fxp, rng_mp,
            errors, level, m_sign):
    n = len(t_fp[0])
    if n == 1:
        # Leaf.
        sigma_fp = tree_fp[0]
        sigma_mp = tree_mp[0]
        mu0_fp = t_fp[0][0].real
        mu1_fp = t_fp[1][0].real
        z0 = samplerz_ref(mu0_fp, sigma_fp, sigmin_fp, rng_fp.randombytes)
        z1 = samplerz_ref(mu1_fp, sigma_fp, sigmin_fp, rng_fp.randombytes)
        # Drain the other rngs by a matching amount (same call, same inputs).
        _ = samplerz_ref(
            t_fxp[0][0].re.to_float(), tree_fxp[0].to_float(), sigmin_fp,
            rng_fxp.randombytes,
        )
        _ = samplerz_ref(
            t_fxp[1][0].re.to_float(), tree_fxp[0].to_float(), sigmin_fp,
            rng_fxp.randombytes,
        )
        _ = samplerz_ref(float(t_mp[0][0].real), float(sigma_mp.real), sigmin_fp,
                          rng_mp.randombytes)
        _ = samplerz_ref(float(t_mp[1][0].real), float(sigma_mp.real), sigmin_fp,
                          rng_mp.randombytes)
        p_used = t_fxp[0][0].re.p
        z_fp_poly = ([complex(z0, 0)], [complex(z1, 0)])
        z_fxp_poly = [
            [FxC(re=FxR.from_int(z0, m=m_sign, p=p_used),
                 im=FxR(x=0, m=m_sign, p=p_used))],
            [FxC(re=FxR.from_int(z1, m=m_sign, p=p_used),
                 im=FxR(x=0, m=m_sign, p=p_used))],
        ]
        z_mp_poly = ([mpmath.mpc(z0, 0)], [mpmath.mpc(z1, 0)])
        return z_fp_poly, z_fxp_poly, z_mp_poly

    ell_fp = tree_fp[0]
    ell_fxp = tree_fxp[0]
    ell_mp = tree_mp[0]
    tree_fp_0, tree_fp_1 = tree_fp[1], tree_fp[2]
    tree_fxp_0, tree_fxp_1 = tree_fxp[1], tree_fxp[2]
    tree_mp_0, tree_mp_1 = tree_mp[1], tree_mp[2]

    # Right recursion on t[1].
    t1_split_fp = _split_float(t_fp[1])
    t1_split_fxp = list(split_complex_fxp(t_fxp[1]))  # split preserves m_sign
    t1_split_mp = list(_mp_split_fft(t_mp[1]))

    z_fp_right, z_fxp_right, z_mp_right = _walker(
        t1_split_fp, t1_split_fxp, t1_split_mp,
        tree_fp_1, tree_fxp_1, tree_mp_1,
        sigmin_fp, rng_fp, rng_fxp, rng_mp,
        errors, level + 1, m_sign,
    )

    z1_fp = _merge_float(z_fp_right)
    z1_fxp = merge_fft_fxp(z_fxp_right, m_sign)   # fixed-m merge, no post-retag
    z1_mp = _mp_merge_fft(*z_mp_right)

    # t_0' = t_0 + (t_1 - z_1) * ell.
    diff_fp = [t_fp[1][i] - z1_fp[i] for i in range(n)]
    prod_fp = [diff_fp[i] * ell_fp[i] for i in range(n)]
    t0p_fp = [t_fp[0][i] + prod_fp[i] for i in range(n)]

    diff_fxp = sub_fft_fxp(t_fxp[1], z1_fxp)
    prod_fxp = mul_fft_to(diff_fxp, ell_fxp, m_sign)  # fused, as deployed
    t0p_fxp = add_fft_fxp(t_fxp[0], prod_fxp)

    diff_mp = [t_mp[1][i] - z1_mp[i] for i in range(n)]
    prod_mp = [diff_mp[i] * ell_mp[i] for i in range(n)]
    t0p_mp = [t_mp[0][i] + prod_mp[i] for i in range(n)]

    # Record per-coefficient errors at this level. We append every
    # |t0p[i] - t0p_mp[i]| so the driver can later compute BOTH the mean
    # and the max-over-(coeff, trial). Earlier versions stored only the
    # per-call max, which conflated "median of per-trial-max" with
    # "true mean error" — see experiments/README.md for the distinction.
    for i in range(n):
        e_fp = float(mpmath.sqrt(
            (mpmath.mpf(t0p_fp[i].real) - t0p_mp[i].real) ** 2
            + (mpmath.mpf(t0p_fp[i].imag) - t0p_mp[i].imag) ** 2
        ))
        e_fxp = float(mpmath.sqrt(
            (mpmath.mpf(t0p_fxp[i].re.x) * mpmath.mpf(2) ** (t0p_fxp[i].re.m - t0p_fxp[i].re.p)
             - t0p_mp[i].real) ** 2
            + (mpmath.mpf(t0p_fxp[i].im.x) * mpmath.mpf(2) ** (t0p_fxp[i].im.m - t0p_fxp[i].im.p)
               - t0p_mp[i].imag) ** 2
        ))
        errors.setdefault(level, {"fp": [], "fxp": []})
        errors[level]["fp"].append(e_fp)
        errors[level]["fxp"].append(e_fxp)

    # Left recursion on t0'.
    t0p_split_fp = _split_float(t0p_fp)
    t0p_split_fxp = list(split_complex_fxp(t0p_fxp))  # split preserves m_sign
    t0p_split_mp = list(_mp_split_fft(t0p_mp))

    z_fp_left, z_fxp_left, z_mp_left = _walker(
        t0p_split_fp, t0p_split_fxp, t0p_split_mp,
        tree_fp_0, tree_fxp_0, tree_mp_0,
        sigmin_fp, rng_fp, rng_fxp, rng_mp,
        errors, level + 1, m_sign,
    )

    z0_fp = _merge_float(z_fp_left)
    z0_fxp = merge_fft_fxp(z_fxp_left, m_sign)   # fixed-m merge, no post-retag
    z0_mp = _mp_merge_fft(*z_mp_left)

    return [z0_fp, z1_fp], [z0_fxp, z1_fxp], [z0_mp, z1_mp]


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


def run_trial(sk, G_mp, tree_fp, tree_fxp_by_p, tree_mp, m_sign, seed_int,
              p_target):
    # Deterministic message/salt.
    r = random.Random(seed_int)
    msg = r.randbytes(32)
    salt = r.randbytes(40)
    c = sk.hash_to_point(msg, salt)

    t_fp, t_mp, t_fxps = build_t_at_all_precisions(sk, c, [p_target], m_sign)
    t_fxp = t_fxps[p_target]
    tree_fxp = tree_fxp_by_p[p_target]

    seed = seed_int.to_bytes(48, "big")
    rng_fp = ChaCha20(seed)
    rng_fxp = ChaCha20(seed)
    rng_mp = ChaCha20(seed)
    sigmin_fp = sk.sigmin

    errors = {}
    _walker(t_fp, t_fxp, t_mp,
            tree_fp, tree_fxp, tree_mp,
            sigmin_fp, rng_fp, rng_fxp, rng_mp,
            errors, level=0, m_sign=m_sign)
    return errors


def _log2_neg(x):
    return -math.log2(x) if x > 0 else float("inf")


def _stats(vals):
    """Return (MSE, RMSE) of a non-empty list of absolute errors. The MSE
    (mean of squares) is the Rényi-relevant aggregate (Prest'17); RMSE = √MSE
    is in error units, for plotting/reading."""
    mse = sum(v * v for v in vals) / len(vals)
    return mse, math.sqrt(mse)


def main(n_trials=1000):
    n = 512
    # Standard target: ‖t̂‖_∞ ≤ n·γ_FG ≈ 2^20.93 worst-case, so m_sign = 21
    # (M_SIGN_STD in m_budgets). With the tweak this would be 18.
    m_sign = 21
    random.seed(2026)
    print(f"Generating Falcon-{n} keypair (full NTRUGen filter)...")
    sk = SecretKey(n)   # ntru_gen applies the full filter (Checks 1b/2/3/4 + 127)

    print("Computing exact Gram + mpmath FFT...")
    G_mp = _gram_int_then_mp_fft(sk)

    print("Building trees at each precision...")
    tree_fp = build_fp_tree(sk)
    tree_mp = build_mp_tree(G_mp, sk, FALCON_Q)
    sigma_mp = mpmath.mpf(sk.sigma)
    mp_normalize_tree(tree_mp, sigma_mp)

    tree_fxp_by_p = {}
    for p in [63, 127]:
        tree_fxp_by_p[p] = build_fxp_tree_at_p_selfconsistent(sk, p=p, m_sign=m_sign)

    print(f"Running {n_trials} trials per precision (per-coefficient errors)...")
    merged_by_p = {}
    for p in [63, 127]:
        merged = {}
        t0 = time.time()
        for t in range(n_trials):
            errs = run_trial(sk, G_mp, tree_fp, tree_fxp_by_p, tree_mp,
                             m_sign, seed_int=100 + t, p_target=p)
            for lvl, d in errs.items():
                merged.setdefault(lvl, {"fp": [], "fxp": []})
                merged[lvl]["fp"].extend(d["fp"])
                merged[lvl]["fxp"].extend(d["fxp"])
            if (t + 1) % 100 == 0:
                dt = time.time() - t0
                rate = (t + 1) / dt
                eta = (n_trials - t - 1) / rate
                print(f"  p={p}: {t+1}/{n_trials}  ({dt:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining at {rate:.1f} trials/s)")
        merged_by_p[p] = merged
        # Sanity: per-level sample count should equal n_trials × n_per_level
        # (where n_per_level summed over all 2^k recursive calls is always
        # the root n). For Falcon-512: 1000 × 512 = 512000 per (level, fmt).
        any_lvl = next(iter(merged))
        print(f"  p={p} done ({len(merged[any_lvl]['fp'])} samples per level)")

    print()
    print("Per-level error of |t_0' − t_0'_mpmath| (absolute, ∞-norm-equivalent),")
    print(f"aggregated as MEAN SQUARED ERROR over {n_trials} trials × all "
          f"coefficients at that level (RMSE = √MSE; reported as 2^-bits).")
    print()
    fmt_hdr = (f"{'level':>5} | "
               f"{'FP MSE':>10} {'FP RMSE':>10} | "
               f"{'F63 MSE':>10} {'F63 RMSE':>10} | "
               f"{'F127 MSE':>10} {'F127 RMSE':>10}")
    print(fmt_hdr)
    print("-" * len(fmt_hdr))
    max_level = max(merged_by_p[63].keys())
    for lvl in range(max_level + 1):
        if lvl not in merged_by_p[63]:
            continue
        fp_mse,  fp_rmse  = _stats(merged_by_p[63][lvl]["fp"])
        e63_mse, e63_rmse = _stats(merged_by_p[63][lvl]["fxp"])
        e127_mse, e127_rmse = _stats(merged_by_p[127][lvl]["fxp"])
        print(f"{lvl:>5} | "
              f"2^-{_log2_neg(fp_mse):>5.2f}   2^-{_log2_neg(fp_rmse):>5.2f}  | "
              f"2^-{_log2_neg(e63_mse):>5.2f}   2^-{_log2_neg(e63_rmse):>5.2f}  | "
              f"2^-{_log2_neg(e127_mse):>5.2f}   2^-{_log2_neg(e127_rmse):>5.2f}")

    # CSV.
    levels = sorted(merged_by_p[63].keys())
    rows = []
    for lvl in levels:
        fp_mse,   fp_rmse   = _stats(merged_by_p[63][lvl]["fp"])
        e63_mse,  e63_rmse  = _stats(merged_by_p[63][lvl]["fxp"])
        e127_mse, e127_rmse = _stats(merged_by_p[127][lvl]["fxp"])
        rows.append([lvl,
                     f"{fp_mse:.6e}",   f"{fp_rmse:.6e}",
                     f"{e63_mse:.6e}",  f"{e63_rmse:.6e}",
                     f"{e127_mse:.6e}", f"{e127_rmse:.6e}"])
    write_csv(HERE / "tables" / f"ffsampling_precision_selfconsistent_n{n}.csv",
              headers=["level",
                       "fp_mse", "fp_rmse",
                       "fxp63_mse", "fxp63_rmse",
                       "fxp127_mse", "fxp127_rmse"],
              rows=rows)

    # Plot RMSE = √MSE per format (one solid curve each). MSE is the Rényi
    # aggregate; RMSE is plotted because it is in error units.
    fp_rmse_l   = [_stats(merged_by_p[63][lvl]["fp"])[1]   for lvl in levels]
    e63_rmse_l  = [_stats(merged_by_p[63][lvl]["fxp"])[1]  for lvl in levels]
    e127_rmse_l = [_stats(merged_by_p[127][lvl]["fxp"])[1] for lvl in levels]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(levels, fp_rmse_l,   "o-", color="C0", linewidth=2, label="float64")
    ax.plot(levels, e63_rmse_l,  "s-", color="C1", linewidth=2, label="FxP-63 (deployed)")
    ax.plot(levels, e127_rmse_l, "^-", color="C2", linewidth=2, label="FxP-127")

    # ULP reference floors at (m_sign, p): LSB = 2^(m_sign - p).
    # FxP-63 at m_sign=21 ⇒ 2^-42; FxP-127 at m_sign=21 ⇒ 2^-106.
    fxp63_ulp = 2 ** (m_sign - 63)
    fxp127_ulp = 2 ** (m_sign - 127)
    ax.axhline(fxp63_ulp,  color="C1", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.axhline(fxp127_ulp, color="C2", linestyle=":", linewidth=0.9, alpha=0.6)

    ax.set_yscale("log", base=2)
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.LogLocator(base=2, numticks=20))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=2))
    ax.set_xlabel("ffsampling level  (0 = root)")
    ax.set_ylabel(r"RMSE of $|t_0'(i) - t_0'_{\mathrm{mpmath}}(i)|$  ($=\sqrt{\mathrm{MSE}}$)")
    ax.set_title(
        f"ffsampling absolute precision (Falcon-{n}, standard target "
        f"$t=(-c{{\\cdot}}F/q,\\,c{{\\cdot}}f/q)$, $m_{{sign}}={m_sign}$, "
        f"{n_trials} trials)\n"
        f"deployed fxp path; RMSE per recursion level (MSE is the Rényi aggregate)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save_fig(fig, f"ffsampling_precision_selfconsistent_n{n}", HERE)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
