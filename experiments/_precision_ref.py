"""
Shared reference helpers for the precision benchmarks: a 256-bit mpmath FFT /
ffLDL* / Gram, the float64 ffLDL reference, fxp→mpmath converters, and a
filtered-NTRU-basis sampler. Imported by `bench_pipeline_precision`,
`bench_ffsampling_precision`, and `bench_ffldl_realcond`.

(The standalone idealized-gram ffLDL precision bench that used to live here was
removed: it fed the fxp ffLDL a per-entry tight-m gram, NOT the deployed
`_gram_fft_fxp`, so its measured precision was optimistic. The DEPLOYED ffLDL
precision is covered by `bench_pipeline_precision` (p=63) and
`bench_ffsampling_precision` (p=63 vs 127).)
"""

import math
import random
from pathlib import Path

import mpmath

HERE = Path(__file__).resolve().parent  # experiments/

import _path_setup  # noqa: F401, E402  (sets up sys.path)

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


def _abs_errs(got_mp, ref_mp):
    """All per-coefficient absolute errors |got − ref| (complex modulus)."""
    return [float(abs(g - r)) for g, r in zip(got_mp, ref_mp)]


def _mse(errs):
    """Mean squared error over a list of absolute errors (the Rényi aggregate)."""
    return sum(e * e for e in errs) / len(errs) if errs else float("nan")


def _log2(x):
    return float("nan") if (x != x or x <= 0) else math.log2(x)


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
