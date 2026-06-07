"""
Tests for ffsampling_fxp (the fxp signing ffsampling).

Strategy:
  1. Build a Falcon-512 keypair (reference).
  2. Build the equivalent fxp ffLDL tree via keygen_fxp on the same basis.
  3. For each trial (the harness runs 200 trials × 2 target modes — standard
     and the Section-5.1 tweak):
     a. Pick a message, hash it to c, build the target t for this mode.
     b. Run reference ffsampling_fft with ChaCha20(seed) → z_ref.
     c. Run ffsampling_fxp with the SAME seed → z_fxp.
     d. Compare z_ref and z_fxp (must be bit-identical per coefficient).
"""

import math
import os
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402  (sets up sys.path)

# Reference (float) Falcon.
from falcon import SecretKey  # noqa: E402
from fft import fft, neg  # noqa: E402
from ffsampling import gram as gram_float, ffsampling_fft as ffsampling_ref  # noqa: E402
from rng import ChaCha20  # noqa: E402
from common import q as FALCON_Q  # noqa: E402

# fxp side.
from fxtypes import FxR, FxC, RootGram  # noqa: E402
from ffldl_fxp import keygen_fxp  # noqa: E402
from ffsampling_fxp import ffsampling_fxp  # noqa: E402
from sign_tweak import _build_t_tweaked  # noqa: E402
# All m budgets come from the single source.
from m_budgets import M_SIGN_DEFAULT, M_SIGN_STD  # noqa: E402


# --------------------------------------------------------------------- #
# Convert between float and fxp polys.
# --------------------------------------------------------------------- #


def complex_poly_to_fxc(poly, m: int, p: int = 63):
    def _one(z):
        return FxC(re=FxR.from_float(float(z.real), m=m, p=p),
                   im=FxR.from_float(float(z.imag), m=m, p=p))
    return [_one(z) for z in poly]


def fxc_poly_to_complex(poly):
    return [complex(z.re.to_float(), z.im.to_float()) for z in poly]


# --------------------------------------------------------------------- #
# Build the fxp tree from the reference basis via keygen_fxp.
# --------------------------------------------------------------------- #


def build_fxp_tree_for_sk(sk, p=63):
    """Build the fxp normalized ffLDL tree on the same basis as sk."""
    B0 = [[sk.g, neg(sk.f)], [sk.G, neg(sk.F)]]
    Gram = gram_float(B0)
    G_fft_float = [[fft(Gram[i][j]) for j in range(2)] for i in range(2)]

    # Per-block tight m for the Gram (from bench_ffldl_precision._run_fxp recipe).
    def _tight_m(poly):
        return max(1, int(math.ceil(math.log2(max(abs(z) for z in poly)))) + 1)

    # RootGram (g00, g10): the diagonal G_00 is Hermitian-real → PolyR (take
    # Re, dropping the float-FFT noise on Im); g10 complex; G_11 recovered by
    # the NTRU-root LDL via the q²/D_00 shortcut.
    g00_f, g10_f = G_fft_float[0][0], G_fft_float[1][0]
    g00 = [FxR.from_float(z.real, m=_tight_m(g00_f), p=p) for z in g00_f]
    g10 = complex_poly_to_fxc(g10_f, m=_tight_m(g10_f), p=p)
    G_fxp = RootGram(g00=g00, g10=g10)

    inv_sigma_fxr = FxR.from_float(1.0 / sk.sigma, m=-7, p=p)   # 1/σ (INV_SIGMA)
    sigmin_fxr = FxR.from_float(sk.sigmin, m=1, p=p)            # σ_min (for ccs_i)
    # m budgets are fixed constants inside keygen_fxp (M_L10_ROOT=5, M_D=18);
    # keygen_fxp default iters=6 (iters=2 under-converges the rsqrt to
    # ε≈3e-3 ≫ 2^-63, corrupting every σ_i).
    tree = keygen_fxp(G_fxp, q=FALCON_Q, inv_sigma=inv_sigma_fxr, sigmin=sigmin_fxr)
    return tree


# --------------------------------------------------------------------- #
# Build t_fft (standard) from c (the hashed-to-point output).
# --------------------------------------------------------------------- #


def build_t_standard_float(sk, c):
    q = FALCON_Q
    [[_a, b], [_c, d]] = sk.B0_fft
    c_fft = fft(c)
    t0 = [(c_fft[i] * d[i]) / q for i in range(sk.n)]
    t1 = [(-c_fft[i] * b[i]) / q for i in range(sk.n)]
    return [t0, t1]


# --------------------------------------------------------------------- #
# KAT comparison: reference vs fxp ffsampling, same seed, same t.
# --------------------------------------------------------------------- #


def extract_leaf_integer_poly_from_complex(z_ref_poly):
    return [int(round(z.real)) for z in z_ref_poly]


def run_kat_test(sk, tree_fxp, n_trials=50, seed_start=1, m_sign=M_SIGN_DEFAULT,
                 use_tweak=False):
    """For each trial, run reference ffsampling and fxp ffsampling with the
    SAME ChaCha20 seed, compare outputs."""

    match_count = 0
    mismatches = []
    for i in range(n_trials):
        # Pick a random message and salt.
        salt = os.urandom(40)
        msg = os.urandom(32)
        c = sk.hash_to_point(msg, salt)

        # t_fft (standard or tweaked).
        if use_tweak:
            t_ref, _ = _build_t_tweaked(sk, c)
        else:
            t_ref = build_t_standard_float(sk, c)

        # Convert to fxp (m = m_sign, p = 63). m=1 suffices for the tweaked case
        # (||t_frac|| ~ sqrt(n/12)*sqrt(2 ln n) < 2^5) but m_sign=18 also works.
        t_fxp = [complex_poly_to_fxc(t_ref[0], m=m_sign),
                 complex_poly_to_fxc(t_ref[1], m=m_sign)]

        # Shared seed.
        trial_seed = (seed_start + i).to_bytes(48, "big")
        prg_ref = ChaCha20(trial_seed)
        prg_fxp = ChaCha20(trial_seed)

        z_ref = ffsampling_ref(t_ref, sk.T_fft, sk.sigmin, prg_ref.randombytes)
        z_fxp_fxc = ffsampling_fxp(t_fxp, tree_fxp,
                                    prg_fxp.randombytes, m_sign=m_sign)

        # Convert both to integer polynomials for comparison.
        z_ref_ints = [extract_leaf_integer_poly_from_complex(z_ref[k])
                      for k in range(2)]
        z_fxp_ints = [[round(v.re.to_float()) for v in z_fxp_fxc[k]]
                      for k in range(2)]

        if z_ref_ints == z_fxp_ints:
            match_count += 1
        else:
            # Characterize the mismatch.
            diffs0 = sum(1 for a, b in zip(z_ref_ints[0], z_fxp_ints[0]) if a != b)
            diffs1 = sum(1 for a, b in zip(z_ref_ints[1], z_fxp_ints[1]) if a != b)
            max_abs_diff = max(
                max(abs(a - b) for a, b in zip(z_ref_ints[0], z_fxp_ints[0])),
                max(abs(a - b) for a, b in zip(z_ref_ints[1], z_fxp_ints[1])),
            )
            mismatches.append((i, diffs0, diffs1, max_abs_diff))
    return match_count, mismatches


def main():
    n = 512
    n_trials = 200

    random.seed(2026)
    print(f"Generating Falcon-{n} reference keypair...")
    sk = SecretKey(n)

    print("Building fxp ffLDL tree via keygen_fxp on the same basis...")
    t0 = time.time()
    tree_fxp = build_fxp_tree_for_sk(sk)
    print(f"Tree built in {time.time()-t0:.1f}s.")

    for label, tweak, m_sign in [("standard (large t)", False, M_SIGN_STD),
                                 ("tweaked (small t_frac)", True, M_SIGN_DEFAULT)]:
        print(f"\n=== KAT: {label}, {n_trials} trials (m_sign={m_sign}) ===")
        t0 = time.time()
        match_count, mismatches = run_kat_test(sk, tree_fxp, n_trials=n_trials,
                                                m_sign=m_sign, use_tweak=tweak)
        dt = time.time() - t0
        print(f"Matches: {match_count}/{n_trials} in {dt:.0f}s")
        if mismatches:
            print(f"Mismatches ({len(mismatches)}):")
            for (idx, d0, d1, maxd) in mismatches[:5]:
                print(f"  trial {idx}: z0 differs at {d0} positions, z1 at {d1}, "
                      f"max |z_ref - z_fxp| = {maxd}")


if __name__ == "__main__":
    main()
