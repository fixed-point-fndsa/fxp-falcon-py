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

import os
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402  (sets up sys.path)

# Reference (float) Falcon.
from falcon import SecretKey  # noqa: E402
from fft import fft  # noqa: E402
from ffsampling import ffsampling_fft as ffsampling_ref  # noqa: E402
from rng import ChaCha20  # noqa: E402
from common import q as FALCON_Q  # noqa: E402

# fxp side.
from fxtypes import FxR, FxC  # noqa: E402
from ffsampling_fxp import ffsampling_fxp  # noqa: E402
from sign_tweak import _build_t_tweaked, _build_fxp_tree_cache  # noqa: E402
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


def build_fxp_tree_for_sk(sk):
    """Build the DEPLOYED fxp normalized ffLDL tree for sk — exactly the path
    production signing uses: the fxp gram via `_gram_fft_fxp(_build_B0_fft_fxp_cache(sk))`
    at the fixed budgets (g00@M_G00, g10@M_G01), then `keygen_fxp` with the
    hardcoded σ constants. NOT an idealized per-entry tight-m gram (which would
    feed the tree a tighter input than production and understate the divergence
    from the float reference)."""
    return _build_fxp_tree_cache(sk)


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
