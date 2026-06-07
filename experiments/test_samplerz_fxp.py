"""
Distribution + KAT test of samplerz_fxp vs reference samplerz (float).

Two complementary tests:

  1. KAT match: with the SAME random byte stream, samplerz_fxp and reference
     samplerz should produce byte-identical outputs (z integer). This works
     because samplerz's rejection loop consumes randomness based on r = mu
     mod 1 only (not on the integer part s = floor(mu)), and both versions
     see the exact same r up to fp-noise on mu. We test this with 10000
     (mu, sigma, sigmin) triples.

  2. Distributional match: empirical mean and variance of z, plus a
     total-variation distance between the output histograms over 50k
     samples. The fxp version must match the reference up to sampling noise.
"""

import math
import random
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from samplerz import samplerz as samplerz_ref  # noqa: E402
from rng import ChaCha20  # noqa: E402
from fxtypes import FxR  # noqa: E402
from samplerz_fxp import samplerz_fxp  # noqa: E402


# --------------------------------------------------------------------- #
# Test 1: KAT match on many (mu, sigma, sigmin) triples, shared randomness.
# --------------------------------------------------------------------- #


def kat_match_test(n_trials: int = 10000, seed_start: int = 1):
    """For each trial, pick (mu, sigma, sigmin), seed a ChaCha20 PRG, run both
    samplerz_ref and samplerz_fxp with that PRG, compare outputs."""
    match = 0
    mismatches = []
    rng = random.Random(2026)
    for i in range(n_trials):
        # Random (mu, sigma) within valid Falcon ranges.
        sigmin = rng.uniform(1.28, 1.35)
        sigma = rng.uniform(sigmin + 0.01, 1.82)
        # mu is arbitrary real — exercise both small (~[-0.5, 0.5], tweaked case)
        # and large (~[-1000, 1000], standard case).
        if i % 2 == 0:
            mu = rng.uniform(-0.5, 0.5)          # tweaked-style
        else:
            mu = rng.uniform(-1000.0, 1000.0)    # standard-style

        # Shared random seed.
        trial_seed = (seed_start + i).to_bytes(48, "big")
        prg_ref = ChaCha20(trial_seed)
        prg_fxp = ChaCha20(trial_seed)

        z_ref = samplerz_ref(mu, sigma, sigmin, randombytes=prg_ref.randombytes)

        # Convert mu, sigma, sigmin to FxR.
        # mu magnitude known → m tight.
        if abs(mu) < 1:
            mu_m = 1
        else:
            mu_m = max(1, int(math.ceil(math.log2(abs(mu) + 1e-12))) + 1)
        mu_fxr = FxR.from_float(mu, m=mu_m, p=63)
        dss_fxr = FxR.from_float(1.0 / (2.0 * sigma * sigma), m=0, p=63)  # 1/(2σ²)
        ccs_fxr = FxR.from_float(sigmin / sigma, m=0, p=63)               # σmin/σ

        z_fxp = samplerz_fxp(mu_fxr, dss_fxr, ccs_fxr,
                             randombytes=prg_fxp.randombytes)

        if z_ref == z_fxp:
            match += 1
        else:
            if len(mismatches) < 5:
                mismatches.append((i, mu, sigma, sigmin, z_ref, z_fxp))
    return match, mismatches


# --------------------------------------------------------------------- #
# Test 2: Distributional match (mean, variance, histogram).
# --------------------------------------------------------------------- #


def distributional_test(mu: float, sigma: float, sigmin: float, n_samples: int = 50000):
    """Run reference and fxp samplerz n_samples times each; compare stats."""
    # Reference.
    zs_ref = [samplerz_ref(mu, sigma, sigmin) for _ in range(n_samples)]
    # fxp.
    mu_fxr = FxR.from_float(mu, m=max(1, int(math.ceil(math.log2(abs(mu) + 1e-12))) + 1),
                            p=63)
    dss_fxr = FxR.from_float(1.0 / (2.0 * sigma * sigma), m=0, p=63)
    ccs_fxr = FxR.from_float(sigmin / sigma, m=0, p=63)
    zs_fxp = [samplerz_fxp(mu_fxr, dss_fxr, ccs_fxr) for _ in range(n_samples)]

    mean_ref = statistics.mean(zs_ref)
    mean_fxp = statistics.mean(zs_fxp)
    stdev_ref = statistics.pstdev(zs_ref)
    stdev_fxp = statistics.pstdev(zs_fxp)

    # Total-variation distance between the histograms.
    zmin = min(min(zs_ref), min(zs_fxp))
    zmax = max(max(zs_ref), max(zs_fxp))
    hist_ref = Counter(zs_ref)
    hist_fxp = Counter(zs_fxp)
    # Total variation distance between the empirical distributions.
    tv = 0.5 * sum(
        abs(hist_ref.get(z, 0) - hist_fxp.get(z, 0)) / n_samples
        for z in range(zmin, zmax + 1)
    )

    return {
        "mu": mu, "sigma": sigma, "sigmin": sigmin, "n": n_samples,
        "mean_ref": mean_ref, "mean_fxp": mean_fxp,
        "stdev_ref": stdev_ref, "stdev_fxp": stdev_fxp,
        "zmin": zmin, "zmax": zmax, "tv_distance": tv,
    }


def main():
    print("=" * 72)
    print("Test 1: KAT match on 10000 shared-rng trials (mixed small/large mu).")
    print("=" * 72)
    match, mismatches = kat_match_test(n_trials=10000)
    print(f"KAT identical: {match}/10000")
    if mismatches:
        print("First mismatches:")
        for i, mu, sigma, sigmin, z_ref, z_fxp in mismatches[:5]:
            print(f"  trial {i}: mu={mu:.4f} sigma={sigma:.4f} sigmin={sigmin:.4f} "
                  f"ref={z_ref} fxp={z_fxp}")
    print()

    print("=" * 72)
    print("Test 2: distributional match (50k samples per config).")
    print("=" * 72)
    configs = [
        # (mu, sigma, sigmin)
        (0.0, 1.5, 1.3),
        (0.3, 1.5, 1.3),
        (-0.25, 1.7, 1.3),
        (7.4, 1.5, 1.3),       # standard-style (larger mu)
        (-100.3, 1.72, 1.3),
        (500.5, 1.8, 1.3),
    ]
    print(f"{'mu':>10} {'sigma':>8} {'sigmin':>8} | "
          f"{'mean_ref':>10} {'mean_fxp':>10} | "
          f"{'std_ref':>10} {'std_fxp':>10} | {'TV':>10}")
    print("-" * 92)
    for mu, sigma, sigmin in configs:
        r = distributional_test(mu, sigma, sigmin, n_samples=50000)
        print(f"{r['mu']:>10.2f} {r['sigma']:>8.4f} {r['sigmin']:>8.4f} | "
              f"{r['mean_ref']:>10.4f} {r['mean_fxp']:>10.4f} | "
              f"{r['stdev_ref']:>10.4f} {r['stdev_fxp']:>10.4f} | {r['tv_distance']:>10.4f}")


if __name__ == "__main__":
    main()
