"""
Simulate NTRUGen, count rejections at every stage, and print summary
statistics of the bounded quantities.

Rejections are tallied for the four paper boundedness Checks (γ_fg,
α_hybrid, γ_FG, γ_root) plus the structural NTRUGen steps that can also
reject a candidate (α_GPV, NTT invertibility, NTRUSolve). Compare the
rejection counts with the paper's rejection table.
"""

import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent  # experiments/
sys.path.insert(0, str(HERE.parent / "falcon_ref"))  # vendored Falcon reference

from ntt import ntt  # noqa: E402
from ntrugen import gen_poly, ntru_solve  # noqa: E402
from ntrugen_filters import (  # noqa: E402
    GAMMA_FG_512,
    GAMMA_HYBRID,
    GAMMA_FG_UPPER_512,
    GAMMA_ROOT,
    alpha_gpv_squared,
    alpha_hybrid_squared,
    norm_fft_fg,
    norm_fft_FG,
    norm_fft_k,
)

REJECTION_LABELS = [
    "gamma_fg",
    "alpha_hybrid",
    "alpha_gpv",
    "ntt",
    "ntrusolve",
    "gamma_FG",
    "gamma_root",
]


def simulate(n: int, n_successes: int, seed: int = 0, verbose_every: int = 5):
    """Run NTRUGen until n_successes completions. Track rejections + observations."""
    random.seed(seed)

    rejections = Counter()
    observations = {
        "norm_fft_fg": [],
        "alpha_hybrid": [],
        "alpha_gpv": [],
        "norm_fft_FG": [],
        "norm_fft_k": [],
    }

    successes = 0
    total_attempts = 0
    t_start = time.time()

    while successes < n_successes:
        total_attempts += 1
        f = gen_poly(n)
        g = gen_poly(n)

        # Check 1: norm of f, g in FFT domain.
        v_fg = norm_fft_fg(f, g)
        observations["norm_fft_fg"].append(v_fg)
        if v_fg > GAMMA_FG_512:
            rejections["gamma_fg"] += 1
            continue

        # Check 2: alpha_hybrid.
        v_hybrid = alpha_hybrid_squared(f, g) ** 0.5
        observations["alpha_hybrid"].append(v_hybrid)
        if v_hybrid > GAMMA_HYBRID:
            rejections["alpha_hybrid"] += 1
            continue

        # Check (existing): alpha_GPV.
        v_gpv = alpha_gpv_squared(f, g) ** 0.5
        observations["alpha_gpv"].append(v_gpv)
        if v_gpv > 1.17:
            rejections["alpha_gpv"] += 1
            continue

        # NTT invertibility.
        if any(v == 0 for v in ntt(f)):
            rejections["ntt"] += 1
            continue

        # NTRU solve.
        try:
            F, G = ntru_solve(f, g)
        except (ValueError, AssertionError):
            rejections["ntrusolve"] += 1
            continue
        F = [int(c) for c in F]
        G = [int(c) for c in G]

        # Check 3: norm of F, G in FFT domain.
        v_FG = norm_fft_FG(F, G)
        observations["norm_fft_FG"].append(v_FG)
        if v_FG > GAMMA_FG_UPPER_512:
            rejections["gamma_FG"] += 1
            continue

        # Check 4: norm of k in FFT domain.
        v_k = norm_fft_k(f, g, F, G)
        observations["norm_fft_k"].append(v_k)
        if v_k > GAMMA_ROOT:
            rejections["gamma_root"] += 1
            continue

        successes += 1
        if verbose_every and successes % verbose_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[{successes}/{n_successes}] {elapsed:.0f}s elapsed, "
                f"{total_attempts} attempts, {sum(rejections.values())} rejections"
            )

    return rejections, observations, total_attempts


def _stats(values):
    if not values:
        return "(empty)"
    return (
        f"n={len(values)} "
        f"min={min(values):.3g} med={statistics.median(values):.3g} "
        f"mean={statistics.mean(values):.3g} max={max(values):.3g}"
    )


def print_report(n, rejections, observations, total_attempts, n_successes):
    print()
    print(f"=== NTRUGen simulation summary (n={n}) ===")
    print(f"Successful runs: {n_successes}")
    print(f"Total sampled (f,g) candidates: {total_attempts}")
    print()
    print("Rejection counts:")
    for label in REJECTION_LABELS:
        count = rejections[label]
        pct = 100 * count / total_attempts if total_attempts else 0.0
        print(f"  {label:14s} {count:6d}  ({pct:5.1f}% of attempts)")
    print()
    print("Observation distributions (unfiltered):")
    for label, values in observations.items():
        print(f"  {label:14s} {_stats(values)}")


def main():
    n = 512
    n_successes = 30
    seed = 42

    print(f"Running NTRUGen simulation: n={n}, n_successes={n_successes}, seed={seed}")
    rejections, observations, attempts = simulate(n, n_successes, seed=seed)
    print_report(n, rejections, observations, attempts, n_successes)


if __name__ == "__main__":
    main()
