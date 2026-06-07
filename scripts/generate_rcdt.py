"""
Generate the reverse cumulative distribution table (RCDT) for the
half-Gaussian base sampler used inside Falcon's samplerz.

Two options are supported, switchable at runtime so the same machinery
covers both Falcon's current spec and the LTYZ countermeasure:

  (i)  ``floor`` (Falcon spec, status quo): the bimodal extension
       ``z = b + (2b−1)·z₀``, b ∈ {0, 1}, where z₀ is sampled from the
       distribution close to D_{Z⁺, σ_max, 0} of Falcon spec §3.9.3.
       NOT halved at z₀ = 0: the bimodal extension maps z₀ = 0 to
       z ∈ {0, 1} via b ∈ {0, 1}, no double-counting.

  (ii) ``round`` (LTYZ Algorithm 4 NewBaseSampler, paper 2024-1709):
       the bimodal extension ``z = (2b−1)·z₀``, b ∈ {0, 1}, where z₀
       is sampled from the distribution close to D_{Z⁺, σ_max, 1/2}.
       D(0) IS halved: the bimodal extension maps z₀ = 0 to z = 0 via
       both values of b (sign of zero is moot), so without halving z = 0
       would be double-counted.

Both options use σ_max = 1.8205 (Falcon spec, all parameter sets) and
19 bins (z₀ ∈ {0, ..., 18}). Output is RCDT[0..17] with RCDT[k] =
P(z₀ ≥ k+1) · 2^72.

PDT rounding follows Howe-Prest-Ricosset-Rossi 2019-1411 §5.2 (the
implementation paper for the Falcon basesampler):

  - for j ≠ argmax PMF, PDT[j] = ⌊2^72 · D(j)⌋  (truncate / floor)
  - for j = argmax PMF,  PDT[j] = 2^72 − Σ_{others}     (mass conservation)

The "clever rounding" trick absorbs the cumulative rounding loss into
the single largest entry where its relative impact is smallest, and
guarantees the integer PDT sums to 2^72 exactly (a proper probability).
For Falcon spec the largest entry is D(0); for NewBaseSampler the
largest is D(1) (twice the halved D(0)).

When run without arguments, this script generates both tables and
checks that option (i) matches bit-for-bit the RCDT hardcoded in
``falcon_ref/samplerz.py``.

Usage:
  python3 scripts/generate_rcdt.py                # both, with diagnostics
  python3 scripts/generate_rcdt.py --option floor # spec table only
  python3 scripts/generate_rcdt.py --option round # LTYZ table only
"""

import argparse
import mpmath


# 256-bit mpmath internal precision is well above the 72-bit RCDT_PREC,
# leaving ~180 bits of headroom for the cumulative subtraction.
mpmath.mp.prec = 512

SIGMA_MAX = mpmath.mpf("1.8205")    # Falcon spec, all parameter sets.
RCDT_PREC = 72
NUM_BINS = 19                       # z₀ ∈ {0, 1, ..., 18}


# Reference table from `falcon_ref/samplerz.py` for the (i) floor case.
# This is what the (i) generator must reproduce bit-for-bit.
RCDT_FLOOR_REF = [
    3024686241123004913666, 1564742784480091954050, 636254429462080897535,
    199560484645026482916,  47667343854657281903,   8595902006365044063,
    1163297957344668388,    117656387352093658,     8867391802663976,
    496969357462633,        20680885154299,         638331848991,
    14602316184,            247426747,              3104126,
    28824,                  198,                    1,
]


def half_gaussian_pmf(mu, halve_zero):
    """Probability mass function (D(0), ..., D(18)) of the half-Gaussian
    centered on `mu` with parameter σ_max.

    `halve_zero` chooses between the two conventions:

      - False (Falcon classic basesampler): D(i) ∝ ρ(i) for all i ∈ {0..18}.
        The downstream bimodal extension is `z ← b + (2b−1)·z₀` so z₀ = 0
        maps to z ∈ {0, 1} via b ∈ {0, 1} — NO double-counting of z = 0.

      - True  (LTYZ NewBaseSampler, Algorithm 4): D(0) ∝ ρ(0) / 2 with the
        rest unchanged. Here the bimodal extension is `y ← (2b−1)·y₊` so
        y₊ = 0 maps to y = 0 via both b ∈ {0, 1} (sign of zero is moot),
        which would double-count z = 0 without the halving.

    `mu` is an mpmath mpf so the entire pipeline stays in arbitrary
    precision until the final round to RCDT_PREC bits.
    """
    rho = lambda z: mpmath.exp(-(mpmath.mpf(z) - mu) ** 2 / (2 * SIGMA_MAX ** 2))
    if halve_zero:
        unnorm = [rho(0) / 2] + [rho(i) for i in range(1, NUM_BINS)]
    else:
        unnorm = [rho(i) for i in range(NUM_BINS)]
    norm = sum(unnorm)
    return [u / norm for u in unnorm]


def make_rcdt(pmf, mass_conserve_idx):
    """RCDT[k] = 2^72 − Σ_{j ≤ k} PDT[j] where PDT[] is built per the
    "clever rounding" of Falcon (Howe-Prest-Ricosset-Rossi 2019-1411 §5.2):

      - for j ≠ mass_conserve_idx, PDT[j] = ⌊2^72 · D(j)⌋ (truncate / floor)
      - for j = mass_conserve_idx, PDT[j] = 2^72 − Σ_{j' ≠ idx} PDT[j']

    This guarantees Σ PDT = 2^72 exactly (the integer-PDT distribution is
    a proper probability) and absorbs the cumulative rounding error into
    the single largest entry, where the relative loss is smallest.

    `mass_conserve_idx` is the index of the largest D(j) — for Falcon spec
    (μ = 0) that's idx = 0; for NewBaseSampler (μ = 1/2) that's idx = 1
    (D(1) ∝ ρ(1/2) is twice as large as the halved D(0) ∝ ρ(1/2)/2).
    """
    scale = 1 << RCDT_PREC
    pdt = [0] * NUM_BINS
    for j in range(NUM_BINS):
        if j == mass_conserve_idx:
            continue
        pdt[j] = int(mpmath.floor(pmf[j] * scale))
    pdt[mass_conserve_idx] = scale - sum(pdt)  # absorbs the rounding loss
    rcdt = []
    cumul = 0
    for k in range(NUM_BINS - 1):
        cumul += pdt[k]
        rcdt.append(scale - cumul)
    return rcdt


def print_rcdt(name, rcdt):
    print(f"{name} = [")
    for v in rcdt:
        print(f"    {v},")
    print("]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--option",
        choices=("floor", "round", "both"),
        default="both",
        help="Which RCDT to generate (default: both, with floor cross-check)",
    )
    args = parser.parse_args()

    if args.option in ("floor", "both"):
        pmf = half_gaussian_pmf(mu=mpmath.mpf(0), halve_zero=False)
        # Largest entry is D(0) ∝ 1 (μ = 0).
        rcdt = make_rcdt(pmf, mass_conserve_idx=0)
        print("=== Option (i) floor : μ = 0 (Falcon spec, status quo) ===")
        print_rcdt("RCDT_FLOOR", rcdt)
        # With the right rounding procedure (truncate-then-mass-conserve
        # on the largest entry, per 2019-1411 §5.2), this should match
        # the samplerz.py RCDT bit-for-bit.
        if rcdt == RCDT_FLOOR_REF:
            print("\nMatches samplerz.py RCDT bit-for-bit ✓")
        else:
            print("\n*** MISMATCH with samplerz.py RCDT — investigate ***")
            for k, (gen, ref) in enumerate(zip(rcdt, RCDT_FLOOR_REF)):
                if gen != ref:
                    print(f"  k={k:2d}: gen={gen}, ref={ref}, diff={gen - ref:+d}")

    if args.option in ("round", "both"):
        if args.option == "both":
            print()
        pmf = half_gaussian_pmf(mu=mpmath.mpf(1) / 2, halve_zero=True)
        # D(1) ∝ ρ(1/2) is the largest entry (twice D(0) ∝ ρ(1/2)/2).
        rcdt = make_rcdt(pmf, mass_conserve_idx=1)
        print("=== Option (ii) round : μ = 1/2 (LTYZ NewBaseSampler) ===")
        print_rcdt("RCDT_ROUND", rcdt)


if __name__ == "__main__":
    main()
