"""
Measure the precision loss of the FFT in three modes: floating-point (double),
FxP-63, and FxP-127. Compare each against a high-precision mpmath reference.

Produces a Markdown-friendly table and a log-scale plot.
"""

import random
from pathlib import Path

import mpmath
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # experiments/

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from fxtypes import FxR  # noqa: E402
from fft import fft as fft_float  # noqa: E402
from fft_fxp import fft_fxp  # noqa: E402
from _outputs import save_fig, write_csv  # noqa: E402


# --------------------------------------------------------------------- #
# High-precision reference FFT via mpmath, mirroring fft.py's recursion
# --------------------------------------------------------------------- #

mpmath.mp.prec = 256  # working precision for the reference


def _mp_roots(n):
    """High-precision roots of x^n + 1 in the same order as roots_dict[n]
    (which matches the sage script: start from phi_4 = [+i, -i] and recurse
    via [sqrt(z), -sqrt(z)] for each z in the previous level)."""
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


def _mp_fft_ref(coeffs):
    """High-precision FFT matching fft.py's recursive structure, in mpmath."""
    n = len(coeffs)
    if n == 2:
        a = mpmath.mpc(coeffs[0])
        b = mpmath.mpc(coeffs[1])
        i_unit = mpmath.mpc(0, 1)
        return [a + i_unit * b, a - i_unit * b]
    f0 = coeffs[0::2]
    f1 = coeffs[1::2]
    return _mp_merge_fft(_mp_fft_ref(f0), _mp_fft_ref(f1))


def _mp_max_err(got, ref_mp):
    """Max over coefficients of |got_i - ref_i| (complex modulus), in mpmath."""
    diffs = [
        mpmath.sqrt((g.real - r.real) ** 2 + (g.imag - r.imag) ** 2)
        for g, r in zip(got, ref_mp)
    ]
    return float(max(diffs))


# --------------------------------------------------------------------- #
# Wrappers for each implementation
# --------------------------------------------------------------------- #


def _fxr_to_mp(a):
    """Convert a FxR to mpmath.mpf exactly (integer mantissa * 2^{m-p})."""
    return mpmath.mpf(a.x) * mpmath.mpf(2) ** (a.m - a.p)


def _fft_fxp_p(coeffs, p: int):
    """FFT at precision p; returns list of mpmath.mpc values preserving full precision."""
    m_in = max(1, max(abs(x) for x in coeffs).bit_length())
    f_fxp = [FxR.from_int(x, m_in, p) for x in coeffs]
    out = fft_fxp(f_fxp)
    return [mpmath.mpc(_fxr_to_mp(z.re), _fxr_to_mp(z.im)) for z in out]


def _fft_float_mp(coeffs):
    """FFT in float64, converted to mpmath.mpc for comparison."""
    return [mpmath.mpc(z.real, z.imag) for z in fft_float(coeffs)]


# --------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------- #


def bench(dims, n_trials: int = 5, seed: int = 0):
    """For each n, return dict with median / max error per implementation."""
    rng = random.Random(seed)
    results = {n: {"fp": [], "fxp63": [], "fxp127": []} for n in dims}

    for n in dims:
        for _ in range(n_trials):
            coeffs = [rng.randint(-8, 8) for _ in range(n)]
            ref_mp = _mp_fft_ref(coeffs)

            got_fp = _fft_float_mp(coeffs)
            got_fxp63 = _fft_fxp_p(coeffs, p=63)
            got_fxp127 = _fft_fxp_p(coeffs, p=127)

            results[n]["fp"].append(_mp_max_err(got_fp, ref_mp))
            results[n]["fxp63"].append(_mp_max_err(got_fxp63, ref_mp))
            results[n]["fxp127"].append(_mp_max_err(got_fxp127, ref_mp))

    return results


def medians(results):
    """Return list of (n, fp, fxp63, fxp127) tuples (medians per dim)."""
    out = []
    for n, r in sorted(results.items()):
        med = {k: sorted(r[k])[len(r[k]) // 2] for k in r}
        out.append((n, med["fp"], med["fxp63"], med["fxp127"]))
    return out


def print_table(rows):
    print(f"{'n':>5} | {'FP (float64)':>14} | {'FxP p=63':>14} | {'FxP p=127':>14}")
    print("-" * 60)
    for n, fp, fxp63, fxp127 in rows:
        print(f"{n:>5} | {fp:>14.3e} | {fxp63:>14.3e} | {fxp127:>14.3e}")


def plot(rows):
    """Build the precision-vs-n loglog plot from medians rows."""
    dims = [r[0] for r in rows]
    fp = [r[1] for r in rows]
    fxp63 = [r[2] for r in rows]
    fxp127 = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(dims, fp, "o-", label="FP (float64, p=53)", color="C0", linewidth=2)
    ax.loglog(dims, fxp63, "s-", label="FxP, p=63", color="C1", linewidth=2)
    ax.loglog(dims, fxp127, "^-", label="FxP, p=127", color="C2", linewidth=2)

    # Reference lines: 2^{-p} scaling.
    ref_n = dims
    ax.loglog(
        ref_n,
        [2 ** (-53) * n**0.5 for n in ref_n],
        "--",
        color="C0",
        alpha=0.4,
        label=r"$\sim \sqrt{n}\, \cdot 2^{-53}$",
    )
    ax.loglog(
        ref_n,
        [2 ** (-63) * n**0.5 for n in ref_n],
        "--",
        color="C1",
        alpha=0.4,
        label=r"$\sim \sqrt{n}\, \cdot 2^{-63}$",
    )
    ax.loglog(
        ref_n,
        [2 ** (-127) * n**0.5 for n in ref_n],
        "--",
        color="C2",
        alpha=0.4,
        label=r"$\sim \sqrt{n}\, \cdot 2^{-127}$",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("n (FFT dimension)")
    ax.set_ylabel(r"$\max_i\, |FFT(f)_i - \mathrm{ref}_i|$")
    ax.set_title(
        "FFT precision: floating-point vs FxP-63 vs FxP-127\n"
        "(random integer polynomial with $|f_i| \\leq 8$, error vs 256-bit mpmath)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def main():
    dims = [8, 16, 32, 64, 128, 256, 512, 1024]
    print(f"Running {len(dims)} dimensions, 5 trials each...")
    results = bench(dims, n_trials=5, seed=42)
    rows = medians(results)
    print()
    print_table(rows)
    print()
    write_csv(HERE / "tables" / "fft_precision.csv",
              headers=["n", "fp", "fxp63", "fxp127"],
              rows=[[n, f"{fp:.6e}", f"{fxp63:.6e}", f"{fxp127:.6e}"]
                    for n, fp, fxp63, fxp127 in rows])
    save_fig(plot(rows), "fft_precision", HERE)


if __name__ == "__main__":
    main()
