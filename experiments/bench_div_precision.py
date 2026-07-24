"""
Measure the precision loss of FFT-domain complex division in three modes:
floating-point (double), FxP-63, and FxP-127. Compare each against a
high-precision mpmath reference.

Produces a Markdown-friendly table and a log-scale plot.
"""

import random
from pathlib import Path

import mpmath
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # experiments/

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from fxtypes import FxC  # noqa: E402
from fft import fft as fft_float  # noqa: E402
from fft_fxp import div_fft_fxp  # noqa: E402
from _outputs import save_fig, write_csv  # noqa: E402

mpmath.mp.prec = 256  # plenty for measuring FxP-127 error


# --------------------------------------------------------------------- #
# High-precision reference FFT (mirrors fft.py's recursion)
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


def _mp_div_fft(f_fft_mp, g_fft_mp):
    # Divide by the REAL part of g: production divides by a Gram diagonal,
    # which is real in FFT domain. Mirrors the complex÷real div_fft_fxp.
    return [a / mpmath.mpf(b.real) for a, b in zip(f_fft_mp, g_fft_mp)]


# --------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------- #


def _fxr_to_mp(a):
    return mpmath.mpf(a.x) * mpmath.mpf(2) ** (a.m - a.p)


def _fxc_list_to_mp(xs):
    return [mpmath.mpc(_fxr_to_mp(z.re), _fxr_to_mp(z.im)) for z in xs]


def _complex_list_to_mp(xs):
    return [mpmath.mpc(z.real, z.imag) for z in xs]


def _fxr_from_mp(v_mp, m: int, p: int):
    """Convert mpmath.mpf to FxR at (m, p), rounding to nearest.

    Using int(mpmath.nint(...)) preserves precision up to the mpmath
    working precision (256 bits here), way beyond float64.
    """
    from fxtypes import FxR
    x = int(mpmath.nint(v_mp * mpmath.mpf(2) ** (p - m)))
    return FxR(x=x, m=m, p=p)


def _fxc_from_mp(z_mp, m: int, p: int):
    return FxC(re=_fxr_from_mp(z_mp.real, m, p), im=_fxr_from_mp(z_mp.imag, m, p))


def _div_fxp(f_fft_mp, g_fft_mp, p, m_out):
    """Convert mpmath inputs to FxC at precision p (preserving precision),
    run div_fft_fxp, return as mpmath for comparison."""
    max_f = max(abs(z) for z in f_fft_mp)
    max_g = max(abs(z) for z in g_fft_mp)
    # Use bit_length on ceil.
    m_f = max(1, int(mpmath.ceil(mpmath.log(max_f, 2))) + 1)
    m_g = max(1, int(mpmath.ceil(mpmath.log(max_g, 2))) + 1)
    f_fxc = [_fxc_from_mp(z, m_f, p) for z in f_fft_mp]
    # Real divisor (PolyR): div_fft_fxp divides a complex poly by a real one,
    # matching the production Gram-diagonal division.
    g_fxr = [_fxr_from_mp(z.real, m_g, p) for z in g_fft_mp]
    return _fxc_list_to_mp(div_fft_fxp(f_fxc, g_fxr, m_out))


def _mp_max_err(got_mp, ref_mp):
    diffs = [
        mpmath.sqrt((g.real - r.real) ** 2 + (g.imag - r.imag) ** 2)
        for g, r in zip(got_mp, ref_mp)
    ]
    return float(max(diffs))


# --------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------- #


def _gen_inputs(n: int, seed: int):
    """Generate test inputs: f random small-coef; g chosen so Re(fft(g)) is a
    Gram-diagonal-like divisor in [q/16, 16q] — Re(fft([q, 4000, 0, ...])) ∈
    [q-4000, q+4000] ≈ [8289, 16289], the domain div_fft_fxp actually inverts."""
    rng = random.Random(seed)
    f = [rng.randint(-8, 8) for _ in range(n)]
    g = [12289, 4000] + [0] * (n - 2)
    return f, g


def bench(dims, n_trials: int = 5, seed0: int = 100):
    results = {n: {"fp": [], "fxp63": [], "fxp127": []} for n in dims}

    for n in dims:
        # |f_fft| <= 8n, divisor >= q-4000 = 8289, so |quotient| <= 8n/8289.
        m_out = max(1, (8 * n // 8289).bit_length() + 1)

        for trial in range(n_trials):
            f, g = _gen_inputs(n, seed0 + n * 100 + trial)

            # Reference: mpmath FFT and mpmath division, both at 256 bits.
            f_fft_mp = _mp_fft(f)
            g_fft_mp = _mp_fft(g)
            ref_mp = _mp_div_fft(f_fft_mp, g_fft_mp)

            # FP: use float's fft then float's div_fft. Inputs downgraded
            # to double, so this is the realistic FP path.
            f_fft_fp = fft_float(f)
            g_fft_fp = fft_float(g)
            # complex ÷ real (float64), matching the fxp/mpmath paths.
            got_fp = _complex_list_to_mp(
                [a / g.real for a, g in zip(f_fft_fp, g_fft_fp)])
            results[n]["fp"].append(_mp_max_err(got_fp, ref_mp))

            # FxP63 / FxP127: inputs taken from mpmath FFT (high precision),
            # converted to the target format before division. This isolates
            # the division-step precision (not the FFT step).
            got_63 = _div_fxp(f_fft_mp, g_fft_mp, p=63, m_out=m_out)
            got_127 = _div_fxp(f_fft_mp, g_fft_mp, p=127, m_out=m_out)
            results[n]["fxp63"].append(_mp_max_err(got_63, ref_mp))
            results[n]["fxp127"].append(_mp_max_err(got_127, ref_mp))

    return results


def print_table(results):
    print(f"{'n':>5} | {'FP (float64)':>14} | {'FxP p=63':>14} | {'FxP p=127':>14}")
    print("-" * 60)
    for n, r in sorted(results.items()):
        med = {k: sorted(r[k])[len(r[k]) // 2] for k in r}
        print(
            f"{n:>5} | {med['fp']:>14.3e} | {med['fxp63']:>14.3e} | {med['fxp127']:>14.3e}"
        )


def plot(results):
    dims = sorted(results)
    fp = [sorted(results[n]["fp"])[len(results[n]["fp"]) // 2] for n in dims]
    fxp63 = [sorted(results[n]["fxp63"])[len(results[n]["fxp63"]) // 2] for n in dims]
    fxp127 = [sorted(results[n]["fxp127"])[len(results[n]["fxp127"]) // 2] for n in dims]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(dims, fp, "o-", label="FP (float64, p=53)", color="C0", linewidth=2)
    ax.loglog(dims, fxp63, "s-", label="FxP, p=63", color="C1", linewidth=2)
    ax.loglog(dims, fxp127, "^-", label="FxP, p=127", color="C2", linewidth=2)

    ax.loglog(dims, [n * 2 ** (-53) for n in dims], "--", color="C0", alpha=0.4,
              label=r"$\sim n \cdot 2^{-53}$")
    ax.loglog(dims, [n * 2 ** (-63) for n in dims], "--", color="C1", alpha=0.4,
              label=r"$\sim n \cdot 2^{-63}$")
    ax.loglog(dims, [n * 2 ** (-127) for n in dims], "--", color="C2", alpha=0.4,
              label=r"$\sim n \cdot 2^{-127}$")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("n (polynomial dimension)")
    ax.set_ylabel(r"$\max_i\, |(f/g)_i^{\mathrm{got}} - (f/g)_i^{\mathrm{ref}}|$")
    ax.set_title(
        "Coefficient-wise complex division precision: FP vs FxP-63 vs FxP-127\n"
        "f = random small-coef polynomial, g = $q + 4000x$; error vs 256-bit mpmath"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def _medians(results):
    out = []
    for n in sorted(results):
        r = results[n]
        med = {k: sorted(r[k])[len(r[k]) // 2] for k in r}
        out.append((n, med["fp"], med["fxp63"], med["fxp127"]))
    return out


def main():
    dims = [8, 16, 32, 64, 128, 256, 512, 1024]
    print(f"Running {len(dims)} dimensions, 5 trials each...")
    results = bench(dims, n_trials=5)
    print()
    print_table(results)
    print()
    rows = _medians(results)
    write_csv(HERE / "tables" / "div_precision.csv",
              headers=["n", "fp", "fxp63", "fxp127"],
              rows=[[n, f"{a:.6e}", f"{b:.6e}", f"{c:.6e}"] for n, a, b, c in rows])
    save_fig(plot(results), "div_precision", HERE)


if __name__ == "__main__":
    main()
