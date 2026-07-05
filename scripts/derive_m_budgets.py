"""Derive the minimal provable m-budgets from the NTRUGen checks and the
paper's lemmas, and compare them to the constants chosen in fxp/m_budgets.py.

Derive-and-verify, NOT codegen: m_budgets.py stays hand-written (with its
derivation comments); this script recomputes each PROVEN lower bound from
(n, q, lambda, sigma, gamma_*, ...) and prints chosen vs derived + slack.
`tests/test_m_budgets_derivation.py` asserts chosen >= derived for every
PROVEN budget. M_S_INTER is EMPIRICAL (no clean worst-case chain: the
Lemma-13 residual bound is ~2^26.6, far above the observed ~2^17.76; the
runtime asserts in `_reconstruct_s_fxp` are its guard) and is only reported.

Usage: uv run python scripts/derive_m_budgets.py  [--gamma-root 12 ...]
What-ifs: every threshold is a CLI flag, so e.g. the keygen-check trade
studies (gamma_FG=3200, gamma_root=12, n=1024) are one-liners.
"""

import argparse
import math
from dataclasses import dataclass


@dataclass
class Params:
    n: int = 512
    q: int = 12289
    bitsec: int = 128           # lambda, tail parameter of Lemma 13
    sigma: float = 165.7366171829776
    gamma_fg: int = 255         # Check 1b (strict <)
    alpha_h: int = 4            # gamma_hybrid, Check 2
    gamma_FG: int = 3500        # Check 3
    gamma_root: int = 24        # Check 4
    alpha_gpv: float = 1.17     # stock gs_norm bound (final leaves)
    cdt_kmax: int = 17          # gen_poly CDT support (gauss_512)
    fg_coef_limit: int = 127    # int8 encoding filter on F, G


def _m_min(bound: float, attainable: bool) -> int:
    """Smallest m such that the FxR invariant |value| < 2^m is guaranteed.

    attainable=True : the bound itself is a reachable value (e.g. an integer
        coefficient with |c| <= 17) -> need bound < 2^m strictly, i.e.
        m = floor(log2 bound) + 1  (17 -> 5, and 16 -> 5, not 4).
    attainable=False: the bound is a strict sup (e.g. ||fft(f)||_inf < 255)
        -> bound <= 2^m suffices, i.e. m = ceil(log2 bound)  (255 -> 8).
    """
    if attainable:
        return math.floor(math.log2(bound)) + 1
    return math.ceil(math.log2(bound))


def derive(p: Params) -> dict:
    """Derived minimal m per PROVEN budget.

    Reading guide — one `add(...)` line per budget:
        add(NAME, bound, attainable, formula, source)
    means: the variable tagged NAME has |value| bounded by `bound` (per
    `formula`, proven by `source`), so its minimal tag is
    _m_min(bound, attainable). Returns {NAME: (m_min, formula, source)},
    plus the Lemma-13 intermediates under underscore keys.
    """
    # Lemma 13 machinery.
    tau = math.sqrt(2 * (p.bitsec + math.log2(2 * p.n * math.log(p.n)) + 1)
                    * math.log(2))
    a_child = 2 * tau * p.sigma * p.alpha_h / math.sqrt(p.q)
    drift = 2 * a_child * math.sqrt(p.n) * (p.gamma_root + math.sqrt(2) + 1)

    d = {}

    def add(name, bound, attainable, formula, source):
        d[name] = (_m_min(bound, attainable), formula, source)

    # Coefficient-domain loads (integer values: bounds are attainable).
    add("M_B0_COEF_FG", p.cdt_kmax, True, "|f,g coefs| <= CDT kmax", "gauss table (c-fn-dsa)")
    add("M_B0_COEF_FG_UP", p.fg_coef_limit, True, "||F,G||_inf <= 127", "int8 encoding filter")
    add("M_QT_COEF", (p.q - 1) / 2, True, "|qt| <= (q-1)/2", "mod+- q centering")
    add("M_POINT_COEF", p.q - 1, True, "point in [0, q)", "hash_to_point")
    add("M_CQ_COEF", 1, False, "c/q <= (q-1)/q < 1", "coefficient-domain /q")
    # FFT-domain B0 rows (check thresholds: strict sups).
    add("M_B_FG", p.gamma_fg, False, "||fft(f,g)|| < gamma_fg", "Check 1b")
    add("M_B_FG_UP", p.gamma_FG, True, "||fft(F,G)|| <= gamma_FG", "Check 3")
    # Root Gram.
    # (M_G00 collapsed into M_D on 2026-07-05: g00 < 2*gamma_fg^2 = 2^16.99 is
    # emitted directly at M_D=18, removing the root D00 widen.)
    add("M_G01", 2 * p.gamma_fg * p.gamma_FG, False, "|g10| < 2*gamma_fg*gamma_FG", "Checks 1b+3")
    # ffLDL tree.
    add("M_L10_ROOT", p.gamma_root, True, "||L10_root|| <= gamma_root", "Check 4")
    add("M_L10_INNER", 1, False, "|L10| < 1", "Lemma 9")
    add("M_D", p.alpha_h ** 2 * p.q, True, "D_ii <= alpha_h^2*q", "Lemma 9 (Check 2)")
    add("M_D_LEAF", p.alpha_gpv ** 2 * p.q, True, "D_leaf <= 1.17^2*q", "gs_norm (final leaves)")
    add("M_NORM_OUT", 1, False, "1/sigma_i, dss, ccs < 1", "sigma_i > 1 (gs_norm)")
    # ffsampling (uniform budgets; Lemma 13 tail bounds, treated as sups).
    add("M_SIGN_DEFAULT", p.n / 2 + drift, False, "n/2 + drift", "Lemma 13 (tweak)")
    add("M_SIGN_STD", p.n * (p.q - 1) * p.gamma_FG / p.q + drift, False,
        "n*gamma_FG + drift", "Lemma 13 (std)")
    # samplerz.
    add("M_SZ_DIFF", 19.5, False, "|z_int - r| < 19.5", "lem:samplerz")

    d["_tau"], d["_A_child"], d["_drift"] = tau, a_child, drift
    return d


# The one budget with no clean worst-case chain (see module docstring).
EMPIRICAL = {"M_S_INTER": "each |diff*B| ~ 2^17.76 observed; guarded by "
                          "runtime asserts in _reconstruct_s_fxp"}


def main():
    ap = argparse.ArgumentParser()
    for f, t in (("n", int), ("q", int), ("bitsec", int), ("sigma", float),
                 ("gamma-fg", int), ("alpha-h", int), ("gamma-FG", int),
                 ("gamma-root", int), ("alpha-gpv", float), ("cdt-kmax", int),
                 ("fg-coef-limit", int)):
        ap.add_argument(f"--{f}", type=t, default=None)
    args = ap.parse_args()
    p = Params(**{k.replace("-", "_"): v for k, v in vars(args).items()
                  if v is not None})

    d = derive(p)
    print(f"Params: n={p.n} q={p.q} lambda={p.bitsec} "
          f"gammas=({p.gamma_fg},{p.alpha_h},{p.gamma_FG},{p.gamma_root}) "
          f"kmax={p.cdt_kmax} lim={p.fg_coef_limit}")
    print(f"tau={d['_tau']:.3f}  A_child={d['_A_child']:.1f}  "
          f"drift={d['_drift']:.0f}=2^{math.log2(d['_drift']):.2f}\n")

    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fxp"))
        import m_budgets
    except ImportError:
        m_budgets = None

    print(f"{'budget':<18} {'derived':>7} {'chosen':>7} {'slack':>5}  formula (source)")
    for name, val in d.items():
        if name.startswith("_"):
            continue
        m_min, formula, source = val
        chosen = getattr(m_budgets, name, None) if m_budgets else None
        slack = "" if chosen is None else f"{chosen - m_min:+d}"
        flag = "" if chosen is None or chosen >= m_min else "  <-- UNDER-PROVISIONED"
        print(f"{name:<18} {m_min:>7} {str(chosen):>7} {slack:>5}  {formula} ({source}){flag}")
    for name, note in EMPIRICAL.items():
        chosen = getattr(m_budgets, name, None) if m_budgets else None
        print(f"{name:<18} {'emp.':>7} {str(chosen):>7} {'':>5}  {note}")


if __name__ == "__main__":
    main()
