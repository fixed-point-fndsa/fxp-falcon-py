"""
Generate ALL fxp numerical constants as integer FxR/FxC mantissas at
configurable precision p (currently p ∈ {63, 127}). Each call writes
`fxp/fxp_constants_p{p}.py` overwriting any prior content.

Usage:
    sage scripts/generate_constants_fxp.sage [p=53] [m_fft=1]

Constants emitted (each at its own m, fixed per constant — see docstrings
of consumers for the rationale):

  - roots_dict_fxp                FFT roots of unity (phi_{2n} per n),
                                  at m=m_fft (1 by default; 0 for "tight"
                                  twiddles — see notes below).
  - INV_SQRT_Q_FXR                1/√q anchor for the Newton-Raphson seed
                                  in `rsqrt`. m=-6 (natural bound on the
                                  rsqrt output range).
  - INV_Q_FXC                     1/q as FxC, used as a multiplicand in
                                  `target_construction._div_by_q_fxc`.
                                  m=-13 (|1/q| ≈ 2^-13.586 < 2^-13).
  - SIGMA_FXR_BY_N                Per-degree gaussian standard deviation
                                  σ_n from the Falcon NIST spec (cf.
                                  `falcon.Params[n]['sigma']`). m=8.
  - SIGMIN_FXR_BY_N               Per-degree sigmin from Falcon spec. m=1.

Q = 12289 (Falcon modulus).

Design notes for the FFT roots:
    - Computed in ComplexField(p + 40) to leave 40 guard bits when
      rounding to the p-bit mantissa.
    - m_fft = 0 is the TIGHT choice: |w| <= 1 for every twiddle, and we
      force strict |w| < 1 by rounding (x_re, x_im) such that the complex
      modulus is strictly below 2^0 = 1. For phi_4 = {+i, -i}, the
      component ±1 gets rounded down by 1 ULP to (2^p − 1) / 2^p < 1.
      Relative error of 2^{-p} on the affected twiddles (~10^{-19} at
      p=63, ~10^{-38} at p=127); the gain is that FFT butterflies no
      longer grow m by +1 per merge level.
    - m_fft = 1 is the default (exact phi_4, no pre-scaling). Switching
      to m_fft=0 was explored in May 2026 and rejected: the lossless
      `retag_value_fxc(f1, m)` introduced in `split_fft_fxp` already
      achieves the same downstream simplification WITHOUT modifying the
      twiddle constants.
    - Convention for roots_dict matches the float reference: roots_dict[n]
      = roots of x^n + 1 = roots of the cyclotomic phi_{2n}.

Design notes for the float-derived constants (INV_Q_FXC, SIGMA_*, SIGMIN_*):
    - Source values are Python float64 (1.0/12289 for inv_q; falcon.Params
      values for sigma/sigmin). We reproduce `FxR.from_float`'s rounding
      EXACTLY via `math.ldexp + round`, so the generated mantissas are
      bit-identical to what the previous hand-written constants gave.
      This avoids any LTYZ-style boundary flip risk in the samplerz when
      switching to the generated file.

Design notes for the integer-derived constant (INV_SQRT_Q_FXR):
    - Pure `Integer.isqrt`-based recipe (no float intermediate): x =
      round(√(2^{2(p-m)} / q)). At p=63 this reproduces the value that
      was previously hardcoded in `ffldl_fxp.Y0_INV_SQRT_Q_FXR`.
"""

import math
import sys
from pathlib import Path

# --- Parse arguments --------------------------------------------------

p = int(sys.argv[1]) if len(sys.argv) > 1 else 53
m_fft = int(sys.argv[2]) if len(sys.argv) > 2 else 1
assert 0 <= m_fft <= p, "require 0 <= m_fft <= p"

Q = 12289

# Per-degree (sigma, sigmin) — copied verbatim from `falcon.Params` (which
# itself follows the NIST FN-DSA spec). These are float64 literals; we
# preserve their float64 rounding when emitting FxR mantissas.
FALCON_PARAMS = {
    2:    (144.81253976308423, 1.1165085072329104),
    4:    (146.83798833523608, 1.1321247692325274),
    8:    (148.83587593064718, 1.147528535373367),
    16:   (151.78340713845503, 1.170254078853483),
    32:   (154.6747794602761,  1.1925466358390344),
    64:   (157.51308555044122, 1.2144300507766141),
    128:  (160.30114421975344, 1.235926056771981),
    256:  (163.04153322607107, 1.2570545284063217),
    512:  (165.7366171829776,  1.2778336969128337),
    1024: (168.38857144654395, 1.298280334344292),
}

# Per-constant m (rationale: see docstrings of consumers).
M_INV_SQRT_Q = -6
M_INV_Q      = -13
M_SIGMA      = 8
M_SIGMIN     = 1
M_INV_SIGMA  = -7    # 1/σ_n ∈ [0.0059, 0.0069] < 2^-7 across all degrees

GUARD_BITS = 40
WORKING_PREC = p + GUARD_BITS
CC = ComplexField(WORKING_PREC)
RR_high = RealField(WORKING_PREC)

# --- Generate the complex roots of unity (mirrors generate_constants.sage)

phi4 = cyclotomic_polynomial(4)
phi4_roots_cc = phi4.roots(CC, multiplicities=False)
phi4_roots_cc.reverse()  # yields [+i, -i]

roots_by_n = {4: phi4_roots_cc}
for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    prev = roots_by_n[n // 2]
    flat = []
    for r in prev:
        s = r.sqrt()
        flat.append(s)
        flat.append(-s)
    roots_by_n[n] = flat

# --- Convert each FFT root to integer mantissas ----------------------

scale_fft = Integer(1) << (p - m_fft)   # mantissa = round(value * 2^{p-m_fft})
bound = Integer(1) << p                  # |mantissa| must be < bound

# For m_fft=0, round-to-nearest on each component may push |z|^2 above
# 2^{2p}. Pre-scale every twiddle by (1 − 4·2^{-p}) to stay safely inside
# the unit disk under any per-component rounding direction.
PRE_SCALE = 1 - Integer(4) / (Integer(1) << p) if m_fft == 0 else Integer(1)


def to_mantissa_fft(v_real):
    """v_real is a ComplexField real; scale (for m_fft=0) then round to FX_{m_fft,p}."""
    scaled = v_real * PRE_SCALE
    x = Integer((scaled * scale_fft).round())
    if abs(x) >= bound:
        x = (bound - 1) if x > 0 else -(bound - 1)
    return int(x)


def from_float_mantissa(v_py, m_const):
    """Reproduce `FxR.from_float(v_py, m_const, p)` EXACTLY via Python's
    math.ldexp + round (banker's rounding). v_py is a Python float64."""
    return int(round(math.ldexp(float(v_py), int(p) - int(m_const))))


def inv_sqrt_int_mantissa(q_int, m_const):
    """Pure integer 1/√q recipe: x = round(√(2^{2(p-m)} / q_int)). Matches
    the `inv_sqrt_int_fxr` helper that previously lived in `ffldl_fxp`."""
    shift = 2 * (p - m_const)
    target = (Integer(1) << shift) // q_int     # floor(2^{2(p-m)} / q)
    s = Integer(target).isqrt()                  # floor(√target)
    # Round to nearest integer (ties down — same rule as runtime helper).
    if (s + 1) ** 2 - target < target - s ** 2:
        s += 1
    return int(s)


# --- Compute the non-FFT constants -----------------------------------

inv_sqrt_q_x = inv_sqrt_int_mantissa(Q, M_INV_SQRT_Q)

inv_q_x  = from_float_mantissa(1.0 / Q, M_INV_Q)

sigma_x_by_n  = {n: from_float_mantissa(s, M_SIGMA)
                 for n, (s, _) in FALCON_PARAMS.items()}
sigmin_x_by_n = {n: from_float_mantissa(sm, M_SIGMIN)
                 for n, (_, sm) in FALCON_PARAMS.items()}
# 1/σ_n: lets samplerz replace its per-call divisions (dss = inv_sigma²/2,
# ccs = sigmin·inv_sigma) by multiplications. The leaf inv_sigma_i = 1/σ_i is
# built at normalize time as √D_ii · INV_SIGMA, division-free.
inv_sigma_x_by_n = {n: from_float_mantissa(1.0 / s, M_INV_SIGMA)
                    for n, (s, _) in FALCON_PARAMS.items()}

# --- Write the output module -----------------------------------------

OUT_PATH = Path(__file__).resolve().parent.parent / "fxp" / ("fxp_constants_p%d.py" % p)

HEADER = '''\
"""
Fixed-point numerical constants at p={p} (Falcon modulus Q=12289).

Generated by scripts/generate_constants_fxp.sage with (m_fft, p) = ({m_fft}, {p}).

Contents (m values are CONSTANT-specific — see the sage script's docstring
for the rationale per constant):

  - roots_dict_fxp        FFT roots of unity (m=m_fft={m_fft})
  - INV_SQRT_Q_FXR        1/√q  at m=-6
  - INV_Q_FXC             1/q   at m=-13
  - SIGMA_FXR_BY_N        Falcon-spec σ_n  at m=8
  - SIGMIN_FXR_BY_N       Falcon-spec sigmin_n at m=1
  - INV_SIGMA_FXR_BY_N    1/σ_n  at m=-7

Do not edit by hand; re-run the sage script to regenerate.
"""

from fxtypes import FxR, FxC


def _mk_fxc(pairs, m):
    """Wrap a list of (x_re, x_im) mantissa pairs into FxC values at (m, p)."""
    return [FxC(re=FxR(x=r, m=m, p={p}), im=FxR(x=i, m=m, p={p}))
            for r, i in pairs]


def _mk_fxr_by_n(d, m):
    """Wrap a dict {{n: x_mantissa}} into a dict {{n: FxR(x, m, p)}}."""
    return {{n: FxR(x=x, m=m, p={p}) for n, x in d.items()}}


'''

with OUT_PATH.open("w") as f:
    f.write(HEADER.format(m_fft=m_fft, p=p))

    # FFT roots, organised per phi_{2n} block.
    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        roots = roots_by_n[n]
        f.write("# ------ phi_%d roots ------\n" % n)
        f.write("_phi%d = [\n" % n)
        for r in roots:
            xr = to_mantissa_fft(r.real())
            xi = to_mantissa_fft(r.imag())
            f.write("    (%d, %d),\n" % (xr, xi))
        f.write("]\n")
        f.write("phi%d_roots_fxp = _mk_fxc(_phi%d, m=%d)\n\n" % (n, n, m_fft))

    f.write("# Dictionary mapping n (dim of the poly ring) to the roots.\n")
    f.write("# Same convention as fft_constants.py: roots_dict[n] = roots of x^n + 1.\n")
    f.write("roots_dict_fxp = {\n")
    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        f.write("    %d: phi%d_roots_fxp,\n" % (n // 2, n))
    f.write("}\n\n")

    # Non-FFT constants.
    f.write("# ------ Non-FFT constants ------\n")
    f.write("INV_SQRT_Q_FXR = FxR(x=%d, m=%d, p=%d)\n\n"
            % (inv_sqrt_q_x, M_INV_SQRT_Q, p))

    f.write("INV_Q_FXC = FxC(re=FxR(x=%d, m=%d, p=%d),\n"
            "                im=FxR(x=0,   m=%d, p=%d))\n\n"
            % (inv_q_x, M_INV_Q, p, M_INV_Q, p))

    f.write("_SIGMA_X_BY_N = {\n")
    for n in sorted(sigma_x_by_n.keys()):
        f.write("    %d: %d,\n" % (n, sigma_x_by_n[n]))
    f.write("}\n")
    f.write("SIGMA_FXR_BY_N = _mk_fxr_by_n(_SIGMA_X_BY_N, m=%d)\n\n" % M_SIGMA)

    f.write("_SIGMIN_X_BY_N = {\n")
    for n in sorted(sigmin_x_by_n.keys()):
        f.write("    %d: %d,\n" % (n, sigmin_x_by_n[n]))
    f.write("}\n")
    f.write("SIGMIN_FXR_BY_N = _mk_fxr_by_n(_SIGMIN_X_BY_N, m=%d)\n\n" % M_SIGMIN)

    f.write("_INV_SIGMA_X_BY_N = {\n")
    for n in sorted(inv_sigma_x_by_n.keys()):
        f.write("    %d: %d,\n" % (n, inv_sigma_x_by_n[n]))
    f.write("}\n")
    f.write("INV_SIGMA_FXR_BY_N = _mk_fxr_by_n(_INV_SIGMA_X_BY_N, m=%d)\n" % M_INV_SIGMA)

print("wrote %s  (m_fft=%d, p=%d, working precision=%d bits)"
      % (OUT_PATH, m_fft, p, WORKING_PREC))
