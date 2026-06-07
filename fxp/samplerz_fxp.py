"""
Fixed-point port of Falcon's samplerz.

Mirrors falcon_ref/samplerz.py exactly in structure, but takes `mu` as
FxR and does the "float arithmetic on mu/r/x" in FxR. Already-integer
parts of the reference (basesampler RCDT, approxexp's 13-coef poly,
berexp's byte-wise compare) stay unchanged.

samplerz only depends on r = mu − floor(mu) ∈ [0, 1), so mu can have
any magnitude — we split into (s = floor(mu), r) and use r throughout.

At p=63 the 63-bit mantissa exceeds float64's 53 bits; with shared
randomness, outputs are byte-identical to the float64 reference.
"""

from os import urandom
from typing import Callable

from beartype import beartype

from fxtypes import FxR, retag_value_fxr as _retag_m


# Constants (Falcon spec / reference samplerz).
_P = 63

# Default mode shared with falcon_ref/samplerz.py: "floor" reproduces the
# Falcon spec sampler bit-for-bit; "round" implements LTYZ NewSamplerZ
# (Algorithm 4 of paper 2024-1709), shifting the round-off-sensitive
# locus from integer centers to half-integer ones (which don't occur
# in Falcon under the paper's keygen restriction ‖(g, −f)‖² odd).
SAMPLERZ_MODE = "floor"

# Reverse CDF table (72-bit precision). Two flavors: floor (Falcon spec
# half-Gaussian D_{Z+, σmax, 0}, no halving) and round (LTYZ
# NewBaseSampler, half-Gaussian D_{Z+, σmax, 1/2} with D(0) halved).
# Both are produced by `scripts/generate_rcdt.py` per Howe-Prest-Ricosset-
# Rossi 2019-1411 §5.2 rounding (truncate + mass-conserve on the largest
# entry).
RCDT_PREC = 72
RCDT_FLOOR = [
    3024686241123004913666,
    1564742784480091954050,
    636254429462080897535,
    199560484645026482916,
    47667343854657281903,
    8595902006365044063,
    1163297957344668388,
    117656387352093658,
    8867391802663976,
    496969357462633,
    20680885154299,
    638331848991,
    14602316184,
    247426747,
    3104126,
    28824,
    198,
    1,
]
RCDT_ROUND = [
    3899470320143983480877,
    2253677994692660015228,
    1036552621035575352532,
    370887374867150348556,
    101649140428940578446,
    21115400469750986562,
    3300668784248344015,
    386323619417263643,
    33739318213646739,
    2193249886848467,
    105935861234397,
    3797090532647,
    100905148891,
    1986734192,
    28967961,
    312672,
    2496,
    14,
]
# Backwards-compat alias (same as samplerz.py).
RCDT = RCDT_FLOOR

# Polynomial coefficients for exp approximation (lifted from FACCT, same as reference).
_C = [
    0x00000004741183A3,
    0x00000036548CFC06,
    0x0000024FDCBF140A,
    0x0000171D939DE045,
    0x0000D00CF58F6F84,
    0x000680681CF796E3,
    0x002D82D8305B0FEA,
    0x011111110E066FD0,
    0x0555555555070F00,
    0x155555555581FF00,
    0x400000000002B400,
    0x7FFFFFFFFFFF4800,
    0x8000000000000000,
]

# Hardcoded FxR constants — x = round(value · 2^{p-m}). Computed once
# offline (see scripts/generate_constants_fxp.sage if you ever need to
# refresh) so that this module loads without a single float operation.
LN2_FXR = FxR(x=6393154322601832448, m=0, p=_P)             # 0.69314718056
ILN2_FXR = FxR(x=6653256548926941184, m=1, p=_P)            # 1.44269504089
INV_2SIGMA2_FXR = FxR(x=1391484473135841792, m=0, p=_P)     # 1/(2·1.8205²) = 0.150865...


# Local FxR helpers. Python's >> on negatives floors, matching
# floor(a.value) = a.x >> (p - m); p - m >= 0 by the m <= p invariant.

def _floor_value(a: FxR) -> int:
    """floor(a.value) as a Python int."""
    scale = a.p - a.m
    return a.x if scale == 0 else a.x >> scale


def _floor_and_frac(a: FxR) -> tuple[int, FxR]:
    """Split a = s + r with s = floor(a.value) ∈ Z, r ∈ [0, 1).
    r is returned in the same format as a (r.x = a.x − s · 2^(p−m))."""
    s = _floor_value(a)
    return s, FxR(x=a.x - (s << (a.p - a.m)), m=a.m, p=a.p)


def _round_value(a: FxR) -> int:
    """Round-half-to-even of a.value as a Python int.

    Equivalent to Python's ``round(a.value)`` but in pure integer
    arithmetic — the same banker's-shift used everywhere in fxtypes.
    """
    from fxtypes import _bankers_shift  # local to avoid module-load cycle
    scale = a.p - a.m
    if scale == 0:
        return a.x
    return _bankers_shift(a.x, scale)


def _round_and_frac(a: FxR) -> tuple[int, FxR]:
    """Split a = s + r with s = round(a.value) ∈ Z, r ∈ [−1/2, 1/2).
    r is returned in the same format as a (r.x = a.x − s · 2^(p−m))."""
    s = _round_value(a)
    return s, FxR(x=a.x - (s << (a.p - a.m)), m=a.m, p=a.p)


def _scale_to_p(a: FxR) -> int:
    """int(a.value · 2^p). At m=0 this is a.x; otherwise shift by ±m."""
    assert a.p == _P
    if a.m == 0:
        return a.x
    return a.x >> (-a.m) if a.m < 0 else a.x << a.m


# --------------------------------------------------------------------- #
# basesampler — identical to reference (pure 72-bit integer arithmetic).
# --------------------------------------------------------------------- #


@beartype
def basesampler(randombytes: Callable[[int], bytes] = urandom,
                mode: str | None = None) -> int:
    """Sample z0 ∈ {0, ..., 18} from a distribution close to a half-Gaussian
    with parameter sigma_max = 1.8205 (Falcon spec). ``mode`` selects the
    table: "floor" → RCDT_FLOOR (μ=0), "round" → RCDT_ROUND (μ=1/2 with
    halved D(0)). When None, falls back to module-level SAMPLERZ_MODE."""
    if mode is None:
        mode = SAMPLERZ_MODE
    rcdt = RCDT_FLOOR if mode == "floor" else RCDT_ROUND
    u = int.from_bytes(randombytes(RCDT_PREC >> 3), "little")
    z0 = 0
    for elt in rcdt:
        z0 += int(u < elt)
    return z0


# --------------------------------------------------------------------- #
# approxexp_fxp — polynomial approximation of 2^63 · ccs · exp(-x).
# --------------------------------------------------------------------- #


@beartype
def approxexp_fxp(x: FxR, ccs: FxR) -> int:
    """2^63 · ccs · exp(-x) as a non-negative 64-bit integer.

    Pre: x.p = ccs.p = 63, x.value ∈ [0, ln 2], ccs.value ∈ (0, 1]. Same
    13-coef polynomial and truncating >>63 shifts as the reference.
    """
    assert x.p == _P and ccs.p == _P
    z_x, z_ccs = _scale_to_p(x), _scale_to_p(ccs)
    y = _C[0]
    for elt in _C[1:]:
        y = elt - ((z_x * y) >> 63)
    return ((z_ccs << 1) * y) >> 63


# --------------------------------------------------------------------- #
# berexp_fxp — Bernoulli with probability ~ ccs · exp(-x).
# --------------------------------------------------------------------- #


@beartype
def berexp_fxp(x: FxR, ccs: FxR,
               randombytes: Callable[[int], bytes] = urandom) -> bool:
    """Return 1 with probability ~ ccs · exp(-x). x.value ≥ 0 expected.

    Splits x = s·ln2 + r with s ∈ N, r ∈ [0, ln2), then exp(-x) =
    2^{-s}·exp(-r). s is capped at 63 so (approxexp >> s) stays nonzero;
    clamped at 0 to absorb tiny negative x from fp drift upstream.
    """
    assert x.p == _P and ccs.p == _P
    # x.value ≥ 0 is a mathematical precondition (the integer-part extraction
    # below uses truncation toward zero, valid only for x ≥ 0). If this fires,
    # ULP drift on `(z-r)²·dss − sigma_correction·INV_2SIGMA2` at the call site
    # pushed x_fxr below 0, where floor-vs-trunc semantics would diverge.
    assert x.x >= 0, (
        f"berexp_fxp: x.value < 0 (x.x={x.x}, m={x.m}, p={x.p}, "
        f"≈ {x.to_float():.3e}); ULP drift on (z-r)²·dss − z0²·INV_2SIGMA2"
    )
    s_int = _floor_value(x * ILN2_FXR)

    # r = x − s·ln2. s·LN2 lives at m = s.bit_length(); align to x.m.
    if s_int == 0:
        r_fxr = x
    else:
        m_s = max(1, s_int.bit_length())
        s_ln2 = FxR.from_int(s_int, m=m_s, p=_P) * LN2_FXR
        m_common = max(x.m, s_ln2.m)
        r_fxr = _retag_m(x, m_common) - _retag_m(s_ln2, m_common)
    # r ∈ [0, ln2) ⊂ [0, 1) → m=0 tight (mandatory for approxexp_fxp).
    # Banker-shift drops ≤ ~6 LSBs (residual ~2^-57, ≫ samplerz sensitivity).
    r_fxr = _retag_m(r_fxr, 0)

    s_int = max(0, min(s_int, 63))
    z = (approxexp_fxp(r_fxr, ccs) - 1) >> s_int

    # Byte-wise Bernoulli comparison (identical to reference).
    for i in range(56, -8, -8):
        p = int.from_bytes(randombytes(1), "little")
        w = p - ((z >> i) & 0xFF)
        if w:
            break
    return w < 0


# --------------------------------------------------------------------- #
# samplerz_fxp — the main entry point.
# --------------------------------------------------------------------- #


@beartype
def samplerz_fxp(mu: FxR, dss: FxR, ccs: FxR,
                 randombytes: Callable[[int], bytes] = urandom,
                 mode: str | None = None) -> int:
    """Sample z from D_{Z, sigma, mu}. Takes the precomputed leaf constants
    dss = 1/(2σ_i²) ≤ 0.305 (m=0) and ccs = σ_min/σ_i ∈ (0, 1] (m=0), built
    once per leaf at normalize time (see `_normalize_leaf_poly`), so the
    sampler itself does no division, square, or σ-related arithmetic.

    Mirrors falcon_ref/samplerz.py in structure. ``mode`` selects between
    the Falcon spec ("floor") and LTYZ NewSamplerZ ("round"); falls back
    to module-level SAMPLERZ_MODE when None.

    Constant-time in the base sampler and polynomial eval; data-dependent
    in the Bernoulli rejection (standard samplerz semantics).
    """
    assert mu.p == _P and dss.p == _P and ccs.p == _P, "samplerz_fxp requires p=63"
    if mode is None:
        mode = SAMPLERZ_MODE

    # mu = s_int + r. floor: r ∈ [0, 1) at m=0. round: r ∈ [−1/2, 1/2) at m=0.
    if mode == "floor":
        s_int, r_raw = _floor_and_frac(mu)
    else:
        s_int, r_raw = _round_and_frac(mu)
    r_fxr = _retag_m(r_raw, 0)

    while True:
        z0 = basesampler(randombytes, mode=mode)             # ∈ [0, 18]
        b = int.from_bytes(randombytes(1), "little") & 1
        # Bimodal extension: floor (Falcon spec) maps z0=0,b∈{0,1} to z∈{0,1};
        # round (LTYZ Algorithm 4) maps z0=0,b∈{0,1} to z=0 (halved D(0)
        # in RCDT_ROUND prevents double-counting).
        if mode == "floor":
            z_int = b + (2 * b - 1) * z0                     # ∈ [-18, 18+1]
            sigma_correction_int = z0 * z0                   # ≤ 324
        else:
            z_int = (2 * b - 1) * z0                         # ∈ [-18, 18]
            # LTYZ Algorithm 4 line 5: y_+² − y_+. Cancels the proxy's
            # μ=1/2 offset exactly: −(z0 − 1/2)² + (z0² − z0) = −1/4 (constant).
            sigma_correction_int = z0 * z0 - z0              # ≤ 18·17 = 306

        # x = (z_int − r)²·dss − sigma_correction·INV_2SIGMA2. Bounds:
        #   |z_int − r| ≤ 18.5 → m=6; squared at m=12; ·dss keeps m=12
        #   sigma_correction ≤ 324 → m=9; ·INV_2SIGMA2 keeps m=9
        diff = FxR.from_int(z_int, m=6, p=_P) - _retag_m(r_fxr, 6)
        term1 = (diff * diff) * dss
        term2 = FxR.from_int(sigma_correction_int, m=9, p=_P) * INV_2SIGMA2_FXR
        m_common = max(term1.m, term2.m)
        x_fxr = _retag_m(term1, m_common) - _retag_m(term2, m_common)

        # x.value ≥ 0 by construction here: term1.x ≥ 0 (banker's-shifts on
        # positive integer products); term2.x ≥ 0 (sigma_correction_int ≥ 0).
        # For z0 = 0, term2 = 0 exactly. For z0 ≥ 1, math gives
        # x ≥ z0²·(dss − INV_2SIGMA2) ≥ 2^-6.4 ≫ ULP under Falcon's σ filter.
        # berexp_fxp asserts x.x ≥ 0 explicitly.
        if berexp_fxp(x_fxr, ccs, randombytes):
            return z_int + s_int
