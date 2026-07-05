"""
Fixed-point port of Falcon's samplerz: takes `mu` as FxR and does the float
arithmetic on mu/r/x in FxR. The already-integer parts (basesampler RCDT,
approxexp's 13-coef poly, berexp's byte-wise compare) are unchanged. With shared
randomness, outputs are byte-identical to the float64 reference.
"""

from os import urandom
from typing import Callable, Literal

from beartype import beartype

from fxtypes import FxR, retag_fxr, _bankers_shift
from m_budgets import M_SZ_DIFF


# Constants (Falcon spec / reference samplerz).
_P = 63

# "floor" reproduces the Falcon spec sampler bit-for-bit; "round" implements LTYZ
# NewSamplerZ (paper 2024-1709), moving the round-off-sensitive locus to
# half-integers (absent in Falcon under the ‖(g, −f)‖² odd keygen restriction).
SAMPLERZ_MODE = "floor"

# Reverse-CDF tables (72-bit). floor: half-Gaussian D_{Z+, σmax, 0}; round: LTYZ
# NewBaseSampler D_{Z+, σmax, 1/2} (D(0) halved). Both from scripts/generate_rcdt.py.
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

# Hardcoded FxR constants (x = round(value·2^{p-m})), so this module loads with no
# float op. Refresh via scripts/generate_constants_fxp.sage.
LN2_FXR = FxR(x=6393154322601832448, m=0, p=_P)             # 0.69314718056
ILN2_FXR = FxR(x=6653256548926941184, m=1, p=_P)            # 1.44269504089
INV_2SIGMA2_FXR = FxR(x=1391484473135841792, m=0, p=_P)     # 1/(2·1.8205²) = 0.150865...


# Local FxR helpers. Python's >> on negatives floors, matching
# floor(a.value) = a.x >> (p - m); p - m >= 0 by the m <= p invariant.

def _floor_value(a: FxR) -> int:
    """floor(a.value) as a Python int."""
    scale = a.p - a.m
    return a.x if scale == 0 else a.x >> scale


def _round_value(a: FxR) -> int:
    """Round-half-to-even of a.value as a Python int.

    Equivalent to Python's ``round(a.value)`` but in pure integer
    arithmetic — the same banker's-shift used everywhere in fxtypes.
    """
    scale = a.p - a.m
    if scale == 0:
        return a.x
    return _bankers_shift(a.x, scale)


def _split(a: FxR, s: int) -> tuple[int, FxR]:
    """Return (s, r) with r = a − s in a's format (r.x = a.x − s · 2^(p−m))."""
    return s, FxR(x=a.x - (s << (a.p - a.m)), m=a.m, p=a.p)


def _floor_and_frac(a: FxR) -> tuple[int, FxR]:
    """a = s + r, s = floor(a) ∈ Z, r ∈ [0, 1)."""
    return _split(a, _floor_value(a))


def _round_and_frac(a: FxR) -> tuple[int, FxR]:
    """a = s + r, s = round(a) ∈ Z, r ∈ [−1/2, 1/2)."""
    return _split(a, _round_value(a))


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
                mode: Literal["floor", "round"] | None = None) -> int:
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
    2^{-s}·exp(-r). s is capped at 63 so (approxexp >> s) stays nonzero
    (s ≥ 0 is guaranteed by the x ≥ 0 precondition below).
    """
    assert x.p == _P and ccs.p == _P
    # x ≥ 0 is a precondition: the integer-part extraction below truncates toward
    # zero. A negative x means upstream ULP drift pushed x_fxr below 0.
    assert x.x >= 0, f"berexp_fxp: x<0 (x.x={x.x}, m={x.m} ≈ {x.to_float():.3e}); upstream ULP drift"
    s_int = _floor_value(x * ILN2_FXR)

    # r = x − s·ln2 (exact when s_int=0: from_int(0)·ln2 = 0, x − 0 = x).
    s_ln2 = FxR.from_int(s_int, m=x.m, p=_P) * LN2_FXR
    r_fxr = x - s_ln2
    # r ∈ [0, ln2) ⊂ [0, 1) → m=0 tight (mandatory for approxexp_fxp).
    # Tighten 10→0: exact left-shift (samplerz has NO rounding retags).
    r_fxr = retag_fxr(r_fxr, 0)

    s_int = min(s_int, 63)
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
                 mode: Literal["floor", "round"] | None = None) -> int:
    """Sample z from D_{Z, σ, mu}. Takes the precomputed leaf constants dss =
    1/(2σ_i²) and ccs = σ_min/σ_i (both m=0, built once per leaf in
    `_normalize_leaf_poly`), so the sampler does no division, square, or σ
    arithmetic. ``mode`` ∈ {"floor" (Falcon spec), "round" (LTYZ)}; None → default.

    Constant-time (control flow) in the base sampler and polynomial eval;
    data-dependent in the Bernoulli rejection (standard samplerz).
    """
    assert mu.p == _P and dss.p == _P and ccs.p == _P, "samplerz_fxp requires p=63"
    # Interface contract (see M_SZ_DIFF): the r retag below must be an exact
    # left-shift, which requires mu to arrive at m ≥ M_SZ_DIFF.
    assert mu.m >= M_SZ_DIFF, f"samplerz_fxp: mu.m={mu.m} < M_SZ_DIFF={M_SZ_DIFF}"
    if mode is None:
        mode = SAMPLERZ_MODE

    # mu = s_int + r, with |r| < 1. Retag r once to M_SZ_DIFF (the format of
    # `diff` below) — exact, so the loop needs no per-iteration retag.
    if mode == "floor":
        s_int, r_raw = _floor_and_frac(mu)
    else:
        s_int, r_raw = _round_and_frac(mu)
    r_fxr = retag_fxr(r_raw, M_SZ_DIFF)

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
        #   |z_int − r| ≤ 19.5 < 2^M_SZ_DIFF; squared → 2·M_SZ_DIFF; ·dss keeps it
        #   sigma_correction ≤ 324 < 2^10; built at 2·M_SZ_DIFF to match term1
        diff = FxR.from_int(z_int, m=M_SZ_DIFF, p=_P) - r_fxr
        term1 = (diff * diff) * dss                       # m = 2·M_SZ_DIFF
        term2 = (FxR.from_int(sigma_correction_int, m=2 * M_SZ_DIFF, p=_P)
                 * INV_2SIGMA2_FXR)                       # aligned to term1
        x_fxr = term1 - term2

        # x ≥ 0 by construction: term1, term2 ≥ 0, and for z0 ≥ 1 math gives
        # x ≥ z0²·(dss − INV_2SIGMA2) ≥ 2^-6.4 ≫ ULP under Falcon's σ filter.
        if berexp_fxp(x_fxr, ccs, randombytes):
            return z_int + s_int
