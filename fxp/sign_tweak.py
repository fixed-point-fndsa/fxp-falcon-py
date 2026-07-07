"""
Modular Sign with optional tweak (Section 5.1 of the paper) and optional
fxp ffsampling path.

Section 5.1 tweak: replace the ffsampling input `t = (−c·F/q, c·f/q)`
(large, ~√n·γ_FG in inf-norm) by its fractional part `t_frac` with
entries in [−1/2, 1/2] (after FFT, ~√(n/12)·√(2 ln n)). Lemma 14 gives
distributional equivalence. Two target builders, via `use_tweak`:

    USE_TWEAK_STD (0) : standard target, no tweak.
    USE_TWEAK_NTT (1) : Section-5.1 NTT-exact tweak.

Public API:

    sample_preimage(sk, point, use_tweak=USE_TWEAK_STD,
                    use_fxp_ffsampling=False, randombytes=os.urandom)
        -> (s_int_poly_pair, extras_dict)
    sign(sk, message, use_tweak=USE_TWEAK_STD, ...) -> bytes

`m_sign` is NOT a public parameter: its optimal value is fully determined
by `use_tweak` (21 for STD, 18 for NTT) and chosen internally. Lower-level
helpers (`ffsampling_fxp`, `_build_t_*_fxp`) still take m_sign explicitly
for benchmark callers. The final s = (t − z)·B0 is reconstructed in pure
integer arithmetic (`_reconstruct_s_int`) — fxp ends at `_ifft_round(z)`.

`use_tweak` also accepts `bool` (False → STD, True → NTT), since `bool`
subclasses `int`. The `falcon.py` `SecretKey` shim provides the stateful
attribute API this module is built against.
"""

import os
from typing import Callable

from beartype import beartype

import _path_setup  # noqa: F401  (prepends falcon_ref/ + fxp/ to sys.path)

from falcon import SecretKey, HEAD_LEN, SALT_LEN  # noqa: E402, F401 (SecretKey re-exported)
from fft import ifft, sub_fft, add_fft, mul_fft  # noqa: E402
from ffsampling import ffsampling_fft  # noqa: E402
from common import q as FALCON_Q  # noqa: E402
from encoding import compress  # noqa: E402
from ntt import mul_zq  # noqa: E402
from ntrugen import karamul  # noqa: E402  (exact Z[x]/(x^n+1) product, tweak t_int)

# fxp side.
from fxtypes import _bankers_shift, RootGram  # noqa: E402
from fft_fxp import (  # noqa: E402
    mul_fft_to, add_fft_fxp, adj_fft_fxp, ifft_fxp,
)
from ffldl_fxp import keygen_fxp  # noqa: E402
from ffsampling_fxp import ffsampling_fxp  # noqa: E402
from fxp_constants_p63 import SIGMIN_FXR_BY_N as _PER_DEGREE_SIGMIN_FXR  # noqa: E402
from fxp_constants_p63 import INV_SIGMA_FXR_BY_N as _PER_DEGREE_INV_SIGMA_FXR  # noqa: E402
from target_construction import (  # noqa: E402
    _build_t_standard, _build_t_tweaked,
    _build_t_standard_fxp, _build_t_tweaked_fxp,
    _build_B0_fft_fxp_cache, _center_signed,
    USE_TWEAK_STD, USE_TWEAK_NTT,
)


# Dispatch table: float-pipeline target builders, indexed by use_tweak.
_FLOAT_BUILDERS = {
    USE_TWEAK_STD: _build_t_standard,
    USE_TWEAK_NTT: _build_t_tweaked,
}


# All fixed-point m budgets live in `m_budgets.py` (single source of truth
# with their derivations). Imported here for the signing pipeline.
from m_budgets import (  # noqa: E402
    M_D, M_G01,
    M_SIGN_DEFAULT, M_SIGN_STD, M_B_FG, M_B_FG_UP,
)


def _gram_fft_fxp(B0_fft):
    """Gram G = B0·adj(B0^T) in FFT domain (pointwise, pure fxp).

    `B0_fft` must come from `_build_B0_fft_fxp_cache` (rows at their tight γ
    bounds M_B_FG / M_B_FG_UP): multiplying at the loose post-FFT m=21 would land
    each product at m=42 and truncate ~12 bits, so tight inputs are load-bearing
    here. g00 is emitted straight at M_D (the shared ffLDL recursion budget:
    G_00 < 2·γ_fg² < 2^17 fits with one bit of slack, and the root D00 = G00
    then needs no widen); g10 at its per-entry sum bound M_G01.

    Returns a `RootGram`: g00 = |a|²+|b|² (real), g10 = adj(G_01); G_11 is not
    computed (see `RootGram`).
    """
    [[a, b], [c, d]] = B0_fft
    assert a[0].m == M_B_FG and c[0].m == M_B_FG_UP, \
        f"_gram_fft_fxp: untight B0 a@{a[0].m} c@{c[0].m} (want {M_B_FG},{M_B_FG_UP})"
    a_adj, b_adj = adj_fft_fxp(a), adj_fft_fxp(b)
    c_adj, d_adj = adj_fft_fxp(c), adj_fft_fxp(d)
    # G_00 = |a|²+|b|² < 2^17 (real → keep .re); G_01 = a·c*+b·d* < 2^21.
    # mul_fft_to emits each product at the target bound (fits the add, single round).
    G00 = [z.re for z in add_fft_fxp(mul_fft_to(a, a_adj, M_D),
                                     mul_fft_to(b, b_adj, M_D))]
    G01 = add_fft_fxp(mul_fft_to(a, c_adj, M_G01),
                      mul_fft_to(b, d_adj, M_G01))
    return RootGram(g00=G00, g10=adj_fft_fxp(G01))


def _per_degree_fxp(sk, table, name):
    """Lookup an FxR constant for sk.n in a hardcoded per-degree table."""
    try:
        return table[sk.n]
    except KeyError:
        raise NotImplementedError(
            f"hardcoded {name} only available for n ∈ {sorted(table)}"
        )


def _inv_sigma_fxp(sk):
    return _per_degree_fxp(sk, _PER_DEGREE_INV_SIGMA_FXR, "inv_sigma")


def _sigmin_fxp(sk):
    return _per_degree_fxp(sk, _PER_DEGREE_SIGMIN_FXR, "sigmin")


def _ifft_round(poly):
    """fxp ifft then banker's-shift to nearest integer (pure int, no float).
    Used on z (integer-valued by construction): the fxp FFT/merge round-off
    ~2^{m_sign−p+ε} ≈ 2^-40 is far below the 0.5 recovery margin."""
    return [_bankers_shift(v.x, v.p - v.m) if v.p > v.m else v.x
            for v in ifft_fxp(poly)]


def _reconstruct_s_int(sk, point, z_int, qt=None):
    """Reconstruct s = (t − z)·B0 in PURE INTEGER arithmetic — no fxp, no
    budget. Since t·B0 = (c, 0) exactly and z is integer,

        s = (c, 0) − z'·B0 = (c − z'0·g − z'1·G,  z'0·f + z'1·F)  mod± q,

    with z' = z (standard target) or z' = z + t_int (tweak: z_std = z_tw +
    t_int, Lemma 14). Everything is mod-q NTT (`mul_zq`) + centered lift: the
    lift is exact iff ‖s‖_∞ < q/2, guaranteed by |s_i| ≤ τσ ≈ 2323 < q/2 =
    6144 (s ~ D_{coset,σ}, lem:upper-dot-product; the 2^-λ failure would be a
    ±q shift that explodes the ‖s‖² ≤ β² check → plain retry, never silent).

    For the tweak, t_int = (q·t_std − qt)/q needs c·F, c·f OVER Z (mod q is
    not enough for the exact /q): `karamul` here; two-prime NTT + CRT in C.
    """
    q = FALCON_Q
    z0, z1 = z_int
    if qt is not None:  # tweak: z' = z + t_int, t_int = (±c·{F,f} − qt)/q
        cF = karamul(list(point), sk.F)
        cf = karamul(list(point), sk.f)
        t_int0 = [_exact_div(-a - b, q) for a, b in zip(cF, qt[0])]
        t_int1 = [_exact_div(a - b, q) for a, b in zip(cf, qt[1])]
        z0 = [a + b for a, b in zip(z0, t_int0)]
        z1 = [a + b for a, b in zip(z1, t_int1)]
    z0_zq = [c % q for c in z0]
    z1_zq = [c % q for c in z1]
    u0 = mul_zq(z0_zq, [c % q for c in sk.g])
    u1 = mul_zq(z1_zq, [c % q for c in sk.G])
    v0 = mul_zq(z0_zq, [c % q for c in sk.f])
    v1 = mul_zq(z1_zq, [c % q for c in sk.F])
    s0 = _center_signed([(c - a - b) % q for c, a, b in zip(point, u0, u1)], q)
    s1 = _center_signed([(a + b) % q for a, b in zip(v0, v1)], q)
    return s0, s1


def _exact_div(a: int, d: int) -> int:
    """a // d, asserting the division is exact."""
    quo, rem = divmod(a, d)
    assert rem == 0, f"_exact_div: {a} not divisible by {d}"
    return quo


def _build_fxp_tree_cache(sk):
    """Build (or return cached) fxp ffLDL tree for this sk. Cached on sk
    via the `_fxp_tree` attribute to avoid rebuilding per signature.

    The tree is independent of m_sign (M_D=18 is fixed by Lemma 9; m_sign
    governs t/z in ffsampling, a separate quantity). M_D = 18 assumes the
    γ_hybrid ≤ 4 filter, which the fxp pipeline requires upstream.
    """
    if sk._fxp_tree is not None:
        return sk._fxp_tree

    # Gram in FFT domain (B0·adj(B0^T)): `_gram_fft_fxp` emits g00 at M_D and
    # g10 at M_G01 by construction; `ldl_fft_fxp_ntru_root` asserts the g00
    # contract downstream.
    gram = _gram_fft_fxp(_build_B0_fft_fxp_cache(sk))

    tree = keygen_fxp(gram, q=FALCON_Q, inv_sigma=_inv_sigma_fxp(sk),
                      sigmin=_sigmin_fxp(sk))
    sk._fxp_tree = tree
    return tree


@beartype
def sample_preimage(sk: SecretKey, point: list[int], use_tweak: int = USE_TWEAK_STD,
                    randombytes: Callable[[int], bytes] | None = None,
                    use_fxp_ffsampling: bool = False) -> tuple[list[list[int]], dict]:
    """Unified sample_preimage with optional tweak and fxp flags.

    `use_tweak` selects the target builder (STD = standard target, NTT =
    Section-5.1 NTT-exact tweak; see module docstring). The two are
    distributionally equivalent (Lemma 14) but diverge sporadically by
    float ULP at the LTYZ-fragile samplerz positions.

    use_fxp_ffsampling=True runs ffsampling_fxp + samplerz_fxp instead of
    the float reference (bit-identical under a shared seed).

    Returns (s, extras) with s = [s0, s1] integer polynomials and
    extras = {t_root_fft, z_coef, q_t_frac_coef (or None)}.
    """
    if use_tweak not in (USE_TWEAK_STD, USE_TWEAK_NTT):
        raise ValueError(
            f"use_tweak must be USE_TWEAK_STD ({USE_TWEAK_STD}) or "
            f"USE_TWEAK_NTT ({USE_TWEAK_NTT}), got {use_tweak}"
        )
    if randombytes is None:
        randombytes = os.urandom

    if use_fxp_ffsampling:
        # Self-consistent fxp pipeline: t built directly in fxp (no float64
        # detour), so std vs tw KAT is bit-identical (vs ~1/1000 in float).
        # The whole fxp pipeline is currently tuned for Falcon-512 (M_D=18,
        # M_L10_ROOT=5, M_B_FG/M_B_FG_UP row tags); n=1024 needs a
        # numerical re-validation of every per-level precision bound.
        assert sk.n == 512, f"fxp ffsampling: Falcon-512 only (sk.n={sk.n})"
        # m_sign is the FxC `m` for t throughout ffsampling — optimal value
        # is fully determined by use_tweak (derivations in m_budgets.py:
        # M_SIGN_STD = 21, M_SIGN_DEFAULT = 18).
        m_sign = M_SIGN_STD if use_tweak == USE_TWEAK_STD else M_SIGN_DEFAULT
        builder = _build_t_tweaked_fxp if use_tweak != USE_TWEAK_STD else _build_t_standard_fxp
        t_fxp, q_t_frac_coef = builder(sk, point, m_sign)
        tree_fxp = _build_fxp_tree_cache(sk)
        z_fxc = ffsampling_fxp(t_fxp, tree_fxp, randombytes, m_sign=m_sign)
        # z back to integers (the last fxp step), then s in pure integer
        # arithmetic — s = (c, 0) − z'·B0 mod± q (see _reconstruct_s_int).
        z0_coef, z1_coef = _ifft_round(z_fxc[0]), _ifft_round(z_fxc[1])
        s0, s1 = _reconstruct_s_int(sk, point, (z0_coef, z1_coef),
                                    qt=q_t_frac_coef)
        t_fft = [[z.to_complex() for z in t_fxp[0]],
                 [z.to_complex() for z in t_fxp[1]]]
    else:
        # Reference float pipeline: s = (t − z)·B via FFT, then ifft + round.
        builder = _FLOAT_BUILDERS[use_tweak]
        t_fft, q_t_frac_coef = builder(sk, point)
        z_fft = ffsampling_fft(t_fft, sk.T_fft, sk.sigmin, randombytes)
        [[a, b], [c, d]] = sk.B0_fft
        diff0 = sub_fft(t_fft[0], z_fft[0])
        diff1 = sub_fft(t_fft[1], z_fft[1])
        s0 = [int(round(z.real)) for z in ifft(add_fft(mul_fft(diff0, a), mul_fft(diff1, c)))]
        s1 = [int(round(z.real)) for z in ifft(add_fft(mul_fft(diff0, b), mul_fft(diff1, d)))]
        z0_coef = [int(round(z.real)) for z in ifft(z_fft[0])]
        z1_coef = [int(round(z.real)) for z in ifft(z_fft[1])]

    extras = {"t_root_fft": t_fft, "z_coef": [z0_coef, z1_coef],
              "q_t_frac_coef": q_t_frac_coef}
    return [s0, s1], extras


@beartype
def sign(sk: SecretKey, message: bytes, use_tweak: int = USE_TWEAK_STD,
         randombytes: Callable[[int], bytes] | None = None,
         max_tries: int = 128, return_extras: bool = False,
         use_fxp_ffsampling: bool = False) -> bytes | tuple[bytes, dict]:
    """Full Sign with optional tweak / fxp flags (see sample_preimage).
    Mirrors SecretKey.sign; retries on norm/encoding rejection."""
    if randombytes is None:
        randombytes = os.urandom
    # n is a power of 2: log2(n) = n.bit_length() − 1 (pure int).
    header = (0x30 + (sk.n.bit_length() - 1)).to_bytes(1, "little")

    # Salt drawn ONCE, before the retry loop (alg:sign / reference falcon.py):
    # a norm/encoding rejection re-runs only ffsampling, same salt and target.
    salt = randombytes(SALT_LEN)
    hashed = sk.hash_to_point(message, salt)
    for _ in range(max_tries):
        s, extras = sample_preimage(sk, hashed, use_tweak=use_tweak,
                                    randombytes=randombytes,
                                    use_fxp_ffsampling=use_fxp_ffsampling)
        norm_sig = sum(c * c for c in s[0]) + sum(c * c for c in s[1])
        if norm_sig > sk.signature_bound:
            continue
        enc_s = compress(s[1], sk.sig_bytelen - HEAD_LEN - SALT_LEN)
        if enc_s is False:
            continue
        signature = header + salt + enc_s
        if return_extras:
            extras.update({"s": s, "norm_sig": norm_sig, "salt": salt,
                           "hashed": hashed})
            return signature, extras
        return signature
    raise RuntimeError("Sign failed after max_tries attempts")
