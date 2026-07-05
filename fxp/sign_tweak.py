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
helpers (`ffsampling_fxp`, `_build_t_*_fxp`, `_reconstruct_s_fxp`) still
take m_sign explicitly for benchmark callers.

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

# fxp side.
from fxtypes import _bankers_shift, RootGram  # noqa: E402
from fft_fxp import (  # noqa: E402
    sub_fft_fxp, mul_fft_to, add_fft_fxp, adj_fft_fxp, ifft_fxp,
)
from ffldl_fxp import keygen_fxp  # noqa: E402
from ffsampling_fxp import ffsampling_fxp  # noqa: E402
from fxp_constants_p63 import SIGMIN_FXR_BY_N as _PER_DEGREE_SIGMIN_FXR  # noqa: E402
from fxp_constants_p63 import INV_SIGMA_FXR_BY_N as _PER_DEGREE_INV_SIGMA_FXR  # noqa: E402
from target_construction import (  # noqa: E402
    _build_t_standard, _build_t_tweaked,
    _build_t_standard_fxp, _build_t_tweaked_fxp,
    _build_B0_fft_fxp_cache,
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
    M_SIGN_DEFAULT, M_SIGN_STD, M_B_FG, M_B_FG_UP, M_S_INTER,
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


def _reconstruct_s_fxp(sk, t_fxp, z_fxc, m_sign):
    """Compute s = (t − z)·B in FxC, then ifft to integer polynomials.

    B0 rows arrive at their tight γ bounds from `_build_B0_fft_fxp_cache`.
    M_S_INTER (= 19) is the common format for the `add` of the two products: it
    holds each product (max |diff·B| ≈ 2^17.76 over the small LTYZ residual t−z),
    not the smaller post-cancellation sum. Sized for Falcon-512 (asserted below).
    """
    assert sk.n == 512, f"_reconstruct_s_fxp: Falcon-512 only (sk.n={sk.n})"
    [a_fxc, b_fxc], [c_fxc, d_fxc] = _build_B0_fft_fxp_cache(sk)  # tight γ bounds (cache)

    diff0 = sub_fft_fxp(t_fxp[0], z_fxc[0])
    diff1 = sub_fft_fxp(t_fxp[1], z_fxc[1])

    # s_j = diff0·B[0][j] + diff1·B[1][j]. `mul_fft_to` emits each product
    # straight at the common M_S_INTER (single round), so the two summands
    # share m for the add — no separate product→M_S_INTER retag.
    s0_fxc = add_fft_fxp(mul_fft_to(diff0, a_fxc, M_S_INTER),
                         mul_fft_to(diff1, c_fxc, M_S_INTER))
    s1_fxc = add_fft_fxp(mul_fft_to(diff0, b_fxc, M_S_INTER),
                         mul_fft_to(diff1, d_fxc, M_S_INTER))

    # ifft then banker's-shift to nearest integer (pure int, no float).
    def _ifft_round(poly):
        return [_bankers_shift(v.x, v.p - v.m) if v.p > v.m else v.x
                for v in ifft_fxp(poly)]
    return (_ifft_round(s0_fxc), _ifft_round(s1_fxc),
            _ifft_round(z_fxc[0]), _ifft_round(z_fxc[1]))


def _build_fxp_tree_cache(sk):
    """Build (or return cached) fxp ffLDL tree for this sk. Cached on sk
    via the `_fxp_tree` attribute to avoid rebuilding per signature.

    The tree is independent of m_sign (M_D=18 is fixed by Lemma 9; m_sign
    governs t/z in ffsampling, a separate quantity). M_D = 18 assumes the
    γ_hybrid ≤ 4 filter, which the fxp pipeline requires upstream.
    """
    if sk._fxp_tree is not None:
        return sk._fxp_tree

    # Gram in FFT domain (B0·adj(B0^T)): `_gram_fft_fxp` already emits g00 at
    # M_D (shared recursion budget, root D00 = G00 without a widen) and g10 at
    # M_G01, so we assert the contract instead of retagging.
    gram = _gram_fft_fxp(_build_B0_fft_fxp_cache(sk))
    assert gram.g00[0].m == M_D and gram.g10[0].m == M_G01

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
        # The whole fxp pipeline is currently tuned for Falcon-512 (m_D=18,
        # M_L10_ROOT=5, m_B_{fg,FG} in _reconstruct_s_fxp); n=1024 needs a
        # numerical re-validation of every per-level precision bound.
        assert sk.n == 512, f"fxp ffsampling: Falcon-512 only (sk.n={sk.n})"
        # m_sign is the FxC `m` for t throughout ffsampling — optimal value
        # is fully determined by use_tweak:
        #   STD : `point` ∈ [0, q-1] ⇒ ‖t̂_root‖_∞ < n·γ_FG = 512·3500 ≈ 2^20.77
        #         (γ_FG tightened from 4096; see ntrugen_filters.py). Lemma 13
        #         drift adds 2·A_child·√n·(γ_root+√2+1) ≈ 2^17.6 (Cholesky
        #         A_child = 2τσ·α_h/√q ≈ 168). Sum ≈ 2^20.93 < 2^21 ⇒ 21.
        #   tweak: q·t_frac = (−c·F, c·f) mod^± q centered ⇒ ‖t̂_root‖_∞ < n/2 = 2^8.
        #         drift dominates ⇒ 18 (cf. ffsampling_fxp.M_SIGN_DEFAULT).
        m_sign = M_SIGN_STD if use_tweak == USE_TWEAK_STD else M_SIGN_DEFAULT
        builder = _build_t_tweaked_fxp if use_tweak != USE_TWEAK_STD else _build_t_standard_fxp
        t_fxp, q_t_frac_coef = builder(sk, point, m_sign)
        tree_fxp = _build_fxp_tree_cache(sk)
        z_fxc = ffsampling_fxp(t_fxp, tree_fxp, randombytes, m_sign=m_sign)
        s0, s1, z0_coef, z1_coef = _reconstruct_s_fxp(sk, t_fxp, z_fxc, m_sign)
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
