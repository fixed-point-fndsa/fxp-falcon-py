"""
Paper-aligned NTRUGen rejection predicates for Falcon-512 (n=512, q=12289).

Branched into `ntrugen.py:ntru_gen` so every returned key satisfies the four
paper Checks. The fxp m-budgets in `fxp/sign_tweak.py` (M_D=18, M_L10_ROOT=5,
m_sign=21/18) assume these bounds; a key violating them would overflow the
fixed-point format (`|x| < 2^p`).

    γ_fg     = 255    ‖FFT(f,g)‖_∞        (Check 1) → m_B_fg = 8 (strict bound)
    γ_hybrid = 4      α_hybrid(f,g)       (Check 2) → m_D = 18 (Lemma 9)
    γ_FG     = 3500   ‖FFT(F,G)‖_∞        (Check 3) → m_sign = 21
    γ_root   = 24     ‖FFT(L_10_root)‖_∞  (Check 4) → M_L10_ROOT = 5
"""

from fft import fft, adj_fft, mul_fft, add_fft, div_fft

# Falcon-512 modulus.
Q = 12289

# Paper thresholds (see module docstring for derivation).
GAMMA_FG_512 = 255        # γ_fg, Check 1 (strict, to give m_B_fg = 8 headroom)
GAMMA_HYBRID = 4          # γ_hybrid, Check 2
GAMMA_FG_UPPER_512 = 3500  # γ_FG (Falcon-512), Check 3
GAMMA_ROOT = 24           # γ_root, Check 4

# ‖F, G‖_∞ limit: the int8 secret-key encoding bound (Pornin's `lim = 127`
# in solve_NTRU). Stock Falcon already rejects at encoding; explicit here so
# the fxp load M_B0_COEF_FG_UP = 8 is guaranteed. Empirical max ≈ 80–105.
FG_COEF_LIMIT = 127


def _norm_inf_fft(coeffs_fft) -> float:
    """‖p‖_∞ in FFT domain: max modulus over all embeddings."""
    return max(abs(z) for z in coeffs_fft)


# --------------------------------------------------------------------- #
# Check 1 helper: ‖FFT(f, g)‖_∞ < γ_fg.
# Branched into `ntru_gen` after `gs_norm` (cheapest pre-filter) and before
# Check 2 (the more expensive α_hybrid). γ_fg = 255 is the strict bound
# behind M_B_FG = 8 (the fft(f), fft(g) row tags in `_build_B0_fft_fxp_cache`).
# --------------------------------------------------------------------- #


def norm_fft_fg(f, g) -> float:
    """Return max(‖FFT(f)‖_∞, ‖FFT(g)‖_∞)."""
    return max(_norm_inf_fft(fft(f)), _norm_inf_fft(fft(g)))


# --------------------------------------------------------------------- #
# Check 2: α_hybrid(f, g) ≤ γ_hybrid (= 4).
# --------------------------------------------------------------------- #


def alpha_hybrid_squared(f, g, q: int = Q) -> float:
    """Return α_hybrid(f, g)² = max_ζ max(u(ζ)/q, q/u(ζ)) where
    u(ζ) = |f(ζ)|² + |g(ζ)|²."""
    f_fft, g_fft = fft(f), fft(g)
    mags = [abs(f_fft[i]) ** 2 + abs(g_fft[i]) ** 2 for i in range(len(f_fft))]
    return max(max(mags) / q, q / min(mags))


# --------------------------------------------------------------------- #
# α_GPV(f, g) ≤ 1.17 — already enforced via `gs_norm` in `ntrugen.py`,
# but kept here for the audit experiments.
# --------------------------------------------------------------------- #


def alpha_gpv_squared(f, g, q: int = Q) -> float:
    """Return α_GPV(f, g)² = max((1/n) Σ u(ζ)/q, (1/n) Σ q/u(ζ))."""
    f_fft, g_fft = fft(f), fft(g)
    mags = [abs(f_fft[i]) ** 2 + abs(g_fft[i]) ** 2 for i in range(len(f_fft))]
    n = len(mags)
    return max(sum(mags) / n / q, sum(q / m for m in mags) / n)


# --------------------------------------------------------------------- #
# Check 3: ‖FFT(F), FFT(G)‖_∞ ≤ γ_FG (= 3500 for Falcon-512).
# --------------------------------------------------------------------- #


def norm_fft_FG(F, G) -> float:
    """Return max(‖FFT(F)‖_∞, ‖FFT(G)‖_∞). F, G are int polynomials
    returned by ntru_solve; we cast to float for the FFT."""
    F_f = [float(c) for c in F]
    G_f = [float(c) for c in G]
    return max(_norm_inf_fft(fft(F_f)), _norm_inf_fft(fft(G_f)))


# --------------------------------------------------------------------- #
# Check 4: ‖FFT(k)‖_∞ ≤ γ_root (= 24), where k = L_10_root.
# --------------------------------------------------------------------- #


def norm_fft_k(f, g, F, G) -> float:
    """Return ‖FFT(k)‖_∞ for k = (F·adj(f) + G·adj(g)) / (f·adj(f) + g·adj(g)).
    k coincides with L_10 at the ffLDL root; γ_root is the bound that
    governs `M_L10_ROOT` in `fxp/sign_tweak.py`."""
    f_fft = fft([float(c) for c in f])
    g_fft = fft([float(c) for c in g])
    F_fft = fft([float(c) for c in F])
    G_fft = fft([float(c) for c in G])
    num = add_fft(mul_fft(F_fft, adj_fft(f_fft)), mul_fft(G_fft, adj_fft(g_fft)))
    den = add_fft(mul_fft(f_fft, adj_fft(f_fft)), mul_fft(g_fft, adj_fft(g_fft)))
    k_fft = div_fft(num, den)
    return _norm_inf_fft(k_fft)
