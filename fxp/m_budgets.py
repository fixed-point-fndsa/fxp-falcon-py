"""
Central table of fixed-point magnitude budgets — the `m` in FX_{m,p} — for the
whole Falcon-512 pipeline. Every `m` chosen in `fxp/` is defined here once, with
its derivation, so the paper's m-budget analysis has a single source of truth.

All values are for Falcon-512 (n = 512, q = 12289) and assume the NTRUGen
rejection filters in `falcon_ref/ntrugen_filters.py`:
    γ_fg = 255   (Check 1b, ‖fft(f,g)‖_∞)       γ_FG   = 3500 (Check 3, ‖fft(F,G)‖_∞)
    γ_hybrid = 4 (Check 2, α_hybrid)            γ_root = 24   (Check 4, ‖L_10_root‖_∞)
The precision is p = 63 throughout the production signing path.
"""

# --------------------------------------------------------------------- #
# Key expansion: ffLDL* on the NTRU Gram (see ffldl_fxp / sign_tweak).
# --------------------------------------------------------------------- #

# Root Gram, per-entry. G_00 = |fft(g)|²+|fft(f)|² < 2·γ_fg² = 130050 < 2^17.
M_G00 = 17
# G_10 (off-diagonal). G_01 = conj(F̂·f̂* + Ĝ·ĝ*) — the Check-4 numerator (see
# norm_fft_k) — so |G_01| = |L_10|·G_00 ≤ |F̂||f̂| + |Ĝ||ĝ| ≤ 2·γ_fg·γ_FG ≈
# 2^20.77 < 2^21. This is essentially tight: over 200 keys max|G_01| = 2^20.23,
# which already needs m=21 (m=20 would overflow). γ_root·G_00 (2^21.6) and
# γ_hybrid·16q (2^22.2) are looser; γ_GPV is an average (not an ∞-norm) bound,
# so it cannot tighten this. The RootGram has no g11 field (D_11 = q²/D_00).
M_G01 = 21

# L_10 at the ffLDL root: ‖L_10_root‖_∞ ≤ γ_root = 24 < 2^5 by construction
# (Check 4), so |L_10| < 32 = 2^5 suffices.
M_L10_ROOT = 5
# L_10 at non-root levels: |·| ≤ 1 (Lemma 9 + α_k interpolation), m=0 is tight.
M_L10_INNER = 0
# Gram diagonal D_ii during the ffLDL recursion: the γ_hybrid filter gives
# |D_ii| ∈ [q/α_h², α_h²·q] ⊂ [2^13, 2^14.5] (Lemma 9), plus drift bits.
# (Use 19 without the filter.) M_D is shared across all levels; it must hold the
# loosest (intermediate) diagonals, where 16q = 2^17.58 forces 18.
M_D = 18
# FINAL ffLDL leaves specifically (the per-coefficient GS norms fed to rsqrt) are
# tighter: D_ii ≤ 1.17²·q = 16822 < 2^15 by the stock NTRUGen gs_norm filter
# (also asserted in nr_fxp.rsqrt). Retagging leaves to M_D_LEAF before rsqrt
# recovers ~3 bits in the rsqrt intermediates (m_xy = M_D_LEAF − 12 = 3) and
# hence in σ_i. NB m=14 would overflow: 1.17²·q = 2^14.04 > 2^14.
M_D_LEAF = 15
# Leaf 1/σ_i after normalization (= √D_ii/σ): |1/σ_i| < 1 = 2^0 (σ_i > 1
# under the stock NTRUGen `gs_norm ≤ 1.17²·q` bound). samplerz multiplies by
# this instead of dividing by σ_i.
M_NORM_OUT = 0

# --------------------------------------------------------------------- #
# Target construction: t = (−c·F/q, c·f/q) (see target_construction).
# --------------------------------------------------------------------- #

# Hashed point: |point|_∞ < q < 2^14.
M_POINT_COEF = 14
# B0 = [[g,−f],[G,−F]] coefficients: ‖f,F‖_∞ ≤ γ_FG < 2^12, kept at 13 (the rest
# of the m chain was tuned for 13; tightening to 12 would require re-tuning).
M_B0_COEF = 13

# --------------------------------------------------------------------- #
# Signing: ffsampling targets + signature reconstruction.
# --------------------------------------------------------------------- #

# m of t/z throughout ffsampling.
#   tweak path: q·t_frac = (−c·F, c·f) mod^± q centered ⇒ ‖t̂_root‖_∞ ≤ n/2 = 2^8; Lemma 13
#     drift ≈ 2^17.61 dominates ⇒ 18.
M_SIGN_DEFAULT = 18
#   std path: point ∈ [0,q) ⇒ ‖t̂_root‖_∞ < n·γ_FG ≈ 2^20.77 + drift ≈ 2^20.93 ⇒ 21.
M_SIGN_STD = 21

# s = (t − z)·B reconstruction (see _reconstruct_s_fxp). Retag B0 rows tightly:
M_B_FG = 8       # a, b = fft(g), fft(−f) — γ_fg = 255 < 2^8 (Check 1b)
M_B_FG_UP = 12   # c, d = fft(G), fft(−F) — γ_FG = 3500 < 2^12 (Check 3)
# Common format for the two products before their sum: must hold each product
# individually (max |diff·B| ≈ 2^17.76), not the smaller post-cancellation sum.
M_S_INTER = 19
