"""
This file implements the section 3.8.2 of Falcon's documentation.
"""
from fft import fft, ifft, add_fft, mul_fft, adj_fft, div_fft
from fft import add, mul, div, adj
from ntt import ntt
from common import sqnorm
from samplerz import samplerz
from ntrugen_filters import (
    GAMMA_FG_512, GAMMA_HYBRID, GAMMA_FG_UPPER_512, GAMMA_ROOT,
    alpha_hybrid_squared, norm_fft_fg, norm_fft_FG, norm_fft_k,
)


q = 12 * 1024 + 1


# LTYZ NewSamplerZ countermeasure (paper 2024-1709, Section 7.1):
# requires ‖(g, −f)‖² to be ODD so that the auxiliary quantities
# m_1, m_2, m_3 stay odd under Heuristic 1 — guarantees no half-integer
# samplerz centers can appear in the recursion levels visible to
# float64. Set FORCE_ODD_GS_NORM = True alongside SAMPLERZ_MODE = "round"
# to enable the full LTYZ countermeasure.
#
# Note: this is independent of the gs_norm magnitude filter (≤ 1.17²·q),
# which acts on a different bound and is always enforced.
FORCE_ODD_GS_NORM = False


def karatsuba(a, b, n):
    """
    Karatsuba multiplication between polynomials.
    The coefficients may be either integer or real.
    """
    if n == 1:
        return [a[0] * b[0], 0]
    else:
        n2 = n // 2
        a0 = a[:n2]
        a1 = a[n2:]
        b0 = b[:n2]
        b1 = b[n2:]
        ax = [a0[i] + a1[i] for i in range(n2)]
        bx = [b0[i] + b1[i] for i in range(n2)]
        a0b0 = karatsuba(a0, b0, n2)
        a1b1 = karatsuba(a1, b1, n2)
        axbx = karatsuba(ax, bx, n2)
        for i in range(n):
            axbx[i] -= (a0b0[i] + a1b1[i])
        ab = [0] * (2 * n)
        for i in range(n):
            ab[i] += a0b0[i]
            ab[i + n] += a1b1[i]
            ab[i + n2] += axbx[i]
        return ab


def karamul(a, b):
    """
    Karatsuba multiplication, followed by reduction mod (x ** n + 1).
    """
    n = len(a)
    ab = karatsuba(a, b, n)
    abr = [ab[i] - ab[i + n] for i in range(n)]
    return abr


def galois_conjugate(a):
    """
    Galois conjugate of an element a in Q[x] / (x ** n + 1).
    Here, the Galois conjugate of a(x) is simply a(-x).
    """
    n = len(a)
    return [((-1) ** i) * a[i] for i in range(n)]


def field_norm(a):
    """
    Project an element a of Q[x] / (x ** n + 1) onto Q[x] / (x ** (n // 2) + 1).
    Only works if n is a power-of-two.
    """
    n2 = len(a) // 2
    ae = [a[2 * i] for i in range(n2)]
    ao = [a[2 * i + 1] for i in range(n2)]
    ae_squared = karamul(ae, ae)
    ao_squared = karamul(ao, ao)
    res = ae_squared[:]
    for i in range(n2 - 1):
        res[i + 1] -= ao_squared[i]
    res[0] += ao_squared[n2 - 1]
    return res


def lift(a):
    """
    Lift an element a of Q[x] / (x ** (n // 2) + 1) up to Q[x] / (x ** n + 1).
    The lift of a(x) is simply a(x ** 2) seen as an element of Q[x] / (x ** n + 1).
    """
    n = len(a)
    res = [0] * (2 * n)
    for i in range(n):
        res[2 * i] = a[i]
    return res


def bitsize(a):
    """
    Compute the bitsize of an element of Z (not counting the sign).
    The bitsize is rounded to the next multiple of 8.
    This makes the function slightly imprecise, but faster to compute.
    """
    val = abs(a)
    res = 0
    while val:
        res += 8
        val >>= 8
    return res


def reduce(f, g, F, G):
    """
    Reduce (F, G) relatively to (f, g).

    This is done via Babai's reduction.
    (F, G) <-- (F, G) - k * (f, g), where k = round((F f* + G g*) / (f f* + g g*)).
    Corresponds to algorithm 7 (Reduce) of Falcon's documentation.
    """
    n = len(f)
    size = max(53, bitsize(min(f)), bitsize(max(f)), bitsize(min(g)), bitsize(max(g)))

    f_adjust = [elt >> (size - 53) for elt in f]
    g_adjust = [elt >> (size - 53) for elt in g]
    fa_fft = fft(f_adjust)
    ga_fft = fft(g_adjust)

    while(1):
        # Because we work in finite precision to reduce very large polynomials,
        # we may need to perform the reduction several times.
        Size = max(53, bitsize(min(F)), bitsize(max(F)), bitsize(min(G)), bitsize(max(G)))
        if Size < size:
            break

        F_adjust = [elt >> (Size - 53) for elt in F]
        G_adjust = [elt >> (Size - 53) for elt in G]
        Fa_fft = fft(F_adjust)
        Ga_fft = fft(G_adjust)

        den_fft = add_fft(mul_fft(fa_fft, adj_fft(fa_fft)), mul_fft(ga_fft, adj_fft(ga_fft)))
        num_fft = add_fft(mul_fft(Fa_fft, adj_fft(fa_fft)), mul_fft(Ga_fft, adj_fft(ga_fft)))
        k_fft = div_fft(num_fft, den_fft)
        k = ifft(k_fft)
        k = [int(round(elt)) for elt in k]
        if all(elt == 0 for elt in k):
            break
        # The two next lines are the costliest operations in ntru_gen
        # (more than 75% of the total cost in dimension n = 1024).
        # There are at least two ways to make them faster:
        # - replace Karatsuba with Toom-Cook
        # - mutualized Karatsuba, see ia.cr/2020/268
        # For simplicity reasons, we didn't implement these optimisations here.
        fk = karamul(f, k)
        gk = karamul(g, k)
        for i in range(n):
            F[i] -= fk[i] << (Size - size)
            G[i] -= gk[i] << (Size - size)
    return F, G


def xgcd(b, n):
    """
    Compute the extended GCD of two integers b and n.
    Return d, u, v such that d = u * b + v * n, and d is the GCD of b, n.
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while n != 0:
        q, b, n = b // n, n, b % n
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0


def ntru_solve(f, g):
    """
    Solve the NTRU equation for f and g.
    Corresponds to NTRUSolve in Falcon's documentation.
    """
    n = len(f)
    if n == 1:
        f0 = f[0]
        g0 = g[0]
        d, u, v = xgcd(f0, g0)
        if d != 1:
            raise ValueError
        else:
            return [- q * v], [q * u]
    else:
        fp = field_norm(f)
        gp = field_norm(g)
        Fp, Gp = ntru_solve(fp, gp)
        F = karamul(lift(Fp), galois_conjugate(g))
        G = karamul(lift(Gp), galois_conjugate(f))
        F, G = reduce(f, g, F, G)
        return F, G


def gs_norm(f, g, q):
    """
    Compute the squared Gram-Schmidt norm of the NTRU matrix generated by f, g.
    This matrix is [[g, - f], [G, - F]].
    This algorithm is equivalent to line 9 of algorithm 5 (NTRUGen).
    """
    sqnorm_fg = sqnorm([f, g])
    ffgg = add(mul(f, adj(f)), mul(g, adj(g)))
    Ft = div(adj(g), ffgg)
    Gt = div(adj(f), ffgg)
    sqnorm_FG = (q ** 2) * sqnorm([Ft, Gt])
    return max(sqnorm_fg, sqnorm_FG)


def gen_poly(n):
    """
    Generate a polynomial of degree at most (n - 1), with coefficients
    following a discrete Gaussian distribution D_{Z, 0, sigma_fg} with
    sigma_fg = 1.17 * sqrt(q / (2 * n)).
    """
    # 1.17 * sqrt(12289 / 8192)
    sigma = 1.43300980528773
    assert(n < 4096)
    f0 = [samplerz(0, sigma, sigma - 0.001) for _ in range(4096)]
    f = [0] * n
    k = 4096 // n
    for i in range(n):
        # We use the fact that adding k Gaussian samples of std. dev. sigma
        # gives a Gaussian sample of std. dev. sqrt(k) * sigma.
        f[i] = sum(f0[i * k + j] for j in range(k))
    return f


def ntru_gen(n):
    """
    Implement the algorithm 5 (NTRUGen) of Falcon's documentation.
    At the end of the function, polynomials f, g, F, G in Z[x]/(x ** n + 1)
    are output, which verify f * G - g * F = q mod (x ** n + 1).

    Four paper Checks are enforced (cf. `ntrugen_filters.py`):
      Check 1a (gs_norm):  pre-filter ‖(g, −f)‖_2² ≤ 1.17²·q.
      Check 1b (γ_fg):     ‖fft(f, g)‖_∞ < γ_fg = 255 strict → m_B_fg = 8
                           safe in `_reconstruct_s_fxp`. Empirical rejection
                           rate post-1a ≈ 13.4% (500 keys). Pre-NTRUSolve,
                           so wall-clock impact ~0.1% (cf. ntrugen_filters).
      Check 2 (γ_hybrid):  bounds α_hybrid → governs `M_D = 18`.
      Check 3 (γ_FG):      bounds ‖fft(F,G)‖_∞ → governs `m_sign = 21`.
      Check 4 (γ_root):    bounds ‖fft(L_10_root)‖_∞ → governs `M_L10_ROOT = 5`.
    Without these branches, an extreme key could violate the fxp
    m-budgets silently and crash a downstream `|x| < 2^p` assert.

    Optional LTYZ Check (FORCE_ODD_GS_NORM = True): require ‖(g, −f)‖²
    to be ODD. Combined with samplerz `mode="round"`, this constitutes
    the full Lin-Tibouchi-Yu-Zhang countermeasure (Section 7.1 of paper
    2024-1709) — guarantees no half-integer samplerz center can appear
    at the float-detectable recursion levels.
    """
    while True:
        f = gen_poly(n)
        g = gen_poly(n)
        # Check order mirrors the paper / C implementation (ng_falcon.c):
        # γ_fg → α_hybrid → α_gpv (gs_norm). The order matters there (fxp:
        # γ_fg bounds the FFT values α_hybrid consumes; α_hybrid lower-bounds
        # the gs_norm divisor); here the filters are pure, so any order
        # accepts the same keys — we mirror the C for consistency.
        # Check 1b: ‖fft(f, g)‖_∞ < γ_fg = 255 strict. Independent of
        # Check 1a (Cauchy-Schwarz on gs_norm gives ‖fft‖_∞ ≤ √n·‖f‖_2 ≈
        # 2933, much looser). Required for m_B_fg = 8 in `_reconstruct_s_fxp`.
        if norm_fft_fg(f, g) >= GAMMA_FG_512:
            continue
        # Check 2: α_hybrid ≤ γ_hybrid = 4.
        if alpha_hybrid_squared(f, g, q) > GAMMA_HYBRID ** 2:
            continue
        # Check 1a: gs_norm filter (the pre-existing Falcon α_GPV test).
        if gs_norm(f, g, q) > (1.17 ** 2) * q:
            continue
        # LTYZ countermeasure: ‖(g, −f)‖² odd. The squared norm equals
        # the sum of f_i² + g_i² over Z; its parity is the parity of
        # sum(f_i + g_i) (since x² ≡ x mod 2). Cheap to test.
        if FORCE_ODD_GS_NORM:
            if (sum(f) + sum(g)) % 2 == 0:
                continue
        f_ntt = ntt(f)
        if any((elem == 0) for elem in f_ntt):
            continue
        try:
            F, G = ntru_solve(f, g)
            F = [int(coef) for coef in F]
            G = [int(coef) for coef in G]
            # Check 3: ‖fft(F, G)‖_∞ ≤ γ_FG = 3500 (Falcon-512).
            if norm_fft_FG(F, G) > GAMMA_FG_UPPER_512:
                continue
            # Check 4: ‖fft(L_10_root)‖_∞ ≤ γ_root = 24.
            if norm_fft_k(f, g, F, G) > GAMMA_ROOT:
                continue
            return f, g, F, G
        # If the NTRU equation cannot be solved, a ValueError is raised
        # In this case, we start again
        except ValueError:
            continue
