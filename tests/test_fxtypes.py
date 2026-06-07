"""
Tests for the fixed-point types `FxR` and `FxC` in `fxtypes.py`.

Run with:
    python3 test_fxtypes.py

Style mirrors `test.py`: each `test_*` function asserts its invariants and
prints a one-line PASS notice. `main()` runs them all and returns a
non-zero exit code on failure.
"""

import random
import sys
from math import ldexp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fxp"))
from fxtypes import FxR, FxC  # noqa: E402

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _ulp_half(m: int, p: int) -> float:
    """Half a ULP of FX_{m,p}, i.e. the round-to-nearest-even error bound."""
    return ldexp(1.0, m - p - 1)


# --------------------------------------------------------------------- #
# `FxR` storage and invariants
# --------------------------------------------------------------------- #


def test_fxr_invariant_p_positive():
    for bad_p in [0, -1, -10]:
        try:
            FxR(x=0, m=0, p=bad_p)
        except AssertionError:
            continue
        raise AssertionError(f"expected AssertionError for p={bad_p}")
    print("test_fxr_invariant_p_positive: PASS")


def test_fxr_invariant_mantissa_bound():
    # |x| must be strictly less than 2^p.
    FxR(x=(1 << 8) - 1, m=0, p=8)  # 255 OK
    FxR(x=-(1 << 8) + 1, m=0, p=8)  # -255 OK
    for bad in [1 << 8, -(1 << 8), (1 << 9)]:
        try:
            FxR(x=bad, m=0, p=8)
        except AssertionError:
            continue
        raise AssertionError(f"expected AssertionError for x={bad}")
    print("test_fxr_invariant_mantissa_bound: PASS")


def test_fxr_value_negative_m():
    # m can be negative: scale 2^{m-p} then smaller than 2^{-p}.
    v = FxR(x=1, m=-2, p=4)
    assert v.to_float() == ldexp(1.0, -2 - 4)
    print("test_fxr_value_negative_m: PASS")


def test_fxr_invariant_m_le_p():
    # m > p is rejected: sub-unit resolution is lost and mul can underflow.
    for bad_m in [5, 10]:
        try:
            FxR(x=0, m=bad_m, p=4)
        except AssertionError:
            continue
        raise AssertionError(f"expected AssertionError for m={bad_m} > p=4")
    # m == p is allowed (scale 1, integer representation).
    FxR(x=0, m=4, p=4)
    print("test_fxr_invariant_m_le_p: PASS")


# --------------------------------------------------------------------- #
# Constructors and recovery
# --------------------------------------------------------------------- #


def test_from_int_exact():
    # m = p -> scale is 1, exact embedding of small integers.
    v = FxR.from_int(42, m=8, p=8)
    assert v.x == 42 and v.to_float() == 42.0
    # m < p -> scale is 2^{m-p} < 1, integer is shifted left into the mantissa.
    v = FxR.from_int(3, m=4, p=8)  # scale 2^-4, x = 3 * 16 = 48
    assert v.x == 48 and v.to_float() == 3.0
    print("test_from_int_exact: PASS")


def test_from_int_overflow():
    # Scale 1, p=4 -> max |x| < 16.
    try:
        FxR.from_int(100, m=4, p=4)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")
    print("test_from_int_overflow: PASS")


def test_from_int_to_int_roundtrip():
    # For every integer a in (-2^m, 2^m), from_int.to_int must return a.
    for m, p in [(4, 4), (4, 8), (8, 8), (2, 16)]:
        for a in range(-(1 << m) + 1, 1 << m):
            assert FxR.from_int(a, m, p).to_int() == a
    print("test_from_int_to_int_roundtrip: PASS")


def test_to_int_rejects_non_integer():
    # FX_{4, 8}: x=1 encodes 2^-4 = 0.0625, not an integer.
    v = FxR(x=1, m=4, p=8)
    try:
        v.to_int()
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for non-integer value")
    print("test_to_int_rejects_non_integer: PASS")


def test_from_float_exact():
    v = FxR.from_float(0.5, m=0, p=8)
    assert v.x == 128 and v.to_float() == 0.5
    v = FxR.from_float(-0.25, m=0, p=4)
    assert v.x == -4 and v.to_float() == -0.25
    print("test_from_float_exact: PASS")


def test_from_float_rounding():
    # 0.3 with scale 2^{-4} = 0.0625: 0.3 / 0.0625 = 4.8 -> round to 5.
    v = FxR.from_float(0.3, m=0, p=4)
    assert v.x == 5
    assert abs(v.to_float() - 0.3) <= _ulp_half(0, 4)
    print("test_from_float_rounding: PASS")


def test_from_float_overflow():
    # p=8, m=0: max |value| < 1.
    try:
        FxR.from_float(1.0, m=0, p=8)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError at |x| == 2^p boundary")
    try:
        FxR.from_float(10.0, m=0, p=8)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")
    print("test_from_float_overflow: PASS")


def test_from_float_to_float_roundtrip(n_iter: int = 10000, seed: int = 4):
    """from_float.to_float must agree with the input within half a ULP."""
    random.seed(seed)
    p = 16
    worst = 0.0
    for _ in range(n_iter):
        f = random.uniform(-0.9, 0.9)
        err = abs(FxR.from_float(f, m=0, p=p).to_float() - f)
        if err > worst:
            worst = err
    bound = _ulp_half(0, p)
    assert worst <= bound + 1e-15, f"worst={worst}, bound={bound}"
    print(
        f"test_from_float_to_float_roundtrip: PASS (worst={worst:.2e} <= {bound:.2e})"
    )


# --------------------------------------------------------------------- #
# Arithmetic: neg, +, -
# --------------------------------------------------------------------- #


def test_fxr_neg_exact():
    a = FxR.from_float(0.3, m=4, p=8)
    assert (-a).x == -a.x and (-a).to_float() == -a.to_float()
    assert -(-a) == a
    print("test_fxr_neg_exact: PASS")


def test_fxr_add_is_exact():
    # + keeps (m, p) and performs an exact integer sum.
    a = FxR.from_float(0.5, m=4, p=8)
    b = FxR.from_float(0.3, m=4, p=8)
    c = a + b
    assert c.m == a.m and c.p == a.p
    assert c.x == a.x + b.x
    print("test_fxr_add_is_exact: PASS")


def test_fxr_add_overflow():
    a = FxR.from_float(0.9, m=0, p=8)
    b = FxR.from_float(0.9, m=0, p=8)
    try:
        a + b
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError")
    print("test_fxr_add_overflow: PASS")


def test_fxr_add_mismatch_mp():
    a = FxR.from_float(0.5, m=4, p=8)
    # p differs
    try:
        a + FxR.from_float(0.5, m=4, p=10)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for mismatched p")
    # m differs
    try:
        a + FxR.from_float(0.5, m=2, p=8)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for mismatched m")
    print("test_fxr_add_mismatch_mp: PASS")


def test_fxr_sub_is_exact():
    a = FxR.from_float(0.5, m=4, p=8)
    b = FxR.from_float(0.3, m=4, p=8)
    c = a - b
    assert c.x == a.x - b.x
    print("test_fxr_sub_is_exact: PASS")


def test_fxr_add_commutative():
    random.seed(1)
    for _ in range(1000):
        fa = random.uniform(-0.4, 0.4)
        fb = random.uniform(-0.4, 0.4)
        a = FxR.from_float(fa, m=0, p=16)
        b = FxR.from_float(fb, m=0, p=16)
        assert a + b == b + a
    print("test_fxr_add_commutative: PASS")


def test_fxr_sub_self_is_zero():
    a = FxR.from_float(0.3, m=4, p=8)
    assert a - a == FxR(x=0, m=4, p=8)
    print("test_fxr_sub_self_is_zero: PASS")


# --------------------------------------------------------------------- #
# Multiplication
# --------------------------------------------------------------------- #


def test_fxr_mul_exact_int():
    # Pick m_a + m_b <= p so the output scale is <= 1 and small integer
    # products are exactly representable.
    #   a = 2 in FX_{2, 4}: x_a = 8 (scale 2^-2), |x_a| = 8 < 16.
    #   b = 3 in FX_{2, 4}: x_b = 12.
    #   x_prod = 96, banker's shift by 4 -> 6, output m = 4, p = 4, scale 1.
    a = FxR.from_int(2, m=2, p=4)
    b = FxR.from_int(3, m=2, p=4)
    c = a * b
    assert c.m == 4 and c.p == 4
    assert c.x == 6 and c.to_float() == 6.0
    print("test_fxr_mul_exact_int: PASS")


def test_fxr_mul_different_m():
    # |a| < 1 (m=0), |b| < 4 (m=2) -> output m = 2.
    a = FxR.from_float(0.5, m=0, p=8)  # x = 128
    b = FxR.from_float(2.5, m=2, p=8)  # x = 160 (at scale 2^-6)
    c = a * b
    assert c.m == 2 and c.p == 8
    assert c.to_float() == 1.25
    print("test_fxr_mul_different_m: PASS")


def test_fxr_mul_rounding():
    # fa = 0.3, p=8, m=0: x_a = round(0.3 * 256) = 77
    # fb = 0.7, p=8, m=0: x_b = round(0.7 * 256) = 179
    # x_prod = 77 * 179 = 13783
    # banker's shift by 8: 13783 / 256 = 53.84... -> 54
    a = FxR.from_float(0.3, m=0, p=8)
    b = FxR.from_float(0.7, m=0, p=8)
    c = a * b
    assert c.x == 54
    assert abs(c.to_float() - a.to_float() * b.to_float()) <= _ulp_half(0, 8) + 1e-15
    print("test_fxr_mul_rounding: PASS")


def test_fxr_mul_ties_to_even():
    # Build products that land exactly on a half to exercise the tie-break.
    # FX_{0, 4}: x=4 -> 0.25, x * x = 16, shift by 4: 16/16 = 1 exactly. Not a tie.
    # FX_{0, 4}: x=6 -> 0.375, x*x=36, 36/16 = 2.25 -> 2. Not a tie.
    # Force a tie: x_prod = half exactly. With p=4, half = 8.
    #   x_a = 4, x_b = 2, x_prod = 8 -> tie, q = 0 (even), stays 0.
    a = FxR(x=4, m=0, p=4)
    b = FxR(x=2, m=0, p=4)
    assert (a * b).x == 0
    # x_a = 4, x_b = 6, x_prod = 24, 24/16 = 1.5 -> tie, q = 1 (odd), rounds to 2.
    a = FxR(x=4, m=0, p=4)
    b = FxR(x=6, m=0, p=4)
    assert (a * b).x == 2
    print("test_fxr_mul_ties_to_even: PASS")


def test_fxr_mul_zero():
    a = FxR.from_float(0.5, m=4, p=8)
    z = FxR(x=0, m=2, p=8)
    c = a * z
    assert c.x == 0 and c.m == 6 and c.p == 8
    print("test_fxr_mul_zero: PASS")


def test_fxr_mul_sign():
    a = FxR.from_float(-0.5, m=0, p=8)
    b = FxR.from_float(0.5, m=0, p=8)
    assert (a * b).to_float() == -0.25
    # (-) * (-) = (+)
    assert (a * a).to_float() == 0.25
    print("test_fxr_mul_sign: PASS")


def test_fxr_mul_mismatch_p():
    a = FxR.from_float(0.5, m=0, p=8)
    b = FxR.from_float(0.5, m=0, p=10)
    try:
        a * b
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for mismatched p")
    print("test_fxr_mul_mismatch_p: PASS")


def test_fxr_mul_commutative():
    random.seed(2)
    for _ in range(1000):
        fa = random.uniform(-0.9, 0.9)
        fb = random.uniform(-0.9, 0.9)
        a = FxR.from_float(fa, m=0, p=16)
        b = FxR.from_float(fb, m=0, p=16)
        assert a * b == b * a
    print("test_fxr_mul_commutative: PASS")


def test_fxr_mul_boundary_no_overflow():
    # Max |x| = 2^p - 1 on both sides must not overflow the output.
    p = 8
    a = FxR(x=(1 << p) - 1, m=0, p=p)
    c = a * a
    # (2^p - 1)^2 / 2^p = 2^p - 2 + 2^{-p} -> rounds to 2^p - 2.
    assert c.x == (1 << p) - 2
    assert c.m == 0 and c.p == p
    print("test_fxr_mul_boundary_no_overflow: PASS")


def test_fxr_mul_ulp_bound_randomized(n_iter: int = 10000, seed: int = 3):
    """Worst-case error of * should not exceed half a ULP of the output."""
    random.seed(seed)
    m_a, m_b, p = 0, 0, 16
    worst = 0.0
    for _ in range(n_iter):
        fa = random.uniform(-0.9, 0.9)
        fb = random.uniform(-0.9, 0.9)
        a = FxR.from_float(fa, m=m_a, p=p)
        b = FxR.from_float(fb, m=m_b, p=p)
        c = a * b
        err = abs(c.to_float() - a.to_float() * b.to_float())
        if err > worst:
            worst = err
    bound = _ulp_half(m_a + m_b, p)
    assert worst <= bound + 1e-15, f"worst={worst}, bound={bound}"
    print(f"test_fxr_mul_ulp_bound_randomized: PASS (worst={worst:.2e} <= {bound:.2e})")


# --------------------------------------------------------------------- #
# `FxC` storage
# --------------------------------------------------------------------- #


def test_fxc_from_complex():
    z = FxC.from_complex(0.5 + 0.25j, m=0, p=8)
    assert z.re.x == 128 and z.im.x == 64
    assert z.to_complex() == 0.5 + 0.25j
    print("test_fxc_from_complex: PASS")


def test_fxc_components_must_match():
    # Hand-build two FxRs with different (m, p); FxC must reject.
    r = FxR(x=1, m=4, p=8)
    i = FxR(x=1, m=2, p=8)
    try:
        FxC(re=r, im=i)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for mismatched (m,p) components")
    print("test_fxc_components_must_match: PASS")


# --------------------------------------------------------------------- #
# FxC arithmetic
# --------------------------------------------------------------------- #


def test_fxc_neg():
    z = FxC.from_complex(0.5 + 0.25j, m=0, p=8)
    assert (-z).to_complex() == -(z.to_complex())
    assert -(-z) == z
    print("test_fxc_neg: PASS")


def test_fxc_add_is_exact():
    a = FxC.from_complex(0.5 + 0.25j, m=0, p=8)
    b = FxC.from_complex(0.1 - 0.1j, m=0, p=8)
    c = a + b
    assert c.re.x == a.re.x + b.re.x
    assert c.im.x == a.im.x + b.im.x
    assert c.m == 0 and c.p == 8
    print("test_fxc_add_is_exact: PASS")


def test_fxc_sub_self_is_zero():
    a = FxC.from_complex(0.3 - 0.2j, m=0, p=8)
    d = a - a
    assert d.re.x == 0 and d.im.x == 0
    print("test_fxc_sub_self_is_zero: PASS")


def test_fxc_add_mismatch_mp():
    a = FxC.from_complex(0.5 + 0.25j, m=0, p=8)
    b = FxC.from_complex(0.5 + 0.25j, m=0, p=10)
    try:
        a + b
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for mismatched p")
    print("test_fxc_add_mismatch_mp: PASS")


def test_fxc_mul_exact():
    # (1 + i) * (1 - i) = 2 exactly.
    # Inputs: |z| = sqrt(2) < 2 = 2^1, so m=1 suffices (complex-modulus).
    #   from_float(1.0, m=1, p=8): x_re = x_im = 2^7 = 128 -> value 1.0 each.
    a = FxC.from_complex(1 + 1j, m=1, p=8)
    b = FxC.from_complex(1 - 1j, m=1, p=8)
    c = a * b
    # Output format under complex-modulus conv: (m_1 + m_2, p) = (2, 8).
    # |z_out| = |z_1| * |z_2| = sqrt(2) * sqrt(2) = 2 < 4 = 2^2. Fits m=2.
    assert c.m == 2 and c.p == 8
    # scale 2^{2-8} = 2^-6. Value 2.0 means x = 2.0 / 2^-6 = 128.
    assert c.re.x == 128 and c.im.x == 0
    assert c.to_complex() == 2.0 + 0j
    print("test_fxc_mul_exact: PASS")


def test_fxc_mul_i_squared():
    # i * i = -1.
    i_ = FxC.from_complex(0 + 1j, m=1, p=8)
    c = i_ * i_
    assert c.to_complex() == -1.0 + 0j
    print("test_fxc_mul_i_squared: PASS")


def test_fxc_mul_zero():
    a = FxC.from_complex(0.5 + 0.25j, m=0, p=8)
    z = FxC(re=FxR(x=0, m=0, p=8), im=FxR(x=0, m=0, p=8))
    c = a * z
    assert c.re.x == 0 and c.im.x == 0
    # Under complex-modulus conv: m_out = m_a + m_z = 0 + 0 = 0.
    assert c.m == 0 and c.p == 8
    print("test_fxc_mul_zero: PASS")


def test_fxc_mul_mismatch_p():
    a = FxC.from_complex(0.5 + 0.25j, m=0, p=8)
    b = FxC.from_complex(0.5 + 0.25j, m=0, p=10)
    try:
        a * b
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for mismatched p")
    print("test_fxc_mul_mismatch_p: PASS")


def test_fxc_mul_commutative():
    # Under complex-modulus conv, need |z| < 2^m. Sampling real and imag
    # in [-0.9, 0.9] can give |z| up to ~1.27, so we use m=1.
    random.seed(6)
    for _ in range(1000):
        a = FxC.from_complex(
            complex(random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)), m=1, p=16
        )
        b = FxC.from_complex(
            complex(random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)), m=1, p=16
        )
        assert a * b == b * a
    print("test_fxc_mul_commutative: PASS")


def test_fxc_mul_ulp_bound_randomized(n_iter: int = 10000, seed: int = 7):
    """Each component's error must stay within half a ULP of the output format."""
    random.seed(seed)
    # Under complex-modulus conv, the m_a, m_b=1 format accommodates
    # |z| up to 2, covering uniform(-0.9, 0.9) sampling (max |z| ~ 1.27).
    m_a, m_b, p = 1, 1, 16
    worst = 0.0
    for _ in range(n_iter):
        a = FxC.from_complex(
            complex(random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)), m_a, p
        )
        b = FxC.from_complex(
            complex(random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)), m_b, p
        )
        c = a * b
        exact = a.to_complex() * b.to_complex()
        err = max(
            abs(c.re.to_float() - exact.real), abs(c.im.to_float() - exact.imag)
        )
        if err > worst:
            worst = err
    # Under complex-modulus conv: output m = m_a + m_b (no +1).
    bound = _ulp_half(m_a + m_b, p)
    assert worst <= bound + 1e-15, f"worst={worst}, bound={bound}"
    print(f"test_fxc_mul_ulp_bound_randomized: PASS (worst={worst:.2e} <= {bound:.2e})")


# --------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------- #

ALL_TESTS = [
    test_fxr_invariant_p_positive,
    test_fxr_invariant_mantissa_bound,
    test_fxr_value_negative_m,
    test_fxr_invariant_m_le_p,
    test_from_int_exact,
    test_from_int_overflow,
    test_from_int_to_int_roundtrip,
    test_to_int_rejects_non_integer,
    test_from_float_exact,
    test_from_float_rounding,
    test_from_float_overflow,
    test_from_float_to_float_roundtrip,
    test_fxr_neg_exact,
    test_fxr_add_is_exact,
    test_fxr_add_overflow,
    test_fxr_add_mismatch_mp,
    test_fxr_sub_is_exact,
    test_fxr_add_commutative,
    test_fxr_sub_self_is_zero,
    test_fxr_mul_exact_int,
    test_fxr_mul_different_m,
    test_fxr_mul_rounding,
    test_fxr_mul_ties_to_even,
    test_fxr_mul_zero,
    test_fxr_mul_sign,
    test_fxr_mul_mismatch_p,
    test_fxr_mul_commutative,
    test_fxr_mul_boundary_no_overflow,
    test_fxr_mul_ulp_bound_randomized,
    test_fxc_from_complex,
    test_fxc_components_must_match,
    test_fxc_neg,
    test_fxc_add_is_exact,
    test_fxc_sub_self_is_zero,
    test_fxc_add_mismatch_mp,
    test_fxc_mul_exact,
    test_fxc_mul_i_squared,
    test_fxc_mul_zero,
    test_fxc_mul_mismatch_p,
    test_fxc_mul_commutative,
    test_fxc_mul_ulp_bound_randomized,
]


def main() -> int:
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"{t.__name__}: FAIL -- {e}")
        except Exception as e:
            failures += 1
            print(f"{t.__name__}: ERROR -- {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{failures} / {len(ALL_TESTS)} test(s) failed.")
        return 1
    print(f"All {len(ALL_TESTS)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
