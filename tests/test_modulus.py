"""Turn FxC's modulus promise into a tested invariant.

FxC declares |z| < 2^m (the complex-modulus convention that keeps the FFT's
+1-per-level bound), but `__post_init__` checks only the per-component FxR
bound |Re|, |Im| < 2^p by default (a big-int square per FxC is too much for
the FFT inner loop). `check_modulus()` enables the exact check
(re.x² + im.x² < 2^{2p}); here we (1) confirm it catches a genuine violation
and is a no-op when off, and (2) run the whole deployed keygen+sign pipeline
under it, asserting that every FxC actually built respects its tag."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fxp"))
sys.path.insert(0, str(_ROOT / "falcon_ref"))

from fxtypes import FxR, FxC, check_modulus       # noqa: E402  (fxp/)
from falcon import SecretKey                       # noqa: E402  (falcon_ref/)
from rng import ChaCha20                           # noqa: E402  (falcon_ref/)
from sign_tweak import sign, USE_TWEAK_STD, USE_TWEAK_NTT  # noqa: E402  (fxp/)

_P = 63


def _saturating_fxc(m):
    """An FxC whose components are just under 2^m but whose MODULUS exceeds it:
    Re = Im = 2^m − ε gives |z| = √2·(2^m − ε) > 2^m. Passes the component
    check, violates the modulus convention."""
    x = (1 << _P) - 1                     # component just under 2^p (|Re| < 2^m)
    return dict(re=FxR(x=x, m=m, p=_P), im=FxR(x=x, m=m, p=_P))


def test_modulus_check_catches_violation_only_when_enabled():
    args = _saturating_fxc(m=5)
    # Off (default): the modulus violation passes — only the FxR bound is checked.
    FxC(**args)
    # On: the same construction must raise.
    raised = False
    try:
        with check_modulus():
            FxC(**args)
    except AssertionError:
        raised = True
    assert raised, "check_modulus() failed to catch a modulus violation"
    # A genuinely valid FxC (|z| < 2^m) passes even with the check on.
    with check_modulus():
        FxC(re=FxR(x=1 << (_P - 2), m=5, p=_P), im=FxR(x=0, m=5, p=_P))


def test_check_modulus_restores_previous_setting():
    with check_modulus(True):
        with check_modulus(False):
            FxC(**_saturating_fxc(m=5))          # inner: off, no raise
        # back to True here — the violation must raise again.
        raised = False
        try:
            FxC(**_saturating_fxc(m=5))
        except AssertionError:
            raised = True
        assert raised
    # fully restored to the module default (off) — no raise.
    FxC(**_saturating_fxc(m=5))


def test_deployed_pipeline_respects_modulus():
    """Every FxC built by keygen (B0 FFT, Gram, ffLDL) and sign (target,
    ffsampling) satisfies |z| < 2^m — on both target modes."""
    sk = SecretKey(512)
    with check_modulus():
        for tweak in (USE_TWEAK_STD, USE_TWEAK_NTT):
            sig = sign(sk, b"modulus-invariant test", use_tweak=tweak,
                       use_fxp_ffsampling=True,
                       randombytes=ChaCha20(bytes(range(48))).randombytes)
            assert sk.verify(b"modulus-invariant test", sig)
