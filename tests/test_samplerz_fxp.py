"""
Byte-identity regression test for `samplerz_fxp`.

`samplerz_fxp` (fxp/) must reproduce the float64 reference sampler
(falcon_ref/samplerz.py) *bit-for-bit* under shared randomness — this is the
property the whole fxp port rests on (the precision analysis assumes an
idealized sampler; the deployed sampler is validated by exact agreement with
the reference). The check spans varied centers, including the near-integer
locus where floor(mu) is fragile, and sigmas across [sigmin, sigmax].

Why this exists: rounding-affecting tweaks (m-budget changes, retag
reorderings) can break byte-identity at a ~1/50k rate. Caught via the
os.urandom end-to-end test, that shows up as un-attributable run-to-run
flicker. This fixed-seed comparison turns it into a loud, reproducible failure.
"""

import hashlib
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fxp"))
sys.path.insert(0, str(_ROOT / "falcon_ref"))

from fxtypes import FxR                          # noqa: E402  (fxp/)
from samplerz_fxp import samplerz_fxp            # noqa: E402  (fxp/)
from samplerz import samplerz as samplerz_ref    # noqa: E402  (falcon_ref/)

_P = 63
_SIGMIN = 1.2778
_SIGMAX = 1.8205
_N_TRIALS = 2000


class _Stream:
    """Deterministic XOF so both samplers consume the same byte sequence."""

    def __init__(self, seed: bytes):
        self.buf = hashlib.shake_256(seed).digest(1 << 14)
        self.pos = 0

    def __call__(self, n: int) -> bytes:
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        assert len(b) == n, "stream exhausted — raise the digest size"
        return b


def test_samplerz_fxp_byte_identical_to_reference():
    rng = random.Random(0xFA1C0)
    # Half the centers are pushed near an integer (the LTYZ-sensitive locus).
    near = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1.0 - 1e-6, 1.0 - 1e-3]
    for i in range(_N_TRIALS):
        # |mu| spans small..large across magnitude tiers: the cancellation in
        # r = mu - floor(mu) worsens with magnitude, and the m=18 sign budget
        # admits |mu| < 2^18. Keeping the near-integer frac at large k stresses
        # the worst case (big subtraction, tiny result).
        k = rng.randint(-(1 << rng.randint(0, 17)), 1 << rng.randint(0, 17))
        frac = rng.random() if i % 2 else rng.choice(near)
        mu = k + frac
        sigma = rng.uniform(_SIGMIN + 1e-3, _SIGMAX - 1e-3)
        seed = rng.getrandbits(128).to_bytes(16, "big")

        z_ref = samplerz_ref(mu, sigma, _SIGMIN, randombytes=_Stream(seed))
        z_fxp = samplerz_fxp(
            FxR.from_float(mu, m=18, p=_P),
            FxR.from_float(1.0 / (2.0 * sigma * sigma), m=0, p=_P),
            FxR.from_float(_SIGMIN / sigma, m=0, p=_P),
            randombytes=_Stream(seed),
        )
        assert z_ref == z_fxp, (
            f"samplerz_fxp diverged from the float64 reference at trial {i}: "
            f"mu={mu!r} sigma={sigma!r} ref={z_ref} fxp={z_fxp}"
        )
