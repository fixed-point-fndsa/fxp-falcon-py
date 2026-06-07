"""
KAT std vs tw at scale, fxp pipeline.

Companion to sign_tweak_kat.py: same paired-signature setup, but with
use_fxp_ffsampling=True (and the new fxp t-builders). Goal is to verify
that the self-consistent fxp pipeline gives bit-identical signatures
between standard and tweaked, with no float64 round-off slippage.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from falcon import SecretKey  # noqa: E402
from rng import ChaCha20  # noqa: E402

from sign_tweak import sign, USE_TWEAK_STD, USE_TWEAK_NTT  # noqa: E402


def main():
    import random
    random.seed(12345)
    n_trials = 1000
    print(f"Generating Falcon-512 secret key…")
    sk = SecretKey(512)
    print(f"Running {n_trials} paired signatures, fxp and float side by side…")

    eq_fxp, diff_fxp = 0, []
    eq_ref, diff_ref = 0, []

    for i in range(n_trials):
        seed = (1 + i).to_bytes(48, "big")
        msg = os.urandom(32)

        # Self-consistent fxp pipeline (NEW).
        sig_std_fxp = sign(sk, msg, use_tweak=USE_TWEAK_STD, use_fxp_ffsampling=True,
                           randombytes=ChaCha20(seed).randombytes)
        sig_tw_fxp = sign(sk, msg, use_tweak=USE_TWEAK_NTT, use_fxp_ffsampling=True,
                          randombytes=ChaCha20(seed).randombytes)
        if sig_std_fxp == sig_tw_fxp:
            eq_fxp += 1
        else:
            diff_fxp.append(i)

        # Reference float pipeline (for the contrast).
        sig_std_ref = sign(sk, msg, use_tweak=USE_TWEAK_STD,
                           randombytes=ChaCha20(seed).randombytes)
        sig_tw_ref = sign(sk, msg, use_tweak=USE_TWEAK_NTT,
                          randombytes=ChaCha20(seed).randombytes)
        if sig_std_ref == sig_tw_ref:
            eq_ref += 1
        else:
            diff_ref.append(i)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_trials}  fxp eq={eq_fxp} diff={len(diff_fxp)}  "
                  f"|  ref eq={eq_ref} diff={len(diff_ref)}")

    print()
    print(f"std_fxp == tw_fxp : {eq_fxp}/{n_trials}  divergent={diff_fxp[:10]}"
          f"{'…' if len(diff_fxp) > 10 else ''}")
    print(f"std_ref == tw_ref : {eq_ref}/{n_trials}  divergent={diff_ref[:10]}"
          f"{'…' if len(diff_ref) > 10 else ''}")
    print()
    if not diff_fxp:
        print("Conclusion: the fxp pipeline preserves KAT bit-for-bit.")
    else:
        print(f"Conclusion: fxp still diverges {len(diff_fxp)}/{n_trials} — investigate.")


if __name__ == "__main__":
    main()
