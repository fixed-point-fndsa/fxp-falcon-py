"""
End-to-end test of fxp Sign: Sign with ffsampling_fxp + samplerz_fxp must
produce byte-identical signatures to reference Sign when given the same
ChaCha20 seed. Tested with and without the Section 5.1 tweak.

Four pipelines, each seeded identically per trial:
  A. standard reference (float ffsampling, standard t)
  B. tweaked reference (float ffsampling, tweaked t_frac via NTT)
  C. fxp ffsampling, standard t          (use_fxp_ffsampling=True, use_tweak=USE_TWEAK_STD)
  D. fxp ffsampling, tweaked t_frac      (use_fxp_ffsampling=True, use_tweak=USE_TWEAK_NTT)

Expected: A == C and B == D (bit-for-bit). Also A == B (proven earlier).
"""

import os
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

import _path_setup  # noqa: F401, E402  (sets up sys.path)

from falcon import SecretKey  # noqa: E402
from rng import ChaCha20  # noqa: E402
from sign_tweak import sign, USE_TWEAK_STD, USE_TWEAK_NTT  # noqa: E402


def kat_end_to_end(sk, n_trials=50, seed_start=1):
    results = {
        "std_ref": [],
        "tw_ref": [],
        "std_fxp": [],
        "tw_fxp": [],
    }
    messages = []  # keep msg per trial so we can verify against the right one
    t0 = time.time()
    for i in range(n_trials):
        trial_seed = (seed_start + i).to_bytes(48, "big")
        msg = os.urandom(32)
        messages.append(msg)

        def fresh_rng():
            return ChaCha20(trial_seed).randombytes

        # A. Standard reference.
        sig_A = sign(sk, msg, use_tweak=USE_TWEAK_STD, use_fxp_ffsampling=False,
                     randombytes=fresh_rng())
        # B. Tweaked reference.
        sig_B = sign(sk, msg, use_tweak=USE_TWEAK_NTT, use_fxp_ffsampling=False,
                     randombytes=fresh_rng())
        # C. Standard t, fxp ffsampling.
        sig_C = sign(sk, msg, use_tweak=USE_TWEAK_STD, use_fxp_ffsampling=True,
                     randombytes=fresh_rng())
        # D. Tweaked t, fxp ffsampling.
        sig_D = sign(sk, msg, use_tweak=USE_TWEAK_NTT, use_fxp_ffsampling=True,
                     randombytes=fresh_rng())

        results["std_ref"].append(sig_A)
        results["tw_ref"].append(sig_B)
        results["std_fxp"].append(sig_C)
        results["tw_fxp"].append(sig_D)

        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            # Running tallies.
            eq_AC = sum(1 for a, c in zip(results["std_ref"], results["std_fxp"]) if a == c)
            eq_BD = sum(1 for b, d in zip(results["tw_ref"], results["tw_fxp"]) if b == d)
            eq_AB = sum(1 for a, b in zip(results["std_ref"], results["tw_ref"]) if a == b)
            print(f"  [{i+1}/{n_trials}] {dt:.0f}s  "
                  f"A==C: {eq_AC}/{i+1}  B==D: {eq_BD}/{i+1}  A==B: {eq_AB}/{i+1}")

    # Equality between pipelines (byte-for-byte).
    eq_AC = sum(1 for a, c in zip(results["std_ref"], results["std_fxp"]) if a == c)
    eq_BD = sum(1 for b, d in zip(results["tw_ref"], results["tw_fxp"]) if b == d)
    eq_AB = sum(1 for a, b in zip(results["std_ref"], results["tw_ref"]) if a == b)
    eq_CD = sum(1 for c, d in zip(results["std_fxp"], results["tw_fxp"]) if c == d)

    # Cryptographic verification of every signature against its OWN message.
    # (`sk.verify(message, signature)` — 2 args after self in the SecretKey
    # shim at falcon.py.)
    verify_counts = {
        label: sum(1 for sig, msg in zip(results[label], messages)
                   if sk.verify(msg, sig))
        for label in ("std_ref", "tw_ref", "std_fxp", "tw_fxp")
    }
    return {
        "n_trials": n_trials,
        "A==B (std_ref vs tw_ref)": eq_AB,
        "A==C (std_ref vs std_fxp)": eq_AC,
        "B==D (tw_ref vs tw_fxp)": eq_BD,
        "C==D (std_fxp vs tw_fxp)": eq_CD,
        "verify_A (std_ref)": verify_counts["std_ref"],
        "verify_B (tw_ref)":  verify_counts["tw_ref"],
        "verify_C (std_fxp)": verify_counts["std_fxp"],
        "verify_D (tw_fxp)":  verify_counts["tw_fxp"],
    }


def main():
    n = 512
    n_trials = 50
    random.seed(2026)
    print(f"Generating Falcon-{n} keypair...")
    sk = SecretKey(n)

    print(f"Running {n_trials} end-to-end KAT trials...")
    print(f"  A = std ref, B = tweaked ref, C = std fxp, D = tweaked fxp")
    res = kat_end_to_end(sk, n_trials=n_trials)
    print()
    print("Final counts:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
