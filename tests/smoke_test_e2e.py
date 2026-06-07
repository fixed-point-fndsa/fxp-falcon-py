"""
End-to-end smoke test: run the key integration scripts and assert that
their stdout contains the expected "all-pass" markers.

This is the layer that would have caught the `or True` placeholder bug
in `experiments/sign_fxp_end_to_end.py` (the script printed
`verify_A_ok: 50` regardless of whether anything actually verified).
The smoke runner is **strict about positive markers**: every script must
emit a line that is unambiguously "all clear", and no line on a
forbidden list (`Squared norm of signature is too large`, `AssertionError`,
`Traceback`, etc.) may appear.

Usage:
    python tests/smoke_test_e2e.py            # run all
    python tests/smoke_test_e2e.py --fast     # skip slow scripts (>30 s)

Exit code 0 iff every required script passes; 1 otherwise.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root

# Each entry:
#   path         : script under the repo root, run from ROOT.
#   required_re  : list of regexes that MUST appear in stdout.
#   forbidden_re : list of regexes that MUST NOT appear in stdout/stderr.
#   slow         : True ⇒ skipped under --fast.
#   timeout_s    : kill-switch; treated as failure if exceeded.

CASES = [
    {
        "name": "check_test_vectors",
        "path": "tests/check_test_vectors.py",
        "required_re": [r"^All vectors verified\.\s*$"],
        "forbidden_re": [r"FAIL\s", r"Traceback", r"AssertionError"],
        "slow": False,
        "timeout_s": 60,
    },
    {
        "name": "sign_fxp_end_to_end",
        "path": "experiments/sign_fxp_end_to_end.py",
        # All four verify counts must hit n_trials (currently 50). We
        # match each pipeline label individually so a single broken one
        # is caught even if the others pass.
        "required_re": [
            r"^\s*verify_A \(std_ref\):\s+50\s*$",
            r"^\s*verify_B \(tw_ref\):\s+50\s*$",
            r"^\s*verify_C \(std_fxp\):\s+50\s*$",
            r"^\s*verify_D \(tw_fxp\):\s+50\s*$",
            r"^\s*A==B \(std_ref vs tw_ref\):\s+50\s*$",
            r"^\s*A==C \(std_ref vs std_fxp\):\s+50\s*$",
            r"^\s*B==D \(tw_ref vs tw_fxp\):\s+50\s*$",
            r"^\s*C==D \(std_fxp vs tw_fxp\):\s+50\s*$",
        ],
        "forbidden_re": [
            r"Squared norm of signature is too large",
            r"Invalid encoding",
            r"Traceback",
            r"AssertionError",
        ],
        "slow": True,  # ~100 s on a laptop
        "timeout_s": 600,
    },
]


def run_one(case: dict) -> tuple[bool, str]:
    """Return (pass, log_excerpt). Excerpt is empty on success, a short
    diagnostic string on failure."""
    script = ROOT / case["path"]
    if not script.exists():
        return False, f"script not found: {script}"

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=case["timeout_s"],
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {case['timeout_s']} s"
    dt = time.time() - t0

    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return False, (
            f"exit code {proc.returncode} (after {dt:.0f}s)\n"
            f"--- last 20 lines of stdout+stderr ---\n"
            + "\n".join(out.splitlines()[-20:])
        )

    missing = [pat for pat in case["required_re"]
               if not re.search(pat, out, re.MULTILINE)]
    if missing:
        return False, (
            f"missing required marker(s) (after {dt:.0f}s):\n"
            + "\n".join(f"  - {p!r}" for p in missing)
            + "\n--- last 30 lines of stdout+stderr ---\n"
            + "\n".join(out.splitlines()[-30:])
        )

    bad = [pat for pat in case["forbidden_re"]
           if re.search(pat, out, re.MULTILINE)]
    if bad:
        offending = []
        for pat in bad:
            for line in out.splitlines():
                if re.search(pat, line):
                    offending.append((pat, line))
                    break
        return False, (
            f"forbidden marker(s) seen (after {dt:.0f}s):\n"
            + "\n".join(f"  - /{p}/ ⇒ {l!r}" for p, l in offending)
        )

    return True, f"ok ({dt:.1f}s)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="skip slow cases (>30s) — useful for pre-commit")
    args = parser.parse_args()

    cases = [c for c in CASES if not (args.fast and c["slow"])]
    n_skipped = len(CASES) - len(cases)

    print(f"Running {len(cases)} smoke case(s)"
          + (f" ({n_skipped} skipped under --fast)" if n_skipped else ""))
    print("=" * 64)

    fails = []
    n_missing = 0
    for case in cases:
        print(f"  [{case['name']:30s}] ... ", end="", flush=True)
        # A case whose script is absent (e.g. experiments/ not present in a
        # trimmed checkout) is skipped, not failed.
        if not (ROOT / case["path"]).exists():
            print(f"SKIP  ({case['path']} not present)")
            n_missing += 1
            continue
        ok, msg = run_one(case)
        if ok:
            print(f"PASS  {msg}")
        else:
            print("FAIL")
            print("    " + msg.replace("\n", "\n    "))
            fails.append(case["name"])

    print("=" * 64)
    if fails:
        print(f"FAILED: {len(fails)}/{len(cases)}  ({', '.join(fails)})")
        return 1
    ran = len(cases) - n_missing
    print(f"All {ran} smoke case(s) passed."
          + (f" ({n_missing} skipped: script not present)" if n_missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
