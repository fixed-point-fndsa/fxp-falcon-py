"""
Shared sys.path setup for the fxp package modules. Each module imports
this FIRST (`import _path_setup  # noqa: F401`) which prepends:

    <repo>/falcon_ref/   (for Prest's falcon.py reference: falcon, fft,
                          ffsampling, ntt, common, encoding, rng, …)

After this, the standard `from falcon import SecretKey`, `from fft
import fft`, etc. resolve correctly against Prest's falcon.py.

Production code in `fxp/*.py` only needs `falcon_ref/` on the path for
the upstream Falcon imports; `fxp/` itself is added by the caller (the
test/entry-point scripts).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent      # repo root
FALCON_REF = REPO_ROOT / "falcon_ref"                    # Prest's falcon.py

for _p in (FALCON_REF, REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
