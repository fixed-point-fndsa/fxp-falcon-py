"""
Shared sys.path setup for the experiments/ scripts. Each script imports
this module FIRST:

    from _path_setup import HERE  # noqa: F401  (also sets up sys.path)

which prepends to `sys.path`:

    <repo>/experiments/    (HERE — for sibling helpers like _outputs)
    <repo>/fxp/            (ROOT/fxp — for fxp module entry points)
    <repo>/falcon_ref/     (for the vendored Falcon: falcon, fft, ...)

After this, `from falcon import SecretKey` etc. resolve correctly. The
`HERE` and `ROOT` Path objects are also exported for scripts that use
them to write tables / figures (e.g. `write_csv(HERE / "tables" / …)`).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <repo>/experiments/
ROOT = HERE.parent                              # repo root

for _p in (HERE, ROOT, ROOT / "fxp", ROOT / "falcon_ref"):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
