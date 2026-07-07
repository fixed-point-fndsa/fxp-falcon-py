"""Machine-check the m_budgets constants against their derived lower bounds.

Every budget must satisfy `chosen >= derived minimum` (inequality, not
equality: some budgets deliberately carry headroom — the slack is reported by
scripts/derive_m_budgets.py, not asserted here). A failure means a threshold
(gamma check, CDT table, encoding limit, lambda, ...) changed without
re-deriving the dependent budgets — the silent-staleness class this test
exists to kill.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "fxp"))
sys.path.insert(0, str(_ROOT / "scripts"))

import m_budgets                      # noqa: E402
from derive_m_budgets import Params, derive  # noqa: E402


def test_chosen_budgets_cover_derived_minima():
    d = derive(Params())
    for name, val in d.items():
        if name.startswith("_"):
            continue
        m_min, formula, source = val
        chosen = getattr(m_budgets, name)
        assert chosen >= m_min, (
            f"{name} = {chosen} < derived minimum {m_min} "
            f"[{formula}, {source}] - a threshold changed without re-deriving"
        )
