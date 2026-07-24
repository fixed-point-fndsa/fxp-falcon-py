# scripts/

Tooling and one-shot scripts for this Falcon fixed-point prototype.
Nothing in here is on the per-signature critical path.

## Contents

### Constant generators

- [`generate_constants_fxp.sage`](generate_constants_fxp.sage) — SageMath
  generator for the fxp constant tables (`fxp/fxp_constants_p63.py`,
  `fxp/fxp_constants_p127.py`): FFT twiddles plus the p-precise 1/√q,
  1/q, σ_n, σ_min constants (seeds of `nr_fxp.rsqrt` /
  `nr_fxp.nr_reciprocal`, multiplicands of the target and leaf math).
  Usage: `sage generate_constants_fxp.sage <p> 1` with `p ∈ {63, 127}`.
  Idempotent — re-running produces byte-identical tables.
- [`generate_rcdt.py`](generate_rcdt.py) — generates the 72-bit RCDT
  tables of the samplerz base sampler: Falcon-spec `floor` (μ = 0,
  cross-checked bit-for-bit against `falcon_ref/samplerz.py`) and LTYZ
  `round` (μ = 1/2, halved D(0)), both via the Howe-Prest-Ricosset-Rossi
  2019-1411 §5.2 rounding procedure.
- [`generate_constants.sage`](generate_constants.sage) — upstream Falcon
  reference generator (float twiddles + NTT constants). Kept for parity
  with the upstream Prest reference; not used by the fxp pipeline directly.
- [`parameters.py`](parameters.py) — upstream Falcon parameter generator
  (Round 3 spec + C-impl constants). Read-only reference.

### Analysis

- [`derive_m_budgets.py`](derive_m_budgets.py) — re-derives every proven
  m-budget lower bound from the NTRUGen thresholds (n, q, λ, σ, γ_*) and
  prints chosen vs derived + slack against `fxp/m_budgets.py`. Every
  threshold is a CLI flag, so what-if studies (e.g. `--gamma-root 12`,
  `--n 1024`) are one-liners. `tests/test_m_budgets_derivation.py`
  asserts chosen ≥ derived for every budget.
- [`callgraph_fxp.py`](callgraph_fxp.py) — static call-graph extractor
  for the `fxp/` package. AST-walks each module, resolves
  intra-package calls (preferring same-module candidates), and emits
  Graphviz DOT grouped by module via subgraph clusters with
  source-colored edges. Produces both a full graph (with dunders) and
  a simplified variant (`*_simplified.{dot,png,svg}`) that drops
  `__add__/__sub__/...` and underscore helpers in `fxtypes.py`.
  Run: `python scripts/callgraph_fxp.py`.

### Upstream Falcon test vectors (read-only reference)

These come from the upstream Prest Falcon Python reference and are
consumed by `falcon_ref/test.py`, not by the fxp pipeline. Listed here
only so the directory inventory is complete:

- `samplerz_KAT512.py`, `samplerz_KAT1024.py` — sampler KATs.
- `sign_KAT.py` — signing KATs.

The fxp test vectors live under [`../tests/`](../tests/) (unit tests in
`test_fxtypes.py` + numerical KATs in `check_test_vectors.py`).

## License

MIT (matches the project).
