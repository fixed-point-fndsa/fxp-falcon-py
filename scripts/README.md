# scripts/

Tooling and one-shot scripts for this Falcon fixed-point prototype.
Nothing in here is on the per-signature critical path.

## Contents

### Constant generators

- [`generate_constants_fxp.sage`](generate_constants_fxp.sage) — SageMath
  generator for the fxp twiddle tables (`fft_constants_fxp_p63.py`,
  `fft_constants_fxp_p127.py`) and the hardcoded `1/√q` anchor used by
  `ffldl_fxp.rsqrt`. Usage: `sage generate_constants_fxp.sage <p> 1`
  with `p ∈ {63, 127}`. Idempotent — re-running produces byte-identical
  tables.
- [`generate_constants.sage`](generate_constants.sage) — upstream Falcon
  reference generator (float twiddles + NTT constants). Kept for parity
  with the upstream Prest reference; not used by the fxp pipeline directly.
- [`parameters.py`](parameters.py) — upstream Falcon parameter generator
  (Round 3 spec + C-impl constants). Read-only reference.

### Analysis

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

- `saga.py` — Gaussian-sampler test suite.
- `samplerz_KAT512.py`, `samplerz_KAT1024.py` — sampler KATs.
- `sign_KAT.py` — signing KATs.

The fxp test vectors live under [`../tests/`](../tests/) (53 unit tests
in `test_fxtypes.py` + ~60 numerical KATs in `check_test_vectors.py`).

## License

MIT (matches the project).
