# Fixed-point Python prototype for Falcon

Reference implementation of the fixed-point arithmetic stack from the paper
*Toward a Secure Fixed-Point Implementation of the Falcon Signature Scheme*
(De Almeida Braga, Fouque, Lachguel, Prest). This repository holds:

- the high-level Python model of the proposed fxp pipeline (`fxp/`),
- Prest's `falcon.py` reference implementation (`falcon_ref/`),
- 52 unit tests + ~60 numerical test vectors (`tests/`),
- validation + precision-benchmark scripts (`experiments/`),
- the helper scripts that generate the FFT twiddle tables (`scripts/`).

`falcon_ref/` is a (lightly modified) clone of Thomas Prest's reference
Falcon Python implementation at <https://github.com/tprest/falcon.py>.
The `fxp/` package replaces only the floating-point arithmetic;
everything else (NTRUGen, hash-to-point, base-sampler tables, …) is
reused from this reference. `falcon_ref/falcon.py` exposes both the
modern `Falcon` manager class (upstream API) and a `SecretKey` shim that
gives the attribute-based stateful API the fxp module is built against.

The one substantive change to that reference is
[`falcon_ref/ntrugen_filters.py`](falcon_ref/ntrugen_filters.py): it adds the
paper's four NTRUGen rejection predicates — tight bounds on `‖FFT(f,g)‖_∞`,
`α_hybrid`, `‖FFT(F,G)‖_∞`, and `‖L_10_root‖_∞` — branched into key generation
so every key stays within the bounds the fixed-point `m`-budgets depend on.
Without them an extreme key could overflow the `|x| < 2^p` fixed-point format.

## Where to look first

| You want to … | Read |
|---|---|
| Understand the fxp pipeline | [`fxp/README.md`](fxp/README.md) |
| See the NTRUGen key filters (the paper's Checks) | [`falcon_ref/ntrugen_filters.py`](falcon_ref/ntrugen_filters.py) |
| Regenerate the FFT twiddle tables | [`scripts/`](scripts/) |

## Requirements

**Python ≥ 3.10** (the `fxp` package uses PEP 604 `X | None` type aliases at
module scope; 3.9 cannot import it).

With [uv](https://docs.astral.sh/uv/) (recommended — nothing to activate):

```bash
uv sync --extra experiments        # creates .venv with core + benchmark deps
```

Then run anything via `uv run python …` / `uv run pytest …`, or just use the
`make` targets below — they auto-detect `uv` and need no activation.

Without uv, install the deps from [`pyproject.toml`](pyproject.toml) into a
venv by hand:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install beartype pycryptodome pytest mpmath matplotlib
```

Core deps (`beartype`, `pycryptodome`, `pytest`) cover the package and tests;
the `experiments/` benchmarks add `mpmath` + `matplotlib` (the `experiments`
extra in `pyproject.toml`).

## Quick start

All commands run from the repository root. Run `make` (no arguments) to
list every target. The `make` targets need no venv activation (they use
`uv run`). Each script prepends `fxp/` and `falcon_ref/` to `sys.path`
automatically — `from falcon import SecretKey` then resolves to Prest's
`falcon.py`.

### Tests (correctness)

```bash
make test                                  # fxp unit tests + numerical KAT (core deps)
python -m pytest tests/test_fxtypes.py     # 52 unit tests, FxR/FxC arithmetic (~0.6 s)
python tests/check_test_vectors.py         # ~60 numerical KAT: FFT/ffLDL/division (~0.6 s)
python tests/smoke_test_e2e.py --fast      # KAT-only integration runner (~1 s)
```

The `check_test_vectors.py` oracles are computed in **float64** from
`falcon_ref/`, so the KAT checks that the fxp pipeline *agrees with the
float64 reference* (a consistency / non-regression check) — not that it is
more precise. Whether fxp beats float64 depends on the precision `p` (a
63-bit mantissa at `p=63` vs 53 for float64, far more at `p=127`) and on
the operation; the precision benchmarks (`experiments/bench_*_precision.py`),
run against an mpmath 256-bit reference, are what quantify that.

`make test-ref` additionally runs the self-test bundled with Prest's
`falcon.py` (FFT/NTT/NTRUGen/ffNP/compress/sign + samplerz KATs).

### Experiments (paper evidence)

Validation scripts (prefix with `uv run`, or activate the venv):

```bash
uv run python experiments/sign_fxp_end_to_end.py # fxp == reference Falcon (A==B==C==D, 50 trials)
uv run python experiments/sign_tweak_kat_fxp.py  # Section-5.1 tweak: 0/1000 divergence in fxp
```

The four precision-benchmark figures have `make` targets (venv auto-detected,
no activation needed). Each writes `experiments/figures/<name>.{png,pdf}` and
`experiments/tables/<name>.csv`:

```bash
make figures          # all four (float64 vs FxP-63 vs FxP-127 vs mpmath-256)
make fig-fft          # or one at a time: fig-fft / fig-div / fig-ffldl / fig-ffsampling
```

See [`experiments/README.md`](experiments/README.md) for the full catalogue.
Needs `mpmath` + `matplotlib`.

### Constant tables / profiling

```bash
sage scripts/generate_constants_fxp.sage   # regenerate twiddle tables + 1/√q → fxp_constants_p*.py
make profile                               # cProfile falcon_ref/profile_action.py → pyprof2calltree
make clean                                 # drop *.pyc / __pycache__ / *.cprof
```

## Layout

```
.
├── fxp/                core fxp package (Python)
├── falcon_ref/         Prest's falcon.py (MIT) + its self-test
├── experiments/        validation suite + precision benchmarks
├── scripts/            constant-table generators
├── tests/              unit tests + test vectors
├── pyproject.toml      dependencies (uv / pip)
├── LICENSE
└── README.md           (this file)
```

## License

MIT — see [`LICENSE`](LICENSE). The `fxp/` contribution and the
`falcon_ref/` reference (Prest's `falcon.py`) are both © Thomas Prest,
released under the MIT license.
