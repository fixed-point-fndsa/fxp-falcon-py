# experiments/ — validation and precision benchmarks for the fxp Falcon

Each script is self-contained. Run it from the repo root with `uv run`
(no activation needed) — or with the venv active:

```bash
uv run --extra experiments python experiments/<name>.py
```

Deterministic seeds by default (scripts that sample fresh Falcon keys via
`random.seed(time)` say so in-file). The `experiments` extra adds `mpmath`
and `matplotlib` on top of the core deps.

The `bench_*` scripts write a figure under `experiments/figures/` and a CSV
under `experiments/tables/` (both git-ignored — regenerate by re-running).
The four also have **make targets** that auto-detect the venv (no activation
needed): `make figures` (all), or `make fig-{fft,div,ffldl,ffsampling}`.

## Validation — the fxp pipeline reproduces reference Falcon

| script | what it checks |
|---|---|
| `sign_fxp_end_to_end.py` | Full Sign, 4 flag combos `A=std/ref, B=tweak/ref, C=std/fxp, D=tweak/fxp`; 50 shared-seed trials verify `A==B==C==D` byte-for-byte. The headline "fxp reproduces reference Falcon" test (also driven by `tests/smoke_test_e2e.py`). |
| `sign_tweak_kat_fxp.py` | 1000 paired sigs (shared ChaCha20 seed): fxp std-vs-tweak → `0/1000` divergence; float64 reference → reproduces the ~1/8000 LTYZ rate. |
| `bench_pipeline_precision.py` | Deployed-path gate (n = 512, production functions on real filtered keys): per-stage absolute precision (gram / ffLDL / σ_i / target) vs 256-bit mpmath. Asserts a per-stage RMSE ceiling + `sig_mism == 0`, so it exits non-zero on a precision regression. Also writes a CSV + figure. |
| `test_ffsampling_fxp.py` | `ffsampling_fxp` vs reference `ffsampling_fft` under shared RNG; 200 trials × 2 modes (std + tweak), bit-for-bit. |
| `simulate_ntrugen.py` | Runs NTRUGen and counts rejections at each paper Check (reproduces the paper's rejection table). |

(samplerz byte-identity is gated in the pytest suite, [`tests/test_samplerz_fxp.py`](../tests/test_samplerz_fxp.py) — 2000 fixed-seed trials over magnitude tiers with near-integer concentration. The former `experiments/test_samplerz_fxp.py` duplicated that property and added a distributional check strictly implied by byte-identity; removed.)

## Precision benchmarks — float64 vs FxP-63 vs FxP-127 vs mpmath-256

Each builds the operation at every precision and compares against a 256-bit
mpmath ground truth. Recurring finding: **FxP-63 ≈ float64** (~2⁻⁴⁵–2⁻⁵³),
**FxP-127 ~60 bits more precise**. Each writes a figure + CSV.

| script | what it benchmarks |
|---|---|
| `bench_fft_precision.py` | `fft_fxp` / `ifft_fxp` precision scaling in `p`. |
| `bench_div_precision.py` | Pointwise FxC ÷ real division (the `L_10 = G_10/G_00` step). |
| `bench_ffsampling_precision.py` | Full self-consistent ffsampling at each precision; per-level per-coefficient error over up to `1000 × 512` samples. |

(Deployed ffLDL-tree precision is covered by `bench_ffldl_realcond.py` (per-level, real keys) and `bench_pipeline_precision.py`. The former standalone `bench_ffldl_precision.py` — an *idealised* tight-m-Gram n-sweep — was removed; its shared mpmath reference helpers now live in `_precision_ref.py`.)

## Helpers

`_path_setup.py` (sys.path) and `_outputs.py` (figure/CSV writers). The
NTRUGen predicates come from
[`falcon_ref/ntrugen_filters.py`](../falcon_ref/ntrugen_filters.py).
