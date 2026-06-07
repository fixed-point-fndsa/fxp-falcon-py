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
| `test_keygen_fxp.py` | End-to-end `keygen_fxp` (ffLDL + `normalize_tree`) vs an mpmath reference, n = 4 … 512. |
| `test_ffsampling_fxp.py` | `ffsampling_fxp` vs reference `ffsampling_fft` under shared RNG; 200 trials × 2 modes (std + tweak), bit-for-bit. |
| `test_samplerz_fxp.py` | samplerz: KAT match over 10k shared-RNG trials + distributional match over 50k samples. |
| `simulate_ntrugen.py` | Runs NTRUGen and counts rejections at each paper Check (reproduces the paper's rejection table). |

## Precision benchmarks — float64 vs FxP-63 vs FxP-127 vs mpmath-256

Each builds the operation at every precision and compares against a 256-bit
mpmath ground truth. Recurring finding: **FxP-63 ≈ float64** (~2⁻⁴⁵–2⁻⁵³),
**FxP-127 ~60 bits more precise**. Each writes a figure + CSV.

| script | what it benchmarks |
|---|---|
| `bench_fft_precision.py` | `fft_fxp` / `ifft_fxp` precision scaling in `p`. |
| `bench_div_precision.py` | Pointwise FxC ÷ real division (the `L_10 = G_10/G_00` step). |
| `bench_ffldl_precision.py` | Full ffLDL\* tree, max coef error per node, on real NTRU bases. |
| `bench_ffsampling_precision.py` | Full self-consistent ffsampling at each precision; per-level per-coefficient error over up to `1000 × 512` samples. |

## Helpers

`_path_setup.py` (sys.path) and `_outputs.py` (figure/CSV writers). The
NTRUGen predicates come from
[`falcon_ref/ntrugen_filters.py`](../falcon_ref/ntrugen_filters.py).
