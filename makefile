# Runner: prefer `uv run` (auto-creates/syncs the env from pyproject.toml —
# no `activate`, no manual install); else the project .venv; else python3.
ifeq ($(shell command -v uv 2>/dev/null),)
  PY     := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
  PY_EXP := $(PY)
else
  PY     := uv run python
  PY_EXP := uv run --extra experiments python
endif
AUX = *.pyc *.cprof */*.pyc

.DEFAULT_GOAL := help

help:   ## list the available targets
	@echo "Usage: make <target>"
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  %-15s %s\n", $$1, $$2}'

test:   ## run the fxp unit tests + numerical KAT
	$(PY) -m pytest tests/test_fxtypes.py tests/test_nr_fxp.py
	$(PY) tests/check_test_vectors.py

test-ref:   ## run Prest's falcon.py reference self-test
	PYTHONPATH=. $(PY) falcon_ref/test.py

figures: fig-fft fig-div fig-ffldl fig-ffsampling   ## generate all 4 precision-benchmark figures

fig-fft:   ## figure - FFT precision (float64 vs FxP-63/127 vs mpmath)
	$(PY_EXP) experiments/bench_fft_precision.py
fig-div:   ## figure - division precision
	$(PY_EXP) experiments/bench_div_precision.py
fig-ffldl:   ## figure - ffLDL precision
	$(PY_EXP) experiments/bench_ffldl_precision.py
fig-ffsampling:   ## figure - ffsampling precision
	$(PY_EXP) experiments/bench_ffsampling_precision.py

profile:   ## cProfile a keygen + sign
	rm -f $(AUX)
	rm -rf __pycache__
	touch profile_action.cprof
	PYTHONPATH=. $(PY) -m cProfile -o profile_action.cprof falcon_ref/profile_action.py
	pyprof2calltree -k -i profile_action.cprof &

clean:   ## remove caches and build cruft
	rm -f $(AUX)
	rm -rf __pycache__ */__pycache__
	@echo "Clean done"
