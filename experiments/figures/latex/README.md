# Paper figures — pgfplots templates

Self-contained `pgfplots` figures that read the benchmark CSVs and render
with the paper's typographic style (no PNG re-exporting). Each `.tex` reads
`../../tables/<name>.csv` — i.e. `experiments/tables/`, relative to this
directory (adjust the path if you `\input` a template into a paper with a
different build root).

## How to use

Drop a fragment into your LaTeX:
```latex
\input{path/to/experiments/figures/latex/fft_precision.tex}
```
or copy/paste the `tikzpicture` block into a figure environment. Each `.tex`
is a standalone fragment for a `\begin{figure}…\end{figure}` env.

Preamble:
```latex
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}     % only if used
```

## Refreshing data

Each template reads `../../tables/<name>.csv`. To (re)generate it, run the
corresponding benchmark from the repo root:

```bash
python experiments/bench_fft_precision.py   # writes experiments/tables/fft_precision.csv
```

The template re-reads it at the next `pdflatex` run.

## Available templates

| Template | Source experiment | CSV |
|---|---|---|
| `fft_precision.tex` | `bench_fft_precision.py` | `fft_precision.csv` |
| `div_precision.tex` | `bench_div_precision.py` | `div_precision.csv` |
| `ffsampling_precision_selfconsistent_n512.tex` | `bench_ffsampling_precision.py` | `ffsampling_precision_selfconsistent_n512.csv` |
| `_template_loglog.tex` | (generic) | — |
| `_template_hist.tex` | (generic) | — |

The generic templates are starting points: copy-paste, change the CSV path
and column names.
