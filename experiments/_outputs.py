"""
Common I/O helpers for experiments: tabular CSV output and dual-format
figure save (PNG for previews, PDF for LaTeX inclusion).

Each experiment writes:
  - tables/<name>.csv     raw numeric data (for pgfplots / pandas / re-rendering)
  - figures/<name>.png    bitmap preview (matplotlib's standard)
  - figures/<name>.pdf    vectorial figure (\\includegraphics{<name>.pdf})

The CSV files double as ground truth and as data sources for the pgfplots
templates under `experiments/figures/latex/`.
"""

import csv
from pathlib import Path


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    """Write headers + rows to a CSV file. Creates parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"wrote {path}")


def save_fig(fig, name: str, base_dir: Path,
             exts: tuple[str, ...] = ("png", "pdf"), dpi: int = 120) -> None:
    """Save `fig` to base_dir/figures/<name>.<ext> for each ext.

    Defaults to PNG (preview) and PDF (LaTeX). Use `exts=("png",)` to skip
    PDF (e.g. for fast iteration during dev).
    """
    figdir = base_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        out = figdir / f"{name}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"wrote {out}")
