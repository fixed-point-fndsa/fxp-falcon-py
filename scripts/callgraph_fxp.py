#!/usr/bin/env python3
"""
Static call graph of the functions defined in `fxp/`.

Parses each .py file with `ast`, collects top-level and class-method
definitions, then for each function body records every call whose target
is either a name or a method we have defined in the package. Cross-module
calls are kept; calls to stdlib / numpy / falcon-ref / beartype are
dropped. Output is a Graphviz DOT file (and a PNG/SVG) grouping nodes by
module via subgraph clusters.

Run:
    python scripts/callgraph_fxp.py

Outputs (next to the script):
    callgraph_fxp.dot
    callgraph_fxp.png
    callgraph_fxp.svg
"""

import ast
import copy
import subprocess
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "fxp"
OUT_DIR = Path(__file__).resolve().parent

# Modules with no callable graph (constant tables) — skipped.
SKIP = {"fxp_constants_p63", "fxp_constants_p127"}

# Modules included in the graph (in display order, top-to-bottom).
ORDER = [
    "fxtypes",
    "fft_fxp",
    "samplerz_fxp",
    "ffldl_fxp",
    "target_construction",
    "ffsampling_fxp",
    "sign_tweak",
]


# --------------------------------------------------------------------- #
# Parse phase: collect every function defined in the package, qualified
# by its module (e.g. "fxtypes.FxR.div" or "ffldl_fxp.rsqrt"). For class
# methods we keep the bare method name as well as the dotted form, since
# call sites typically look like `x.div(...)` rather than `FxR.div(x,...)`.
# --------------------------------------------------------------------- #


def collect_defs():
    """Returns (defs, by_short_name).

    defs            : dict qualified_name -> dict(file=..., lineno=..., kind=..., short=...)
    by_short_name   : dict short_name -> set of qualified_name (for fuzzy resolution).
    """
    defs = {}
    by_short = {}

    for path in sorted(PKG.glob("*.py")):
        mod = path.stem
        if mod in SKIP or mod == "__init__":
            continue
        tree = ast.parse(path.read_text())

        # Top-level functions.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                short = node.name
                qual = f"{mod}.{short}"
                defs[qual] = dict(file=mod, lineno=node.lineno,
                                  kind="func", short=short)
                by_short.setdefault(short, set()).add(qual)
            elif isinstance(node, ast.ClassDef):
                cls = node.name
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        short = sub.name
                        qual = f"{mod}.{cls}.{short}"
                        defs[qual] = dict(file=mod, lineno=sub.lineno,
                                          kind="method", short=short, cls=cls)
                        by_short.setdefault(short, set()).add(qual)
    return defs, by_short


# --------------------------------------------------------------------- #
# Resolve phase: walk every function body, find calls, resolve to the
# qualified package name when possible.
# --------------------------------------------------------------------- #


def collect_calls(defs, by_short):
    """For each qualified function, return the set of qualified callees
    that are defined in the package."""
    edges = {q: set() for q in defs}

    for path in sorted(PKG.glob("*.py")):
        mod = path.stem
        if mod in SKIP or mod == "__init__":
            continue
        tree = ast.parse(path.read_text())

        # Build a list of (qual_name, body_node) for every defined function.
        bodies = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bodies.append((f"{mod}.{node.name}", node))
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        bodies.append((f"{mod}.{node.name}.{sub.name}", sub))

        for qual, body in bodies:
            for call in ast.walk(body):
                if not isinstance(call, ast.Call):
                    continue
                # Two cases: simple name (foo(...)) or attribute (x.foo(...)).
                target_short = None
                if isinstance(call.func, ast.Name):
                    target_short = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    target_short = call.func.attr
                if target_short is None:
                    continue
                cands = by_short.get(target_short)
                if not cands:
                    continue
                # Prefer same-module candidate, otherwise take all.
                same_mod = {c for c in cands if defs[c]["file"] == mod}
                chosen = same_mod if same_mod else cands
                # Drop self-loops.
                for c in chosen:
                    if c != qual:
                        edges[qual].add(c)
    return edges


# --------------------------------------------------------------------- #
# Render phase: emit DOT.
# --------------------------------------------------------------------- #


# Pastel fill colors for nodes (light, for readability of the labels).
COLORS = {
    "fxtypes":              "#fde0dd",
    "fft_fxp":              "#fff7bc",
    "samplerz_fxp":         "#ccebc5",
    "ffldl_fxp":            "#bdd7e7",
    "target_construction":  "#decbe4",
    "ffsampling_fxp":       "#fbb4ae",
    "sign_tweak":           "#b3cde3",
}

# Saturated edge colors (matching but darker than the fill, so arrows pop
# against the white background). Edges are colored by their *source* module,
# so it's easy to see which module is calling out.
EDGE_COLORS = {
    "fxtypes":              "#c51b8a",  # pink/magenta
    "fft_fxp":              "#d95f0e",  # orange
    "samplerz_fxp":         "#31a354",  # green
    "ffldl_fxp":            "#2171b5",  # blue
    "target_construction":  "#6a51a3",  # purple
    "ffsampling_fxp":       "#cb181d",  # red
    "sign_tweak":           "#08519c",  # dark blue
}

# Cross-check that the per-module palette is exhaustive: every module that
# can appear in the graph must have a fill *and* edge color, and no stale
# entries linger when a module is removed from `ORDER`.
assert set(COLORS) == set(EDGE_COLORS) == set(ORDER), (
    "COLORS / EDGE_COLORS / ORDER are out of sync"
)


def render_dot(defs, edges):
    by_mod = {}
    for q, info in defs.items():
        by_mod.setdefault(info["file"], []).append(q)

    lines = [
        "digraph callgraph_fxp {",
        '  graph [rankdir=LR, fontname="Helvetica", fontsize=11, '
        'splines=true, nodesep=0.18, ranksep=0.55, compound=true];',
        '  node  [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=9, margin="0.06,0.03"];',
        '  edge  [fontname="Helvetica", fontsize=8, arrowsize=0.7];',
        "",
    ]

    for i, mod in enumerate(ORDER):
        if mod not in by_mod:
            continue
        color = COLORS.get(mod, "#eeeeee")
        lines.append(f'  subgraph cluster_{mod} {{')
        lines.append(f'    label="{mod}.py"; labeljust="l"; '
                     f'style="rounded,filled"; fillcolor="white"; '
                     f'color="{color}"; penwidth=2; fontsize=12;')
        for q in sorted(by_mod[mod]):
            info = defs[q]
            label = info.get("cls", "") + ("." if info.get("cls") else "") + info["short"]
            lines.append(f'    "{q}" [label="{label}", fillcolor="{color}"];')
        lines.append("  }")
        lines.append("")

    # Edges colored by source module (so it's easy to track who's calling
    # whom). Cross-module edges drawn slightly heavier.
    seen = set()
    for src, dsts in edges.items():
        for dst in sorted(dsts):
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)
            src_mod = defs[src]["file"]
            same_mod = src_mod == defs[dst]["file"]
            color = EDGE_COLORS.get(src_mod, "#555555")
            penwidth = 0.9 if same_mod else 1.3
            lines.append(
                f'  "{src}" -> "{dst}" '
                f'[color="{color}", penwidth={penwidth}];'
            )

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def _drop_isolated(defs, edges):
    in_deg = {q: 0 for q in defs}
    for src, dsts in edges.items():
        for dst in dsts:
            in_deg[dst] = in_deg.get(dst, 0) + 1
    isolated = {q for q in defs
                if not edges.get(q) and in_deg.get(q, 0) == 0}
    KEEP = {"sign_tweak.sign", "sign_tweak.sample_preimage"}
    isolated -= KEEP
    for q in isolated:
        defs.pop(q, None)
        edges.pop(q, None)


def _strip_dunders_and_helpers(defs, edges):
    """Drop FxR/FxC dunder methods (__add__, __sub__, ...) and the internal
    `_` helpers that pollute the graph without adding signal — they are
    pure arithmetic plumbing. We also drop the bare ``__init__`` /
    ``__post_init__`` of FxR / FxC since beartype obscures them."""
    drop = set()
    for q, info in defs.items():
        s = info["short"]
        if s.startswith("__") and s.endswith("__"):
            drop.add(q)
        elif info["file"] == "fxtypes" and s.startswith("_"):
            drop.add(q)
    for q in drop:
        defs.pop(q, None)
    for src, dsts in list(edges.items()):
        if src in drop:
            del edges[src]
            continue
        edges[src] = {d for d in dsts if d not in drop}


def _write_graph(stem, defs, edges):
    """Render `defs`/`edges` as DOT under OUT_DIR/<stem>.{dot,png,svg}."""
    dot_path = OUT_DIR / f"{stem}.dot"
    dot_path.write_text(render_dot(defs, edges))
    print(f"Wrote {dot_path}")
    for ext in ("png", "svg"):
        out = OUT_DIR / f"{stem}.{ext}"
        subprocess.run(["dot", f"-T{ext}", str(dot_path), "-o", str(out)],
                       check=True)
        print(f"Wrote {out}")


def main():
    # Parse once; the simplified pass works on deep copies so the two
    # filter pipelines do not interfere.
    defs, by_short = collect_defs()
    edges = collect_calls(defs, by_short)

    # ---------------- full graph (with dunders) ----------------
    defs_full, edges_full = copy.deepcopy(defs), copy.deepcopy(edges)
    _drop_isolated(defs_full, edges_full)
    _write_graph("callgraph_fxp", defs_full, edges_full)

    # ---------------- simplified (no dunders / helpers) ----------------
    defs_simple, edges_simple = copy.deepcopy(defs), copy.deepcopy(edges)
    _strip_dunders_and_helpers(defs_simple, edges_simple)
    _drop_isolated(defs_simple, edges_simple)
    _write_graph("callgraph_fxp_simplified", defs_simple, edges_simple)

    # Print a quick summary (based on the full graph).
    defs, edges = defs_full, edges_full
    n_nodes = len(defs)
    n_edges = sum(len(dsts) for dsts in edges.values())
    print(f"\n{n_nodes} nodes, {n_edges} edges.")
    print("Modules included:")
    by_mod_count = {}
    for q, info in defs.items():
        by_mod_count[info["file"]] = by_mod_count.get(info["file"], 0) + 1
    for mod in ORDER:
        c = by_mod_count.get(mod, 0)
        if c:
            print(f"  {mod:<22}  {c:>3} fn")


if __name__ == "__main__":
    main()
