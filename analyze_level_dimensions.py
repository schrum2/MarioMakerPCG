#!/usr/bin/env python3
"""
analyze_level_dimensions.py
===========================
Measure the width and height of every COMPLETE level in a training input and
plot them as a scatter (width = x, height = y) so the distribution of level
sizes across 2D space is visible.

This deliberately works on the full, un-cropped levels -- the same
"(source_num)"-delimited levels that build_dataset_with_ascii.py reads -- and
NOT the 20x20 windows the dataset builder slices out of them. The point is to
see how big the source levels actually are before any windowing/padding, to
inform whether smaller levels should be padded up rather than larger levels
chopped down.

Width  = number of columns = the longest row in the level (trailing newlines
         stripped, matching build_dataset_with_ascii.py).
Height = number of rows = the level's row count after leading/trailing fully
         blank lines are removed.

Usage:
    python analyze_level_dimensions.py --input <file-or-folder> --output dist.png
"""

import argparse
import os
import sys

# Reuse the exact level splitter the dataset builder uses so "a level" here
# means the same thing it means downstream.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset_with_ascii import (  # noqa: E402
    parse_source_file,
    collect_input_files,
)


def level_dimensions(rows, empty_chars=" "):
    """Return (width, height) of one level's CONTENT bounding box.

    Sky padding is excluded on all four sides: after stripping trailing
    newlines, the empty (sky/air) characters in `empty_chars` are treated as
    blank, and the tightest box that still contains every non-empty tile is
    measured. So a level stored on the full canvas but whose tiles only span,
    say, 100 columns and 14 rows measures 100x14, not the padded canvas size.

    In mm2_tileset_we.json the air tile is the space character (and '-' is a
    real Lift tile, NOT sky), so the default treats only spaces as empty.
    Returns (0, 0) for a level with no content at all.
    """
    rows = [r.rstrip("\r\n") for r in rows]
    empty = set(empty_chars)

    def row_has_content(r):
        return any(ch not in empty for ch in r)

    # Vertical extent: first and last rows that hold any non-empty tile.
    top = next((i for i, r in enumerate(rows) if row_has_content(r)), None)
    if top is None:
        return 0, 0
    bottom = next(i for i in range(len(rows) - 1, -1, -1) if row_has_content(rows[i]))
    content_rows = rows[top: bottom + 1]
    height = len(content_rows)

    # Horizontal extent: leftmost and rightmost columns holding a non-empty
    # tile across the content rows.
    left = min(
        next(j for j, ch in enumerate(r) if ch not in empty)
        for r in content_rows if row_has_content(r)
    )
    right = max(
        max(j for j, ch in enumerate(r) if ch not in empty)
        for r in content_rows if row_has_content(r)
    )
    width = right - left + 1
    return width, height


def collect_dimensions(input_path, empty_chars=" "):
    """Walk every input .txt file, returning a list of
    (name, width, height) for each complete level found."""
    input_files = collect_input_files(input_path)
    results = []
    for input_file in input_files:
        levels = parse_source_file(input_file)
        multi = len(input_files) > 1
        for name, rows in levels.items():
            w, h = level_dimensions(rows, empty_chars=empty_chars)
            if w == 0 or h == 0:
                continue
            full_name = f"{input_file.stem}/{name}" if multi else name
            results.append((full_name, w, h))
    return results


def _summary(values):
    import numpy as np

    arr = np.asarray(values)
    return (
        f"min {arr.min()}, max {arr.max()}, "
        f"mean {arr.mean():.1f}, median {np.median(arr):.0f}"
    )


def make_plot(dims, output_path, title=None):
    import matplotlib

    matplotlib.use("Agg")  # headless: never pop a window, just write the file
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from collections import Counter
    import numpy as np

    widths = [w for _, w, _ in dims]
    heights = [h for _, _, h in dims]

    # Widths/heights are whole tiles, so hundreds of levels can land on the exact
    # same (w, h) and hide behind a single dot. Collapse to one point per unique
    # size and let the count drive both the marker area and the colour, so a
    # popular size reads as a big bright dot instead of vanishing under the pile.
    counts = Counter(zip(widths, heights))
    xs = np.array([w for (w, _h) in counts])
    ys = np.array([h for (_w, h) in counts])
    n = np.array([counts[(w, h)] for (w, h) in counts])

    # Area scales with the square root of the count so a cell with 100 levels
    # isn't 100x the radius of a cell with one -- that would swamp the plot.
    sizes = 25 + 200 * np.sqrt(n / n.max())

    # Extra-wide so the closely-spaced widths spread out and small differences
    # between sizes are readable.
    fig, ax = plt.subplots(figsize=(19, 9))
    cmap = plt.get_cmap("viridis")
    sc = ax.scatter(
        xs, ys, s=sizes, c=n, cmap=cmap,
        norm=LogNorm(vmin=1, vmax=n.max()),
        alpha=0.85, edgecolors="black", linewidths=0.3,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Number of levels at this exact size")

    ax.set_xlabel("Level width (columns / tiles)")
    ax.set_ylabel("Level height (rows / tiles)")
    ax.set_title(title or "Complete level size distribution")
    ax.grid(True, linestyle=":", alpha=0.4)
    # Give the points a little breathing room from the axes so the busiest
    # cells aren't clipped against the frame.
    ax.margins(0.04)

    # A short stats line in the corner so the plot stands on its own.
    stats = (
        f"levels: {len(dims)}\n"
        f"width:  {_summary(widths)}\n"
        f"height: {_summary(heights)}"
    )
    ax.text(
        0.02, 0.98, stats, transform=ax.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Scatter-plot the width/height of every complete training level."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a .txt level file or a folder of .txt files.")
    parser.add_argument("--output", required=True,
                        help="Output PNG path for the scatter plot.")
    parser.add_argument("--title", default=None,
                        help="Optional plot title.")
    parser.add_argument("--csv", default=None,
                        help="Optional path to also dump per-level dimensions as CSV.")
    parser.add_argument("--empty-chars", default=" ",
                        help="Characters treated as sky/air when trimming the content "
                             "bounding box. Default: a single space (the mm2_tileset_we "
                             "air tile). Note '-' is a real Lift tile, so it is NOT sky.")
    args = parser.parse_args()

    dims = collect_dimensions(args.input, empty_chars=args.empty_chars)
    if not dims:
        print(f"ERROR: No non-empty levels found in {args.input}")
        sys.exit(1)

    widths = [w for _, w, _ in dims]
    heights = [h for _, _, h in dims]
    print(f"Measured {len(dims)} complete level(s) from {args.input}")
    print(f"  width:  {_summary(widths)}")
    print(f"  height: {_summary(heights)}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("name,width,height\n")
            for name, w, h in dims:
                safe = name.replace(",", "_")
                f.write(f"{safe},{w},{h}\n")
        print(f"Per-level dimensions written to {args.csv}")

    make_plot(dims, args.output, title=args.title)
    print(f"Scatter plot written to {args.output}")


if __name__ == "__main__":
    main()
