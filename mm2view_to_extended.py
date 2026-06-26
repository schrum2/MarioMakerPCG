#!/usr/bin/env python3
"""
mm2view_to_extended.py
Collapse a Mario Maker 2 ASCII level from the full 69-tile mm2_tileset_we vocab
down to a smaller "extended" tileset, so diffusion training isn't spread thin over
dozens of near-empty tile classes.

The input levels are already written in the mm2_tileset_we glyph space (# ground,
k semisolid, g goomba, K koopa, ...). Reducing to a target tileset is then a pure
per-character remap: every glyph the target keeps stays as-is, and every glyph it
drops is folded onto its closest survivor.

"Closest" is decided from the tags in mm2_tileset_we.json -- each glyph lists a
broad collision/behaviour class (solid / passable / enemy / collectable / hazard)
followed by more specific tags. A dropped tile is matched to a kept tile of the
same class, preferring the one that shares the most specific tags (so koopa folds
onto another ground enemy, a cloud block onto another solid platform, a power-up
onto another collectable, an environmental hazard onto spikes, and so on). A few
genuinely ambiguous folds are pinned in REPLACEMENT_OVERRIDES below.

Target tilesets, most aggressive to least (see build_extended_tilesets.py):
    extended_tiles_20.json ... extended_tiles_60.json   (frequency-ranked)
    extended_tiles.json                                  (hand-picked ~17, default)

Pick one with --tileset (CLI) or set_target() (when imported, e.g. by
bucket_levels_by_size.py / build_dataset_with_ascii.py).

Usage:
    python mm2view_to_extended.py input_level.txt [output_level.txt] --tileset extended_tiles_30.json
    python mm2view_to_extended.py --tileset extended_tiles_20.json --show_map
"""

import sys
import os
import json
import argparse
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))

VGLC_HEIGHT = 20
OUT_EMPTY = " "

# Full source vocabulary the input levels are written in.
SOURCE_TILESET = os.path.join(HERE, "mm2_tileset_we.json")
# Default reduction target when nobody calls set_target() -- keeps the existing
# extended_tiles.json pipeline working unchanged.
DEFAULT_TARGET = os.path.join(HERE, "extended_tiles.json")

# The json->ascii export emits a direction-specific arrow per pipe; the tilesets
# keep only the single pipe glyph "|". Fold the arrows back before remapping so a
# pipe reduces as a pipe instead of an unknown tile.
PIPE_ARROWS = {"→": "|", "←": "|", "↑": "|", "↓": "|"}

# Broad class each glyph belongs to, read off the leading tags. Order matters:
# an enemy that also carries "hazard" is classed as an enemy, not a hazard.
_CLASS_TAGS = ["enemy", "collectable", "hazard", "solid", "passable"]
# Where a class falls back when the target keeps nothing of that kind. These
# glyphs (generic enemy, coin, spikes, ground, air) sit near the top of the
# frequency ranking, so they survive in every target tileset.
_CLASS_FALLBACK = {
    "enemy": "g",
    "collectable": "c",
    "hazard": "^",
    "solid": "#",
    "passable": OUT_EMPTY,
}

# Hand-pinned folds for tiles whose nearest-by-tags answer is a judgement call.
# A door is a passable warp but shares its only meaningful tag ("warp") with the
# pipe, so we keep the warp structure rather than dissolving it to open air.
REPLACEMENT_OVERRIDES = {
    "D": "|",   # door -> pipe (both warps; keep a structure, don't blank it out)
}


def _load_tiles(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["tiles"]


def _class_of(tags):
    for cls in _CLASS_TAGS:
        if cls in tags:
            return cls
    return "passable"


def build_replacement_map(target_path, source_path=SOURCE_TILESET):
    """Return {source_glyph: target_glyph} folding the full source vocab onto the
    glyphs the target tileset keeps. Pipe arrows are included as pipe aliases."""
    source = _load_tiles(source_path)
    target = _load_tiles(target_path)
    kept = list(target.keys())
    kept_set = set(kept)
    # Earlier-listed kept glyphs are more frequent (the tilesets are written in
    # frequency order); used to break ties toward the more generic survivor.
    rank = {g: i for i, g in enumerate(kept)}

    def best_fold(glyph, tags):
        if glyph in REPLACEMENT_OVERRIDES and REPLACEMENT_OVERRIDES[glyph] in kept_set:
            return REPLACEMENT_OVERRIDES[glyph]
        cls = _class_of(tags)
        # Candidate survivors of the same class. Hazards exclude living enemies so
        # a saw/burner folds onto spikes, not onto a goomba that merely shares
        # the "hazard" tag.
        pool = [g for g in kept if cls in source.get(g, target[g])]
        if cls == "hazard":
            pool = [g for g in pool if "enemy" not in target[g]]
        src_tags = set(tags)
        best_g, best_score = None, -1
        for g in pool:
            if g == glyph:
                continue
            shared = src_tags & set(target[g])
            score = len(shared - {cls})   # reward specific-tag overlap
            if score > best_score or (score == best_score and rank[g] < rank.get(best_g, 1 << 30)):
                best_g, best_score = g, score
        # A passable object with nothing specific in common (a lone door-less,
        # platform-less oddity) is just open space.
        if cls == "passable" and best_score < 1:
            return OUT_EMPTY
        if best_g is not None:
            return best_g
        return _CLASS_FALLBACK[cls]

    mapping = {}
    for glyph, tags in source.items():
        mapping[glyph] = glyph if glyph in kept_set else best_fold(glyph, tags)
    # Pipe-direction arrows ride along with whatever "|" maps to.
    for arrow, pipe in PIPE_ARROWS.items():
        mapping[arrow] = mapping.get(pipe, pipe if pipe in kept_set else OUT_EMPTY)
    return mapping


# Active reduction map. Lazily built against DEFAULT_TARGET so importing modules
# that just call convert_level() keep working; call set_target() to switch.
_active_map = None
_active_target = None


def set_target(target_path):
    """Point the converter at a target tileset (path to an extended_tiles*.json)."""
    global _active_map, _active_target
    _active_map = build_replacement_map(target_path)
    _active_target = target_path
    return _active_map


def _ensure_map():
    if _active_map is None:
        set_target(DEFAULT_TARGET)
    return _active_map


def load_level(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f.readlines()]


def crop_and_pad(grid, target=VGLC_HEIGHT):
    """Fix the level to `target` rows: drop empty rows off the top, keep the
    bottom `target`, and pad short levels with air on top (bottom-aligned)."""
    while len(grid) > target and grid[0].strip() == "":
        grid = grid[1:]
    if len(grid) > target:
        grid = grid[-target:]
    width = max((len(r) for r in grid), default=0)
    grid = [r.ljust(width) for r in grid]
    while len(grid) < target:
        grid.insert(0, " " * width)
    return grid


def convert_level(lines):
    """Reduce one level (list of rows) to the active target tileset, returning the
    converted rows. Fixed to VGLC_HEIGHT rows; trailing all-air columns trimmed."""
    mapping = _ensure_map()
    if not lines:
        return [OUT_EMPTY * 10] * VGLC_HEIGHT

    grid = crop_and_pad(list(lines))
    out_rows = ["".join(mapping.get(ch, OUT_EMPTY) for ch in row) for row in grid]

    # Trim trailing all-air columns so the content box stays tight (the bucketer
    # measures width off this).
    max_col = -1
    for row in out_rows:
        stripped = row.rstrip(OUT_EMPTY)
        if len(stripped) - 1 > max_col:
            max_col = len(stripped) - 1
    out_rows = [row[:max_col + 1] for row in out_rows]
    return out_rows


def main():
    parser = argparse.ArgumentParser(
        description="Reduce a Mario Maker 2 ASCII level to a smaller extended tileset."
    )
    parser.add_argument("input", nargs="?", help="Path to the MM2 ASCII level .txt file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output path (default: print to stdout)")
    parser.add_argument("--tileset", default=DEFAULT_TARGET,
                        help="Target tileset JSON (extended_tiles_20.json ... _60.json, "
                             f"or the default {os.path.basename(DEFAULT_TARGET)}).")
    parser.add_argument("--show_map", action="store_true",
                        help="Print the full source->target replacement map and exit.")
    args = parser.parse_args()

    set_target(args.tileset)

    if args.show_map:
        source = _load_tiles(SOURCE_TILESET)
        kept = set(_load_tiles(args.tileset).keys())
        print(f"# Replacement map for {os.path.basename(args.tileset)} "
              f"({len(kept)} tiles)")
        for glyph, tags in source.items():
            dst = _active_map[glyph]
            name = tags[-1]
            if glyph in kept:
                print(f"  {glyph!r:6} {name:22} KEPT")
            else:
                dst_name = _load_tiles(args.tileset)[dst][-1] if dst in kept else "air"
                print(f"  {glyph!r:6} {name:22} -> {dst!r} ({dst_name})")
        return

    if not args.input:
        parser.error("input level is required unless --show_map is given")

    out_rows = convert_level(load_level(args.input))
    output_text = "\n".join(out_rows) + "\n"
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Saved to: {args.output}")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
