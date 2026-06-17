#!/usr/bin/env python3
"""Reverse of mm2_json_to_ascii.py: turn an ASCII level grid (e.g. the output
of the diffusion model) back into a readable .json.

Usage:
    python mm2_ascii_to_json.py input_folder output_folder
    python mm2_ascii_to_json.py level.txt out/          # single file also works
"""
import json, argparse
from pathlib import Path

from mm2_json_to_ascii import OBJ_META, GROUND_CHAR

TILE = 160  # JSON coordinate units per tile (matches the forward script)


def build_char_to_name():
    """Invert OBJ_META char -> name. First name listed for a glyph wins, which
    yields sensible canonical picks (Ground for '#', Block for 'B', Hard Block
    for 'H', Piranha Plant for 'P', Big Mushroom for 'E', ...). '_unknown' and
    its '?' glyph are skipped here; unmapped glyphs fall back to '_unknown'."""
    char_to_name = {}
    for name, (char, _color, _cat) in OBJ_META.items():
        if name == "_unknown":
            continue
        char_to_name.setdefault(char, name)
    return char_to_name


CHAR_TO_NAME = build_char_to_name()


def parse_ascii(text):
    """Return (rows, width). Row 0 is the bottom of the level (game row 0),
    matching the forward script's `grid[max_ty - 1 - row_game]` layout."""
    lines = text.split("\n")
    # Drop only the single trailing empty line produced by the final "\n";
    # interior/legitimate blank rows are preserved.
    if lines and lines[-1] == "":
        lines.pop()
    width = max((len(l) for l in lines), default=0)
    # File is written top-to-bottom (highest game row first), so reverse it to
    # index from the bottom: rows[0] == game row 0.
    rows = list(reversed(lines))
    return rows, width


def ascii_to_level(text, source_file=None):
    rows, width = parse_ascii(text)
    height = len(rows)

    ground = []
    objects = []
    unknown_glyphs = {}

    for row_game, line in enumerate(rows):
        for col, ch in enumerate(line):
            if ch == " ":
                continue
            if ch == GROUND_CHAR:
                ground.append({"x": col, "y": row_game})
                continue
            name = CHAR_TO_NAME.get(ch)
            if name is None:
                name = "_unknown"
                unknown_glyphs[ch] = unknown_glyphs.get(ch, 0) + 1
            objects.append({
                "name": name.lower(),
                "x": col * TILE,
                "y": row_game * TILE,
                "w": 1,
                "h": 1,
            })

    level = {
        "is_overworld": False,            # suppress synthetic ground/goal on re-render
        "right_boundary": width * 16,     # forward script: max_tx = right_boundary // 16
        "top_boundary": height * 16,      #                  max_ty = top_boundary // 16
        "ground": ground,
        "objects": objects,
        "_note": ("Reconstructed from ASCII by mm2_ascii_to_json.py. Lossy: all "
                  "objects are 1x1 and names are the canonical glyph mapping."),
    }
    if source_file is not None:
        level["_source_file"] = str(source_file)
    if unknown_glyphs:
        level["_unknown_glyphs"] = unknown_glyphs
    return level


def convert_file(infile, outdir):
    text = Path(infile).read_text(encoding="utf-8")
    level = ascii_to_level(text, source_file=infile)
    outfile = Path(outdir) / f"{Path(infile).stem}.json"
    outfile.write_text(json.dumps(level, indent=2), encoding="utf-8")
    if level.get("_unknown_glyphs"):
        print(f"  warning: unmapped glyphs in {Path(infile).name}: "
              f"{level['_unknown_glyphs']}")


def main():
    ap = argparse.ArgumentParser(description="Convert ASCII Mario Maker grids back to JSON.")
    ap.add_argument("input", help="folder of .txt files, or a single .txt file")
    ap.add_argument("output_folder")
    args = ap.parse_args()

    outdir = Path(args.output_folder)
    outdir.mkdir(parents=True, exist_ok=True)

    inpath = Path(args.input)
    files = [inpath] if inpath.is_file() else sorted(inpath.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {args.input}")
        return

    for tf in files:
        try:
            convert_file(tf, outdir)
            print(f"Converted {tf.name}")
        except Exception as e:
            print(f"Failed {tf.name}: {e}")


if __name__ == "__main__":
    main()
