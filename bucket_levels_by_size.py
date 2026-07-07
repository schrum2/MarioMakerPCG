#!/usr/bin/env python3
"""
bucket_levels_by_size.py
========================
Sort complete levels into fixed-size buckets for bucketed (variable-size)
training.

analyze_level_dimensions.py plots where the levels sit in width x height space;
this is the companion that acts on that picture. Given one or more size buckets
-- each a (min/max width, min/max height) box -- every complete level whose
content bounding box falls inside a bucket is encoded to tile ids, padded up to
that bucket's target size, and written to the bucket's own JSON. Pick the bucket
boxes off the scatter plot so each one captures a cluster of comparable-size
levels.

Each output file is the "Levels" stage (name + scene, no captions yet), exactly
like datasets/MM_Levels-regular.json, so the rest of the pipeline is unchanged:
caption the per-bucket files (MarioMaker_create_ascii_captions.py) and/or stitch
them together with combine_data.py, then train. The diffusion trainer's
BucketBatchSampler keeps every batch to a single scene size, so a merged file
that mixes several sizes trains fine.

The tile ids come from mm2pipeline.dataset.load_tileset, the same mapping
the windowed dataset builder uses, so a level encoded here lands in the exact
same id space as datasets/MM_Levels-regular.json (air " " = 0, unknown chars ->
the extra "_" tile).

The raw json->ascii export writes a direction-specific arrow for each pipe (see
mm2pipeline.ascii._PIPE_DIR_CHAR), but the training tileset keeps only the single
pipe glyph "|". Those arrows are folded back onto "|" before the id lookup (see
PIPE_ARROW_TO_GLYPH) so a pipe encodes as the pipe tile instead of the unknown
tile; on an export that already uses "|" the fold is a no-op.

Padding always keeps the level bottom-left aligned: air is added above and to
the right, so the ground stays on the bottom rows and the start of the level on
the left -- matching how mm2pipeline.dataset lays scenes out in a window.

A bucket's target size is rounded up to --size-multiple (default 4) because the
UNet halves the scene's width and height at each downsampling block; a target
that isn't a clean multiple can't be reconstructed by the matching upsampling
and the model fails to build. 4 covers the default three-level UNet (two halving
steps); deepen the net and you'll want 8, 16, ...

By default a level is bucketed whole, so any level wider than the largest bucket
is dropped. Pass --sliding_window to instead chop each level into consecutive,
NON-overlapping windows and bucket every window on its own: the whole level is
covered exactly once -- no overlap, no repeats -- so a level wider than the
largest bucket contributes several differently-sized windows (e.g. 500 wide ->
w240 + w240 + w48) instead of being dropped. The chop width defaults to the
widest bucket, so the leading windows fill the biggest bucket and only the
remainder spills into a smaller one.

Usage (default scheme -- the tuned MM2 buckets in DEFAULT_BUCKETS below):
    python bucket_levels_by_size.py --input <file-or-folder> --output_dir datasets/buckets

Usage (sliding window -- cover wide levels with non-overlapping bucket-sized windows):
    python bucket_levels_by_size.py --input <file-or-folder> --output_dir datasets/buckets \\
        --sliding_window

Usage (single bucket):
    python bucket_levels_by_size.py --input <file-or-folder> --output_dir datasets/buckets \\
        --min_w 33 --max_w 96 --min_h 1 --max_h 28

Usage (several buckets from a config file):
    python bucket_levels_by_size.py --input <file-or-folder> --output_dir datasets/buckets \\
        --buckets buckets.json

where buckets.json is a list of boxes, e.g.

    [
        {"name": "small",  "min_w": 1,   "max_w": 64,  "min_h": 1, "max_h": 28},
        {"name": "medium", "min_w": 65,  "max_w": 128, "min_h": 1, "max_h": 28},
        {"name": "large",  "min_w": 129, "max_w": 240, "min_h": 1, "max_h": 28}
    ]
"""

import argparse
import json
import os
import sys

# Reuse the exact level splitter, tileset loader and bounding-box measurement the
# rest of the pipeline uses, so "a level", "a tile id" and "a size" all mean the
# same thing here as they do in mm2pipeline.dataset / the scatter plot.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm2pipeline.dataset import (  # noqa: E402
    collect_input_files,
    parse_source_file,
    load_tileset,
    detect_empty_char,
    check_unmapped_chars,
    load_converter,
)
from analyze_level_dimensions import level_content_box  # noqa: E402
import util.common_settings as common_settings  # noqa: E402
import combine_data  # noqa: E402  (Fletcher's merge helper, reused for --merged_output)

# Warn about a level when this fraction of its characters aren't in the tileset;
# the same threshold mm2pipeline.dataset uses to flag a likely wrong tileset.
UNMAPPED_WARN_RATIO = 0.2

# The four pipe-direction arrows the json->ascii export emits (see
# mm2pipeline.ascii._PIPE_DIR_CHAR) all map to the one pipe glyph the tileset
# keeps, so a pipe encodes as "|" rather than collapsing to the unknown tile.
PIPE_ARROW_TO_GLYPH = str.maketrans({"→": "|", "←": "|", "↑": "|", "↓": "|"})

# Default bucket scheme, used when neither --buckets nor --max_w/--max_h is given.
# Tuned to the MM2 course distribution: horizontal courses cap at 240 tiles wide
# and 28 tall, so a single height (28) covers every level and the width is split
# into five even bands. On the full level set this keeps ~5 comparably-sized
# buckets and cuts padding to ~27% air (a single 240-wide bucket wastes ~58%).
# Override with --buckets for other data (e.g. tall vertical courses).
DEFAULT_BUCKETS = [
    {"name": "w48",  "min_w": 1,   "max_w": 48,  "min_h": 1, "max_h": 28},
    {"name": "w96",  "min_w": 49,  "max_w": 96,  "min_h": 1, "max_h": 28},
    {"name": "w144", "min_w": 97,  "max_w": 144, "min_h": 1, "max_h": 28},
    {"name": "w192", "min_w": 145, "max_w": 192, "min_h": 1, "max_h": 28},
    {"name": "w240", "min_w": 193, "max_w": 240, "min_h": 1, "max_h": 28},
]


def round_up(value, multiple):
    """Round value up to the nearest multiple (>=1). multiple<=1 leaves it as is."""
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def normalize_buckets(raw_buckets, size_multiple):
    """Validate the raw bucket dicts and attach each one's padded target size.

    Returns a list of dicts with name/min_w/max_w/min_h/max_h plus target_w and
    target_h (the max dims rounded up to size_multiple). Raises ValueError on a
    malformed bucket or on a size clash the width-only benchmark code can't tell
    apart (two buckets padded to the same width but a different height).
    """
    buckets = []
    for i, raw in enumerate(raw_buckets):
        try:
            max_w = int(raw["max_w"])
            max_h = int(raw["max_h"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Bucket {i} must define integer 'max_w' and 'max_h': {raw!r}")
        min_w = int(raw.get("min_w", 1))
        min_h = int(raw.get("min_h", 1))
        if min_w > max_w or min_h > max_h:
            raise ValueError(f"Bucket {i} has an empty range (min > max): {raw!r}")

        name = raw.get("name")
        target_w = round_up(max_w, size_multiple)
        target_h = round_up(max_h, size_multiple)
        buckets.append({
            "name": name,
            "min_w": min_w, "max_w": max_w,
            "min_h": min_h, "max_h": max_h,
            "target_w": target_w, "target_h": target_h,
        })

    # The trainer groups batches by scene size, but the benchmark/sample-width
    # bookkeeping is keyed on width alone. Two buckets that pad to the same width
    # but a different height would be indistinguishable there, so refuse it.
    width_to_height = {}
    for b in buckets:
        prev = width_to_height.get(b["target_w"])
        if prev is not None and prev != b["target_h"]:
            raise ValueError(
                f"Buckets disagree on height for padded width {b['target_w']}: "
                f"{prev} vs {b['target_h']}. Give each bucket a distinct width range "
                f"so every padded width maps to a single height."
            )
        width_to_height[b["target_w"]] = b["target_h"]

    return buckets


def load_bucket_specs(args):
    """Build the raw bucket list from --buckets, the single-bucket flags, or the default."""
    if args.buckets:
        with open(args.buckets, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            sys.exit(f"ERROR: {args.buckets} must be a non-empty JSON list of bucket objects.")
        return raw

    # Nothing specified at all: fall back to the tuned MM2 default scheme.
    if args.max_w is None and args.max_h is None:
        print("No --buckets or --max_w/--max_h given; using the built-in default scheme.")
        return DEFAULT_BUCKETS

    if args.max_w is None or args.max_h is None:
        sys.exit("ERROR: a single bucket needs BOTH --max_w and --max_h "
                 "(or pass --buckets, or omit both to use the default scheme).")
    return [{
        "min_w": args.min_w, "max_w": args.max_w,
        "min_h": args.min_h, "max_h": args.max_h,
    }]


def bucket_for(width, height, buckets):
    """Return the first bucket whose box contains (width, height), or None.

    First match wins, so overlapping buckets never duplicate a level; order the
    buckets from most to least specific if their boxes overlap.
    """
    for b in buckets:
        if b["min_w"] <= width <= b["max_w"] and b["min_h"] <= height <= b["max_h"]:
            return b
    return None


def partition_windows(box, window_w):
    """Chop a content box (list of equal-width strings) into consecutive,
    NON-overlapping windows at most window_w wide, so the whole level is covered
    exactly once -- every column is used, with no overlap and no repeats.

    The leading windows are window_w wide (sized for the widest bucket) and the
    final window is whatever remains, which then falls into a smaller bucket. A
    level no wider than window_w is returned unchanged as a single window. Each
    window is later re-trimmed to its own content box, so a window that lands on
    a sparse stretch shrinks into a smaller bucket rather than wasting padding.
    """
    width = len(box[0]) if box else 0
    if width <= window_w:
        return [box]

    return [[row[x:x + window_w] for row in box] for x in range(0, width, window_w)]


def encode_and_pad(box, target_w, target_h, tile_to_id, extra_id, empty_id):
    """Encode a content box (list of equal-width strings) to a target_h x target_w
    grid of tile ids, bottom-left aligned and air-padded on the top and right."""
    scene = [[tile_to_id.get(ch, extra_id) for ch in row] for row in box]

    content_h = len(scene)
    content_w = len(scene[0]) if scene else 0
    # The bucket filter guarantees the content already fits inside the target.
    assert content_w <= target_w and content_h <= target_h

    # Right-pad each content row, then drop in full air rows above it.
    padded = [row + [empty_id] * (target_w - content_w) for row in scene]
    air_row = [empty_id] * target_w
    return [list(air_row) for _ in range(target_h - content_h)] + padded


def main():
    parser = argparse.ArgumentParser(
        description="Sort complete levels into fixed-size buckets and write a padded scene JSON per bucket."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a .txt level file or a folder of .txt files "
                             "(the same input analyze_level_dimensions.py reads).")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write the per-bucket JSON files into.")
    parser.add_argument("--tileset", default=common_settings.MM2_TILESET,
                        help="Tileset JSON used to map characters to tile ids. "
                             f"Default: {common_settings.MM2_TILESET}.")
    parser.add_argument("--prefix", default="MM_Levels",
                        help="Filename prefix for the per-bucket JSONs. Default: MM_Levels.")
    parser.add_argument("--convert_to_extended", action="store_true",
                        help="Run each level through mm2view_to_extended.py before measuring/bucketing, "
                             "so the buckets land in the simplified extended_tiles.json id space (pass "
                             "--tileset extended_tiles.json with this). The converter fixes every level "
                             "to 20 rows tall and leaves the width free, so the width buckets still apply.")

    parser.add_argument("--buckets", default=None,
                        help="Path to a JSON list of bucket boxes "
                             "({min_w,max_w,min_h,max_h, optional name}). Overrides the single-bucket "
                             "flags. If neither this nor --max_w/--max_h is given, the built-in "
                             "DEFAULT_BUCKETS scheme is used.")
    parser.add_argument("--min_w", type=int, default=1, help="Single-bucket minimum width (inclusive).")
    parser.add_argument("--max_w", type=int, default=None, help="Single-bucket maximum width (inclusive).")
    parser.add_argument("--min_h", type=int, default=1, help="Single-bucket minimum height (inclusive).")
    parser.add_argument("--max_h", type=int, default=None, help="Single-bucket maximum height (inclusive).")

    parser.add_argument("--sliding_window", action="store_true",
                        help="Instead of bucketing each level as one scene, chop it into consecutive, "
                             "NON-overlapping windows and bucket each on its own (re-trimmed back to its "
                             "content box first). The whole level is covered exactly once -- no overlap, "
                             "no repeats -- so a level wider than the largest bucket contributes several "
                             "differently-sized windows (e.g. 500 wide -> w240 + w240 + w48) instead of "
                             "being dropped.")
    parser.add_argument("--window_w", type=int, default=None,
                        help="Window width in tiles for --sliding_window: the chop size for the leading "
                             "windows (the remainder falls into a smaller bucket). Default: the largest "
                             "bucket's max width, so the leading windows fill the biggest bucket.")

    parser.add_argument("--size_multiple", type=int, default=4,
                        help="Round each bucket's target width/height up to this multiple so the "
                             "UNet's downsampling divides evenly. Default: 4.")
    parser.add_argument("--empty_chars", default=" ",
                        help="Characters treated as sky/air when trimming the content box and when "
                             "padding. Default: a single space (the mm2_tileset_we air tile).")
    parser.add_argument("--merged_output", default=None,
                        help="If set, also merge every bucket's levels into this one JSON (via "
                             "combine_data.py) -- the single mixed-size file the trainer consumes.")
    args = parser.parse_args()

    try:
        buckets = normalize_buckets(load_bucket_specs(args), args.size_multiple)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    # Resolve the chop width: default to the widest bucket, so the leading
    # (full) windows fill the biggest bucket and only the remainder spills into a
    # smaller one. The partition is non-overlapping, so there is no stride.
    window_w = args.window_w
    if args.sliding_window:
        if window_w is None:
            window_w = max(b["max_w"] for b in buckets)
        if window_w < 1:
            sys.exit("ERROR: --window_w must be >= 1.")

    tile_to_id, extra_tile = load_tileset(args.tileset)
    extra_id = tile_to_id[extra_tile]
    empty_char = args.empty_chars[0]
    if empty_char not in tile_to_id:
        sys.exit(f"ERROR: air character {empty_char!r} is not in {args.tileset}; "
                 f"cannot decide the padding tile.")
    empty_id = tile_to_id[empty_char]

    print(f"Tileset {os.path.basename(args.tileset)}: {len(tile_to_id)} ids, "
          f"air {empty_char!r}=>{empty_id}, unknown chars =>{extra_id} ({extra_tile!r}).")
    print("Buckets:")
    for b in buckets:
        label = b["name"] or f"{b['target_w']}x{b['target_h']}"
        print(f"  [{label}] width {b['min_w']}-{b['max_w']}, height {b['min_h']}-{b['max_h']} "
              f"-> padded to {b['target_w']}x{b['target_h']}")
    if args.sliding_window:
        print(f"Sliding window: levels are chopped into non-overlapping {window_w}-wide windows "
              f"(remainder into a smaller bucket); the whole level is covered once, no repeats. "
              f"Each window is re-trimmed and bucketed on its own.")

    # Optionally collapse each raw mm2view level into the simplified extended
    # tile alphabet before it's measured and encoded, so the buckets land in the
    # extended_tiles.json id space rather than the full mm2 one.
    converter = None
    if args.convert_to_extended:
        converter = load_converter("mm2view_to_extended.py", "mm2view_to_extended")
        # Reduce onto the same tileset we measure/encode against, so the converter's
        # surviving glyphs are exactly this tileset's ids (extended_tiles_20.json ...).
        converter.set_target(args.tileset)
        print(f"Converting levels to the extended tile format ({os.path.basename(args.tileset)}) "
              f"via mm2view_to_extended.py before bucketing.")

    input_files = collect_input_files(args.input)
    multi = len(input_files) > 1

    for b in buckets:
        b["entries"] = []
    total = 0
    unmatched = 0
    empty_levels = 0
    total_chars = 0
    total_unmapped = 0

    for input_file in input_files:
        for name, rows in parse_source_file(input_file).items():
            total += 1
            if converter is not None:
                rows = converter.convert_level(rows)
            box = level_content_box(rows, empty_chars=args.empty_chars)
            if not box:
                empty_levels += 1
                continue

            full_name = f"{input_file.stem}/{name}" if multi else name

            # A whole level is one "window" by default; with --sliding_window a wide
            # level is chopped into consecutive, non-overlapping windows that cover it
            # exactly once. Each window is re-trimmed to its own content box (the chop
            # can leave fresh air at the top/sides) and then measured, bucketed, padded
            # and named exactly like a standalone level.
            if args.sliding_window:
                raw_windows = partition_windows(box, window_w)
            else:
                raw_windows = [box]
            multi_win = len(raw_windows) > 1

            for w_idx, raw in enumerate(raw_windows):
                sub = level_content_box(raw, empty_chars=args.empty_chars) if multi_win else raw
                if not sub:
                    continue  # window landed on an all-air gap; nothing to bucket

                width, height = len(sub[0]), len(sub)
                b = bucket_for(width, height, buckets)
                if b is None:
                    unmatched += 1
                    continue

                # Fold the pipe-direction arrows onto "|" before encoding (a 1:1 glyph
                # swap, so it doesn't disturb the measured size or bucket choice).
                sub = [row.translate(PIPE_ARROW_TO_GLYPH) for row in sub]

                # Keep an eye on how much falls outside the tileset; a high ratio
                # usually means the wrong --tileset was passed for this input.
                chars, unmapped, unmapped_set = check_unmapped_chars(sub, tile_to_id, extra_tile)
                total_chars += chars
                total_unmapped += unmapped
                win_name = f"{full_name}_w{w_idx}" if multi_win else full_name
                if chars and unmapped / chars > UNMAPPED_WARN_RATIO:
                    print(f"  [WARNING] {win_name}: {unmapped}/{chars} chars "
                          f"({unmapped / chars:.0%}) not in the tileset and collapsed to {extra_tile!r}. "
                          f"Unmapped: {' '.join(repr(c) for c in sorted(unmapped_set))}")

                scene = encode_and_pad(sub, b["target_w"], b["target_h"], tile_to_id, extra_id, empty_id)
                b["entries"].append({"name": win_name, "scene": scene})

    os.makedirs(args.output_dir, exist_ok=True)
    written = []
    print()
    for b in buckets:
        entries = b["entries"]
        suffix = b["name"] or f"{b['target_w']}x{b['target_h']}"
        if not entries:
            print(f"  [{suffix}] no levels matched; nothing written.")
            continue
        out_path = os.path.join(args.output_dir, f"{args.prefix}_{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        written.append(out_path)
        print(f"  [{suffix}] {len(entries)} level(s) -> {out_path}")

    # With sliding windows one level yields many entries, so "bucketed"/"outside"
    # count windows; without it they're per-level and match the old behaviour.
    matched = sum(len(b["entries"]) for b in buckets)
    unit = "window(s)" if args.sliding_window else "level(s)"
    print(f"\nScanned {total} level(s): {matched} {unit} bucketed, {unmatched} {unit} outside every "
          f"bucket, {empty_levels} empty.")
    if total_chars:
        print(f"Unmapped characters: {total_unmapped}/{total_chars} "
              f"({total_unmapped / total_chars:.1%}) across the bucketed levels.")

    if args.merged_output:
        # Stitch the per-bucket files into the single mixed-size training file with
        # Fletcher's merge helper, so it matches a hand-run combine_data.py exactly.
        if not written:
            sys.exit("ERROR: no levels matched any bucket, so there is nothing to merge.")
        combine_data.combine_json_files(args.merged_output, written)
    elif len(written) > 1:
        # The per-bucket files share a scene schema, so combine_data.py merges them
        # straight into the single mixed-size file the trainer consumes.
        merged = os.path.join(args.output_dir, f"{args.prefix}_merged.json")
        print("\nMerge the buckets into one training file with:")
        print(f"  python combine_data.py {merged} " + " ".join(written))


if __name__ == "__main__":
    main()
