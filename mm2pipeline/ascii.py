"""Convert between MM2 level JSON (toost's export) and the simplified ASCII grid
the diffusion model trains on.

Forward  (json -> ascii):  ``json_to_ascii_file`` / ``main_json_to_ascii``
Reverse  (ascii -> json):  ``ascii_to_level`` / ``ascii_to_json_file`` / ``main_ascii_to_json``

Both directions read their object metadata from ``mm2pipeline.tiles`` (built
around mm2_tileset_we.json). The forward path is lossy at the glyph level
(ASCII_REPLACEMENTS / ASCII_DROP); the reverse path is lossy at the object level
(every glyph comes back as a 1x1 baseline object). See the per-function notes.
"""
import json
import os
import argparse
from pathlib import Path

from .tiles import (
    GROUND_CHAR,
    ASCII_REPLACEMENTS,
    ASCII_DROP,
    CHAR_TO_NAME,
    NAME_TO_ID,
    GAMESTYLE_RAW,
    GAMESTYLE_NAME,
    THEME_RAW,
    THEME_NAME,
    resolve_obj_name,
    get_meta,
)

# ===========================================================================
# Forward: JSON -> ASCII   (was mm2_json_to_ascii.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Pipe direction helpers (flag % 0x80: 0x00=R, 0x20=L, 0x40=U, 0x60=D)
# ---------------------------------------------------------------------------
def _pipe_direction(flag: int) -> str:
    d = flag % 0x80
    if d == 0x00: return 'R'
    if d == 0x20: return 'L'
    if d == 0x40: return 'U'
    return 'D'

_PIPE_DIR_CHAR = {'R': '→', 'L': '←', 'U': '↑', 'D': '↓'}


# ---------------------------------------------------------------------------
# Tile size helper — uses w/h from JSON directly (already tile counts)
# ---------------------------------------------------------------------------
def obj_tile_size(obj: dict):
    """Return (w_tiles, h_tiles). The JSON w/h fields are direct tile counts.

    Pipes use h as the pipe length (C++ objH) regardless of direction;
    the cross-section is always 2 tiles wide/tall.
    """
    if obj.get("name") == "Pipe":
        direction = _pipe_direction(obj.get("flag", 0))
        length = max(1, obj.get("h", 1))
        if direction in ('U', 'D'):
            return 2, length
        else:
            return length, 2
    w = max(1, obj.get("w", 1))
    h = max(1, obj.get("h", 1))
    return w, h


# Objects whose x coordinate is the left-tile center (x = col*160 + 80).
# The C++ drawer uses the per-tile formula  j - 0.5 + x/160  for these,
# so  col = x // 160  is already correct — no w//2 correction.
# Everything else (Thwomp, Skewer, Lift, Saw, Arrow, Donut, …) uses the
# center-of-span formula  -w/2 + x/160  →  col = x//160 - w//2.
_LEFT_ANCHOR = frozenset({
    "Pipe",
    "Bridge",
    "Conveyor Belt",
    "Fast Conveyor Belt",
    "Mushroom Platform",
    "Semisolid Platform",
    "Slight Slope",
    "Steep Slope",
    "Half-Collision Platform",
    # synthetic objects injected by _normalize_level use tile coords directly
    "Ground",
    "Starting Brick",
    "Goal",
})


def obj_anchor(obj: dict):
    """Return (col, row) — bottom-left tile of the object.

    Left-anchor objects store x as the left-tile center (x = col*160 + 80)
    and are drawn per-tile with  j - 0.5 + x/160  in the C++ renderer.
    All other objects store x as the center of their full bounding span and
    are drawn with  -w/2 + x/160  — equivalent to  x//160 - w//2  for both
    even-width (x%160==0) and odd-width (x%160==80) cases.
    y is always the bottom-tile center for all JSON objects, so
    row = y // 160 is always correct.

    Pipes require direction-specific anchor adjustment derived from the C++
    rendering offsets for each direction case.
    """
    if obj.get("name") == "Pipe":
        direction = _pipe_direction(obj.get("flag", 0))
        base_col = obj["x"] // 160
        base_row = obj["y"] // 160
        w, h = obj_tile_size(obj)
        if direction == 'U':
            # columns [col, col+1], rows [base_row, base_row+h-1]
            return base_col, base_row
        elif direction == 'D':
            # x offset -1 tile; pipe extends downward (decreasing row)
            return base_col - 1, base_row - h + 1
        elif direction == 'R':
            # columns [col, col+w-1], rows [base_row, base_row]
            return base_col, base_row - 1
        else:  # L
            # pipe extends left; y stays the same
            return base_col - w + 1, base_row

    w, h = obj_tile_size(obj)
    x = obj["x"]
    if obj.get("name", "") in _LEFT_ANCHOR:
        col = x // 160
    else:
        col = x // 160 - w // 2
    row = obj["y"] // 160
    return col, row


# ---------------------------------------------------------------------------
# Level normalization + ASCII grid construction
# ---------------------------------------------------------------------------
_SLOPE_NAMES = frozenset({"Slight Slope", "Steep Slope"})


def normalize_level(lvl):
    if lvl.get("_normalized"):
        return
    lvl["_normalized"] = True

    objects = lvl.get("objects", [])

    for g in lvl.get("ground", []):
        objects.append({
            "name":"Ground","x":g["x"]*160,"y":g["y"]*160,"w":1,"h":1
        })

    if "is_overworld" in lvl:
        is_overworld = bool(lvl["is_overworld"])
    else:
        src_name = os.path.basename(lvl.get("_source_file","")).lower()
        is_overworld = "_subworld" not in src_name

    if not is_overworld:
        lvl["objects"] = objects
        return

    start_y = lvl.get("start_y",0)

    for col in range(0,7):
        for row in range(0,start_y):
            objects.append({
                "name":"Ground",
                "x":col*160,
                "y":row*160,
                "w":1,
                "h":1
            })

    goal_x = lvl.get("goal_x",0)
    goal_y = lvl.get("goal_y",0)
    goal_col = goal_x // 10

    is_castle = (lvl.get("theme_raw",-1)==2 or lvl.get("theme","")=="Castle")
    is_3dw = (lvl.get("gamestyle","")=="SM3DW" or lvl.get("gamestyle_raw",0)==22323)

    if is_castle and not is_3dw:
        objects.append({
            "name":"Goal","x":goal_col*160,"y":goal_y*160,"w":2,"h":4
        })
    else:
        objects.append({
            "name":"Goal","x":goal_col*160,"y":goal_y*160,"w":1,"h":11
        })

    for col in range(goal_col, goal_col+13):
        for row in range(0, goal_y):
            objects.append({
                "name":"Ground",
                "x":col*160,
                "y":row*160,
                "w":1,
                "h":1
            })

    lvl["objects"] = objects


def grid_bounds(lvl):
    top_b = lvl.get("top_boundary",0)
    right_b = lvl.get("right_boundary",0)
    if top_b > 0 and right_b > 0:
        max_tx = right_b // 16
        max_ty = top_b // 16
    else:
        max_tx,max_ty = 40,20
        for o in lvl.get("objects",[]):
            col,row = obj_anchor(o)
            w,h = obj_tile_size(o)
            max_tx = max(max_tx, col+w+1)
            max_ty = max(max_ty, row+h+1)
    return min(max_tx,240), min(max_ty,28)


def build_ascii_grid(level):
    objects = level.get("objects", [])
    max_tx,max_ty = grid_bounds(level)

    grid = [[" "] * max_tx for _ in range(max_ty)]

    def set_cell(col,row_game,ch):
        if 0 <= col < max_tx and 0 <= row_game < max_ty:
            grid[max_ty - 1 - row_game][col] = ch

    BG_TYPES = {"Semisolid Platform","Mushroom Platform"}

    for pass_n in range(2):
        for obj in objects:
            obj_name = obj.get("name","_unknown")

            # Objects with no tileset glyph are dropped entirely (writing an
            # empty string into cells would misalign the row on "".join()).
            if obj_name in ASCII_DROP:
                continue

            is_bg = obj_name in BG_TYPES
            if pass_n == 0 and not is_bg:
                continue
            if pass_n == 1 and is_bg:
                continue

            char,_,_ = get_meta(resolve_obj_name(obj_name, level.get("gamestyle_raw", 0)))

            if obj_name in ASCII_REPLACEMENTS:
                char = ASCII_REPLACEMENTS[obj_name]

            if obj_name in _SLOPE_NAMES:
                # Slopes have no flat-ASCII diagonal equivalent, so fill their
                # footprint as a solid ascending/descending staircase of
                # ground -- mirrors slope_fill_cells() in mm2pipeline.swe.
                col, row = obj_anchor(obj)
                w, h = obj_tile_size(obj)
                step = 2 if obj.get("id") == 87 else 1
                descending = (obj.get("flag", 0) & 0x100000) != 0
                for x in range(w):
                    run = (w - x) if descending else (x + 1)
                    height = min((run + step - 1) // step, h)
                    for y in range(height):
                        set_cell(col + x, row + y, GROUND_CHAR)
            elif obj_name == "Mushroom Platform":
                col,row = obj_anchor(obj)
                w,h = obj_tile_size(obj)
                sc = col + w // 2
                # stem: centered column, all rows below cap
                for dy in range(h - 1):
                    set_cell(sc, row + dy, char)
                # cap: full width at top row
                for dx in range(w):
                    set_cell(col + dx, row + h - 1, char)
            elif obj_name == "Bridge":
                col,row = obj_anchor(obj)
                w,_ = obj_tile_size(obj)
                # Only the bottom (walkable) row is kept; the rope/chain
                # row above it is decorative and gets dropped.
                for dx in range(w):
                    set_cell(col+dx,row,char)
            elif obj_name == "Big Coin":
                # Draw the Big Coin as a 2x2 cluster of regular coins so it
                # reads as "more than one coin" in the grid.
                col,row = obj_anchor(obj)
                coin_char,_,_ = get_meta("Coin")
                for dx in range(2):
                    for dy in range(2):
                        set_cell(col+dx,row+dy,coin_char)
            else:
                col,row = obj_anchor(obj)
                w,h = obj_tile_size(obj)

                for dx in range(w):
                    for dy in range(h):
                        set_cell(col+dx,row+dy,char)

    return ["".join(r).rstrip() for r in grid]


def json_to_ascii_file(infile, outdir):
    data = json.loads(Path(infile).read_text(encoding="utf-8"))
    levels = data if isinstance(data, list) else [data]

    for lvl in levels:
        lvl.setdefault("_source_file", str(infile))
        normalize_level(lvl)

    for idx,lvl in enumerate(levels, start=1):
        stem = Path(infile).stem
        suffix = f"_{idx}" if len(levels) > 1 else ""
        outfile = Path(outdir) / f"{stem}{suffix}.txt"
        outfile.write_text("\n".join(build_ascii_grid(lvl)) + "\n", encoding="utf-8")


def main_json_to_ascii():
    ap = argparse.ArgumentParser(description="Convert MM2 level JSON to ASCII grids.")
    ap.add_argument("input_folder")
    ap.add_argument("output_folder")
    args = ap.parse_args()

    outdir = Path(args.output_folder)
    outdir.mkdir(parents=True, exist_ok=True)

    for jf in sorted(Path(args.input_folder).glob("*.json")):
        try:
            json_to_ascii_file(jf, outdir)
            print(f"Converted {jf.name}")
        except Exception as e:
            print(f"Failed {jf.name}: {e}")


# ===========================================================================
# Reverse: ASCII -> JSON   (was mm2_ascii_to_json.py)
# ===========================================================================

TILE = 160          # JSON sub-pixel units per tile (160 = 1 tile)
TILE_CENTER = 80    # real .bcd objects anchor at the tile CENTER (col*160 + 80)
GROUND_TILE_PX = 16  # boundary fields are in pixels (16 px / tile), per toost

# Baseline object flag (0x6000040): the "normal", no-modifiers value seen on the
# vast majority of objects in real levels (see mm2pipeline.swe). Orientation,
# wings, parachutes, big-form, etc. can't be recovered from flat ASCII, so every
# reconstructed object gets the baseline.
DEFAULT_FLAG = 0x6000040


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


def make_object(name, col, row):
    """Build a base-schema object entry for a 1x1 tile at (col, row).

    Coordinates use the real .bcd tile-center convention
    (x = col*160 + 80, y = row*160 + 80); json_to_bcd.py / mm2_viewer_json.py /
    mm2pipeline.swe all expect this. flags are the baseline value; cid/lid/sid
    are unlinked (-1)."""
    return {
        "name": name,
        "x": col * TILE + TILE_CENTER,
        "y": row * TILE + TILE_CENTER,
        "w": 1,
        "h": 1,
        "flag": DEFAULT_FLAG,
        "cflag": DEFAULT_FLAG,
        "ex": 0,
        "id": NAME_TO_ID[name],
        "cid": -1,
        "lid": -1,
        "sid": -1,
        "link_type": 0,
    }


def ascii_to_level(text, source_file=None, *, gamestyle_raw=22349, theme_raw=0,
                   timer=300):
    rows, width = parse_ascii(text)
    height = len(rows)

    ground = []
    objects = []
    unknown_glyphs = {}
    goal_cells = []

    for row_game, line in enumerate(rows):
        for col, ch in enumerate(line):
            if ch == " ":
                continue
            if ch == GROUND_CHAR:
                # SMM2 ground autotile id/bid aren't recoverable from ASCII;
                # toost re-derives the tile graphic from tile occupancy, so 0/0
                # loads and renders as solid ground.
                ground.append({"x": col, "y": row_game, "id": 0, "bid": 0})
                continue
            name = CHAR_TO_NAME.get(ch)
            if name is None or name not in NAME_TO_ID:
                unknown_glyphs[ch] = unknown_glyphs.get(ch, 0) + 1
                continue
            if name == "Goal":
                # The goal/flagpole is LEVEL METADATA in this schema, not an
                # objects[] entry: it lives in the header's goal_x/goal_y and is
                # consumed from there by mm2pipeline.swe (build_metadata -> S1)
                # and json_to_bcd (pack_level_header). The forward path paints it
                # as a vertical column of 'G' glyphs -- normalize_level injects an
                # h=11 pole anchored at goal_x/goal_y. So collect those cells and
                # recover the pole's base below, instead of emitting id=27
                # objects: mm2pipeline.swe drops id=27 outright
                # (OBJ_ID_MAP[27] = None), which is exactly the "the end doesn't
                # make it into the .swe" symptom.
                goal_cells.append((col, row_game))
                continue
            objects.append(make_object(name, col, row_game))

    # Recover goal_x/goal_y from the painted flagpole. goal_x is stored in
    # TENTHS of a tile (both build_metadata in mm2pipeline.swe and
    # normalize_level compute goal_col = goal_x // 10); goal_y is the pole's base
    # row in whole tiles from the bottom. Goal is left-anchored and its height
    # grows upward, so the anchor is the left-most, bottom-most cell -- take the
    # min column / min row. With no 'G' glyphs the level has no goal and both
    # stay 0.
    if goal_cells:
        goal_col = min(c for c, _ in goal_cells)
        goal_row = min(r for _, r in goal_cells)
    else:
        goal_col = 0
        goal_row = 0

    stem = Path(source_file).stem if source_file else "level"

    level = {
        "name": stem,
        "description": "Reconstructed from ASCII by mm2pipeline.ascii",
        "gamestyle": GAMESTYLE_NAME.get(gamestyle_raw, "SMW"),
        "gamestyle_raw": gamestyle_raw,
        "theme": THEME_NAME.get(theme_raw, "Ground"),
        "theme_raw": theme_raw,
        # is_overworld=False suppresses the synthetic start/goal ground that
        # normalize_level() injects for overworld maps -- the generated level
        # carries no real goal/start metadata, so injecting one would corrupt it.
        "is_overworld": False,
        "night_time": False,
        "clear_time": 0,
        "clear_attempts": 0,
        "game_version": "0.0.0",
        "game_version_raw": 0,
        "timer": timer,
        "start_y": 0,
        "goal_x": goal_col * 10,
        "goal_y": goal_row,
        "clear_condition_type": "None",
        "clear_condition_type_raw": 0,
        "clear_condition_magnitude": 0,
        "clear_condition": "None",
        "clear_condition_category": "None",
        "clear_condition_category_raw": 0,
        "autoscroll_speed": "x1",
        "autoscroll_speed_raw": 0,
        "autoscroll_type": "None",
        "autoscroll_type_raw": 0,
        "orientation": "Horizontal",
        "orientation_raw": 0,
        "liquid_start_height": 0,
        "liquid_end_height": 0,
        "liquid_mode": "None",
        "liquid_speed": "x1",
        "boundary_type": "Built Above Line",
        "liquid_mode_raw": 0,
        "liquid_speed_raw": 0,
        "boundary_type_raw": 0,
        # Boundaries are in pixels (16 px / tile), per toost (mm2_viewer_json's
        # _grid_bounds: max_tx = right_boundary // 16).
        "right_boundary": width * GROUND_TILE_PX,
        "top_boundary": height * GROUND_TILE_PX,
        "left_boundary": 0,
        "bottom_boundary": 0,
        "object_count": len(objects),
        "ground_count": len(ground),
        "upload_id": 0,
        "creation_id": 0,
        "gamever": 0,
        "management_flags": 0,
        "objects": objects,
        "ground": ground,
        "track": [],
        "clear_pipes": [],
        "snakes": [],
        "piranha_creepers": [],
    }
    if source_file is not None:
        level["_source_file"] = str(source_file)
    if unknown_glyphs:
        level["_unknown_glyphs"] = unknown_glyphs
    return level


def ascii_to_json_file(infile, outdir, **kwargs):
    # utf-8-sig transparently strips a leading BOM if one is present.
    text = Path(infile).read_text(encoding="utf-8-sig")
    level = ascii_to_level(text, source_file=infile, **kwargs)
    # toost / json_to_bcd discover the subworld companion by an _overworld /
    # _subworld suffix; tag generated levels as overworld so the pipeline picks
    # them up (json_to_bcd.find_companion).
    stem = Path(infile).stem
    if not (stem.endswith("_overworld") or stem.endswith("_subworld")):
        stem += "_overworld"
    outfile = Path(outdir) / f"{stem}.json"
    outfile.write_text(json.dumps(level, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    if level.get("_unknown_glyphs"):
        print(f"  warning: unmapped glyphs in {Path(infile).name}: "
              f"{level['_unknown_glyphs']}")


def main_ascii_to_json():
    ap = argparse.ArgumentParser(description="Convert ASCII Mario Maker grids back to MM2 JSON.")
    ap.add_argument("input", help="folder of .txt files, or a single .txt file")
    ap.add_argument("output_folder")
    ap.add_argument("--gamestyle", choices=sorted(GAMESTYLE_RAW), default="smw",
                    help="game style for the rebuilt level (default: smw)")
    ap.add_argument("--theme", choices=sorted(THEME_RAW), default="overworld",
                    help="course theme (default: overworld)")
    ap.add_argument("--timer", type=int, default=300, help="level timer (default: 300)")
    args = ap.parse_args()

    outdir = Path(args.output_folder)
    outdir.mkdir(parents=True, exist_ok=True)

    inpath = Path(args.input)
    files = [inpath] if inpath.is_file() else sorted(inpath.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {args.input}")
        return

    kwargs = dict(
        gamestyle_raw=GAMESTYLE_RAW[args.gamestyle],
        theme_raw=THEME_RAW[args.theme],
        timer=args.timer,
    )
    for tf in files:
        try:
            ascii_to_json_file(tf, outdir, **kwargs)
            print(f"Converted {tf.name}")
        except Exception as e:
            print(f"Failed {tf.name}: {e}")


# Running the module directly defaults to the forward (json -> ascii) direction,
# matching the historical `python mm2_json_to_ascii.py` entrypoint.
def main():
    main_json_to_ascii()


if __name__ == "__main__":
    main()
