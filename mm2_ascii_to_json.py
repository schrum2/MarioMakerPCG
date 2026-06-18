#!/usr/bin/env python3
"""Reverse of mm2_json_to_ascii.py: turn an ASCII level grid (e.g. the output
of the diffusion model) back into a Mario Maker 2 level .json.

The output matches the base JSON schema that the rest of this toolchain speaks
(the same format toost.exe emits with --overworldJson and that json_to_bcd.py /
json_to_swe.py / mm2_viewer_json.py consume): a full level header, an `objects`
array with numeric `id` + flag fields, a `ground` array with `id`/`bid`, and the
empty entity arrays (track / clear_pipes / snakes / piranha_creepers).

It does NOT emit `drawing_instructions` -- those are produced downstream by
toost.exe when it renders the .bcd (json_to_bcd.py -> toost). The viewer falls
back to its rectangle renderer when they're absent, and toost regenerates them.

The conversion is lossy: ASCII is one glyph per tile, so every object comes back
as a 1x1 tile with baseline flags (orientation / wings / links / multi-tile
sizes are not recoverable). json_to_bcd.py re-merges/-grows blocks and platforms
(_fix_extended_objects / _fix_platform_objects) on the way to a .bcd.

Usage:
    python mm2_ascii_to_json.py input_folder output_folder
    python mm2_ascii_to_json.py level.txt out/          # single file also works
    python mm2_ascii_to_json.py in/ out/ --gamestyle smw --theme overworld
"""
import json, argparse
from pathlib import Path

from mm2_json_to_ascii import OBJ_META, GROUND_CHAR

TILE = 160          # JSON sub-pixel units per tile (160 = 1 tile)
TILE_CENTER = 80    # real .bcd objects anchor at the tile CENTER (col*160 + 80)
GROUND_TILE_PX = 16  # boundary fields are in pixels (16 px / tile), per toost

# Baseline object flag (0x6000040): the "normal", no-modifiers value seen on the
# vast majority of objects in real levels (see json_to_swe.py). Orientation,
# wings, parachutes, big-form, etc. can't be recovered from flat ASCII, so every
# reconstructed object gets the baseline.
DEFAULT_FLAG = 0x6000040

# OBJ_META display name -> MM2 object id (level.ksy obj_id enum; mirrors OBJ_ID
# in extract_mm2_bcd.py). Keyed by the canonical names in OBJ_META so the values
# CHAR_TO_NAME produces resolve directly. Gamestyle-variant aliases (Super Leaf,
# Link, Yoshi's Egg, ...) point at their shared slot id. "Ground" is intentionally
# absent: the '#' glyph goes to the `ground` array, never `objects`.
NAME_TO_ID = {
    # terrain / blocks
    "Block": 4, "Hard Block": 6, "Stone": 6, "? Block": 5, "Hidden Block": 29,
    "Note Block": 23, "Donut Block": 21, "Ice Block": 63, "P Block": 79,
    "ON/OFF Block": 99, "Dotted-Line Block": 100, "Blinking Block": 108,
    "Spike Block": 110, "Crate": 112, "Goal Ground": 26, "Starting Brick": 37,
    "Castle Bridge": 49, "Tree": 106, "Slight Slope": 87, "Steep Slope": 88,
    # doors / warps
    "Pipe": 9, "Door": 55, "Warp Box": 97, "Key": 95, "Checkpoint Flag": 90,
    "Goal": 27, "Clear Pipe": 93,
    # enemies
    "Goomba": 0, "Koopa": 1, "Piranha Plant": 2, "Piranha Flower": 2,
    "Hammer Bro": 3, "Thwomp": 12, "Bob-omb": 15, "Spiny": 25,
    "Buzzy Beetle": 28, "Lakitu": 30, "Lakitu's Cloud": 31, "Banzai Bill": 32,
    "Bullet Bill Blaster": 13, "Magikoopa": 39, "Spike Top": 40, "Boo": 41,
    "Bowser": 62, "Bowser Jr.": 98, "Bowser Jr": 98, "Chain Chomp": 61,
    "Cheep Cheep": 56, "Blooper": 48, "Wiggler": 52, "Pokey": 78,
    "Piranha Creeper": 107, "Porcupuffer": 114, "Fish Bone": 103,
    "Lava Bubble": 60, "Rocky Wrench": 58, "Muncher": 57, "Ant Trooper": 96,
    "Monty Mole": 102, "Mechakoopa": 111, "Boom Boom": 77, "Dry Bones": 46,
    "Skipsqueak": 51, "Stingby": 65, "Angry Sun": 104, "Charvaargh": 86,
    "Bully": 117, "Lemmy": 120, "Morton": 121, "Larry": 122, "Wendy": 123,
    "Iggy": 124, "Roy": 125, "Ludwig": 126,
    # items
    "Coin": 8, "Red Coin": 92, "1-Up Mushroom": 33, "1UP": 33,
    "Fire Flower": 34, "Super Star": 35, "Super Mushroom": 20,
    "Big Mushroom": 44, "SMB2 Mushroom": 81, "Super Leaf": 44,
    "Cape Feather": 44, "Propeller Mushroom": 44, "Link": 81, "Frog Suit": 81,
    "Power Balloon": 81, "Super Acorn": 81, "Super Hammer": 116, "Big Coin": 70,
    "P Switch": 18, "POW Block": 19, "POW": 19, "Spring": 10,
    "Goomba's Shoe": 45, "Yoshi's Egg": 45, "Cannon Box": 127,
    "Propeller Box": 128, "Goomba Mask": 129, "Bullet Bill Mask": 130,
    "Red POW Box": 131,
    # platforms
    "Lift": 11, "Mushroom Platform": 14, "Semisolid Platform": 16,
    "Bridge": 17, "Lava Lift": 36, "Snake Block": 84, "Track Block": 85,
    "Conveyor Belt": 94, "Fast Conveyor Belt": 53, "Sprint Platform": 80,
    "Seesaw": 91, "Swinging Claw": 105, "ON/OFF Trampoline": 132,
    "Mushroom Trampoline": 113, "Jumping Machine": 50,
    "Half-Collision Platform": 71, "Donut": 82,
    # hazards
    "Fire Bar": 24, "Saw": 68, "Burner": 54, "Spikes": 43, "Spike Ball": 74,
    "Skewer": 83, "Twister": 76, "Icicle": 118,
    # deco / other
    "Cloud": 22, "Vine": 64, "Water Marker": 101, "Arrow": 66,
    "One-Way Wall": 67, "One-Way": 67, "Reel Camera": 89, "Sound Effect": 109,
    "Player": 69, "Clown Car": 42, "Koopa Clown Car": 72, "Track": 59,
    "Starting Arrow": 38, "Cannon": 47, "! Block": 119,
}

GAMESTYLE_RAW = {
    "smb1": 12621, "smb3": 13133, "smw": 22349, "nsmbu": 21847, "sm3dw": 22323,
}
GAMESTYLE_NAME = {
    12621: "SMB1", 13133: "SMB3", 22349: "SMW", 21847: "NSMBU", 22323: "SM3DW",
}
THEME_RAW = {
    "overworld": 0, "underground": 1, "castle": 2, "airship": 3,
    "underwater": 4, "ghost": 5, "snow": 6, "desert": 7, "sky": 8, "forest": 9,
}
THEME_NAME = {
    0: "Ground", 1: "Underground", 2: "Castle", 3: "Airship", 4: "Underwater",
    5: "Ghost House", 6: "Snow", 7: "Desert", 8: "Sky", 9: "Forest",
}


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


def make_object(name, col, row):
    """Build a base-schema object entry for a 1x1 tile at (col, row).

    Coordinates use the real .bcd tile-center convention
    (x = col*160 + 80, y = row*160 + 80); json_to_bcd.py / mm2_viewer_json.py /
    json_to_swe.py all expect this. flags are the baseline value; cid/lid/sid
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
                # consumed from there by json_to_swe (build_metadata -> S1) and
                # json_to_bcd (pack_level_header). The forward script paints it
                # as a vertical column of 'G' glyphs -- mm2_json_to_ascii's
                # normalize_level injects an h=11 pole anchored at goal_x/goal_y.
                # So collect those cells and recover the pole's base below,
                # instead of emitting id=27 objects: json_to_swe drops id=27
                # outright (OBJ_ID_MAP[27] = None), which is exactly the "the end
                # doesn't make it into the .swe" symptom.
                goal_cells.append((col, row_game))
                continue
            objects.append(make_object(name, col, row_game))

    # Recover goal_x/goal_y from the painted flagpole. goal_x is stored in
    # TENTHS of a tile (both build_metadata in json_to_swe and normalize_level
    # in mm2_json_to_ascii compute goal_col = goal_x // 10); goal_y is the
    # pole's base row in whole tiles from the bottom. Goal is left-anchored and
    # its height grows upward, so the anchor is the left-most, bottom-most cell
    # -- take the min column / min row. With no 'G' glyphs the level has no goal
    # and both stay 0.
    if goal_cells:
        goal_col = min(c for c, _ in goal_cells)
        goal_row = min(r for _, r in goal_cells)
    else:
        goal_col = 0
        goal_row = 0

    stem = Path(source_file).stem if source_file else "level"

    level = {
        "name": stem,
        "description": "Reconstructed from ASCII by mm2_ascii_to_json.py",
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


def convert_file(infile, outdir, **kwargs):
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


def main():
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
            convert_file(tf, outdir, **kwargs)
            print(f"Converted {tf.name}")
        except Exception as e:
            print(f"Failed {tf.name}: {e}")


if __name__ == "__main__":
    main()
