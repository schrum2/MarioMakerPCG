"""
Rebuild a .bcd course file from the JSON exported by Toost
(toost.exe --overworldJson / --subworldJson, see mm2pipeline.toost).

Toost exports one JSON file per map (overworld/subworld), each duplicating
the level-wide header fields (name, description, gamestyle, clear
conditions, timer, goal, etc). To rebuild a full Level (per level.ksy) we
need both *_overworld.json and *_subworld.json; if only one is given, the
other map is written out empty.

Field layout follows level.ksy exactly (376768-byte payload =
512-byte level header + two 188128-byte maps).

A few fields aren't in Toost's JSON and get written back as 0, so the
output won't match the original .bcd byte-for-byte: header unk1/unk2 and
creation date, per-object unk1, per-map unk_flag bits besides night_time
and unk1/unk2 padding, the sound/exclamation/track-block/icicle arrays
(no JSON for these), and the clear-pipe "unk" marker (set to 1 if any
pipe is present, else 0).

The result loads fine in Toost / SMM2 but is a best-effort reconstruction,
not a byte-for-byte round trip.

Usage
-----
    python json_to_bcd.py bcd_levels/json/<id>_overworld.json
    python json_to_bcd.py bcd_levels/json/<id>_overworld.json -o out/<id>.bcd

    # Drop/clamp objects this build of toost can't render (see toost_compat.py),
    # so the resulting .bcd can be previewed with toost without crashing:
    python json_to_bcd.py bcd_levels/json/<id>_overworld.json --toost-compat
"""

import argparse
import json
import struct
from pathlib import Path

from mm2pipeline.bcd import build_bcd, PAYLOAD_SIZE

# ---------------------------------------------------------------------------
# Fixed array sizes / element sizes, per level.ksy
# ---------------------------------------------------------------------------

OBJ_MAX           = 2600
SOUND_MAX         = 300
SNAKE_MAX         = 5
CLEAR_PIPE_MAX    = 200
PIRANHA_MAX       = 10
EXCLAMATION_MAX   = 10
TRACK_BLOCK_MAX   = 10
GROUND_MAX        = 4000
TRACK_MAX         = 1500
ICICLE_MAX        = 300

OBJ_SIZE          = 32   # x,y(s4*2) unk1(s2) w,h(u1*2) flag,cflag,ex(s4*3) id,cid,lid,sid(s2*4)
SOUND_SIZE        = 4
SNAKE_NODE_SIZE   = 8    # index,direction(u2*2) unk1(u4)
SNAKE_SIZE        = 4 + 120 * SNAKE_NODE_SIZE
CLEAR_PIPE_NODE_SIZE = 8  # type,index,x,y,width,height,unk1,direction (u1*8)
CLEAR_PIPE_SIZE   = 4 + 36 * CLEAR_PIPE_NODE_SIZE
PIRANHA_NODE_SIZE = 4    # unk1,direction(u1*2) unk2(u2)
PIRANHA_SIZE      = 4 + 20 * PIRANHA_NODE_SIZE
EXCLAMATION_SIZE  = 4 + 10 * 4
TRACK_BLOCK_SIZE  = 4 + 10 * 4
GROUND_SIZE       = 4
TRACK_SIZE        = 12   # unk1(u2) flags,x,y,type(u1*4) lid,unk2,unk3(u2*3)
ICICLE_SIZE       = 4
MAP_UNK2_SIZE     = 0xDBC  # 3516

LEVEL_HEADER_SIZE = 512
MAP_HEADER_SIZE   = 72


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def pack_str_utf16(s, size_bytes):
    raw = (s or "").encode("utf-16-le")
    # Toost reads these fields as null-terminated char16_t* strings, so
    # always leave room for a trailing u"\x00" even when truncating.
    max_bytes = size_bytes - 2
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        if len(raw) % 2:
            raw = raw[:-1]
    return raw.ljust(size_bytes, b"\x00")


def _snap_to_tile_anchor(value, offset=80, tile=160):
    """Round `value` to the nearest grid position satisfying value % tile == offset."""
    return (value - offset + tile // 2) // tile * tile + offset


# ---------------------------------------------------------------------------
# Level header (512 bytes)
# ---------------------------------------------------------------------------

def pack_level_header(j):
    goal_x = j.get("goal_x", 0)
    goal_y = j.get("goal_y", 0)
    right_boundary = j.get("right_boundary", 0)
    # Toost stores goal_y in whole tiles and goal_x in tenths of a tile.
    # Some exporters report both in object-coordinate units (160/tile)
    # instead, which overflows goal_y's u1 field - rescale back to raw units.
    if goal_y > 255:
        goal_x //= 160
        goal_y //= 160
    elif (
        right_boundary
        and goal_x // 10 > right_boundary // 16
        and goal_x // 160 <= right_boundary // 16
    ):
        # Other exporters get goal_y right but still leave goal_x in
        # 160/tile units. Reading it as tenths-of-a-tile would place the
        # goal past the right boundary, so rescale to 10/tile instead.
        goal_x = (goal_x // 160) * 10

    # goal_x sits on the tile-center grid (col*10 + 5); snap stragglers
    # that land on a tile boundary instead.
    goal_x = _snap_to_tile_anchor(goal_x, offset=5, tile=10)

    fixed = struct.pack(
        "<BBhhhhbbbbBBiiiiiIqi",
        j.get("start_y", 0),
        goal_y,
        goal_x,
        j.get("timer", 0),
        j.get("clear_condition_magnitude", 0),
        0,  # year
        0,  # month
        0,  # day
        0,  # hour
        0,  # minute
        j.get("autoscroll_speed_raw", 0),
        j.get("clear_condition_category_raw", 0),
        j.get("clear_condition_type_raw", 0),
        j.get("gamever", 0),
        j.get("management_flags", 0),
        j.get("clear_attempts", 0),
        j.get("clear_time", 0),
        j.get("creation_id", 0),
        j.get("upload_id", 0),
        j.get("game_version_raw", 0),
    )
    out = (
        fixed
        + b"\x00" * 189  # unk1
        + struct.pack("<h", j.get("gamestyle_raw", 0))
        + b"\x00"  # unk2
        + pack_str_utf16(j.get("name", ""), 66)
        + pack_str_utf16(j.get("description", ""), 202)
    )
    assert len(out) == LEVEL_HEADER_SIZE, len(out)
    return out


# ---------------------------------------------------------------------------
# Map entities
# ---------------------------------------------------------------------------

def pack_obj(o):
    return struct.pack(
        "<iihBBiiihhhh",
        o.get("x", 0),
        o.get("y", 0),
        0,  # unk1, not exported by toost
        o.get("w", 0),
        o.get("h", 0),
        o.get("flag", 0),
        o.get("cflag", 0),
        o.get("ex", 0),
        o.get("id", 0),
        o.get("cid", -1),
        o.get("lid", -1),
        o.get("sid", -1),
    )


def pack_ground(g):
    return struct.pack("<BBBB", g.get("x", 0), g.get("y", 0), g.get("id", 0), g.get("bid", 0))


def pack_snake(s):
    nodes = s.get("nodes", [])
    out = struct.pack("<BBH", s.get("index", 0), s.get("node_count", len(nodes)), 0)
    for i in range(120):
        if i < len(nodes):
            n = nodes[i]
            out += struct.pack("<HHI", n.get("index", 0), n.get("direction", 0), 0)
        else:
            out += b"\x00" * SNAKE_NODE_SIZE
    return out


def pack_clear_pipe(cp):
    nodes = cp.get("nodes", [])
    # "unk" is used by toost as an "is this slot populated" marker; any
    # non-zero value works, so use 1 for pipes present in the JSON.
    out = struct.pack("<BBH", cp.get("index", 0), cp.get("node_count", len(nodes)), 1)
    for i in range(36):
        if i < len(nodes):
            n = nodes[i]
            out += struct.pack(
                "<BBBBBBBB",
                n.get("type", 0),
                n.get("index", 0),
                n.get("x", 0),
                n.get("y", 0),
                n.get("w", 0),
                n.get("h", 0),
                0,  # unk1
                n.get("direction", 0),
            )
        else:
            out += b"\x00" * CLEAR_PIPE_NODE_SIZE
    return out


def pack_piranha_creeper(c):
    nodes = c.get("nodes", [])
    out = struct.pack("<BBBB", 0, c.get("index", 0), c.get("node_count", len(nodes)), 0)
    for i in range(20):
        if i < len(nodes):
            out += struct.pack("<BBH", 0, int(nodes[i]), 0)
        else:
            out += b"\x00" * PIRANHA_NODE_SIZE
    return out


def pack_track(t):
    x = t.get("x", 0)
    y = t.get("y", 0)
    # Inverse of toost's TX==255 -> 0, else TX+1 transform.
    raw_x = 255 if x == 0 else (x - 1) & 0xFF
    raw_y = 255 if y == 0 else (y - 1) & 0xFF
    return struct.pack(
        "<HBBBBHHH",
        t.get("un", 0),
        t.get("flag", 0),
        raw_x,
        raw_y,
        t.get("type", 0),
        t.get("lid", 0),
        t.get("k0", 0),
        t.get("k1", 0),
    )


# ---------------------------------------------------------------------------
# Map (188128 bytes)
# ---------------------------------------------------------------------------

def _pack_array(items, max_count, size, pack_fn, label):
    if len(items) > max_count:
        raise ValueError(f"too many {label}: {len(items)} > {max_count}")
    out = bytearray()
    for i in range(max_count):
        out += pack_fn(items[i]) if i < len(items) else b"\x00" * size
    return bytes(out)


# Left-anchored (x = left_col*160 + 80) rather than center-of-span;
# _fix_object_anchors snaps these straight to the grid instead of applying
# the odd-width naive/real shift.
LEFT_ANCHOR_OBJ_IDS = {
    9,   # Pipe
    27,  # Goal
}

# Always placed as a single 1x1 tile in SMM2. A generator emitting one of
# these with w>1/h>1 really means a stack/row of individual blocks; see
# _fix_extended_objects.
ATOMIC_BLOCK_IDS = {
    4,    # Block
    5,    # ? Block
    6,    # Hard Block
    21,   # Donut Block
    23,   # Note Block
    29,   # Hidden Block
    63,   # Ice Block
    79,   # P Block
    99,   # ON/OFF Block
    100,  # Dotted-Line Block
    108,  # Blinking Block
    110,  # Spike Block
}

# "Stretchy" platforms whose w/h directly describe their footprint: Mushroom
# Platform draws a stem for the lower rows plus a full-width cap on top;
# Semisolid/Half-Collision fill the whole w x h box (see build_ascii_grid in
# mm2pipeline.ascii). SMM2 won't place any of these smaller than 3x3; see
# _fix_platform_objects.
PLATFORM_FILL_IDS = {14, 16, 71}  # Mushroom / Semisolid / Half-Collision Platform
PLATFORM_MIN_SIZE = 3


def _fix_object_anchors(objects, label="map"):
    """Real .bcd objects store x/y as the CENTER of their tile footprint:
        x = (left_col + w/2) * 160
        y = bottom_row * 160 + 80
    Some exporters instead use a naive "x = col*160, y = row*160" grid with
    no center offset, which lands the object in the right tile but draws it
    floating half a tile off the ground grid. Detect that from odd-width
    objects (where naive vs. real differ mod 160) and correct both axes.

    Pipe/Goal (LEFT_ANCHOR_OBJ_IDS) are left-anchored, not center-of-span,
    so they skip the above and just get snapped onto the tile grid.
    """
    odd_w = [o for o in objects if o.get("id") not in LEFT_ANCHOR_OBJ_IDS and o.get("w", 1) % 2 == 1]
    x_naive = bool(odd_w) and sum(1 for o in odd_w if o.get("x", 0) % 160 == 0) > len(odd_w) // 2

    fixed = []
    n_x = n_y = n_left = 0
    for o in objects:
        o = dict(o)
        if o.get("id") in LEFT_ANCHOR_OBJ_IDS:
            new_x = _snap_to_tile_anchor(o.get("x", 0))
            new_y = _snap_to_tile_anchor(o.get("y", 0))
            if new_x != o.get("x", 0) or new_y != o.get("y", 0):
                o["x"], o["y"] = new_x, new_y
                n_left += 1
            fixed.append(o)
            continue

        if x_naive:
            o["x"] = o.get("x", 0) + o.get("w", 1) * 80
            n_x += 1
        if o.get("y", 0) % 160 == 0:
            o["y"] = o.get("y", 0) + 80
            n_y += 1
        fixed.append(o)

    if n_x or n_y or n_left:
        msg = (f"  [{label}] toost-anchor fix: shifted {n_x} object(s) on X, "
               f"{n_y} object(s) on Y onto toost's tile-center grid")
        if n_left:
            msg += f"; snapped {n_left} pipe/goal object(s) onto the tile grid"
        print(msg)
    return fixed


def _fix_extended_objects(objects, label="map"):
    """Split any ATOMIC_BLOCK_IDS object with w>1/h>1 into a grid of 1x1
    objects, one per tile. Some generators emit e.g. a single w=1,h=4 Hard
    Block instead of 4 stacked 1x1 blocks; toost would draw that as one
    sprite stretched over the footprint. Assumes x/y are already normalized
    by _fix_object_anchors.
    """
    fixed = []
    n_split = n_total = 0
    for o in objects:
        w = o.get("w", 1) or 1
        h = o.get("h", 1) or 1
        if o.get("id") not in ATOMIC_BLOCK_IDS or (w <= 1 and h <= 1):
            fixed.append(o)
            continue

        left_col = o.get("x", 0) // 160 - w // 2
        bottom_row = o.get("y", 0) // 160
        for dx in range(w):
            for dy in range(h):
                tile = dict(o)
                tile["w"] = 1
                tile["h"] = 1
                tile["x"] = (left_col + dx) * 160 + 80
                tile["y"] = (bottom_row + dy) * 160 + 80
                fixed.append(tile)
        n_split += 1
        n_total += w * h

    if n_split:
        print(f"  [{label}] extended-object fix: split {n_split} object(s) "
              f"into {n_total} 1x1 tile(s)")
    return fixed


def _col_w_to_x(col, w):
    """Inverse of  col = x // 160 - w // 2  (the center-of-span convention
    used after _fix_object_anchors)."""
    return (col + w // 2) * 160 + (80 if w % 2 else 0)


def _row_to_y(row):
    """Inverse of  row = y // 160  (y is always row*160 + 80)."""
    return row * 160 + 80


def _fix_platform_objects(objects, ground, label="map"):
    """Merge adjacent 1x1 PLATFORM_FILL_IDS "cap" objects on the same row
    into one, then grow anything smaller than SMM2's 3x3 minimum downward.

    Some generators place only a platform's top-row "cap" with no
    body/stem beneath it. Columns are added symmetrically, rows are added
    below the cap (keeping its walkable top where it was placed) up to the
    3x3 minimum, then further until the platform actually rests on ground.
    """
    ground_set = {(g.get("x"), g.get("y")) for g in ground}

    others = []
    groups = {}
    for o in objects:
        if o.get("id") not in PLATFORM_FILL_IDS:
            others.append(o)
            continue
        w = o.get("w", 1) or 1
        col = o.get("x", 0) // 160 - w // 2
        row = o.get("y", 0) // 160
        groups.setdefault((o.get("id"), row), []).append((col, w, o))

    fixed = list(others)
    n_merged = n_grown = 0
    for (oid, row), entries in groups.items():
        entries.sort(key=lambda e: e[0])
        runs = []
        for col, w, o in entries:
            h = o.get("h", 1) or 1
            if (h == 1 and runs and runs[-1]["h"] == 1
                    and runs[-1]["end"] == col):
                runs[-1]["end"] = col + w
                runs[-1]["count"] += 1
            else:
                runs.append({"start": col, "end": col + w, "h": h,
                              "count": 1, "obj": o})

        for run in runs:
            o = dict(run["obj"])
            col, w, h = run["start"], run["end"] - run["start"], run["h"]
            cap_row = row + h - 1

            if run["count"] > 1:
                n_merged += run["count"] - 1
            if w < PLATFORM_MIN_SIZE or h < PLATFORM_MIN_SIZE:
                n_grown += 1

            if w < PLATFORM_MIN_SIZE:
                col -= (PLATFORM_MIN_SIZE - w) // 2
                w = PLATFORM_MIN_SIZE
            if h < PLATFORM_MIN_SIZE:
                new_row = max(0, cap_row - (PLATFORM_MIN_SIZE - 1))
                h = cap_row - new_row + 1
                while new_row > 0 and any(
                        (c, new_row - 1) not in ground_set
                        for c in range(col, col + w)):
                    new_row -= 1
                    h += 1
            else:
                new_row = row

            o["x"] = _col_w_to_x(col, w)
            o["y"] = _row_to_y(new_row)
            o["w"] = w
            o["h"] = h
            fixed.append(o)

    if n_merged or n_grown:
        print(f"  [{label}] platform-fill fix: merged {n_merged} cap object(s), "
              f"grew {n_grown} platform object(s) to the "
              f"{PLATFORM_MIN_SIZE}x{PLATFORM_MIN_SIZE} minimum "
              f"(extended further to clear voids beneath them)")
    return fixed


def pack_map(j, label="map"):
    ground           = j.get("ground", [])
    objects          = _fix_object_anchors(j.get("objects", []), label)
    objects          = _fix_platform_objects(objects, ground, label)
    objects          = _fix_extended_objects(objects, label)
    snakes           = j.get("snakes", [])
    clear_pipes      = j.get("clear_pipes", [])
    piranha_creepers = j.get("piranha_creepers", [])
    tracks           = j.get("track", [])

    night_time = j.get("night_time", False)

    header = struct.pack(
        "<BBBBBBBB",
        j.get("theme_raw", 0),
        j.get("autoscroll_type_raw", 0),
        j.get("boundary_type_raw", 0),
        j.get("orientation_raw", 0),
        j.get("liquid_end_height", 0),
        j.get("liquid_mode_raw", 0),
        j.get("liquid_speed_raw", 0),
        j.get("liquid_start_height", 0),
    ) + struct.pack(
        "<iiiiiiiiiiiiiiii",
        j.get("right_boundary", 0),
        j.get("top_boundary", 0),
        j.get("left_boundary", 0),
        j.get("bottom_boundary", 0),
        1 if night_time else 0,  # unk_flag
        len(objects),
        0,  # sound_effect_count (no sounds array in JSON)
        len(snakes),
        j.get("clear_pipe_count", len(clear_pipes)),
        len(piranha_creepers),
        0,  # exclamation_mark_block_count (no array in JSON)
        0,  # track_block_count (no array in JSON)
        0,  # unk1
        len(ground),
        len(tracks),
        0,  # ice_count (no icicles array in JSON)
    )
    assert len(header) == MAP_HEADER_SIZE, len(header)

    out = bytearray(header)
    out += _pack_array(objects, OBJ_MAX, OBJ_SIZE, pack_obj, "objects")
    out += b"\x00" * (SOUND_SIZE * SOUND_MAX)
    out += _pack_array(snakes, SNAKE_MAX, SNAKE_SIZE, pack_snake, "snakes")
    out += _pack_array(clear_pipes, CLEAR_PIPE_MAX, CLEAR_PIPE_SIZE, pack_clear_pipe, "clear_pipes")
    out += _pack_array(piranha_creepers, PIRANHA_MAX, PIRANHA_SIZE, pack_piranha_creeper, "piranha_creepers")
    out += b"\x00" * (EXCLAMATION_SIZE * EXCLAMATION_MAX)
    out += b"\x00" * (TRACK_BLOCK_SIZE * TRACK_BLOCK_MAX)
    out += _pack_array(ground, GROUND_MAX, GROUND_SIZE, pack_ground, "ground")
    out += _pack_array(tracks, TRACK_MAX, TRACK_SIZE, pack_track, "tracks")
    out += b"\x00" * (ICICLE_SIZE * ICICLE_MAX)
    out += b"\x00" * MAP_UNK2_SIZE

    return bytes(out)


# ---------------------------------------------------------------------------
# Top level: combine header + two maps into the full payload
# ---------------------------------------------------------------------------

def build_payload(overworld_json, subworld_json):
    header_source = overworld_json or subworld_json or {}
    payload = (
        pack_level_header(header_source)
        + pack_map(overworld_json or {}, label="overworld")
        + pack_map(subworld_json or {}, label="subworld")
    )
    assert len(payload) == PAYLOAD_SIZE, len(payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_companion(path: Path):
    """Given an *_overworld.json or *_subworld.json path, return
    (overworld_path, subworld_path, base_stem), with paths set to None
    if that file doesn't exist."""
    stem = path.stem
    if stem.endswith("_overworld"):
        base = stem[: -len("_overworld")]
        ow, sub = path, path.with_name(base + "_subworld" + path.suffix)
    elif stem.endswith("_subworld"):
        base = stem[: -len("_subworld")]
        ow, sub = path.with_name(base + "_overworld" + path.suffix), path
    else:
        base = stem
        ow, sub = path, None

    ow = ow if ow and ow.exists() else None
    sub = sub if sub and sub.exists() else None
    return ow, sub, base


def parse_args():
    p = argparse.ArgumentParser(
        description="Rebuild a .bcd course file from toost's JSON export."
    )
    p.add_argument("json_path", help="Path to a *_overworld.json or *_subworld.json file")
    p.add_argument("-o", "--output", help="Output .bcd path (default: <stem>.bcd next to the input)")
    p.add_argument("--toost-compat", action="store_true",
                   help="Drop/clamp objects toost's local sprite atlas can't render (see toost_compat.py)")
    p.add_argument("--leveldata", help="Path to toost's LevelData.hpp (used with --toost-compat)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    in_path = Path(args.json_path)
    ow_path, sub_path, base = find_companion(in_path)

    overworld_json = load_json(ow_path) if ow_path else None
    subworld_json = load_json(sub_path) if sub_path else None

    if overworld_json is None and subworld_json is None:
        raise SystemExit(f"Could not find JSON data at {in_path}")

    if overworld_json is None:
        print(f"  [WARN] no overworld JSON found, writing an empty overworld map")
    if subworld_json is None:
        print(f"  [WARN] no subworld JSON found, writing an empty subworld map")

    if args.toost_compat:
        import toost_compat

        leveldata_path = Path(args.leveldata) if args.leveldata else toost_compat.DEFAULT_LEVELDATA
        if not leveldata_path.exists():
            raise SystemExit(f"Could not find LevelData.hpp at {leveldata_path} (pass --leveldata)")

        constants, location_keys = toost_compat.parse_leveldata(leveldata_path)
        overworld_json = toost_compat.sanitize_map_json(overworld_json, constants, location_keys, "overworld")
        subworld_json = toost_compat.sanitize_map_json(subworld_json, constants, location_keys, "subworld")

    payload = build_payload(overworld_json, subworld_json)
    bcd_bytes = build_bcd(payload)

    out_path = Path(args.output) if args.output else in_path.with_name(base + ".bcd")
    out_path.write_bytes(bcd_bytes)
    print(f"Wrote {out_path} ({len(bcd_bytes)} bytes)")
