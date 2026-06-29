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

from mm2_json_to_ascii import OBJ_META, GROUND_CHAR, CAT_ITEM

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

# Blocks that can hold an item in their `cid` (Block=4, ? Block=5, Hidden=29),
# by the canonical CHAR_TO_NAME names. An item glyph drawn directly above one of
# these is folded into the block's cid instead of becoming a free item -- the
# reverse of mm2_json_to_ascii's item-above-block stamping. See ascii_to_level.
_CONTAINER_BLOCK_NAMES = {"Block", "? Block", "Hidden Block"}

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


# Pipe direction is the low bits of `flag` (% 0x80): 0x00=R, 0x40=U. DEFAULT_FLAG
# already carries 0x40, so a reconstructed vertical pipe is mouth-up by default;
# a horizontal one clears those bits back to 0x00 (mouth-right). See coalesce().
PIPE_FLAG_U = DEFAULT_FLAG               # 0x..40  -> Up   (build_pipes default)
PIPE_FLAG_R = DEFAULT_FLAG & ~0x60       # 0x..00  -> Right


def make_object(name, col, row, w=1, h=1, flag=DEFAULT_FLAG):
    """Build a base-schema object entry of size w x h with bottom-left tile at
    (col, row).

    Coordinates use the real .bcd convention the rest of the toolchain
    (json_to_bcd.py / mm2_viewer_json.py / json_to_swe.py) consumes:

      * left-anchored objects (Pipe, Mushroom/Semisolid/Bridge, One-Way Wall;
        _REV_LEFT_ANCHOR) store x as the LEFT-tile centre  ->  x = col*160 + 80,
        matching col = x // 160 in the decoders.
      * everything else stores x as the CENTRE of its w-wide span  ->
        x = (col + w//2)*160 + (80 if w odd else 0), matching col = x//160 - w//2.

    y is always the bottom-row centre (row = y // 160). flags are the baseline
    value (with the pipe direction folded in for pipes); cid/lid/sid unlinked."""
    if name in _REV_LEFT_ANCHOR:
        x = col * TILE + TILE_CENTER
    else:
        x = (col + w // 2) * TILE + (TILE_CENTER if w % 2 else 0)
    return {
        "name": name,
        "x": x,
        "y": row * TILE + TILE_CENTER,
        "w": w,
        "h": h,
        "flag": flag,
        "cflag": flag,
        "ex": 0,
        "id": NAME_TO_ID[name],
        "cid": -1,
        "lid": -1,
        "sid": -1,
        "link_type": 0,
    }


# ---------------------------------------------------------------------------
# Multi-tile object coalescing (ASCII -> JSON)
#
# The forward painter (mm2_json_to_ascii.build_ascii_grid) draws each object as a
# w x h block of its glyph using the real MM2 JSON w/h. Reading ASCII back one
# glyph at a time would turn every cell into its own 1x1 object: a 2x2 Thwomp
# comes back as FOUR thwomps, a 5-wide Mushroom Platform as five separate 1x1
# platforms (each drawn as a whole tiny mushroom by SMM:WE instead of stem + cap
# pieces), a 2-wide pipe column as 2N length-1 pipes, two-tile doors as mispaired
# half-doors, and so on.
#
# coalesce() instead groups 4-connected same-glyph cells back into correctly
# sized multi-tile objects, so json_to_swe / json_to_bcd / the viewer all see the
# real footprint -- and SMM:WE renders ONE object with its proper internal pieces.
#
# Footprints/policies are grounded in the real toost JSON export (Bullet Bill
# Blaster 1x2, Mushroom Platform variable, Big Coin 2x2, Checkpoint 2x2, Koopa
# 1x1 -- all confirmed against the reference levels in big doc/1/json),
# level_dataset.py's documented *painted* footprints (Thwomp 2x2, Skewer 4x4,
# Swinging Claw 3x4) and json_to_swe.py (Thwomp's h=2 anchor, the platform S7
# families). Objects NOT listed here stay 1x1 -- correct for blocks, coins,
# spikes and every single-tile enemy. Listing a fixed-size object that turns out
# to actually be 1x1 is harmless: clamping (below) still yields exactly one
# object for an isolated cell. Only genuinely multi-tile objects are listed, so a
# long row of (1x1) spikes/coins is never wrongly merged.
# ---------------------------------------------------------------------------
_FIXED, _BBOX, _MUSHROOM, _HRUN, _VRUN, _PIPE = (
    "fixed", "bbox", "mushroom", "hrun", "vrun", "pipe")

# Objects whose x is the LEFT-tile centre (col = x // 160), matching
# OBJ_LEFT_ANCHOR_IDS in json_to_swe.py and _LEFT_ANCHOR in the forward path.
_REV_LEFT_ANCHOR = frozenset({
    "Mushroom Platform", "Semisolid Platform", "Bridge", "Pipe",
    "One-Way Wall", "One-Way",
})

# name -> coalescing policy. _FIXED tiles a glyph block into fw x fh stamps (so a
# row of two 2x2 thwomps becomes two thwomps, but an isolated <2x2 block clamps
# to one); _BBOX emits one object covering the whole block; _MUSHROOM recovers a
# cap+stem platform; _HRUN/_VRUN split into 1-tall / 1-wide runs; _PIPE rebuilds a
# 2-wide directional pipe. Names are the canonical CHAR_TO_NAME outputs.
COALESCE_POLICY = {
    # Fixed multi-tile sprites. Footprints confirmed against real toost JSON
    # exports (runtest/json + bigdoc): Thwomp / Bowser / Big Coin / Checkpoint /
    # Shoe-Egg are 2x2, Saw is 3x3, Swinging Claw 3x4, Skewer 4x4 (the last
    # documented in level_dataset.py). Boom Boom / Banzai Bill / Angry Sun /
    # Bowser Jr / Clown Car had no sample to confirm but are big single bosses
    # assumed 2x2 -- they are never placed adjacent, so a wrong guess can't
    # merge neighbours, and clamping makes an isolated <2x2 instance one object.
    "Thwomp":             (_FIXED, 2, 2),   # confirmed
    "Bowser":             (_FIXED, 2, 2),   # confirmed
    "Big Coin":           (_FIXED, 2, 2),   # confirmed
    "Checkpoint Flag":    (_FIXED, 2, 2),   # confirmed
    "Goomba's Shoe":      (_FIXED, 2, 2),   # confirmed (decoder: "Shoe Goomba")
    "Yoshi's Egg":        (_FIXED, 2, 2),   # id 45, SMW/NSMBU form of the above
    "Saw":                (_FIXED, 3, 3),   # confirmed 3x3 (NOT 2x2)
    "Swinging Claw":      (_FIXED, 3, 4),   # confirmed
    "Skewer":             (_FIXED, 4, 4),   # documented (level_dataset.py)
    "Donut":              (_FIXED, 3, 3),   # id 82 Donut Block Platform, 3x3
    "Boom Boom":          (_FIXED, 2, 2),   # assumed boss 2x2
    "Banzai Bill":        (_FIXED, 2, 2),   # assumed 2x2
    "Angry Sun":          (_FIXED, 2, 2),   # assumed 2x2
    "Bowser Jr.":         (_FIXED, 2, 2),   # assumed boss 2x2
    "Bowser Jr":          (_FIXED, 2, 2),
    "Clown Car":          (_FIXED, 2, 2),   # assumed 2x2
    # door: 1 wide x 2 tall; grouping the two cells stops build_doors mispairing
    # a single door's halves into a warp
    "Door":               (_FIXED, 1, 2),
    # Intentionally NOT coalesced: Wiggler (id 52) and Chain Chomp (id 61) are
    # 1x1 in real data (Chain Chomp only occasionally 2x2) and are commonly
    # placed in rows, so a fixed 2x2 guess wrongly merged adjacent ones.
    # variable-size platforms / structures
    "Mushroom Platform":  (_MUSHROOM,),    # cap (top row) + centred stem
    "Semisolid Platform": (_BBOX,),   # box repair, see repair_semisolid_cells
    "Half-Collision Platform": (_BBOX,),
    "One-Way Wall":       (_BBOX,),
    "One-Way":            (_BBOX,),
    "Bridge":             (_HRUN,),        # one walkable row, any width
    "Lift":               (_HRUN,),
    "Bullet Bill Blaster":(_VRUN,),        # 1 wide, stacked any height
    "Vine":               (_VRUN,),
    "Pipe":               (_PIPE,),        # 2 wide, directional
}


def _connected_components(cells):
    """4-connected components of a set of (col, row) cells."""
    cells = set(cells)
    seen = set()
    comps = []
    for start in cells:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            c, r = stack.pop()
            comp.append((c, r))
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (c + dc, r + dr)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _runs(values):
    """Yield (start, length) for each maximal run of consecutive ints."""
    values = sorted(values)
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
        else:
            yield start, prev - start + 1
            start = prev = v
    yield start, prev - start + 1


# ---------------------------------------------------------------------------
# Semisolid platform repair (ASCII -> JSON)
#
# A Semisolid Platform (id 16) is a solid rectangular BOX in SMM2: you stand on
# its top and pass through it from below / the sides. The diffusion model doesn't
# know that and sprays the platform's 'k' glyph as lone tiles and ragged blobs
# that don't form clean rectangles. coalesce()'s plain _BBOX (one box per
# component) then leaves a swarm of 1x1 / sliver "platforms".
#
# repair_semisolid_cells() rebuilds a real box for EACH 4-connected blob. Blobs
# at different heights stay SEPARATE platforms -- they are never merged together:
#   1. Take the blob's bounding columns and its TOP row (the walkable cap).
#   2. Stretch the width to SMM2's 3-tile minimum (PLATFORM_MIN_SIZE), centred,
#      so even a lone tile becomes a placeable 3-wide platform.
#   3. Drag the box's bottom straight DOWN until it rests on ground (or the level
#      floor), keeping the cap where the model put it -- a semisolid reads as a
#      solid block from its top surface down to the floor, not a floating sliver.
#   4. Enforce the 3-tall minimum if the cap already sits right above the ground.
# ---------------------------------------------------------------------------
PLATFORM_MIN_SIZE = 3       # SMM2 won't place a semisolid smaller than 3x3


def repair_semisolid_cells(cells, ground=None):
    """Turn (col, row) Semisolid-Platform 'k' cells into a list of
    (col, row, w, h) box rectangles -- one box per 4-connected blob, widened to
    the 3-tile minimum and dragged down to `ground` (a set of (col, row) solid
    cells; falsy = drag to the floor). See the section comment for rationale."""
    ground = ground or set()
    boxes = []
    for comp in _connected_components(cells):
        cols = [c for c, _ in comp]
        rows = [r for _, r in comp]
        c0, c1 = min(cols), max(cols)
        cap_row = max(rows)                        # top / walkable surface
        w = c1 - c0 + 1
        if w < PLATFORM_MIN_SIZE:                  # widen symmetrically
            c0 = max(0, c0 - (PLATFORM_MIN_SIZE - w) // 2)
            w = PLATFORM_MIN_SIZE
        c1 = c0 + w - 1
        # Drag the bottom down until every column directly beneath the base rests
        # on ground (or we hit the floor) -- the same rest-on-ground rule
        # json_to_bcd._fix_platform_objects uses.
        bottom = min(rows)
        while bottom > 0 and any((c, bottom - 1) not in ground
                                 for c in range(c0, c1 + 1)):
            bottom -= 1
        if cap_row - bottom + 1 < PLATFORM_MIN_SIZE:   # 3-tall minimum
            bottom = max(0, cap_row - (PLATFORM_MIN_SIZE - 1))
        boxes.append((c0, bottom, w, cap_row - bottom + 1))
    return boxes


def coalesce(name, cells, out, ground=None):
    """Append reconstructed object(s) for every cell tagged `name` to `out`.

    `cells` is the set of (col, row) cells carrying this object's glyph; the
    policy in COALESCE_POLICY decides how connected blocks become objects.
    `ground` is the set of (col, row) solid ground cells, used to drag semisolid
    platforms down to the floor. Objects with no policy stay 1x1 (one per cell)."""
    policy = COALESCE_POLICY.get(name)
    if policy is None:
        for col, row in cells:
            out.append(make_object(name, col, row))
        return

    kind = policy[0]
    # Semisolid platforms get the dedicated box repair above rather than a plain
    # per-component bbox -- the model's 'k' glyph is too scattered for that.
    if name == "Semisolid Platform":
        for col, row, w, h in repair_semisolid_cells(cells, ground):
            out.append(make_object(name, col, row, w, h))
        return
    for comp in _connected_components(cells):
        cols = [c for c, _ in comp]
        rows = [r for _, r in comp]
        c0, c1, r0, r1 = min(cols), max(cols), min(rows), max(rows)

        if kind == _FIXED:
            fw, fh = policy[1], policy[2]
            cr = r0
            while cr <= r1:                # cr is each stamp's bottom row
                cc = c0
                while cc <= c1:
                    out.append(make_object(
                        name, cc, cr,
                        min(fw, c1 - cc + 1), min(fh, r1 - cr + 1)))
                    cc += fw
                cr += fh
        elif kind == _BBOX:
            out.append(make_object(name, c0, r0, c1 - c0 + 1, r1 - r0 + 1))
        elif kind == _MUSHROOM:
            # A mushroom is a wide cap (top row) over a 1-wide centred stem. Two
            # stacked mushrooms form one blob with two caps joined by a stem. A
            # cap is a wide row (>=2 cells) whose row ABOVE is narrow/empty -- the
            # actual top of a mushroom; a wide row sitting under another wide row
            # is absorbed into the mushroom above (so overdrawn / overlapping
            # caps don't split into 1-tall slivers). Each cap's platform hangs
            # down to just above the next cap below it (or the blob's bottom).
            rows_by = {}
            for c, r in comp:
                rows_by.setdefault(r, []).append(c)
            cap_rows = sorted(
                (r for r, cs in rows_by.items()
                 if len(cs) >= 2 and len(rows_by.get(r + 1, ())) <= 1),
                reverse=True)
            if not cap_rows:                       # stem-only speck: treat top
                cap_rows = [r1]
            for i, rc in enumerate(cap_rows):
                cs = rows_by[rc]
                cl = min(cs)
                bottom = cap_rows[i + 1] + 1 if i + 1 < len(cap_rows) else r0
                out.append(make_object(name, cl, bottom,
                                       max(cs) - cl + 1, rc - bottom + 1))
        elif kind == _HRUN:
            by_row = {}
            for c, r in comp:
                by_row.setdefault(r, []).append(c)
            for r, rcols in by_row.items():
                for start, length in _runs(rcols):
                    out.append(make_object(name, start, r, length, 1))
        elif kind == _VRUN:
            by_col = {}
            for c, r in comp:
                by_col.setdefault(c, []).append(r)
            for c, crows in by_col.items():
                for start, length in _runs(crows):
                    out.append(make_object(name, c, start, 1, length))
        elif kind == _PIPE:
            bw, bh = c1 - c0 + 1, r1 - r0 + 1
            if bw == 2 and bh >= 2:        # vertical -> mouth up (default)
                out.append(make_object(name, c0, r0, 2, bh, PIPE_FLAG_U))
            elif bh == 2 and bw >= 2:      # horizontal -> mouth right
                out.append(make_object(name, c0, r1, 2, bw, PIPE_FLAG_R))
            else:                          # malformed -> leave as 1x1 cells
                for col, row in comp:
                    out.append(make_object(name, col, row))


def ascii_to_level(text, source_file=None, *, gamestyle_raw=22349, theme_raw=0,
                   timer=300):
    rows, width = parse_ascii(text)
    height = len(rows)

    ground = []
    objects = []
    unknown_glyphs = {}
    goal_cells = []
    # name -> set of (col, row) cells, coalesced into multi-tile objects below.
    obj_cells = {}

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
            obj_cells.setdefault(name, set()).add((col, row_game))

    # Fold an item glyph sitting directly ABOVE a Brick / ? / Hidden block back
    # into that block's `cid` (the inverse of the forward painter's item-above-
    # block stamping), rather than emitting a free-floating item object. The
    # block then round-trips as a container: json_to_bcd packs `cid` into the
    # .bcd and json_to_swe maps it to the block's sprout. For human-authored
    # ASCII this also makes "item resting on a block" mean "item inside it".
    block_cells = set()
    for bname in _CONTAINER_BLOCK_NAMES:
        block_cells |= obj_cells.get(bname, set())
    block_item = {}  # (col, row) of a block -> contained item's MM2 object id
    for iname in list(obj_cells):
        meta = OBJ_META.get(iname)
        if not meta or meta[2] != CAT_ITEM:
            continue
        remaining = set()
        for cell in obj_cells[iname]:
            below = (cell[0], cell[1] - 1)
            if below in block_cells and below not in block_item:
                block_item[below] = NAME_TO_ID[iname]
            else:
                remaining.add(cell)
        obj_cells[iname] = remaining

    # Coalesce same-glyph cells back into correctly-sized multi-tile objects
    # (a 2x2 Thwomp -> one object, not four; an N-wide Mushroom Platform -> one
    # platform, not N tiny whole mushrooms; a 2-wide pipe -> one pipe). Objects
    # with no policy fall through to one 1x1 object per cell. See coalesce().
    ground_cells = {(g["x"], g["y"]) for g in ground}
    for name, cells in obj_cells.items():
        if name in _CONTAINER_BLOCK_NAMES:
            # Container blocks have no multi-tile policy (one object per cell);
            # emit them here so the recovered contained-item cid can be attached.
            for col, row in cells:
                obj = make_object(name, col, row)
                cid = block_item.get((col, row))
                if cid is not None:
                    obj["cid"] = cid
                objects.append(obj)
            continue
        coalesce(name, cells, objects, ground_cells)

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

    # Recover the player spawn height. The player always starts at the left edge,
    # standing on top of the left-edge ground column (the start platform the
    # forward painter injects for overworld maps, cols 0-6). Measure the height
    # of the contiguous ground stack at the left-most column that has ground.
    # Without this start_y stays 0, which json_to_swe maps to one tile BELOW the
    # bottom ground row -- the player spawns in the void and falls out (every
    # level's spawn "messed up"). Default to 2 (the common SMM2 spawn) if the
    # left edge has no ground.
    ground_at = {(g["x"], g["y"]) for g in ground}
    start_y = 0
    for c in (0, 1, 2):
        h = 0
        while (c, h) in ground_at:
            h += 1
        if h:
            start_y = h
            break
    if start_y == 0:
        start_y = 2

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
        "start_y": start_y,
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
