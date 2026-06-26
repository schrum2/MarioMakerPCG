#!/usr/bin/env python3
"""
build_extended_tilesets.py
==========================
Generate the frequency-ranked "extended" tilesets (20/30/40/50/60 tiles) used by
mm2view_to_extended.py to shrink the 69-tile mm2_tileset_we vocabulary down to the
handful of tiles that actually carry the dataset.

The idea is simple: a diffusion model that has to learn 69 near-empty tile classes
wastes most of its capacity on tiles that appear a fraction of a percent of the
time. So we walk the real tile-frequency distribution from most to least common,
keep the first N distinct tiles, and let the converter fold everything rarer onto
its closest survivor (see mm2view_to_extended.py for the tag-based replacement).

FREQUENCY_ORDER below is the tile_counts ordering straight out of
MM_Levels-regular_tile_distribution.json (descending). Each name is also a tag in
mm2_tileset_we.json, which is how we recover the glyph + the full tag list to copy
into each generated tileset (so captioning and Block2Vec see the same tags they
would in the full vocab).

Run:
    python build_extended_tilesets.py
writes extended_tiles_20.json ... extended_tiles_60.json next to this script.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_TILESET = os.path.join(HERE, "mm2_tileset_we.json")
SIZES = [20, 30, 40, 50, 60]

# Tile names in descending frequency order, copied from the tile_counts block of
# MM_Levels-regular_tile_distribution.json. Walk this list, keep the first N
# distinct glyphs -> that's the N-tile tileset. "style power-up" shares the glyph
# "E" with "propeller mushroom" (which ranks higher), so it never adds a new tile.
FREQUENCY_ORDER = [
    "air", "ground", "semisolid", "hard block", "coin", "brick", "pipe", "spikes",
    "mushroom platform", "flagpole", "dotted line block", "goomba", "question block",
    "thwomp", "swinging claw", "banzai bill", "bridge", "ice block", "vine", "lift",
    "koopa", "cloud block", "saw", "piranha plant", "skewer", "hidden block",
    "donut block", "bullet bill blaster", "door", "one way", "spiny", "note block",
    "cheep cheep", "bowser", "lava bubble", "boom boom", "mushroom", "one up",
    "fire flower", "checkpoint flag", "spring", "clown car", "on off block",
    "hammer bro", "buzzy beetle", "burner", "cannon", "dry bones", "fire bar", "star",
    "propeller mushroom", "bob-omb", "chain chomp", "boo", "monty mole",
    "goomba's shoe", "wiggler", "blooper", "angry sun", "twister", "p switch",
    "lava lift", "magikoopa", "bowser jr.", "rocky wrench", "lakitu", "pow",
    "style power-up",
]


def name_to_glyph(tiles):
    """Map each tile name (a tag) to the glyph that carries it.

    A glyph can carry several names (e.g. "E" is both propeller mushroom and the
    generic style power-up); the first name wins so the lookup is 1:1 per name."""
    lookup = {}
    for glyph, tags in tiles.items():
        for tag in tags:
            lookup.setdefault(tag, glyph)
    return lookup


def select_glyphs(name_order, name_glyph, n):
    """Walk name_order keeping the first n distinct glyphs (frequency-ranked)."""
    chosen = []
    seen = set()
    for name in name_order:
        glyph = name_glyph.get(name)
        if glyph is None:
            raise KeyError(f"distribution name {name!r} has no glyph in mm2_tileset_we.json")
        if glyph in seen:
            continue
        seen.add(glyph)
        chosen.append(glyph)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} distinct glyphs available; cannot reach {n}")
    return chosen


def main():
    with open(SOURCE_TILESET, encoding="utf-8") as f:
        tiles = json.load(f)["tiles"]

    name_glyph = name_to_glyph(tiles)

    for n in SIZES:
        glyphs = select_glyphs(FREQUENCY_ORDER, name_glyph, n)
        # Preserve the source tag lists verbatim so the reduced tileset stays
        # tag-compatible with mm2_tileset_we (captioning, Block2Vec, etc.).
        out = {"tiles": {g: tiles[g] for g in glyphs}}
        out_path = os.path.join(HERE, f"extended_tiles_{n}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
        print(f"extended_tiles_{n}.json  ({len(glyphs)} tiles): {''.join(glyphs)!r}")


if __name__ == "__main__":
    main()
