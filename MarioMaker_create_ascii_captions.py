import json
import os
import sys
import argparse

# Tags used to describe tile properties rather than identity; everything in a
# tile's tag list other than these is treated as part of its name.
PROPERTY_TAGS = {
    "passable", "solid", "empty", "air", "breakable", "collectable", "enemy",
    "damaging", "hazard", "moving", "flying", "projectile", "explosive",
    "shooter", "power-up", "style ride", "platform",
    "interactive", "climbable", "togglable", "slippery", "falling", "warp",
    "door", "vehicle",
}

# Tiles that represent empty space and should never appear in a caption.
EMPTY_TAGS = {"empty", "air"}


def build_id_to_char(tileset_path):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tile_chars = sorted(data["tiles"].keys())
    if "_" not in tile_chars:
        tile_chars.append("_")
    return {idx: char for idx, char in enumerate(tile_chars)}


def get_char_names(tileset_path):
    """Derive char -> human-readable name directly from the tileset file.

    Each tile's value is a tag list whose trailing entries are its name (e.g.
    ["passable", "collectable", "coin"] -> "Coin"). Reading names straight from
    the tileset keeps captions aligned with whatever tileset is passed in,
    instead of a hardcoded table that can silently disagree with the file's
    actual char assignments (e.g. "c" = coin here vs. "Clear Pipe" in the full
    set).
    """
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char_names = {}
    for char, tags in data["tiles"].items():
        if not tags or any(t in EMPTY_TAGS for t in tags):
            continue
        # Name = trailing tags that aren't property descriptors. Fall back to
        # the last tag if a tile is described purely by properties.
        name_tags = [t for t in tags if t not in PROPERTY_TAGS]
        name = name_tags[0] if name_tags else tags[-1]
        char_names[char] = name.title()
    return char_names


def assign_caption(scene, id_to_char, char_names):
    seen = set()
    names = []
    for row in scene:
        for tile_id in row:
            char = id_to_char.get(tile_id)
            if char is None:
                continue
            name = char_names.get(char)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return " ".join(f"{n}." for n in names)


def generate_captions(dataset_path, tileset_path, output_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)

    captioned = []
    for item in dataset:
        scene = item["scene"] if isinstance(item, dict) else item
        caption = assign_caption(scene, id_to_char, char_names)
        entry = {"scene": scene, "caption": caption}
        if isinstance(item, dict):
            if "name" in item:
                entry["name"] = item["name"]
            if "prompt" in item:
                entry["prompt"] = item["prompt"]
        captioned.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captioned, f, indent=2, ensure_ascii=False)

    print(f"Captioned {len(captioned)} scenes -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple presence-based captions for MM2 ASCII datasets.")
    parser.add_argument("--dataset", required=True, help="Input dataset JSON.")
    parser.add_argument("--tileset", required=True, help="Tileset JSON (e.g. mm2_tileset_we.json); names are read from its tile tags.")
    parser.add_argument("--output", required=True, help="Output captioned JSON.")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        print(f"Error: dataset not found: {args.dataset}")
        sys.exit(1)
    if not os.path.isfile(args.tileset):
        print(f"Error: tileset not found: {args.tileset}")
        sys.exit(1)

    generate_captions(args.dataset, args.tileset, args.output)
