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

# Per-level metadata fields (folded into each dataset entry by
# build_dataset_with_ascii.py from the export's metadata.json) that get added to
# the caption, paired with the word that turns the raw value into a phrase.
# level_name is intentionally left out: it names the source level, not anything
# about its contents.
CAPTION_METADATA_FIELDS = [
    ("gamestyle", "style"),
    ("theme", "theme"),
    ("difficulty", "difficulty"),
]


def metadata_phrases(item):
    """Build caption phrases from an item's level metadata. Skips missing/empty
    values, the "Unknown" difficulty placeholder, and the "None" tag slot."""
    phrases = []
    for field, suffix in CAPTION_METADATA_FIELDS:
        value = item.get(field)
        if value in (None, "") or str(value).lower() == "unknown":
            continue
        phrases.append(f"{value} {suffix}")
    for tag in item.get("tags") or []:
        if tag and str(tag).lower() != "none":
            phrases.append(str(tag))
    return phrases


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
        # The name is the last tag. A tile can carry a category word ahead of it
        # (Bowser is [..., "boss", "bowser"]), so take the last non-property tag
        # rather than the first, or every boss ends up named "boss".
        name_tags = [t for t in tags if t not in PROPERTY_TAGS]
        name = name_tags[-1] if name_tags else tags[-1]
        char_names[char] = name.title()
    return char_names


def get_tile_categories(tileset_path):
    """Sort tile chars into (enemies, items, ground) by their tags. Enemies are
    tagged "enemy", items are collectables, ground is the terrain we call floor."""
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enemy_chars, item_chars, ground_chars = set(), set(), set()
    for char, tags in data["tiles"].items():
        tagset = set(tags)
        if "enemy" in tagset:
            enemy_chars.add(char)
        if "collectable" in tagset:
            item_chars.add(char)
        if "ground" in tagset:
            ground_chars.add(char)
    return enemy_chars, item_chars, ground_chars


def describe_quantity(count):
    # Same coarse buckets MarioDiffusion's caption code uses.
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    if count < 5:
        return "a few"
    if count < 10:
        return "several"
    return "many"


def count_phrase(count, singular, plural):
    if count <= 0:
        return None
    noun = plural if count > 1 else singular
    return f"{describe_quantity(count)} {noun}".capitalize()


def describe_ground(scene, id_to_char, ground_chars):
    if not ground_chars:
        return None
    if not any(id_to_char.get(t) in ground_chars for row in scene for t in row):
        return None

    bottom = [id_to_char.get(t) in ground_chars for t in scene[-1]]
    if all(bottom):
        return "Full ground floor"
    if not any(bottom):
        return "Scattered ground"

    # Count the runs of missing ground along the bottom row.
    gaps = 0
    in_gap = False
    for is_ground in bottom:
        if not is_ground:
            gaps += not in_gap
            in_gap = True
        else:
            in_gap = False
    return f"Ground floor with {describe_quantity(gaps)} gap" + ("s" if gaps > 1 else "")


def assign_caption(scene, id_to_char, char_names, enemy_chars, item_chars,
                   ground_chars, meta_phrases=None):
    # Metadata, then the distinct tiles present, then the ground and how many
    # enemies and items there are, e.g. "SMB1 style. Goomba. Coin. Full ground
    # floor. A few enemies. Several items."
    phrases = list(meta_phrases) if meta_phrases else []

    enemy_count = 0
    item_count = 0
    seen = set()
    for row in scene:
        for tile_id in row:
            char = id_to_char.get(tile_id)
            if char is None:
                continue
            if char in enemy_chars:
                enemy_count += 1
            if char in item_chars:
                item_count += 1
            if char in ground_chars:
                continue  # Ground is covered by the floor phrase below.
            name = char_names.get(char)
            if name and name not in seen:
                seen.add(name)
                phrases.append(name)

    for phrase in (describe_ground(scene, id_to_char, ground_chars),
                   count_phrase(enemy_count, "enemy", "enemies"),
                   count_phrase(item_count, "item", "items")):
        if phrase:
            phrases.append(phrase)

    return " ".join(f"{p}." for p in phrases)


def generate_captions(dataset_path, tileset_path, output_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)
    enemy_chars, item_chars, ground_chars = get_tile_categories(tileset_path)

    captioned = []
    for item in dataset:
        is_dict = isinstance(item, dict)
        scene = item["scene"] if is_dict else item
        meta_phrases = metadata_phrases(item) if is_dict else []
        caption = assign_caption(scene, id_to_char, char_names, enemy_chars,
                                 item_chars, ground_chars, meta_phrases)
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
    parser = argparse.ArgumentParser(description="Simple captions for MM2 ASCII datasets: tiles present, a ground/floor summary, and enemy/item counts.")
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
