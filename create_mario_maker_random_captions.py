"""Generate random MM2 test captions for evaluating caption adherence.

Like MarioDiffusion's per-game generators (captions/caption_generator.py for SMB,
captions/LR_caption_generator.py for Lode Runner), this builds captions by grouping
phrases into topics and picking one variant per topic, so a caption never asks for
both "two goombas" and "many goombas". MM topics aren't fixed, so we learn them from
a dataset using the same grouping as the MM2 scorer (MM2_caption_match.phrase_topic).
"""
import argparse
import json
import random
from collections import OrderedDict

from captions.MM2_caption_match import (
    QUANTITY_TERMS, build_name_lookup, phrase_topic, topic_entity,
)
from MarioMaker_create_ascii_captions import CAPTION_METADATA_FIELDS
import util.common_settings as common_settings

# style/theme/difficulty, in the order MarioMaker_create_ascii_captions emits them.
METADATA_ORDER = [suffix for _field, suffix in CAPTION_METADATA_FIELDS]

# Longest first so "a ton of" is stripped before "a few"/"a".
_QUANTITIES_LONGEST_FIRST = sorted(QUANTITY_TERMS, key=len, reverse=True)


def _singularize(name):
    """Strip a trailing plural 's' so "goombas" and "goomba" group together."""
    name = name.strip()
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def _fallback_kind_entity(phrase):
    """Group a phrase phrase_topic() didn't recognize (entity not in the tileset,
    or a free-form tag). Count/blob phrases still group by entity; anything else
    stays as its own topic."""
    p = phrase.strip().lower().rstrip(".")
    if p.startswith("a blob of "):
        return "blob", _singularize(p[len("a blob of "):])
    for qty in _QUANTITIES_LONGEST_FIRST:
        if p.startswith(qty + " "):
            return "count", _singularize(p[len(qty) + 1:])
    return "other", p


def classify(phrase, name_lookup):
    """Return (kind, primary_key, entity) for a phrase. kind is one of
    metadata/ground/count/blob/other; primary_key is what we group on when picking
    topics (None for blobs, which attach to their entity's count). Unrecognized
    phrases fall back to a structural parse."""
    topic, kind = phrase_topic(phrase, name_lookup)
    if kind == "metadata":
        return "metadata", ("metadata", topic), None
    if kind == "ground":
        return "ground", ("ground",), None
    if kind == "count":
        return "count", ("count", topic), topic
    if kind == "blob":
        return "blob", None, topic_entity(topic)

    fallback_kind, entity = _fallback_kind_entity(phrase)
    if fallback_kind == "blob":
        return "blob", None, entity
    if fallback_kind == "count":
        return "count", ("count", entity), entity
    return "other", ("other", entity), None


def group_vocabulary(captions, name_lookup):
    """Collect the phrase variants seen for each topic.

    Returns (primary, blobs): primary maps a topic key to its list of phrases;
    blobs maps an entity to its "a blob of ..." phrases (kept separate since a blob
    only makes sense next to its count). Phrases are stored verbatim, first seen first.
    """
    primary = OrderedDict()
    blobs = OrderedDict()

    def add(store, key, phrase):
        bucket = store.setdefault(key, [])
        if phrase not in bucket:
            bucket.append(phrase)

    for caption in captions:
        for raw in caption.split("."):
            phrase = raw.strip()
            if not phrase:
                continue
            verbatim = phrase.rstrip(".")
            kind, key, entity = classify(phrase, name_lookup)
            if kind == "blob":
                add(blobs, entity, verbatim)
            else:
                add(primary, key, verbatim)

    return primary, blobs


def generate_caption(primary, blobs, rng, min_topics, max_topics, blob_prob):
    """Draw one coherent caption: pick a random handful of topics, one phrase
    variant each, then maybe add a blob to any entity that got a count."""
    keys = list(primary.keys())
    if not keys:
        return ""

    lo = min(min_topics, len(keys))
    hi = min(max_topics, len(keys))
    k = rng.randint(lo, hi)
    chosen_keys = set(rng.sample(keys, k))
    chosen_phrase = {key: rng.choice(primary[key]) for key in chosen_keys}

    # A blob only tags along if its entity was picked as a count.
    chosen_blob = {}
    for key in chosen_keys:
        if key[0] == "count":
            entity = key[1]
            if entity in blobs and rng.random() < blob_prob:
                chosen_blob[entity] = rng.choice(blobs[entity])

    # Order it like a real caption: metadata, ground, then each count with its blob,
    # then any leftover phrases.
    ordered = []
    for suffix in METADATA_ORDER:
        key = ("metadata", suffix)
        if key in chosen_phrase:
            ordered.append(chosen_phrase[key])
    if ("ground",) in chosen_phrase:
        ordered.append(chosen_phrase[("ground",)])
    for key in primary:  # stable order
        if key[0] == "count" and key in chosen_phrase:
            ordered.append(chosen_phrase[key])
            if key[1] in chosen_blob:
                ordered.append(chosen_blob[key[1]])
    for key in primary:
        if key[0] == "other" and key in chosen_phrase:
            ordered.append(chosen_phrase[key])

    return " ".join(f"{p}." for p in ordered)


def main():
    parser = argparse.ArgumentParser(description="Generate random MM2 test captions by grouping a dataset's phrases into topics and drawing one variant per chosen topic.")
    parser.add_argument("--json", required=True, help="Captioned dataset JSON (used to learn the phrase vocabulary).")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument("--tileset", default=common_settings.MARIO_TILESET, help="Tileset JSON used to recognize entity names. Should match the tileset the captions were built with.")
    parser.add_argument("--num_captions", type=int, default=100)
    parser.add_argument("--min_topics", "--min_tiles", dest="min_topics", type=int, default=1, help="Minimum number of topics per caption.")
    parser.add_argument("--max_topics", "--max_tiles", dest="max_topics", type=int, default=8, help="Maximum number of topics per caption.")
    parser.add_argument("--blob_prob", type=float, default=0.5, help="Chance of adding a 'blob' phrase to an entity that already has a count.")
    parser.add_argument("--seed", type=int, default=512)
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    captions = [item["caption"] for item in data if isinstance(item, dict) and item.get("caption")]

    name_lookup = build_name_lookup(args.tileset)
    primary, blobs = group_vocabulary(captions, name_lookup)

    num_entities = sum(1 for k in primary if k[0] == "count")
    print(f"Learned {len(primary)} topics from {len(captions)} captions "
          f"({num_entities} entities, {len(blobs)} with blobs).")

    rng = random.Random(args.seed)
    captions_out = []
    seen = set()
    attempts = 0
    max_attempts = args.num_captions * 200
    while len(captions_out) < args.num_captions and attempts < max_attempts:
        attempts += 1
        caption = generate_caption(primary, blobs, rng, args.min_topics,
                                   args.max_topics, args.blob_prob)
        if caption and caption not in seen:
            seen.add(caption)
            captions_out.append({"caption": caption})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(captions_out, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(captions_out)} random captions -> {args.output}")


if __name__ == "__main__":
    main()
