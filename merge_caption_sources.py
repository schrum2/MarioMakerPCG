#!/usr/bin/env python3
"""
merge_caption_sources.py
========================
Merge already-captioned datasets of the SAME scenes into one multi-source dataset (the schema
--caption_source_keys reads), matching scenes by "name". Each input contributes one source key,
taken from that input's "caption"/"caption1"/... fields; every other attribute is carried over.
Scenes missing from an input get an empty list for its key.

Usage:
    python merge_caption_sources.py \
        --inputs gemma.json qwen.json det.json \
        --keys   gemma4:26b_captions qwen3:32b_captions deterministic_captions \
        --output merged.json
"""

import argparse
import json
import sys


def legacy_captions(item):
    """Captions stored in an item's "caption"/"caption1"/"caption2"/... fields, in order."""
    caps = []
    if item.get("caption"):
        caps.append(item["caption"])
    idx = 1
    while f"caption{idx}" in item:
        value = item[f"caption{idx}"]
        if value:
            caps.append(value)
        idx += 1
    return caps


def is_legacy_caption_key(key):
    return key == "caption" or (key.startswith("caption") and key[len("caption"):].isdigit())


def merge(inputs, keys, output_path, match_on="name"):
    if len(inputs) != len(keys):
        print(f"Error: got {len(inputs)} --inputs but {len(keys)} --keys; they must pair up.")
        sys.exit(1)

    merged = {}   # match value -> entry
    order = []    # preserve first-seen order

    for path, source_key in zip(inputs, keys):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            match = item[match_on]
            if match not in merged:
                merged[match] = {}
                order.append(match)
            entry = merged[match]

            for k, v in item.items():  # carry over scene/metadata/existing lists, first writer wins
                if is_legacy_caption_key(k):
                    continue
                entry.setdefault(k, v)

            entry.setdefault(source_key, [])
            entry[source_key].extend(legacy_captions(item))  # this input's captions become this source key's pool

    for entry in merged.values():  # every scene gets every key, even ones no input mentioned
        for source_key in keys:
            entry.setdefault(source_key, [])

    result = [merged[m] for m in order]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(inputs)} source(s) over {len(result)} unique scene(s) -> {output_path}")
    for source_key in keys:
        with_caps = sum(1 for e in result if e.get(source_key))
        print(f"  {source_key}: {with_caps}/{len(result)} scenes captioned")


def main():
    parser = argparse.ArgumentParser(
        description="Merge same-scene captioned datasets into one multi-source dataset."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Captioned dataset JSON files (one per source).")
    parser.add_argument("--keys", nargs="+", required=True,
                        help="Source key for each input, in the same order "
                             "(e.g. gemma4:26b_captions qwen3:32b_captions).")
    parser.add_argument("--output", required=True, help="Output merged JSON.")
    parser.add_argument("--match-on", default="name",
                        help="Item field used to match the same scene across inputs. Default: name")
    args = parser.parse_args()

    merge(args.inputs, args.keys, args.output, match_on=args.match_on)


if __name__ == "__main__":
    main()
