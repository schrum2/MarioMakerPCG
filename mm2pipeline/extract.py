"""Extract properly-formatted .bcd level files from the
TheGreatRambler/mm2_level HuggingFace dataset for use with Toost.

Usage
-----
    # Stream a small sample (recommended for testing):
    python -m mm2pipeline.extract --output_dir ./bcd_levels --limit 100

    # Extract specific data_ids:
    python -m mm2pipeline.extract --ids 3000004 3000007

    # Extract everything (streaming, ~100 GB):
    python -m mm2pipeline.extract --output_dir ./bcd_levels

Requirements:
    pip install datasets pycryptodome
"""

import argparse
from pathlib import Path

from .bcd import (
    PAYLOAD_SIZE,
    GAMESTYLE_SM3DW,
    build_bcd,
    decompress_level_data,
    get_gamestyle_raw,
    level_contains_skip_object,
    subworld_has_items,
)


def extract_levels(
    output_dir: str,
    limit=None,
    streaming: bool = True,
    data_id_filter=None,
    name_filter=None,
    name_count=None,
    skip_3dworld: bool = False,
    skip_items: bool = False,
    skip_subworld_items: bool = False,
):
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset (streaming={streaming}) …")
    ds = load_dataset("TheGreatRambler/mm2_level", streaming=streaming, split="train")

    saved = skipped = errors = 0
    skipped_3dw = 0
    skipped_items = 0
    skipped_subworld = 0
    name_saved = 0

    for row in ds:
        data_id = row["data_id"]

        if data_id_filter is not None and data_id not in data_id_filter:
            continue

        if name_filter is not None:
            level_name = str(row.get("name", ""))
            if name_filter.lower() not in level_name.lower():
                continue

        raw = row["level_data"]
        if raw is None:
            skipped += 1
            continue

        try:
            plaintext = decompress_level_data(raw)
        except ValueError as e:
            print(f"  [WARN] data_id={data_id}: decompress error: {e}")
            errors += 1
            continue

        if len(plaintext) != PAYLOAD_SIZE:
            print(f"  [WARN] data_id={data_id}: unexpected size {len(plaintext)}, skipping")
            errors += 1
            continue

        if skip_3dworld and get_gamestyle_raw(plaintext) == GAMESTYLE_SM3DW:
            print(f"  [SKIP] data_id={data_id}: skipped because level is 3D World")
            skipped_3dw += 1
            continue

        if skip_items and level_contains_skip_object(plaintext):
            print(f"  [SKIP] data_id={data_id}: skipped because level contains a banned item")
            skipped_items += 1
            continue

        if skip_subworld_items and subworld_has_items(plaintext):
            print(f"  [SKIP] data_id={data_id}: skipped because subworld has items")
            skipped_subworld += 1
            continue

        try:
            bcd = build_bcd(plaintext)
        except Exception as e:
            print(f"  [WARN] data_id={data_id}: encrypt error: {e}")
            errors += 1
            continue

        if name_filter is not None:
            name_saved += 1
            filename = f"{data_id}_{name_saved}.bcd"
        else:
            filename = f"{data_id}.bcd"

        (out / filename).write_bytes(bcd)
        saved += 1

        if name_filter is not None and name_count is not None:
            if name_saved >= name_count:
                break

        if saved % 500 == 0 or saved == 1:
            print(f"  Saved {saved} levels …  (last: {data_id}.bcd)")

        if limit is not None and saved >= limit:
            break

    print(
        f"\nDone. Saved: {saved}  |  Skipped (null): {skipped}  |  "
        f"Skipped (3D World): {skipped_3dw}  |  Skipped (banned item): {skipped_items}  |  "
        f"Skipped (subworld has items): {skipped_subworld}  |  "
        f"Errors: {errors}"
    )
    print(f"Output dir: {out.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract properly-formatted .bcd level files from the "
            "TheGreatRambler/mm2_level HuggingFace dataset for use with Toost."
        )
    )
    p.add_argument("--output_dir", "-o", default="./bcd_levels")
    p.add_argument("--limit", "-n", type=int, default=None,
                   help="Max levels to extract (default: all)")
    p.add_argument("--no_stream", action="store_true",
                   help="Download full dataset first (~100 GB)")
    p.add_argument("--ids", nargs="+", type=int, default=None,
                   metavar="DATA_ID", help="Only extract these data_ids")
    p.add_argument("--name", type=str, default=None, help="Extract levels whose name contains this text")
    p.add_argument("--name_count", type=int, default=None, help="Number of matching levels to extract")
    p.add_argument("--skip_3dworld", action="store_true",
                   help="Skip levels whose gamestyle is Super Mario 3D World")
    p.add_argument("--skip_items", action="store_true",
                   help="Skip levels containing any object listed in SKIP_ITEM_NAMES")
    p.add_argument("--skip_subworld_items", action="store_true",
                   help="Skip levels whose subworld contains any items")
    return p.parse_args()


def main():
    args = parse_args()
    extract_levels(
        output_dir=args.output_dir,
        limit=args.limit,
        streaming=not args.no_stream,
        data_id_filter=set(args.ids) if args.ids else None,
        name_filter=args.name,
        name_count=args.name_count,
        skip_3dworld=args.skip_3dworld,
        skip_items=args.skip_items,
        skip_subworld_items=args.skip_subworld_items,
    )


if __name__ == "__main__":
    main()
