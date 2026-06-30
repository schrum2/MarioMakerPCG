import argparse
import json
import os
import random

"""
Cut a dataset (a JSON array of entries) into N separate .json datasets of
roughly equal size. Every entry ends up in exactly one part and no entry is
dropped or duplicated, so concatenating the parts reproduces the input set.

COMMAND LINE: python split_dataset_into_n.py --json ds_captioned_smolvlm.json --parts 4
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut a dataset JSON array into N separate .json datasets of equal size."
    )
    parser.add_argument("--json", type=str, required=True, help="Path to dataset JSON file (a JSON array)")
    parser.add_argument("--parts", type=int, required=True, help="Number of parts to cut the dataset into")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle entries before cutting")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when --shuffle is set")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory for the output parts (defaults to the input file's directory)")
    return parser.parse_args()


def even_chunk_sizes(n, parts):
    """Return a list of `parts` sizes that sum to n and differ by at most one.

    The first `n % parts` chunks get one extra entry so nothing is left over."""
    base = n // parts
    remainder = n % parts
    return [base + 1 if i < remainder else base for i in range(parts)]


def split_into_parts(data, parts):
    """Slice `data` into `parts` contiguous chunks using even_chunk_sizes."""
    chunks = []
    start = 0
    for size in even_chunk_sizes(len(data), parts):
        chunks.append(data[start:start + size])
        start += size
    return chunks


def main():
    args = parse_args()

    if args.parts < 1:
        raise ValueError("--parts must be at least 1")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{args.json} does not contain a JSON list/array of entries")

    if args.parts > len(data):
        raise ValueError(f"Cannot cut {len(data)} entries into {args.parts} parts; "
                         f"--parts must be <= number of entries")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(data)

    chunks = split_into_parts(data, args.parts)

    base, ext = os.path.splitext(args.json)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        base = os.path.join(args.outdir, os.path.basename(base))

    # Zero-pad the index so the parts sort naturally (part01, part02, ... part10).
    width = len(str(args.parts))
    for i, chunk in enumerate(chunks, start=1):
        path = f"{base}-part{i:0{width}d}{ext}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2)
        print(f"Saved {len(chunk)} entries -> {path}")

    print(f"Cut {len(data)} entries into {args.parts} parts.")


if __name__ == "__main__":
    main()
