import argparse
import glob
import json
import os
import random

"""
Merge a bunch of dataset JSON arrays into one. Inverse of split_dataset_into_n.py:
feed the parts back in and you get the original set (order aside, unless you shuffle).

Inputs can be files, glob patterns, or directories (any .json inside gets picked up).

COMMAND LINE: python merge_datasets_into_one.py ds-part*.json --out ds_merged.json
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge several dataset JSON arrays into one .json dataset."
    )
    parser.add_argument("inputs", nargs="+",
                        help="Input datasets: files, glob patterns, or directories of .json files")
    parser.add_argument("--out", type=str, required=True, help="Path to write the merged dataset")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the merged entries")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when --shuffle is set")
    parser.add_argument("--dedup", action="store_true", help="Drop exact-duplicate entries")
    return parser.parse_args()


def resolve_inputs(patterns):
    # Glob everything ourselves so wildcards work even when the shell won't expand
    # them (Windows). Directories mean "every .json in here".
    paths = []
    seen = set()
    for pattern in patterns:
        if os.path.isdir(pattern):
            matches = sorted(glob.glob(os.path.join(pattern, "*.json")))
        else:
            matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"No files matched '{pattern}'")
        for path in matches:
            key = os.path.abspath(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def load_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list/array of entries")
    return data


def main():
    args = parse_args()

    out_key = os.path.abspath(args.out)
    paths = [p for p in resolve_inputs(args.inputs) if os.path.abspath(p) != out_key]
    if not paths:
        raise ValueError("No input datasets to merge")

    merged = []
    for path in paths:
        entries = load_entries(path)
        merged.extend(entries)
        print(f"Read {len(entries)} entries <- {path}")

    if args.dedup:
        unique = []
        seen = set()
        for entry in merged:
            key = json.dumps(entry, sort_keys=True)  # make dicts/lists hashable
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        if len(unique) < len(merged):
            print(f"Dropped {len(merged) - len(unique)} duplicate entries")
        merged = unique

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(merged)

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(paths)} datasets into {len(merged)} entries -> {args.out}")


if __name__ == "__main__":
    main()
