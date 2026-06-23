import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Split a Mario Maker dataset into train/val/test sets.")
    parser.add_argument("--json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--val_pct", type=float, default=0.1)
    parser.add_argument("--test_pct", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if abs(args.train_pct + args.val_pct + args.test_pct - 1.0) > 1e-6:
        raise ValueError("Train/val/test percentages must sum to 1.0")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    n = len(data)
    # int() truncation can zero out val/test entirely on small datasets (e.g. n=6,
    # val_pct=0.1 -> int(0.6)=0). Round instead, and guarantee at least one sample
    # for any split with a non-zero requested percentage, borrowing from
    # whichever split currently has the most so train still gets priority.
    n_val = max(1, round(args.val_pct * n)) if args.val_pct > 0 and n > 0 else 0
    n_test = max(1, round(args.test_pct * n)) if args.test_pct > 0 and n > 0 else 0
    n_train = n - n_val - n_test
    while n_train < 1 and (n_val > 0 or n_test > 0) and n > 0:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        n_train = n - n_val - n_test

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    base, ext = os.path.splitext(args.json)
    splits = {
        f"{base}-train{ext}": train_data,
        f"{base}-validate{ext}": val_data,
        f"{base}-test{ext}": test_data,
    }

    for path, split in splits.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        print(f"Saved {len(split)} samples -> {path}")


if __name__ == "__main__":
    main()
