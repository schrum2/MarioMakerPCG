#!/usr/bin/env python3
"""
summarize_dataset.py
====================
Write a short info file summarizing a finished pipeline run: the level-name
filter passed to extract_levels_to_ascii.bat, how many levels were extracted
(one .txt each), how many samples ended up in the captioned dataset, and which
model captioned them ("simple captions" or "N/A" when no LLM was used).

extract_levels_to_ascii.bat writes the name and extracted-level count to an
extract_info.txt in the extraction output folder (the one holding the
bcd/json/images/ascii subfolders). This script finds that file by walking up
from --input, counts the samples in the captioned JSON, and writes the summary
next to the dataset.
"""

import argparse
import json
import os


def find_extract_info(start_path):
    """Walk up from start_path looking for an extract_info.txt.

    start_path is the pipeline's --input (the ascii folder, or a single ascii
    file). The info file sits in the extraction output folder, one level above
    the ascii folder, so checking a couple of parents is enough.
    """
    path = os.path.abspath(start_path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    for _ in range(3):
        candidate = os.path.join(path, "extract_info.txt")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def parse_info_file(path):
    """Read simple key=value lines (one per line) into a dict."""
    info = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            key, sep, value = line.partition("=")
            if sep:
                info[key.strip()] = value.strip()
    return info


def count_ascii_levels(input_path):
    """Fallback level count: number of .txt files in the ascii input folder."""
    if os.path.isdir(input_path):
        return sum(1 for f in os.listdir(input_path) if f.lower().endswith(".txt"))
    if os.path.isfile(input_path):
        return 1
    return 0


def count_samples(dataset_path):
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        return len(json.load(f))


def main():
    parser = argparse.ArgumentParser(
        description="Write a dataset info summary (source name, level/sample "
                    "counts, caption model) next to a finished dataset."
    )
    parser.add_argument("--input", required=True,
                        help="The ascii folder (or file) the pipeline ran on, used to "
                             "locate the extract_info.txt left by extract_levels_to_ascii.bat.")
    parser.add_argument("--dataset", required=True,
                        help="The captioned dataset JSON whose samples are counted.")
    parser.add_argument("--caption-model", default="",
                        help="Label for the captioner: an LLM model name, "
                             "'simple captions', or left empty for 'N/A'.")
    parser.add_argument("--output", required=True,
                        help="Path of the info file to write.")
    args = parser.parse_args()

    name = "unknown"
    levels_extracted = None
    info_path = find_extract_info(args.input)
    if info_path:
        info = parse_info_file(info_path)
        name = info.get("name") or name
        try:
            levels_extracted = int(info["levels_extracted"])
        except (KeyError, ValueError):
            pass

    # No extract_info.txt (or it lacked a count): fall back to counting files.
    if levels_extracted is None:
        levels_extracted = count_ascii_levels(args.input)

    samples = count_samples(args.dataset)
    caption_model = args.caption_model.strip() or "N/A"

    lines = [
        f"name: {name}",
        f"levels extracted: {levels_extracted}",
        f"dataset samples: {samples}",
        f"caption model: {caption_model}",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Dataset info written to {args.output}")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
