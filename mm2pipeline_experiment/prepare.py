"""Turn a captioned dataset JSON into what the trainers and evaluators need:
train / validate / test splits, a tokenizer, and a random-caption test set.

Wraps the same three steps the run_full_pipeline batch scripts run before
training -- mm2pipeline_data.dataset split, tokenizer.py, and
create_mario_maker_random_captions.py -- so a captioned dataset from
mm2pipeline_data is ready to hand to the mlm / diffusion stages.
"""
import argparse

from . import steps


def prepare(dataset, *, tileset="mm2_tileset_we.json", tokenizer=None,
            random_json=None, seed=0, skip_random=False):
    """Split the dataset, build its tokenizer, and (unless skipped) a random
    caption test set. Returns the derived dataset paths."""
    p = steps.dataset_paths(dataset, tokenizer, random_json)
    steps.run_module("mm2pipeline_data.dataset",
                     ["split", "--input", p.dataset, "--seed", seed])
    steps.run_script("tokenizer.py",
                     ["save", "--json_file", p.train, "--pkl_file", p.tokenizer])
    if not skip_random:
        steps.run_script("create_mario_maker_random_captions.py",
                         ["--json", p.dataset, "--output", p.random, "--tileset", tileset])
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Split a captioned dataset and build its tokenizer + random-caption test set.")
    parser.add_argument("--dataset", required=True, help="Captioned dataset JSON.")
    parser.add_argument("--tileset", default="mm2_tileset_we.json",
                        help="Tileset the captions were built with (for the random test set).")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer pkl path. Default: <dataset>_tokenizer.pkl.")
    parser.add_argument("--random_json", default=None,
                        help="Random-caption test set path. Default: <dataset>_random.json.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_random", action="store_true",
                        help="Don't build the random-caption test set.")
    args = parser.parse_args(argv)
    prepare(args.dataset, tileset=args.tileset, tokenizer=args.tokenizer,
            random_json=args.random_json, seed=args.seed, skip_random=args.skip_random)


if __name__ == "__main__":
    main()
