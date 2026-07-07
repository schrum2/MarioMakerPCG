"""Run the model test suite on a trained diffusion model (and its MLM encoder):

    captions   caption adherence, on the real captions and a random-caption set
    tiles      generated-vs-training tile distribution
    metrics    MM2 playability / diversity metrics on the generated levels
    mlm        masked-token prediction, scoring the MLM encoder itself

By default every applicable test runs (mlm only when an MLM encoder dir is given).
Each wraps the matching evaluate_*.py script, the same ones the run_full_pipeline
batch scripts call after training.
"""
import argparse
import os

from . import steps

TESTS = ("captions", "tiles", "metrics", "mlm")


def _caption_dir(model_path, game, kind):
    # evaluate_caption_adherence.py writes output_dir inside the model dir.
    name = f"samples-from-{kind}-{game}-captions"
    return name, os.path.join(model_path, name)


def caption_adherence(model_path, dataset, random_json, *, game="MM", num_tiles=None,
                      tileset="mm2_tileset_we.json", compare_checkpoints=False):
    """Score how well generated levels match their prompt, on the real captions
    and on a random-caption set. Returns the real run's all_levels.json path (the
    input the metrics test wants)."""
    num_tiles = steps.resolve_num_tiles(num_tiles, tileset)
    real_name, real_path = _caption_dir(model_path, game, "real")
    rand_name, _ = _caption_dir(model_path, game, "random")

    def run(json_path, out_name):
        # The eval script refuses to overwrite; clear the regenerable dir first.
        steps.wipe(os.path.join(model_path, out_name))
        steps.run_script("evaluate_caption_adherence.py",
                         ["--model_path", model_path, "--json", json_path, "--save_as_json",
                          "--output_dir", out_name, "--num_tiles", num_tiles, "--tileset", tileset])

    run(dataset, real_name)
    run(random_json, rand_name)

    if compare_checkpoints:
        # The heavier per-checkpoint sweep the evaluate_caption_adherence_multi
        # batch script runs; --resume lets an interrupted sweep pick back up.
        p = steps.dataset_paths(dataset)
        for json_path in (dataset, p.test, random_json):
            steps.run_script("evaluate_caption_adherence.py",
                             ["--model_path", model_path, "--json", json_path, "--save_as_json",
                              "--compare_checkpoints", "--resume",
                              "--num_tiles", num_tiles, "--tileset", tileset])

    return os.path.join(real_path, "all_levels.json")


def tile_distribution(model_path, dataset, *, num_tiles=None,
                      tileset="mm2_tileset_we.json", seed=0):
    """Compare the generated tile distribution against the training set's."""
    num_tiles = steps.resolve_num_tiles(num_tiles, tileset)
    steps.run_script("evaluate_tile_distribution.py",
                     ["--model_path", model_path, "--num_tiles", num_tiles,
                      "--tileset", tileset, "--captions_json", dataset, "--seed", seed])


def mm2_metrics(generated_json, *, real_json=None, tileset="mm2_tileset_we.json"):
    """MM2 playability / diversity metrics on a generated all_levels.json."""
    if not os.path.isfile(generated_json):
        raise SystemExit(
            f"ERROR: no generated levels at {generated_json}. Run the 'captions' test "
            "first, or pass --generated_json to point at an all_levels.json.")
    args = ["--json", generated_json, "--tileset", tileset, "--override"]
    if real_json:
        args += ["--real_json", real_json]
    steps.run_script("evaluate_mm2_metrics.py", args)


def masked_token(mlm_dir, test_json, *, num_samples=10):
    """Masked-token prediction accuracy of the trained MLM encoder."""
    steps.run_script("evaluate_masked_token_prediction.py",
                     ["--model_path", mlm_dir, "--json", test_json,
                      "--num_samples", num_samples, "--compare_checkpoints"])


def run_all(model_path, dataset, *, tests=None, mlm_dir=None, random_json=None,
            game="MM", num_tiles=None, tileset="mm2_tileset_we.json", seed=0,
            compare_checkpoints=False, generated_json=None, real_json=None):
    """Run the requested tests (default: all that apply). The metrics test reuses
    the all_levels.json the captions test produces, so ordering matters."""
    tests = list(tests) if tests else [t for t in TESTS if t != "mlm" or mlm_dir]
    p = steps.dataset_paths(dataset, random_json=random_json)
    num_tiles = steps.resolve_num_tiles(num_tiles, tileset)

    if "captions" in tests:
        generated = caption_adherence(model_path, p.dataset, p.random, game=game,
                                      num_tiles=num_tiles, tileset=tileset,
                                      compare_checkpoints=compare_checkpoints)
        generated_json = generated_json or generated
    if "tiles" in tests:
        tile_distribution(model_path, p.dataset, num_tiles=num_tiles,
                          tileset=tileset, seed=seed)
    if "metrics" in tests:
        _, real_path = _caption_dir(model_path, game, "real")
        gen = generated_json or os.path.join(real_path, "all_levels.json")
        mm2_metrics(gen, real_json=real_json or p.train, tileset=tileset)
    if "mlm" in tests:
        if not mlm_dir:
            raise SystemExit("ERROR: the 'mlm' test needs --mlm_dir (the trained MLM encoder).")
        masked_token(mlm_dir, p.test)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the model test suite.")
    parser.add_argument("tests", nargs="*", choices=TESTS + ("all",), default=[],
                        help="Which tests to run (default: all that apply).")
    parser.add_argument("--model_path", required=True, help="Trained diffusion model dir.")
    parser.add_argument("--dataset", required=True, help="Captioned dataset JSON.")
    parser.add_argument("--mlm_dir", default=None,
                        help="Trained MLM encoder dir. Required for (and enables) the 'mlm' test.")
    parser.add_argument("--random_json", default=None,
                        help="Random-caption test set. Default: <dataset>_random.json.")
    parser.add_argument("--generated_json", default=None,
                        help="all_levels.json for the 'metrics' test. Default: the "
                             "captions test's real-caption output.")
    parser.add_argument("--real_json", default=None,
                        help="Training-set json baseline for the 'metrics' test. Default: the train split.")
    parser.add_argument("--compare_checkpoints", action="store_true",
                        help="Also sweep caption adherence across every checkpoint (slow).")
    steps.add_common_args(parser)
    args = parser.parse_args(argv)

    tests = [] if "all" in args.tests else args.tests
    run_all(args.model_path, args.dataset, tests=tests, mlm_dir=args.mlm_dir,
            random_json=args.random_json, game=args.game, num_tiles=args.num_tiles,
            tileset=args.tileset, seed=args.seed, compare_checkpoints=args.compare_checkpoints,
            generated_json=args.generated_json, real_json=args.real_json)


if __name__ == "__main__":
    main()
