"""Run a whole modelling experiment end to end from a captioned dataset:

    prepare -> (train MLM, if using the MLM encoder) -> train diffusion
            -> generate samples -> run the test suite.

This is the Python equivalent of run_full_pipeline.bat steps 2-6; the data-build
and captioning steps that come before it live in mm2pipeline_data.
"""
import argparse

from . import diffusion, evaluate, generate, mlm, prepare, steps


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train + evaluate a diffusion model end to end from a captioned dataset.")
    parser.add_argument("--dataset", required=True, help="Captioned dataset JSON.")
    parser.add_argument("--text_encoder", default="MLM",
                        help="Conditioning encoder: MLM (train our own), MiniLM, GTE, "
                             "or a raw HuggingFace id.")
    parser.add_argument("--mlm_dir", default=None,
                        help="MLM encoder dir. Default: <game>-MLM.")
    parser.add_argument("--diffusion_dir", default=None,
                        help="Diffusion output dir. Default: <game>-conditional.")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer pkl. Default: <dataset>_tokenizer.pkl.")
    parser.add_argument("--mlm_epochs", type=int, default=300)
    parser.add_argument("--num_epochs", type=int, default=500, help="Diffusion training epochs.")
    parser.add_argument("--save_image_epochs", type=int, default=20)
    parser.add_argument("--save_model_epochs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=100, help="Samples per generation batch.")
    parser.add_argument("--no_prepare", action="store_true",
                        help="Assume the splits, tokenizer and random test set already exist.")
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe the model output dirs before training instead of resuming.")
    parser.add_argument("--compare_checkpoints", action="store_true",
                        help="Include the slow per-checkpoint caption-adherence sweep.")
    parser.add_argument("--tests", nargs="*", choices=evaluate.TESTS + ("all",), default=[],
                        help="Which tests to run at the end (default: all that apply).")
    steps.add_common_args(parser)
    args = parser.parse_args(argv)

    num_tiles = steps.resolve_num_tiles(args.num_tiles, args.tileset)
    kind, _ = steps.resolve_text_encoder(args.text_encoder)
    mlm_dir = args.mlm_dir or f"{args.game}-MLM"
    diffusion_dir = args.diffusion_dir or f"{args.game}-conditional"

    p = steps.dataset_paths(args.dataset, args.tokenizer)

    print("\n=== [1/5] preparing dataset (split + tokenizer + random captions) ===")
    if args.no_prepare:
        print("  skipped (--no_prepare)")
    else:
        prepare.prepare(p.dataset, tileset=args.tileset, tokenizer=p.tokenizer,
                        random_json=p.random, seed=args.seed)

    if kind == "mlm":
        print("\n=== [2/5] training MLM text encoder ===")
        mlm.train(p.train, p.test, p.tokenizer, mlm_dir,
                  epochs=args.mlm_epochs, seed=args.seed)
    else:
        print(f"\n=== [2/5] MLM training skipped (frozen encoder '{args.text_encoder}') ===")

    print("\n=== [3/5] training conditional diffusion ===")
    diffusion.train(p.train, p.validate, p.tokenizer, diffusion_dir, game=args.game,
                    tileset=args.tileset, num_tiles=num_tiles, text_encoder=args.text_encoder,
                    mlm_dir=mlm_dir, num_epochs=args.num_epochs,
                    save_image_epochs=args.save_image_epochs,
                    save_model_epochs=args.save_model_epochs, seed=args.seed, fresh=args.fresh)

    print("\n=== [4/5] generating samples ===")
    generate.generate(diffusion_dir, tileset=args.tileset, game=args.game,
                      num_samples=args.num_samples, seed=args.seed)

    print("\n=== [5/5] running the test suite ===")
    tests = [] if "all" in args.tests else args.tests
    evaluate.run_all(diffusion_dir, p.dataset, tests=tests,
                     mlm_dir=mlm_dir if kind == "mlm" else None, random_json=p.random,
                     game=args.game, num_tiles=num_tiles, tileset=args.tileset, seed=args.seed,
                     compare_checkpoints=args.compare_checkpoints, real_json=p.train)

    print("\n=== done ===")
    if kind == "mlm":
        print(f"  MLM:       {mlm_dir}")
    print(f"  Diffusion: {diffusion_dir}")


if __name__ == "__main__":
    main()
