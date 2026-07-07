"""Train the masked-language-model text encoder (train_mlm.py) that conditions
the diffusion model.

Only the "MLM" text encoder needs this stage; the frozen pretrained encoders
(MiniLM, GTE, ...) skip it -- see mm2pipeline_experiment.run.
"""
import argparse

from . import steps


def train(train_json, test_json, tokenizer, output_dir, *, epochs=300, seed=0,
          save_checkpoints=True):
    """Train the MLM encoder into output_dir from the dataset's train split (with
    test_json used for the final evaluation)."""
    args = ["--json", train_json, "--pkl", tokenizer, "--output_dir", output_dir,
            "--epochs", epochs, "--seed", seed]
    if test_json:
        args += ["--test_json", test_json]
    if save_checkpoints:
        args.append("--save_checkpoints")
    steps.run_script("train_mlm.py", args)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the MLM text encoder.")
    parser.add_argument("--dataset", required=True,
                        help="Captioned dataset JSON; its -train / -test splits are used.")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer pkl. Default: <dataset>_tokenizer.pkl.")
    parser.add_argument("--output_dir", default="mlm", help="Where to write the trained encoder.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_checkpoints", action="store_true",
                        help="Don't save periodic checkpoints (needed by the masked-token eval).")
    args = parser.parse_args(argv)

    p = steps.dataset_paths(args.dataset, args.tokenizer)
    train(p.train, p.test, p.tokenizer, args.output_dir,
          epochs=args.epochs, seed=args.seed, save_checkpoints=not args.no_checkpoints)


if __name__ == "__main__":
    main()
