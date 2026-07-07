"""Train the text-conditional diffusion model (train_diffusion.py), conditioned
on either our own trained MLM encoder or a frozen pretrained one (MiniLM, GTE).
"""
import argparse

from . import steps


def train(train_json, val_json, tokenizer, output_dir, *, game="MM",
          tileset="mm2_tileset_we.json", num_tiles=None, text_encoder="MLM",
          mlm_dir="mlm", num_epochs=500, save_image_epochs=20, save_model_epochs=20,
          seed=0, fresh=False):
    """Train the diffusion model into output_dir. text_encoder picks the
    conditioning encoder: "MLM" uses mlm_dir, anything else is resolved to a
    frozen pretrained model."""
    num_tiles = steps.resolve_num_tiles(num_tiles, tileset)
    kind, model_id = steps.resolve_text_encoder(text_encoder)
    if fresh:
        steps.wipe(output_dir)

    args = ["--game", game, "--num_tiles", num_tiles, "--tileset", tileset,
            "--augment", "--text_conditional",
            "--json", train_json, "--val_json", val_json, "--pkl", tokenizer,
            "--output_dir", output_dir, "--num_epochs", num_epochs,
            "--save_image_epochs", save_image_epochs, "--save_model_epochs", save_model_epochs,
            "--plot_validation_caption_score", "--seed", seed]
    if kind == "mlm":
        args += ["--mlm_model_dir", mlm_dir]
    else:
        args += ["--pretrained_language_model", model_id]
    # auto_yes answers the resume-from-checkpoint prompt when output_dir survives.
    steps.run_script("train_diffusion.py", args, auto_yes=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the text-conditional diffusion model.")
    parser.add_argument("--dataset", required=True,
                        help="Captioned dataset JSON; its -train / -validate splits are used.")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer pkl. Default: <dataset>_tokenizer.pkl.")
    parser.add_argument("--output_dir", default="diffusion", help="Where to write the model.")
    parser.add_argument("--text_encoder", default="MLM",
                        help="Conditioning encoder: MLM (our trained one), MiniLM, GTE, "
                             "or a raw HuggingFace id.")
    parser.add_argument("--mlm_dir", default="mlm",
                        help="Trained MLM encoder dir (used when --text_encoder MLM).")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--save_image_epochs", type=int, default=20)
    parser.add_argument("--save_model_epochs", type=int, default=20)
    parser.add_argument("--fresh", action="store_true",
                        help="Delete an existing output dir first instead of resuming from it.")
    steps.add_common_args(parser)
    args = parser.parse_args(argv)

    p = steps.dataset_paths(args.dataset, args.tokenizer)
    train(p.train, p.validate, p.tokenizer, args.output_dir, game=args.game,
          tileset=args.tileset, num_tiles=args.num_tiles, text_encoder=args.text_encoder,
          mlm_dir=args.mlm_dir, num_epochs=args.num_epochs,
          save_image_epochs=args.save_image_epochs, save_model_epochs=args.save_model_epochs,
          seed=args.seed, fresh=args.fresh)


if __name__ == "__main__":
    main()
