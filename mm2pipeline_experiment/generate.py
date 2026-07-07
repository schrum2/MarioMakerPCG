"""Sample levels from a trained diffusion model (run_diffusion.py).

Mirrors run_diffusion_multi.bat: an unconditional batch at the model's native
width and a second, wider batch, each saved as images plus an all_levels.json.
"""
import argparse

from . import steps


def generate(model_path, *, tileset="mm2_tileset_we.json", game="MM",
             num_samples=100, long_width=128, seed=0):
    """Generate two unconditional sample sets (native width + a wide one) next to
    the model, as <model_path>-unconditional-samples-{short,long}."""
    base = f"{model_path}-unconditional-samples"
    common = ["--model_path", model_path, "--num_samples", num_samples,
              "--save_as_json", "--output_format", "image",
              "--tileset", tileset, "--game", game, "--seed", seed]
    steps.run_script("run_diffusion.py", common + ["--output_dir", f"{base}-short"])
    steps.run_script("run_diffusion.py",
                     common + ["--output_dir", f"{base}-long", "--level_width", long_width])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate levels from a trained diffusion model.")
    parser.add_argument("--model_path", required=True, help="Trained diffusion model dir.")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--long_width", type=int, default=128,
                        help="Width (tiles) of the second, wider sample batch.")
    parser.add_argument("--game", default="MM", choices=["MM", "Mario"])
    parser.add_argument("--tileset", default="mm2_tileset_we.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    generate(args.model_path, tileset=args.tileset, game=args.game,
             num_samples=args.num_samples, long_width=args.long_width, seed=args.seed)


if __name__ == "__main__":
    main()
