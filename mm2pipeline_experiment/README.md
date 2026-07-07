# mm2pipeline_experiment

Where [`mm2pipeline_data`](../mm2pipeline_data/README.md) turns real levels into a
captioned training dataset, this package takes that dataset the rest of the way:
it trains the text encoder and diffusion model, generates samples, and runs the
evaluation suite. Run any stage through the single entry point (add `--help` for
its options):

```
python -m mm2pipeline_experiment <command> [options]
```

Each stage shells out to the existing repo-root script (`train_mlm.py`,
`train_diffusion.py`, `run_diffusion.py`, `evaluate_*.py`) — the package is an
orchestrator around them, not a rewrite. It's the Python equivalent of
`bat\run_full_pipeline.bat` steps 2–6.

| Command     | Does                                                        | Wraps                          |
|-------------|-------------------------------------------------------------|--------------------------------|
| `prepare`   | captioned dataset → split + tokenizer + random test set     | `mm2pipeline_data.dataset split`, `tokenizer.py`, `create_mario_maker_random_captions.py` |
| `mlm`       | train the MLM text encoder                                  | `train_mlm.py`                 |
| `diffusion` | train the text-conditional diffusion model                 | `train_diffusion.py`           |
| `generate`  | sample levels from a trained model                          | `run_diffusion.py`             |
| `evaluate`  | caption adherence / tile distribution / MM2 metrics / MLM   | `evaluate_*.py`                |
| `run`       | all of the above, end to end                                | — |

Each command maps to a module you can also run directly, e.g.
`python -m mm2pipeline_experiment.diffusion`.

## The whole experiment in one command

Point `run` at a captioned dataset (the output of the `mm2pipeline_data` pipeline
plus a captioning pass) and it does prepare → train MLM → train diffusion →
generate → evaluate:

```bat
python -m mm2pipeline_experiment run --dataset datasets\MM_LevelsAndCaptions-regular.json
```

`--text_encoder` picks how the diffusion model is conditioned, and decides whether
the MLM training step runs at all:

```bat
REM Train our own MLM encoder (default).
python -m mm2pipeline_experiment run --dataset datasets\MM_LevelsAndCaptions-regular.json --text_encoder MLM

REM Freeze a pretrained encoder instead — no MLM training step.
python -m mm2pipeline_experiment run --dataset datasets\MM_LevelsAndCaptions-regular.json --text_encoder MiniLM
python -m mm2pipeline_experiment run --dataset datasets\MM_LevelsAndCaptions-regular.json --text_encoder Alibaba-NLP/gte-large-en-v1.5
```

`MLM`, `MiniLM` and `GTE` are recognised names; anything else is passed through as
a raw HuggingFace id. By default the models land in `<game>-MLM` and
`<game>-conditional`; override with `--mlm_dir` / `--diffusion_dir`.

Useful `run` options: `--no_prepare` (splits/tokenizer already exist), `--fresh`
(wipe the model dirs instead of resuming), `--num_epochs`, `--mlm_epochs`,
`--tests captions tiles` (run only some tests), `--compare_checkpoints` (add the
slow per-checkpoint caption-adherence sweep).

## Running a stage on its own

Every stage takes the same `--dataset` and derives the split / tokenizer / random
paths from it (matching what `prepare` writes), so they compose:

```bat
REM Build the splits + tokenizer + random-caption test set.
python -m mm2pipeline_experiment prepare --dataset datasets\MM_LevelsAndCaptions-regular.json

REM Train just the MLM encoder, then just the diffusion model.
python -m mm2pipeline_experiment mlm --dataset datasets\MM_LevelsAndCaptions-regular.json --output_dir MM-MLM
python -m mm2pipeline_experiment diffusion --dataset datasets\MM_LevelsAndCaptions-regular.json --output_dir MM-conditional --text_encoder MLM --mlm_dir MM-MLM

REM Sample from a trained model.
python -m mm2pipeline_experiment generate --model_path MM-conditional

REM Run the whole test suite (or a subset) against a trained model.
python -m mm2pipeline_experiment evaluate --model_path MM-conditional --dataset datasets\MM_LevelsAndCaptions-regular.json --mlm_dir MM-MLM
python -m mm2pipeline_experiment evaluate --model_path MM-conditional --dataset datasets\MM_LevelsAndCaptions-regular.json captions tiles
```

## The tests

`evaluate` runs, by default, every test that applies (the `mlm` test only when a
`--mlm_dir` is given):

- **captions** — `evaluate_caption_adherence.py` on the real captions and on a
  random-caption set. Also produces the `all_levels.json` the `metrics` test reads.
- **tiles** — `evaluate_tile_distribution.py`: generated vs. training tile mix.
- **metrics** — `evaluate_mm2_metrics.py`: MM2 playability / diversity on the
  generated levels, against the training split.
- **mlm** — `evaluate_masked_token_prediction.py`: masked-token accuracy of the
  trained MLM encoder.

Run `metrics` on its own only after `captions` has produced an `all_levels.json`,
or point it at one with `--generated_json`.

## A few conventions

Paths key off the captioned dataset. For `--dataset .../foo.json` the stages use
`.../foo-train.json`, `-validate`, `-test` (the split names
`mm2pipeline_data.dataset split` writes), `.../foo_tokenizer.pkl`, and
`.../foo_random.json` — each overridable with an explicit flag.

`--num_tiles` defaults to the tileset's glyph count plus one padding id (68 for
`mm2_tileset_we.json`), and `run` threads one resolved value through both training
and evaluation so they always share a tile-id space. Override it only to match a
model trained with a different count.
