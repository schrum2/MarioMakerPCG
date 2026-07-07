# Multi-source captions

Two caption schemes are supported end to end (captioning, merging, and training):

- **legacy** — a single `"caption"` field per scene (plus optional `"caption1"`, `"caption2"`,
  ...). This is the default everywhere and what every existing pipeline uses.
- **keyed (multi-source)** — no `"caption"` field. Each scene carries one or more source keys,
  each holding a **list** of captions:

  ```json
  {
    "name": "3182316_overworld/source_0_2",
    "scene": [[...], ...],
    "gemma3:27b_captions": ["...", "..."],
    "qwen3:32b_captions": ["..."],
    "deterministic_captions": ["..."]
  }
  ```

The point is to build up **one** dataset of scenes with captions from many sources, then train
on whatever combination of sources you want by naming them with `--caption_source_keys`.

## Building the dataset

Every captioner copies all input attributes to its output, so you feed the output of one pass in
as the input of the next and the sources accumulate. Metadata rides along for free.

1. **Deterministic captions first** (cheap, covers everything):

   ```
   python MarioMaker_create_ascii_captions.py --dataset raw.json --tileset mm2_tileset_we.json \
       --output captioned.json --caption-mode keyed
   ```

   Writes `"deterministic_captions"`. (Default key; override with `--caption-key`.)

2. **Local LLMs** (Ollama). Point `--dataset` at the previous output and `--caption-mode keyed`.
   The key defaults to `"<model>_captions"`:

   ```
   python MarioMaker_llm_captions.py --dataset captioned.json --output captioned.json \
       --backend ollama --model gemma3:27b --caption-mode keyed \
       --tileset mm2_tileset_we.json --grid-format tokens --tileset-we mm2_tileset_we.json
   ```

   Run again with a different `--model` (e.g. `qwen3:32b`) to add another source. Reading and
   writing the same file works because unfinished scenes resume and finished ones are skipped.

3. **Paid LLMs** (Claude / OpenAI / Gemini). Same command with `--backend claude --model
   claude-sonnet-4-6 --api-key-file key.txt`.

Order does not matter — you can add deterministic captions before or after the LLM passes. If you
build sources in separate files instead of one accumulating file, combine them afterward with
`merge_caption_sources.py` (matches scenes by `name`):

```
python merge_caption_sources.py --inputs gemma.json qwen.json det.json \
    --keys gemma3:27b_captions qwen3:32b_captions deterministic_captions --output merged.json
```

## Training on it

Pass the source keys to draw captions from; one caption is picked at random per access across all
of them. Omit the flag to stay in legacy mode.

```
python train_diffusion.py --text_conditional --json train.json --val_json val.json ... \
    --caption_source_keys gemma3:27b_captions qwen3:32b_captions deterministic_captions
```

`train_mlm.py` takes the same `--caption_source_keys`.

Two ways to use a partially-captioned source (e.g. a source that only covers some scenes):

- **Only that source** — `--caption_source_keys claude_sonnet4.6_captions`. Scenes with an empty
  list are dropped, so you train only on Claude-captioned scenes.
- **That source plus fallbacks** — `--caption_source_keys claude_sonnet4.6_captions
  deterministic_captions`. Scenes with a Claude caption can use it; the rest fall back to whatever
  other sources they have.
