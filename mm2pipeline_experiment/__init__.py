"""mm2pipeline_experiment — the Mario Maker 2 modelling pipeline.

Where mm2pipeline_data moves real levels into a training dataset, this package
takes that captioned dataset the rest of the way: it trains the text encoder and
diffusion model, generates samples, and runs the evaluation suite. Each stage
shells out to the existing repo-root script (train_mlm.py, train_diffusion.py,
run_diffusion.py, evaluate_*.py), so the package is an orchestrator, not a rewrite.

Stages
------
    prepare    captioned dataset -> train/val/test split + tokenizer + random test set
    mlm        train the MLM text encoder            (train_mlm.py)
    diffusion  train the text-conditional diffusion  (train_diffusion.py)
    generate   sample levels from a trained model    (run_diffusion.py)
    evaluate   caption adherence / tile distribution / MM2 metrics / masked-token
    run        all of the above, end to end          (mm2pipeline_experiment.experiment)

``python -m mm2pipeline_experiment <command>`` (see mm2pipeline_experiment.__main__)
dispatches to the stages; each stage module also exposes a ``main()`` so it runs
directly as ``python -m mm2pipeline_experiment.<stage>``. See
mm2pipeline_experiment/README.md for full usage.
"""

__version__ = "1.0.0"
