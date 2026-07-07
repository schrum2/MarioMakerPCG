"""Single entry point for the whole modelling pipeline.

    python -m mm2pipeline_experiment <command> [options]

Commands (in pipeline order):
    prepare     captioned dataset -> split + tokenizer + random-caption test set
    mlm         train the MLM text encoder
    diffusion   train the text-conditional diffusion model
    generate    sample levels from a trained diffusion model
    evaluate    run the test suite (caption adherence, tile distribution, MM2 metrics, masked-token)
    run         everything above, end to end

Each command forwards to its stage module, so `python -m mm2pipeline_experiment mlm`
and `python -m mm2pipeline_experiment.mlm` are the same thing. Run any command with
--help for its options.
"""
import sys


def _usage():
    print(__doc__.strip())


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _usage()
        sys.exit(0 if argv else 2)

    command, rest = argv[0], argv[1:]

    # Import lazily so one stage's dependencies aren't pulled in to run another.
    if command == "prepare":
        from . import prepare
        prepare.main(rest)
    elif command == "mlm":
        from . import mlm
        mlm.main(rest)
    elif command == "diffusion":
        from . import diffusion
        diffusion.main(rest)
    elif command == "generate":
        from . import generate
        generate.main(rest)
    elif command == "evaluate":
        from . import evaluate
        evaluate.main(rest)
    elif command == "run":
        from . import experiment
        experiment.main(rest)
    else:
        print(f"Unknown command: {command!r}\n")
        _usage()
        sys.exit(2)


if __name__ == "__main__":
    main()
