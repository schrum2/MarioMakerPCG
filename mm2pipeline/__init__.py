"""mm2pipeline — the Mario Maker 2 format-conversion pipeline.

One package for every step that moves data through the project, anchored on the
canonical training tileset ``mm2_tileset_we.json`` (see ``mm2pipeline.tiles``).

Stages
------
    extract   HuggingFace mm2_level dataset   -> .bcd        (mm2pipeline.extract)
    toost     .bcd                            -> .json + .png (mm2pipeline.toost)
    ascii     .json  <->  ASCII grid          (mm2pipeline.ascii)
    dataset   ASCII  ->  tile-id dataset + split (mm2pipeline.dataset)
    swe       .json  ->  .swe (playable)      (mm2pipeline.swe)

``python -m mm2pipeline <command>`` (see ``mm2pipeline.__main__``) dispatches to
the stages; each stage module also exposes a ``main()`` so it runs directly as
``python -m mm2pipeline.<stage>``. The shared object metadata lives in
``mm2pipeline.tiles``; the binary .bcd codec lives in ``mm2pipeline.bcd``.
See mm2pipeline/README.md for full usage instructions.
"""

__version__ = "1.1.0"
