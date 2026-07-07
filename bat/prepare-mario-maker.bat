@echo off
REM Usage: prepare-mario-maker.bat <input> [seed]
REM <input> is a path to a .txt ASCII level file or a folder of .txt files
REM [seed] is optional, defaults to 0
cd ..

set INPUT=%1
if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)

set SEED=%2
if "%SEED%"=="" set SEED=0

set GAME=MM
set TYPE=regular
set TILESET=extended_tiles.json

set RAW_OUTPUT=datasets\%GAME%_Levels-%TYPE%.json
set CAPTIONED_OUTPUT=datasets\%GAME%_LevelsAndCaptions-%TYPE%.json

python analyze_level_dimensions.py --input "%INPUT%" --output datasets\%GAME%_LevelSizeDistribution-%TYPE%.png --csv datasets\%GAME%_LevelSizeDistribution-%TYPE%.csv --title "%GAME% complete level size distribution (%TYPE%)"
REM Fold in the level-metadata sidecar the extract pipeline writes next to the
REM ascii files (folder input -> INPUT\metadata.json; single file -> its folder),
REM matching where bat\extract_levels_to_ascii.bat puts it. Passed only when it
REM exists, so inputs prepared without the extract pipeline still build.
set "METADATA="
if exist "%INPUT%\" (
    set "METADATA=%INPUT%\metadata.json"
) else (
    for %%I in ("%INPUT%") do set "METADATA=%%~dpImetadata.json"
)
set "META_ARG="
if defined METADATA if exist "%METADATA%" set "META_ARG=--metadata "%METADATA%""
python -m mm2pipeline.dataset build --input_file %INPUT% --output %RAW_OUTPUT% --tileset %TILESET% --sliding_window --stride 20 --convert_to_extended %META_ARG%
python MarioMaker_create_ascii_captions.py --dataset %RAW_OUTPUT% --tileset %TILESET% --output %CAPTIONED_OUTPUT%
python -m mm2pipeline.dataset split --json %CAPTIONED_OUTPUT% --seed %SEED%
python tokenizer.py save --json_file datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --pkl_file datasets\%GAME%_Tokenizer-%TYPE%.pkl
python create_mario_maker_random_captions.py --json %CAPTIONED_OUTPUT% --output datasets\%GAME%_RandomTest-%TYPE%.json
