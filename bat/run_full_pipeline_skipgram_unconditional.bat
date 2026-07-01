@echo off
REM Usage: run_full_pipeline_skipgram_unconditional.bat <input> [type] [game] [seed] [embedding_dim] [window_size]
REM Same as run_full_pipeline_skipgram.bat but the diffusion model is trained
REM UNCONDITIONALLY (no text conditioning).
REM
REM <input>        path to a .txt ASCII level file or folder of .txt files
REM [type]         defaults to "regular"
REM [game]         defaults to "MM"
REM [seed]         defaults to 0
REM [embedding_dim] block embedding size, defaults to 32 (MM has ~69 tile types)
REM [window_size]  odd integer window size, defaults to 3
cd ..

set INPUT=%~1
set TYPE=%~2
set GAME=%~3
set SEED=%~4
set EMBEDDING_DIM=%~5
set WINDOW_SIZE=%~6

if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular
if "%GAME%"=="" set GAME=MM
if "%SEED%"=="" set SEED=0
if "%EMBEDDING_DIM%"=="" set EMBEDDING_DIM=32
if "%WINDOW_SIZE%"=="" set WINDOW_SIZE=3

set TILESET=mm2_tileset_we.json
set NUM_TILES=69
set EVAL_TILESET=mm2_tileset_we.json
if /I "%GAME%"=="MM" (
    set TILESET=mm2_tileset_we.json
    set NUM_TILES=69
    set EVAL_TILESET=mm2_tileset_we.json
)

set RAW_OUTPUT=datasets\%GAME%_Levels-%TYPE%.json
set CAPTIONED_OUTPUT=datasets\%GAME%_LevelsAndCaptions-%TYPE%.json

REM Skip-gram embedding artifacts.
set TILES_JSON=datasets\%GAME%_%WINDOW_SIZE%x%WINDOW_SIZE%_tiles-%TYPE%.json
set SG_OUTPUT=%GAME%-skipgram-embeddings-%EMBEDDING_DIM%-w%WINDOW_SIZE%

set DIFF_OUTPUT=%GAME%-unconditional-skipgram%EMBEDDING_DIM%-w%WINDOW_SIZE%-%TYPE%%SEED%

REM Used to auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\mariover_yes.txt
echo y> "%YES_FILE%"

echo === Step 0: Plotting complete-level size distribution ===
python analyze_level_dimensions.py --input "%INPUT%" --output datasets\%GAME%_LevelSizeDistribution-%TYPE%.png --csv datasets\%GAME%_LevelSizeDistribution-%TYPE%.csv --title "%GAME% complete level size distribution (%TYPE%)"
if %ERRORLEVEL% neq 0 ( echo ERROR: analyze_level_dimensions.py failed. & exit /b 1 )

echo === Step 1: Preparing dataset with tile-presence captions ===
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
python build_dataset_with_ascii.py --input_file "%INPUT%" --output %RAW_OUTPUT% --tileset %TILESET% --sliding_window --stride 20 %META_ARG%
if %ERRORLEVEL% neq 0 ( echo ERROR: build_dataset_with_ascii.py failed. & exit /b 1 )
python MarioMaker_create_ascii_captions.py --dataset %RAW_OUTPUT% --tileset %TILESET% --output %CAPTIONED_OUTPUT%
if %ERRORLEVEL% neq 0 ( echo ERROR: MarioMaker_create_ascii_captions.py failed. & exit /b 1 )
python split_mario_maker_data.py --json %CAPTIONED_OUTPUT% --seed %SEED%
if %ERRORLEVEL% neq 0 ( echo ERROR: split_mario_maker_data.py failed. & exit /b 1 )
python tokenizer.py save --json_file datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --pkl_file datasets\%GAME%_Tokenizer-%TYPE%.pkl
if %ERRORLEVEL% neq 0 ( echo ERROR: tokenizer.py failed. & exit /b 1 )
python create_mario_maker_random_captions.py --json %CAPTIONED_OUTPUT% --output datasets\%GAME%_RandomTest-%TYPE%.json
if %ERRORLEVEL% neq 0 ( echo ERROR: create_mario_maker_random_captions.py failed. & exit /b 1 )

echo === Step 1b: Training Skip-Gram tile embeddings ===
python create_tile_level_json_data.py --from_dataset datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --output %TILES_JSON% --tile_size %WINDOW_SIZE%
if %ERRORLEVEL% neq 0 ( echo ERROR: create_tile_level_json_data.py failed. & exit /b 1 )
python train_skipgram.py --json_file %TILES_JSON% --output_dir "%SG_OUTPUT%" --embedding_dim %EMBEDDING_DIM% --vocab_size %NUM_TILES% --batch_size 32 --negative_samples 10
if %ERRORLEVEL% neq 0 ( echo ERROR: train_skipgram.py failed. & exit /b 1 )

echo === Step 2: Training unconditional diffusion model (with skipgram embeddings) ===
python train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --save_image_epochs 20 --augment --block_embedding_model_path "%SG_OUTPUT%" --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --seed %SEED% < "%YES_FILE%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: train_diffusion.py failed.
    exit /b 1
)

echo === Step 3: Running diffusion generation ===
call bat\run_diffusion_multi.bat "%DIFF_OUTPUT%" %TYPE% %GAME%
if %ERRORLEVEL% neq 0 (
    echo ERROR: run_diffusion_multi.bat failed.
    exit /b 1
)

echo === Step 4: Evaluating tile distribution ===
python evaluate_tile_distribution.py --model_path "%DIFF_OUTPUT%" --num_tiles %NUM_TILES% --tileset %EVAL_TILESET% --captions_json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json --seed %SEED%
if %ERRORLEVEL% neq 0 ( echo ERROR: evaluate_tile_distribution.py failed. & exit /b 1 )

echo === Pipeline complete! Model saved to: %DIFF_OUTPUT% ===
