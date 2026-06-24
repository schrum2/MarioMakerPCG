@echo off
REM Usage: run_full_pipeline_block2vec_unconditional.bat <input> [model] [type] [game] [seed] [embedding_dim]
REM Same as run_full_pipeline_block2vec.bat (LLM-captioned dataset, learned
REM Block2Vec tile embeddings) but the diffusion model is trained UNCONDITIONALLY
REM (no text conditioning). There is no text encoder, so the MLM/MiniLM/GTE step
REM and the caption-adherence evaluation are dropped.
REM
REM <input>        path to a .txt ASCII level file or folder of .txt files
REM [model]        Ollama model for captioning, defaults to "qwen2.5:14b"
REM [type]         defaults to "regular"
REM [game]         defaults to "MM"
REM [seed]         defaults to 0
REM [embedding_dim] block embedding size, defaults to 32 (MM has ~69 tile types)
cd ..

set INPUT=%~1
set MODEL=%~2
set TYPE=%~3
set GAME=%~4
set SEED=%~5
set EMBEDDING_DIM=%~6

if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)
if "%MODEL%"=="" set MODEL=qwen2.5:14b
if "%TYPE%"=="" set TYPE=regular
if "%GAME%"=="" set GAME=MM
if "%SEED%"=="" set SEED=0
if "%EMBEDDING_DIM%"=="" set EMBEDDING_DIM=32

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

for %%I in ("%INPUT%") do set "INPUT_DIR=%%~dpI"
set LLM_ASCII_DIR=%INPUT_DIR%ascii_tokens

REM Block embedding artifacts.
set TILES_JSON=datasets\%GAME%_3x3_tiles-%TYPE%.json
set B2V_OUTPUT=%GAME%-block2vec-embeddings-%EMBEDDING_DIM%

REM "-b2v" in the name keeps these models separate from the one-hot runs.
set DIFF_OUTPUT=%GAME%-unconditional-b2v%EMBEDDING_DIM%-%TYPE%%SEED%

REM Used to auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\mariover_yes.txt
echo y> "%YES_FILE%"

REM -- Ollama setup ---------------------------------------------------------
ollama list >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ollama not running. Starting ollama server...
    start /min "" ollama serve
    echo Waiting for Ollama to initialise...
    timeout /t 6 /nobreak >nul
)

echo Pulling %MODEL% ^(no-op if already present^)...
ollama pull %MODEL%
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to pull %MODEL%. Is Ollama running?
    exit /b 1
)

echo === Step 1: Preparing dataset with LLM captions ===
python build_dataset_with_ascii.py --input_file "%INPUT%" --output %RAW_OUTPUT% --tileset %TILESET% --sliding_window --stride 20
if %ERRORLEVEL% neq 0 ( echo ERROR: build_dataset_with_ascii.py failed. & exit /b 1 )
python MarioMaker_llm_captions.py --dataset %RAW_OUTPUT% --tileset %TILESET% --output %CAPTIONED_OUTPUT% --model %MODEL% --ascii-output-dir "%LLM_ASCII_DIR%" --num-captions 1 --prompt-log MM2_Prompt.txt
if %ERRORLEVEL% neq 0 ( echo ERROR: MarioMaker_llm_captions.py failed. & exit /b 1 )
python split_mario_maker_data.py --json %CAPTIONED_OUTPUT% --seed %SEED%
if %ERRORLEVEL% neq 0 ( echo ERROR: split_mario_maker_data.py failed. & exit /b 1 )
python tokenizer.py save --json_file datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --pkl_file datasets\%GAME%_Tokenizer-%TYPE%.pkl
if %ERRORLEVEL% neq 0 ( echo ERROR: tokenizer.py failed. & exit /b 1 )
python create_mario_maker_random_captions.py --json %CAPTIONED_OUTPUT% --output datasets\%GAME%_RandomTest-%TYPE%.json
if %ERRORLEVEL% neq 0 ( echo ERROR: create_mario_maker_random_captions.py failed. & exit /b 1 )

echo === Step 1b: Training Block2Vec tile embeddings ===
REM Slice 3x3 windows straight from the integer-encoded train scenes so the tile
REM ids line up with the diffusion model, then learn an embedding per tile.
REM --vocab_size pins the table to the full tileset so every id has a row.
python create_tile_level_json_data.py --from_dataset datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --output %TILES_JSON% --tile_size 3
if %ERRORLEVEL% neq 0 ( echo ERROR: create_tile_level_json_data.py failed. & exit /b 1 )
python train_block2vec.py --json_file %TILES_JSON% --output_dir "%B2V_OUTPUT%" --embedding_dim %EMBEDDING_DIM% --vocab_size %NUM_TILES% --epochs 1000 --batch_size 32
if %ERRORLEVEL% neq 0 ( echo ERROR: train_block2vec.py failed. & exit /b 1 )

echo === Step 2: Training unconditional diffusion model (with block embeddings) ===
python train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --save_image_epochs 20 --augment --block_embedding_model_path "%B2V_OUTPUT%" --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --seed %SEED% < "%YES_FILE%"
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
if %ERRORLEVEL% neq 0 (
    echo ERROR: evaluate_tile_distribution.py failed.
    exit /b 1
)

echo === Pipeline complete! Model saved to: %DIFF_OUTPUT% ===
