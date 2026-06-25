@echo off
REM Usage: run_full_pipeline_block2vec.bat <input> [model] [text_encoder] [type] [game] [seed] [embedding_dim] [window_size]
REM Same as run_full_pipeline.bat (LLM-captioned, text-conditional diffusion) but
REM the diffusion model is trained on learned Block2Vec tile embeddings instead of
REM one-hot tile encodings. The extra trailing arguments are the embedding size and window size.
REM
REM <input>        path to a .txt ASCII level file or folder of .txt files
REM [model]        Ollama model for captioning, defaults to "qwen2.5:14b"
REM [text_encoder] "MLM" (default), "MiniLM", or "GTE"
REM [type]         defaults to "regular"
REM [game]         defaults to "MM"
REM [seed]         defaults to 0
REM [embedding_dim] block embedding size, defaults to 64 (MM has ~69 tile types)
REM [window_size]  odd integer window size, defaults to 3
cd ..

set INPUT=%~1
set MODEL=%~2
set TEXT_ENCODER=%~3
set TYPE=%~4
set GAME=%~5
set SEED=%~6
set EMBEDDING_DIM=%~7
set WINDOW_SIZE=%~8

if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)
if "%MODEL%"=="" set MODEL=qwen2.5:14b
if "%TYPE%"=="" set TYPE=regular
if "%GAME%"=="" set GAME=MM
if "%SEED%"=="" set SEED=0
if "%TEXT_ENCODER%"=="" set TEXT_ENCODER=MLM
if "%EMBEDDING_DIM%"=="" set EMBEDDING_DIM=32
if "%WINDOW_SIZE%"=="" set WINDOW_SIZE=3

REM Map a text encoder name to its HuggingFace model id.
set PRETRAINED_MODEL_NAME=
if /I "%TEXT_ENCODER%"=="MiniLM" set PRETRAINED_MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
if /I "%TEXT_ENCODER%"=="GTE" set PRETRAINED_MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5

if /I not "%TEXT_ENCODER%"=="MLM" if "%PRETRAINED_MODEL_NAME%"=="" (
    echo ERROR: Unknown text_encoder "%TEXT_ENCODER%". Expected MLM, MiniLM, or GTE.
    exit /b 1
)

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
set TILES_JSON=datasets\%GAME%_%WINDOW_SIZE%x%WINDOW_SIZE%_tiles-%TYPE%.json
set B2V_OUTPUT=%GAME%-block2vec-embeddings-%EMBEDDING_DIM%-w%WINDOW_SIZE%

set MLM_OUTPUT=%GAME%-MLM-%TYPE%%SEED%
REM "-b2v" in the name keeps these models separate from the one-hot runs.
if /I "%TEXT_ENCODER%"=="MLM" (
    set DIFF_OUTPUT=%GAME%-conditional-b2v%EMBEDDING_DIM%-w%WINDOW_SIZE%-%TYPE%%SEED%
) else (
    set DIFF_OUTPUT=%GAME%-conditional-%TEXT_ENCODER%-b2v%EMBEDDING_DIM%-w%WINDOW_SIZE%-%TYPE%%SEED%
)

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

echo === Step 0: Plotting complete-level size distribution ===
python analyze_level_dimensions.py --input "%INPUT%" --output datasets\%GAME%_LevelSizeDistribution-%TYPE%.png --csv datasets\%GAME%_LevelSizeDistribution-%TYPE%.csv --title "%GAME% complete level size distribution (%TYPE%)"
if %ERRORLEVEL% neq 0 ( echo ERROR: analyze_level_dimensions.py failed. & exit /b 1 )

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
REM Slice %WINDOW_SIZE%x%WINDOW_SIZE% windows straight from the integer-encoded train scenes so the tile
REM ids line up with the diffusion model, then learn an embedding per tile.
REM --vocab_size pins the table to the full tileset so every id has a row.
python create_tile_level_json_data.py --from_dataset datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --output %TILES_JSON% --tile_size %WINDOW_SIZE%
if %ERRORLEVEL% neq 0 ( echo ERROR: create_tile_level_json_data.py failed. & exit /b 1 )
python train_block2vec.py --json_file %TILES_JSON% --output_dir "%B2V_OUTPUT%" --embedding_dim %EMBEDDING_DIM% --vocab_size %NUM_TILES% --epochs 1000 --batch_size 32
if %ERRORLEVEL% neq 0 ( echo ERROR: train_block2vec.py failed. & exit /b 1 )

set TEXT_ENCODER_FLAGS=
if /I "%TEXT_ENCODER%"=="MLM" (
    echo === Step 2: Training MLM model ===
    python train_mlm.py --epochs 300 --save_checkpoints --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --test_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
    if %ERRORLEVEL% neq 0 (
        echo ERROR: train_mlm.py failed.
        exit /b 1
    )
    set TEXT_ENCODER_FLAGS=--mlm_model_dir %MLM_OUTPUT%
) else (
    echo === Step 2: Skipped - using pretrained text encoder "%TEXT_ENCODER%" ^(%PRETRAINED_MODEL_NAME%^) ===
    set TEXT_ENCODER_FLAGS=--pretrained_language_model "%PRETRAINED_MODEL_NAME%"
)

echo === Step 3: Training diffusion model (with block embeddings) ===
python train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --save_image_epochs 20 --augment --text_conditional --block_embedding_model_path "%B2V_OUTPUT%" --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl %TEXT_ENCODER_FLAGS% --plot_validation_caption_score --seed %SEED% < "%YES_FILE%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: train_diffusion.py failed.
    exit /b 1
)

echo === Step 4: Running diffusion generation ===
call bat\run_diffusion_multi.bat "%DIFF_OUTPUT%" %TYPE% %GAME%
if %ERRORLEVEL% neq 0 (
    echo ERROR: run_diffusion_multi.bat failed.
    exit /b 1
)

echo === Step 5: Evaluating caption adherence ===
call bat\evaluate_caption_adherence_multi.bat "%DIFF_OUTPUT%" %TYPE% %GAME%
if %ERRORLEVEL% neq 0 (
    echo ERROR: evaluate_caption_adherence_multi.bat failed.
    exit /b 1
)

echo === Step 6: Evaluating tile distribution ===
python evaluate_tile_distribution.py --model_path "%DIFF_OUTPUT%" --num_tiles %NUM_TILES% --tileset %EVAL_TILESET% --captions_json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json --seed %SEED%
if %ERRORLEVEL% neq 0 (
    echo ERROR: evaluate_tile_distribution.py failed.
    exit /b 1
)

echo === Pipeline complete! Model saved to: %DIFF_OUTPUT% ===
