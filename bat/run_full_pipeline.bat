@echo off
REM Usage: run_full_pipeline.bat <input> [model] [text_encoder] [type] [game] [seed]
REM <input>        path to a .txt ASCII level file or folder of .txt files
REM [model]        Ollama model for captioning, defaults to "qwen2.5:14b"
REM [text_encoder] which text encoder conditions the diffusion model, defaults to "MLM"
REM                "MLM"    -> train our own masked-language-model encoder (train_mlm.py)
REM                "MiniLM" -> sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (frozen, pretrained)
REM                "GTE"    -> Alibaba-NLP/gte-large-en-v1.5 (frozen, pretrained)
REM [type]         defaults to "regular"
REM [game]         defaults to "MM"
REM [seed]         defaults to 0
cd ..

set INPUT=%~1
set MODEL=%~2
set TEXT_ENCODER=%~3
set TYPE=%~4
set GAME=%~5
set SEED=%~6

if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)
if "%MODEL%"=="" set MODEL=qwen2.5:14b
if "%TYPE%"=="" set TYPE=regular
if "%GAME%"=="" set GAME=MM
if "%SEED%"=="" set SEED=0
if "%TEXT_ENCODER%"=="" set TEXT_ENCODER=MLM

REM Map a text encoder name to its HuggingFace model id. Add new entries here
REM as more pretrained text encoders are tried.
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
set CAPTION_ARGS=
if /I "%GAME%"=="MM" (
    set TILESET=mm2_tileset_we.json
    set NUM_TILES=69
    set EVAL_TILESET=mm2_tileset_we.json
    set CAPTION_ARGS=--grid-format tokens --tileset-we mm2_tileset_we.json
)

set RAW_OUTPUT=datasets\%GAME%_Levels-%TYPE%.json
set CAPTIONED_OUTPUT=datasets\%GAME%_LevelsAndCaptions-%TYPE%.json

REM -- Folder for the grid .txt files actually sent to the LLM, placed next
REM    to the input ASCII folder/file.
for %%I in ("%INPUT%") do set "INPUT_DIR=%%~dpI"
set LLM_ASCII_DIR=%INPUT_DIR%ascii_tokens

set MLM_OUTPUT=%GAME%-MLM-%TYPE%%SEED%
if /I "%TEXT_ENCODER%"=="MLM" (
    set DIFF_OUTPUT=%GAME%-conditional-%TYPE%%SEED%
) else (
    set DIFF_OUTPUT=%GAME%-conditional-%TEXT_ENCODER%-%TYPE%%SEED%
)

REM Used to auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
REM A redirected file (rather than a pipe) keeps %ERRORLEVEL% checks working afterward.
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

echo === Step 3: Training diffusion model ===
python train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --save_image_epochs 20 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl %TEXT_ENCODER_FLAGS% --plot_validation_caption_score --seed %SEED% < "%YES_FILE%"
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
