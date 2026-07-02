@echo off
setlocal
REM ===========================================================================
REM train_from_dataset_llm_clip.bat  --  like train_from_dataset.bat but uses
REM LLM-generated captions (MarioMaker_llm_captions.py, Ollama) and a FROZEN
REM CLIP text encoder instead of a trained MLM. Because CLIP is pretrained
REM there is no MLM training step; --pretrained_language_model overrides it.
REM   LLM captions -> split -> tokenizer -> train diffusion (CLIP) -> generate.
REM
REM CLIP's text tower caps prompts at 77 tokens (longer captions are truncated),
REM which is plenty for these short tile captions.
REM
REM Requires a running Ollama server with the caption model pulled, e.g.:
REM   ollama pull qwen2.5:14b
REM
REM Usage: train_from_dataset_llm_clip.bat <dataset.json>
REM ===========================================================================
cd /d "%~dp0"

REM Full path to the SURF conda python. Bare "python" can resolve to the
REM Windows Store stub in some shells; the absolute path avoids that.
set PY="C:\Users\mckeonp\AppData\Local\miniconda3\envs\SURF\python.exe"
set DATASET=%~1
if "%DATASET%"=="" (
    echo Usage: %~nx0 ^<dataset.json^>
    exit /b 1
)
for %%I in ("%DATASET%") do set "DATASET=%%~fI"

set TILESET=mm2_tileset_we.json
set GAME=MM
set NUM_TILES=68
set SEED=0
REM Frozen pretrained CLIP text encoder (no MLM training needed). Swap for the
REM b-32/b-16 variants (or an openai/clip-* repo) to trade quality for size.
set TEXT_ENCODER=sentence-transformers/clip-vit-l-14
REM Ollama model used to write the captions. Pull it first (see header).
set LLM_MODEL=qwen2.5:14b
REM Captions generated per level. >1 stores caption/caption1/... and diffusion
REM picks one at random per access via --multiple_captions.
set NUM_CAPTIONS=5

for %%I in ("%DATASET%") do set "OUTDIR=%%~dpI"
set CAPTIONED=%OUTDIR%dataset_captioned_llm.json
set BASE=%CAPTIONED:.json=%
set TOKENIZER=%OUTDIR%dataset_tokenizer_llm.pkl
set DIFF_DIR=%OUTDIR%diffusion_llm_clip
set GEN_DIR=%OUTDIR%generated_llm_clip

REM Auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\train_from_dataset_llm_clip_yes.txt
echo y> "%YES_FILE%"

REM Wipe prior model folder so training never fails on an existing dir.
if exist "%DIFF_DIR%" rd /s /q "%DIFF_DIR%"

echo.
echo === [1/5] LLM captions via Ollama (%LLM_MODEL%, %NUM_CAPTIONS% per level) -> dataset_captioned_llm.json ===
%PY% MarioMaker_llm_captions.py --dataset "%DATASET%" --tileset %TILESET% --output "%CAPTIONED%" --backend ollama --model %LLM_MODEL% --num-captions %NUM_CAPTIONS% --grid-format ascii
if errorlevel 1 goto error

echo.
echo === [2/5] train/validate/test split + tokenizer ===
%PY% split_mario_maker_data.py --json "%CAPTIONED%" --seed %SEED%
if errorlevel 1 goto error
%PY% tokenizer.py save --json_file "%BASE%-train.json" --pkl_file "%TOKENIZER%"
if errorlevel 1 goto error

echo.
echo === [3/5] training conditional diffusion (frozen CLIP; samples every 10, checkpoints every 20) ===
%PY% train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --augment --text_conditional --multiple_captions --pretrained_language_model "%TEXT_ENCODER%" --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --pkl "%TOKENIZER%" --output_dir "%DIFF_DIR%" --save_image_epochs 10 --save_model_epochs 20 --seed %SEED% < "%YES_FILE%"
if errorlevel 1 goto error

echo.
echo === [4/5] generating levels from the trained diffusion model ===
%PY% run_diffusion.py --model_path "%DIFF_DIR%" --game %GAME% --tileset %TILESET% --output_dir "%GEN_DIR%" --output_format both --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [5/5] done ===
echo   Captioned: "%CAPTIONED%"
echo   Diffusion: "%DIFF_DIR%"
echo   Generated: "%GEN_DIR%"
goto end

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%). Fix that step, then re-run. ***

:end
endlocal
