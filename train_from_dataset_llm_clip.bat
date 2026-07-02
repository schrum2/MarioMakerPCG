@echo off
setlocal
REM ===========================================================================
REM train_from_dataset_llm_clip.bat  --  train text-conditional diffusion on an
REM ALREADY-CAPTIONED dataset.json using a FROZEN CLIP text encoder. The dataset
REM is assumed to already contain captions, so there is no captioning step and
REM (CLIP being pretrained) no MLM training step. Flow:
REM   split -> tokenizer -> train diffusion (CLIP) -> generate.
REM
REM CLIP's text tower caps prompts at 77 tokens (longer captions are truncated),
REM which is plenty for these short tile captions.
REM
REM Usage: train_from_dataset_llm_clip.bat <captioned_dataset.json>
REM ===========================================================================
cd /d "%~dp0"

REM Full path to the SURF conda python. Bare "python" can resolve to the
REM Windows Store stub in some shells; the absolute path avoids that.
set PY="C:\Users\mckeonp\AppData\Local\miniconda3\envs\SURF\python.exe"
set DATASET=%~1
if "%DATASET%"=="" (
    echo Usage: %~nx0 ^<captioned_dataset.json^>
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

REM Outputs land next to the dataset. Split writes <dataset>-train.json etc.
for %%I in ("%DATASET%") do set "OUTDIR=%%~dpI"
set BASE=%DATASET:.json=%
set TOKENIZER=%OUTDIR%dataset_tokenizer_clip.pkl
set DIFF_DIR=%OUTDIR%diffusion_clip
set GEN_DIR=%OUTDIR%generated_clip

REM Auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\train_from_dataset_clip_yes.txt
echo y> "%YES_FILE%"

REM Wipe prior output folders so nothing fails on an existing dir.
if exist "%DIFF_DIR%" rd /s /q "%DIFF_DIR%"
if exist "%GEN_DIR%" rd /s /q "%GEN_DIR%"

echo.
echo === [1/4] train/validate/test split + tokenizer (dataset assumed captioned) ===
%PY% split_mario_maker_data.py --json "%DATASET%" --seed %SEED%
if errorlevel 1 goto error
%PY% tokenizer.py save --json_file "%BASE%-train.json" --pkl_file "%TOKENIZER%"
if errorlevel 1 goto error

echo.
echo === [2/4] training conditional diffusion (frozen CLIP; samples every 10, checkpoints every 20) ===
%PY% train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --augment --text_conditional --pretrained_language_model "%TEXT_ENCODER%" --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --pkl "%TOKENIZER%" --output_dir "%DIFF_DIR%" --save_image_epochs 10 --save_model_epochs 20 --seed %SEED% < "%YES_FILE%"
if errorlevel 1 goto error

echo.
echo === [3/4] generating levels from the trained diffusion model ===
%PY% run_diffusion.py --model_path "%DIFF_DIR%" --game %GAME% --tileset %TILESET% --output_dir "%GEN_DIR%" --output_format both --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [4/4] done ===
echo   Diffusion: "%DIFF_DIR%"
echo   Generated: "%GEN_DIR%"
goto end

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%). Fix that step, then re-run. ***

:end
endlocal
