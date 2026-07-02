@echo off
setlocal
REM ===========================================================================
REM train_from_dataset_llm_t5.bat  --  train text-conditional diffusion on an
REM ALREADY-CAPTIONED dataset.json using a FROZEN T5 text encoder. The dataset
REM is assumed to already contain captions, so there is no captioning step and
REM (T5 being pretrained) no MLM training step. Flow:
REM   split -> tokenizer -> train diffusion (T5) -> generate.
REM
REM Only T5's encoder tower is used (its hidden states are mean-pooled). t5-base
REM ships a fast tokenizer (tokenizer.json), so no SentencePiece install is
REM needed; some other T5 variants may require: pip install sentencepiece
REM
REM Usage: train_from_dataset_llm_t5.bat <captioned_dataset.json>
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
REM Frozen pretrained T5 text encoder (no MLM training needed). Swap for t5-small
REM (lighter) or google/flan-t5-base, etc.
set TEXT_ENCODER=t5-base

REM Outputs land next to the dataset. Split writes <dataset>-train.json etc.
for %%I in ("%DATASET%") do set "OUTDIR=%%~dpI"
set BASE=%DATASET:.json=%
set TOKENIZER=%OUTDIR%dataset_tokenizer_t5.pkl
set DIFF_DIR=%OUTDIR%diffusion_t5
set GEN_DIR=%OUTDIR%generated_t5

REM Auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\train_from_dataset_t5_yes.txt
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
echo === [2/4] training conditional diffusion (frozen T5; samples every 10, checkpoints every 20) ===
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
