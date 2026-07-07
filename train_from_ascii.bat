@echo off
setlocal
REM ===========================================================================
REM train_from_ascii.bat  --  end-to-end from a single ascii level file:
REM   sliding-window dataset (20x20 windows, stride 20, goal stripped, with the
REM   cropped PNGs) -> simple presence captions -> split -> tokenizer ->
REM   train MLM text encoder -> train conditional diffusion.
REM
REM Usage: train_from_ascii.bat <ascii_file>
REM   dataset.json / dataset_captioned.json land in the ascii file's folder.
REM ===========================================================================
cd /d "%~dp0"

REM Full path to the SURF conda python. Bare "python" can resolve to the
REM Windows Store stub in some shells; the absolute path avoids that.
set PY="C:\Users\mckeonp\AppData\Local\miniconda3\envs\SURF\python.exe"
set ASCII=%~1
if "%ASCII%"=="" (
    echo Usage: %~nx0 ^<ascii_file^>
    exit /b 1
)
for %%I in ("%ASCII%") do set "ASCII=%%~fI"

set TILESET=mm2_tileset_we.json
set GAME=MM
set NUM_TILES=68
set SEED=0
set WINDOW=20
set STRIDE=20

for %%I in ("%ASCII%") do set "OUTDIR=%%~dpI"
set DATASET=dataset.json
set CAPTIONED=dataset_captioned.json
set BASE=%CAPTIONED:.json=%
set TOKENIZER=dataset_tokenizer.pkl
set MLM_DIR=mlm
set DIFF_DIR=diffusion
set GEN_DIR=generated

REM Auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\train_from_ascii_yes.txt
echo y> "%YES_FILE%"

REM Wipe prior model folders so training never fails on an existing dir.
if exist "%MLM_DIR%" rd /s /q "%MLM_DIR%"
if exist "%DIFF_DIR%" rd /s /q "%DIFF_DIR%"
if exist "%GEN_DIR%" rd /s /q "%GEN_DIR%"

echo.
echo === [1/6] building sliding-window dataset (%WINDOW%x%WINDOW%, stride %STRIDE%, goal stripped, +images) ===
%PY% -m mm2pipeline.dataset build --input "%ASCII%" --output_folder "%DATASET%" --tileset %TILESET% --sliding_window --stride %STRIDE% --window_h %WINDOW% --window_w %WINDOW% --strip_goal --with_images
if errorlevel 1 goto error

echo.
echo === [2/6] simple presence captions -> dataset_captioned.json ===
%PY% MarioMaker_create_ascii_captions.py --dataset "%DATASET%" --tileset %TILESET% --output "%CAPTIONED%"
if errorlevel 1 goto error

echo.
echo === [3/6] train/validate/test split + tokenizer ===
%PY% -m mm2pipeline.dataset split --input "%CAPTIONED%" --seed %SEED%
if errorlevel 1 goto error
%PY% tokenizer.py save --json_file "%BASE%-train.json" --pkl_file "%TOKENIZER%"
if errorlevel 1 goto error

echo.
echo === [4/6] training MLM text encoder ===
%PY% train_mlm.py --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --pkl "%TOKENIZER%" --output_dir "%MLM_DIR%" --max_seq_length 128 --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [5/7] training conditional diffusion (MLM-conditioned; samples every 10, checkpoints every 20) ===
%PY% train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --augment --text_conditional --mlm_model_dir "%MLM_DIR%" --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --pkl "%TOKENIZER%" --output_dir "%DIFF_DIR%" --save_image_epochs 10 --save_model_epochs 20 --seed %SEED% < "%YES_FILE%"
if errorlevel 1 goto error

echo.
echo === [6/7] generating levels from the trained diffusion model ===
%PY% run_diffusion.py --model_path "%DIFF_DIR%" --game %GAME% --tileset %TILESET% --output_dir "%GEN_DIR%" --output_format both --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [7/7] done ===
echo   Dataset:   "%DATASET%"
echo   Captioned: "%CAPTIONED%"
echo   MLM:       "%MLM_DIR%"
echo   Diffusion: "%DIFF_DIR%"
echo   Generated: "%GEN_DIR%"
goto end

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%). Fix that step, then re-run. ***

:end
endlocal
