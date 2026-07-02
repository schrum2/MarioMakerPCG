@echo off
setlocal
REM ===========================================================================
REM train_from_dataset_test.bat  --  quick smoke test of train_from_dataset.bat.
REM Same pipeline (captions -> split -> tokenizer -> MLM -> diffusion ->
REM generate) but MLM and diffusion run for only 2 epochs so you can verify the
REM whole chain wires up before committing to a real run.
REM
REM Usage: train_from_dataset_test.bat <dataset.json>
REM ===========================================================================
cd /d "%~dp0"

set PY=python
set DATASET=%~1
if "%DATASET%"=="" (
    echo Usage: %~nx0 ^<dataset.json^>
    exit /b 1
)
for %%I in ("%DATASET%") do set "DATASET=%%~fI"

set TILESET=mm2_tileset_we.json
set GAME=MM
set NUM_TILES=69
set SEED=0
set EPOCHS=2

for %%I in ("%DATASET%") do set "OUTDIR=%%~dpI"
set CAPTIONED=%OUTDIR%dataset_captioned.json
set BASE=%CAPTIONED:.json=%
set TOKENIZER=%OUTDIR%dataset_tokenizer.pkl
set MLM_DIR=%OUTDIR%mlm_test
set DIFF_DIR=%OUTDIR%diffusion_test
set GEN_DIR=%OUTDIR%generated_test

REM Auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\train_from_dataset_test_yes.txt
echo y> "%YES_FILE%"

echo.
echo === [1/6] simple presence captions -> dataset_captioned.json ===
%PY% MarioMaker_create_ascii_captions.py --dataset "%DATASET%" --tileset %TILESET% --output "%CAPTIONED%"
if errorlevel 1 goto error

echo.
echo === [2/6] train/validate/test split + tokenizer ===
%PY% split_mario_maker_data.py --json "%CAPTIONED%" --seed %SEED%
if errorlevel 1 goto error
%PY% tokenizer.py save --json_file "%BASE%-train.json" --pkl_file "%TOKENIZER%"
if errorlevel 1 goto error

echo.
echo === [3/6] training MLM text encoder (%EPOCHS% epochs - TEST) ===
%PY% train_mlm.py --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --pkl "%TOKENIZER%" --output_dir "%MLM_DIR%" --epochs %EPOCHS% --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [4/6] training conditional diffusion (%EPOCHS% epochs - TEST) ===
%PY% train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --augment --text_conditional --mlm_model_dir "%MLM_DIR%" --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --pkl "%TOKENIZER%" --output_dir "%DIFF_DIR%" --num_epochs %EPOCHS% --save_image_epochs 1 --save_model_epochs 1 --seed %SEED% < "%YES_FILE%"
if errorlevel 1 goto error

echo.
echo === [5/6] generating levels from the trained diffusion model ===
%PY% run_diffusion.py --model_path "%DIFF_DIR%" --game %GAME% --tileset %TILESET% --output_dir "%GEN_DIR%" --output_format both --num_samples 4 --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [6/6] TEST RUN DONE ===
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
