@echo off
setlocal
REM ===========================================================================
REM run_extended_size.bat <INPUT> [SIZE]
REM   <INPUT>  path to an ascii .txt level file or a folder of them (required)
REM   [SIZE]   tile-vocab size: 20 / 30 / 40 / 50 / 60   (optional, default 20)
REM
REM Unconditional diffusion run (no captions, no text encoder) on the
REM frequency-ranked extended_tiles_<SIZE>.json vocab, to compare loss across
REM vocab sizes. The tilesets already exist in the repo root -- this script just
REM uses them. Lives in bat\, so it cd's up to the repo root first.
REM
REM   bat\run_extended_size.bat C:\path\to\ascii_levels 30
REM ===========================================================================
cd /d "%~dp0.."

set INPUT=%1
if "%INPUT%"=="" (
    echo ERROR: give the ascii level file/folder as the first argument.
    echo   bat\run_extended_size.bat ^<INPUT^> [SIZE]
    goto end
)
set SIZE=%2
if "%SIZE%"=="" set SIZE=20
set /a NUM_TILES=%SIZE%+1
set TILESET=extended_tiles_%SIZE%.json
if not exist "%TILESET%" (
    echo ERROR: %TILESET% not found. Valid sizes: 20 30 40 50 60.
    goto end
)

REM -- Knobs (PY is the full SURF conda path; bare "python" is the broken stub) -
set PY=C:\Users\mckeonp\AppData\Local\miniconda3\envs\SURF\python.exe
set SEED=0
set BATCH_SIZE=16
set EMBEDDING_DIM=16
set WINDOW_SIZE=3
set B2V_EPOCHS=200
set DIFF_EPOCHS=500

REM -- Throwaway output (one folder per size so runs don't collide) -----------
set OUTDIR=run_extended_%SIZE%
set RAW=%OUTDIR%\MM_Levels_extended.json
set BASE=%RAW:.json=%
set TILES_JSON=%OUTDIR%\MM_tiles_%WINDOW_SIZE%x%WINDOW_SIZE%.json
set B2V_OUTPUT=%OUTDIR%\block2vec
set DIFF_OUTPUT=%OUTDIR%\diff

if exist "%OUTDIR%" rd /s /q "%OUTDIR%"
mkdir "%OUTDIR%"

REM Auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\run_extended_%SIZE%_yes.txt
echo y> "%YES_FILE%"

echo.
echo === SIZE=%SIZE% (%TILESET%, NUM_TILES=%NUM_TILES%)  INPUT=%INPUT% ===

echo.
echo === [1/4] bucket-sort + reduce to the %TILESET% id space ===
%PY% bucket_levels_by_size.py --input "%INPUT%" --output_dir "%OUTDIR%\size_buckets" --tileset %TILESET% --convert_to_extended --merged_output "%RAW%"
if errorlevel 1 goto error

echo.
echo === [2/4] train/validate/test split (unconditional, no captions) ===
REM LevelDataset still wants a "caption" key on every item even unconditionally,
REM so stamp an empty one; the trainer returns it but never uses it.
%PY% -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));[e.setdefault('caption','') for e in d];json.dump(d,open(sys.argv[1],'w',encoding='utf-8'))" "%RAW%"
if errorlevel 1 goto error
%PY% split_mario_maker_data.py --json "%RAW%" --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [3/4] Block2Vec tile embeddings (dim %EMBEDDING_DIM%, vocab %NUM_TILES%) ===
%PY% create_tile_level_json_data.py --from_dataset "%BASE%-train.json" --output "%TILES_JSON%" --tile_size %WINDOW_SIZE%
if errorlevel 1 goto error
%PY% train_block2vec.py --json_file "%TILES_JSON%" --output_dir "%B2V_OUTPUT%" --embedding_dim %EMBEDDING_DIM% --vocab_size %NUM_TILES% --epochs %B2V_EPOCHS% --batch_size 32
if errorlevel 1 goto error

echo.
echo === [4/4] unconditional diffusion (%DIFF_EPOCHS% epochs) ===
%PY% train_diffusion.py --game MM --num_tiles %NUM_TILES% --tileset %TILESET% --augment --block_embedding_model_path "%B2V_OUTPUT%" --output_dir "%DIFF_OUTPUT%" --num_epochs %DIFF_EPOCHS% --save_image_epochs 1000000 --validate_epochs 1 --batch_size %BATCH_SIZE% --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --seed %SEED% < "%YES_FILE%"
if errorlevel 1 goto error

echo.
echo === done (SIZE=%SIZE%)  loss curve: "%DIFF_OUTPUT%\training_loss.png" ===
goto end

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%). ***

:end
endlocal
