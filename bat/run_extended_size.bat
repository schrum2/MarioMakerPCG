@echo off
setlocal
REM ===========================================================================
REM run_extended_size.bat <SIZE>  --  full pipeline on a frequency-ranked extended
REM tileset, where SIZE is one of 20 / 30 / 40 / 50 / 60 (default 20).
REM
REM UNCONDITIONAL run (no text conditioning) -- size buckets + Block2Vec tile
REM embeddings only. This is a throwaway loss-sweep harness, so the MiniLM text
REM encoder / captions are dropped; we just want the diffusion loss per vocab size.
REM The tile vocabulary is the top-SIZE tiles by frequency
REM (extended_tiles_<SIZE>.json, built by build_extended_tilesets.py). Everything
REM rarer is folded onto its closest survivor by mm2view_to_extended.py.
REM
REM The point is the loss sweep: run it for 20/30/40/50/60 and compare the final
REM diffusion loss to find the smallest vocab that still captures the data, so the
REM model stops wasting capacity on dozens of near-empty tile classes.
REM
REM   bat\run_extended_size.bat 30
REM
REM Lives in bat\ but all scripts/tilesets are at the repo root, so cd up one level
REM (works whether launched from the repo root or double-clicked).
REM ===========================================================================
cd /d "%~dp0.."

REM -- Size argument (the tileset has SIZE tiles; +1 padding id => NUM_TILES) ---
set SIZE=%1
if "%SIZE%"=="" set SIZE=20
set /a NUM_TILES=%SIZE%+1

REM -- Knobs ------------------------------------------------------------------
REM PY: full path to the SURF conda python (bare "python" is the broken Store stub).
set PY=C:\Users\mckeonp\AppData\Local\miniconda3\envs\SURF\python.exe
set INPUT=C:\Users\mckeonp\Documents\d
set TILESET=extended_tiles_%SIZE%.json
set GAME=MM
set SEED=0
set BATCH_SIZE=16
set EMBEDDING_DIM=16
set WINDOW_SIZE=3
set B2V_EPOCHS=100
set DIFF_EPOCHS=100

if not exist "%TILESET%" (
    echo *** Tileset %TILESET% not found. Valid sizes: 20 30 40 50 60.
    echo *** Build them first:  %PY% build_extended_tilesets.py
    goto end
)

REM -- Throwaway output locations (one folder per size so sizes don't collide) -
set OUTDIR=run_extended_%SIZE%
set BUCKET_DIR=%OUTDIR%\size_buckets
set RAW=%OUTDIR%\MM_Levels_extended.json
set BASE=%RAW:.json=%
set TILES_JSON=%OUTDIR%\MM_tiles_%WINDOW_SIZE%x%WINDOW_SIZE%.json
set B2V_OUTPUT=%OUTDIR%\block2vec
set DIFF_OUTPUT=%OUTDIR%\diff

if exist "%OUTDIR%" rd /s /q "%OUTDIR%"
mkdir "%OUTDIR%"

set YES_FILE=%TEMP%\run_extended_%SIZE%_yes.txt
echo y> "%YES_FILE%"

echo.
echo === extended tileset sweep: SIZE=%SIZE% (%TILESET%, NUM_TILES=%NUM_TILES%) ===

echo.
echo === [1/5] bucket-sorting %INPUT% into the %TILESET% id space ===
%PY% bucket_levels_by_size.py --input "%INPUT%" --output_dir "%BUCKET_DIR%" --tileset %TILESET% --convert_to_extended --merged_output "%RAW%"
if errorlevel 1 goto error

echo.
echo === [2/5] train/validate/test split (no captioning -- unconditional) ===
REM No captions are generated. LevelDataset still requires a "caption" key on every
REM item (level_dataset.py reads item["caption"] even unconditionally), so stamp an
REM empty one; the unconditional trainer returns it but never uses it.
%PY% -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));[e.setdefault('caption','') for e in d];json.dump(d,open(sys.argv[1],'w',encoding='utf-8'))" "%RAW%"
if errorlevel 1 goto error
%PY% split_mario_maker_data.py --json "%RAW%" --seed %SEED%
if errorlevel 1 goto error

echo.
echo === [3/5] slicing %WINDOW_SIZE%x%WINDOW_SIZE% tile windows for Block2Vec ===
%PY% create_tile_level_json_data.py --from_dataset "%BASE%-train.json" --output "%TILES_JSON%" --tile_size %WINDOW_SIZE%
if errorlevel 1 goto error

echo.
echo === [4/5] training Block2Vec embeddings (dim %EMBEDDING_DIM%, vocab %NUM_TILES%, %B2V_EPOCHS% epochs) ===
%PY% train_block2vec.py --json_file "%TILES_JSON%" --output_dir "%B2V_OUTPUT%" --embedding_dim %EMBEDDING_DIM% --vocab_size %NUM_TILES% --epochs %B2V_EPOCHS% --batch_size 32
if errorlevel 1 goto error

echo.
echo === [5/5] training UNCONDITIONAL diffusion: Block2Vec, no text encoder (%DIFF_EPOCHS% epochs) ===
%PY% train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --augment --block_embedding_model_path "%B2V_OUTPUT%" --output_dir "%DIFF_OUTPUT%" --num_epochs %DIFF_EPOCHS% --save_image_epochs 1000000 --validate_epochs 1 --batch_size %BATCH_SIZE% --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --seed %SEED% < "%YES_FILE%"
if errorlevel 1 goto error

echo.
echo === done (SIZE=%SIZE%) ===
echo Final loss curve: "%DIFF_OUTPUT%\training_loss.png"
echo Compare the final loss across sizes 20/30/40/50/60 (and the 69-tile mm2 run).
goto end

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%). Fix that step, then re-run. ***

:end
endlocal
