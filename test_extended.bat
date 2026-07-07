@echo off
setlocal
REM ===========================================================================
REM test_extended.bat  --  TEMPORARY end-to-end run of the FULL pipeline on the
REM extended_tiles.json representation (simplified 18-id vocab), with size-bucket
REM sorting, Block2Vec tile embeddings, and a frozen MiniLM text encoder -- the
REM same feature set as the real runs, so the diffusion loss is comparable.
REM Goal: see whether the simpler extended vocab brings the loss down.
REM Throwaway: delete this file and OUTDIR when done. Run from the repo root
REM (or just double-click it; it cd's to its own folder).
REM
REM Relies on three small code changes that make the extended vocab a first-class
REM citizen of the existing pipeline:
REM   - mm2view_to_extended.py  : output glyphs realigned to extended_tiles.json
REM   - bucket_levels_by_size.py: new --convert_to_extended flag
REM   - train_diffusion.py      : --game MM now honors an explicit --num_tiles/--tileset
REM ===========================================================================
cd /d "%~dp0"

REM -- Knobs ------------------------------------------------------------------
REM PY: set to the full path of your conda python.exe if bare "python" isn't on
REM PATH in this shell.
set PY=python
set INPUT=C:\Users\mckeonp\Documents\d
set TILESET=extended_tiles.json
set GAME=MM
set SEED=0
REM extended_tiles.json has 17 tiles + the "_" padding tile = 18 ids.
set NUM_TILES=18
set BATCH_SIZE=16
REM Frozen pretrained text encoder (MiniLM) -- no MLM training needed.
set TEXT_ENCODER=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
REM Block2Vec tile embeddings (smaller dim than the 69-tile runs since the vocab
REM is much smaller here). WINDOW_SIZE is the odd tile-context window.
set EMBEDDING_DIM=16
set WINDOW_SIZE=3
REM Epoch counts are intentionally small for a quick "is the loss going down?"
REM test. Bump to 500 to match bat\run_full_pipeline_block2vec.bat for a real run.
set B2V_EPOCHS=100
set DIFF_EPOCHS=100

REM -- Throwaway output locations (all under one folder so cleanup is easy) ---
set OUTDIR=test_extended
set BUCKET_DIR=%OUTDIR%\size_buckets
set RAW=%OUTDIR%\MM_Levels_extended.json
set CAPTIONED=%OUTDIR%\MM_LAC_extended.json
set BASE=%CAPTIONED:.json=%
set TOKENIZER=%OUTDIR%\MM_Tokenizer_extended.pkl
set TILES_JSON=%OUTDIR%\MM_tiles_%WINDOW_SIZE%x%WINDOW_SIZE%.json
set B2V_OUTPUT=%OUTDIR%\block2vec
set DIFF_OUTPUT=%OUTDIR%\diff

REM Wipe any previous run so re-runs never mix old + new data.
if exist "%OUTDIR%" rd /s /q "%OUTDIR%"
mkdir "%OUTDIR%"

REM Auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
REM A redirected file (not a pipe) keeps %ERRORLEVEL% checks working afterward.
set YES_FILE=%TEMP%\test_extended_yes.txt
echo y> "%YES_FILE%"

echo.
echo === [1/6] bucket-sorting %INPUT% into the extended tile space (%TILESET%) ===
%PY% bucket_levels_by_size.py --input "%INPUT%" --output_dir "%BUCKET_DIR%" --tileset %TILESET% --convert_to_extended --merged_output "%RAW%"
if errorlevel 1 goto error

echo.
echo === [2/6] presence-based ascii captions + train/validate/test split + tokenizer ===
%PY% MarioMaker_create_ascii_captions.py --dataset "%RAW%" --tileset %TILESET% --output "%CAPTIONED%"
if errorlevel 1 goto error
%PY% -m mm2pipeline.dataset split --input "%CAPTIONED%" --seed %SEED%
if errorlevel 1 goto error
%PY% tokenizer.py save --json_file "%BASE%-train.json" --pkl_file "%TOKENIZER%"
if errorlevel 1 goto error

echo.
echo === [3/6] slicing %WINDOW_SIZE%x%WINDOW_SIZE% tile windows for Block2Vec ===
%PY% create_tile_level_json_data.py --from_dataset "%BASE%-train.json" --output "%TILES_JSON%" --tile_size %WINDOW_SIZE%
if errorlevel 1 goto error

echo.
echo === [4/6] training Block2Vec embeddings (dim %EMBEDDING_DIM%, vocab %NUM_TILES%, %B2V_EPOCHS% epochs) ===
%PY% train_block2vec.py --json_file "%TILES_JSON%" --output_dir "%B2V_OUTPUT%" --embedding_dim %EMBEDDING_DIM% --vocab_size %NUM_TILES% --epochs %B2V_EPOCHS% --batch_size 32
if errorlevel 1 goto error

echo.
echo === [5/6] training conditional diffusion: Block2Vec + MiniLM (%DIFF_EPOCHS% epochs) ===
REM --save_image_epochs is set beyond --num_epochs on purpose: the sample renderer
REM is the 69-tile mm2 one, so we skip rendering and just track the loss curve.
%PY% train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --tileset %TILESET% --augment --text_conditional --block_embedding_model_path "%B2V_OUTPUT%" --pretrained_language_model "%TEXT_ENCODER%" --output_dir "%DIFF_OUTPUT%" --num_epochs %DIFF_EPOCHS% --save_image_epochs 1000000 --validate_epochs 1 --batch_size %BATCH_SIZE% --json "%BASE%-train.json" --val_json "%BASE%-validate.json" --pkl "%TOKENIZER%" --seed %SEED% < "%YES_FILE%"
if errorlevel 1 goto error

echo.
echo === [6/6] done ===
echo Watch the per-epoch loss in the console output above, and open the curve:
echo   "%DIFF_OUTPUT%\training_loss.png"  (Training Loss vs Validation Loss)
echo Compare its final loss against a 69-tile mm2_tileset_we run to see if extended_tiles helps.
echo (Throwaway outputs live in "%OUTDIR%"; delete it and this .bat when done.)
goto end

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%). Fix that step, then re-run. ***

:end
endlocal
