@echo off
REM Usage: train-mario-maker-block2vec.bat [embedding_dim] [seed]
REM   [embedding_dim] optional, defaults to 32 (Mario Maker has ~69 tile types,
REM                   so it needs a larger embedding than SMB's default of 16).
REM   [seed]          optional, defaults to 0.
REM
REM Trains an UNCONDITIONAL diffusion model on Mario Maker levels using learned
REM Block2Vec tile embeddings instead of one-hot tile encodings.
REM Run prepare-mario-maker.bat first to build and split the dataset.
cd ..

set EMBEDDING_DIM=%1
if "%EMBEDDING_DIM%"=="" set EMBEDDING_DIM=32

set SEED=%2
if "%SEED%"=="" set SEED=0

set GAME=MM
set TYPE=regular
REM Number of tile types in mm2_tileset_we.json (68 tiles + the "_" padding tile).
set NUM_TILES=69
set TILESET=mm2_tileset_we.json

set TILES_JSON=datasets\%GAME%_3x3_tiles-%TYPE%.json
set B2V_OUTPUT=%GAME%-block2vec-embeddings-%EMBEDDING_DIM%
set DIFF_OUTPUT=%GAME%-unconditional-block2vec-%EMBEDDING_DIM%%SEED%
set SAMPLES_OUTPUT=%DIFF_OUTPUT%-samples

REM Used to auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
set YES_FILE=%TEMP%\mariover_yes.txt
echo y> "%YES_FILE%"

REM 1) Build the 3x3 tile-window dataset straight from the integer-encoded scenes
REM    so the tile-id space matches the diffusion model exactly.
python create_tile_level_json_data.py --from_dataset datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --output %TILES_JSON% --tile_size 3

REM 2) Train the Block2Vec embeddings. --vocab_size pins the embedding table to the
REM    full tileset so every tile id has a row even if it never appears in a window.
python train_block2vec.py --json_file %TILES_JSON% --output_dir "%B2V_OUTPUT%" --embedding_dim %EMBEDDING_DIM% --vocab_size %NUM_TILES% --epochs 500 --batch_size 32

REM 3) Train the unconditional diffusion model on top of those embeddings.
python train_diffusion.py --game %GAME% --num_tiles %NUM_TILES% --augment --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --block_embedding_model_path "%B2V_OUTPUT%" --seed %SEED% < "%YES_FILE%"

REM 4) Generate levels. The trained pipeline stores its own block embeddings and
REM    decodes its embedding output back to tile space internally, so no embedding
REM    path is needed here.
python run_diffusion.py --model_path "%DIFF_OUTPUT%" --num_samples 100 --save_as_json --output_dir "%SAMPLES_OUTPUT%" --tileset %TILESET%
