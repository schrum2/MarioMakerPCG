@echo off
REM Usage: train-mario-maker-big-unconditional.bat [seed]
REM [seed] is optional, defaults to 0
REM Trains an unconditional diffusion model on Mario Maker levels
REM with a larger model architecture
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=MM

set DIFF_OUTPUT=%GAME%-big-unconditional%SEED%

REM Need to set this! Must be 32x32 if 4 dim_mults are used
set DATASET=1-_10k_levels_32x32_new.json

python train_diffusion.py --batch_size 8 --model_dim 256 --dim_mults 1 2 4 4 --attention_head_dim 32 --down_block_types "DownBlock2D" "AttnDownBlock2D" "AttnDownBlock2D" "AttnDownBlock2D" --up_block_types "AttnUpBlock2D" "AttnUpBlock2D" "AttnUpBlock2D" "UpBlock2D" --game %GAME% --save_image_epochs 20 --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%DATASET% --seed %SEED%


REM call run_diffusion too

REM wrong tile set?
REM python evaluate_tile_distribution.py --model_path %DIFF_OUTPUT% --num_tiles 17 --tileset extended_tiles.json



