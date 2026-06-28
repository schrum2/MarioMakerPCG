python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V16_05" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.05
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V16_03" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V16_07" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.07
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V16_10" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.1

python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V08_05" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.05
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V08_03" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.03
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V08_07" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.07
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V08_10" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.1


python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip16_05" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.05
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip16_03" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip16_07" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.07
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip16_10" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.1


python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip08_05" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.05
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip08_03" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.03
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip08_07" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.07
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip08_10" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.1


REM ==========================================================================
REM SWEEP 2: follow-up runs based on results from sweep 1.
REM
REM Findings from sweep 1 that motivate these:
REM   - dim=16 beat dim=8 cleanly on every metric (no near-duplicate tile
REM     pairs at dim=16 vs several at dim=8) -- so we push dim higher (20, 24)
REM     to see whether that trend continues or has already plateaued, since
REM     dim=16 was only using ~91% of its dimensions (not fully saturated).
REM   - subsample_threshold 0.03 beat 0.05/0.07/0.10 monotonically in every
REM     method/dim group, and the trend hadn't visibly bottomed out -- so we
REM     push lower (0.01, 0.02) to find where it actually plateaus.
REM   - --no_subsampling was never tried -- direct test of turning the whole
REM     mechanism off, since the dataset's frequency skew (mentioned by the
REM     user) is exactly the scenario subsampling targets, and it's worth
REM     knowing whether subsampling is helping at all vs. just changing which
REM     tiles get the most updates.
REM   - --negative_samples was never swept away from its default (10).
REM     Crowded similarity space (the near-duplicate tile pairs seen at
REM     dim=8, and even some spread issues at dim=16) is exactly the failure
REM     mode harder negative sampling is supposed to help with.
REM   - --use_class_weights and --focal_gamma (block2vec only -- not
REM     available in train_skipgram.py) directly target the update-count
REM     imbalance (max/min update ratio was in the millions-to-tens-of-
REM     millions range and climbing with higher subsample_threshold, the
REM     opposite of subsampling's intent) -- never tried.
REM
REM Naming: B2V_dim{D}_thresh{T}[_neg{N}][_cw][_focal{G}] / Skip_dim{D}_thresh{T}[_neg{N}]
REM No-subsampling runs use _nosub instead of _thresh since the threshold is
REM unused in that mode. compare_embeddings.py parses these descriptive names
REM directly (no positional encoding needed).
REM ==========================================================================

REM --- 2a: lower subsample thresholds, at the dim that won (16), both methods ---
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh01" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.01
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh02" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.02
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim16_thresh01" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.01
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim16_thresh02" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.02

REM --- 2b: no subsampling at all, both methods, both dims (cheap to also recheck dim=8 here) ---
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_nosub" --embedding_dim 16 --vocab_size 69 --no_subsampling
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim8_nosub" --embedding_dim 8 --vocab_size 69 --no_subsampling
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim16_nosub" --embedding_dim 16 --vocab_size 69 --no_subsampling
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim8_nosub" --embedding_dim 8 --vocab_size 69 --no_subsampling

REM --- 2c: push embedding_dim higher than 16, at the best threshold so far (0.03) ---
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim20_thresh03" --embedding_dim 20 --vocab_size 69 --subsample_threshold 0.03
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim24_thresh03" --embedding_dim 24 --vocab_size 69 --subsample_threshold 0.03
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim20_thresh03" --embedding_dim 20 --vocab_size 69 --subsample_threshold 0.03
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim24_thresh03" --embedding_dim 24 --vocab_size 69 --subsample_threshold 0.03

REM --- 2d: harder negative sampling, at dim=16/thresh=0.03 (best combo so far) and dim=8 (where crowding was worst) ---
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh03_neg15" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03 --negative_samples 15
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh03_neg20" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03 --negative_samples 20
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim8_thresh03_neg20" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.03 --negative_samples 20
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim16_thresh03_neg15" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03 --negative_samples 15
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim16_thresh03_neg20" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03 --negative_samples 20
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim8_thresh03_neg20" --embedding_dim 8 --vocab_size 69 --subsample_threshold 0.03 --negative_samples 20

REM --- 2e: block2vec-only -- class weights / focal loss to directly attack update-count imbalance ---
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh03_cw" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03 --use_class_weights
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh03_focal2" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03 --focal_gamma 2.0
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh03_cw_focal2" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.03 --use_class_weights --focal_gamma 2.0
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_nosub_cw" --embedding_dim 16 --vocab_size 69 --no_subsampling --use_class_weights

REM --- 2f: combine the two strongest single-axis wins (lower threshold + harder negatives) to check for interaction effects ---
python train_block2vec.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "B2V_dim16_thresh01_neg15" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.01 --negative_samples 15
python train_skipgram.py --json_file datasets\Tile3x3_dataset_10k_1-_bucket.json --output_dir "Skip_dim16_thresh01_neg15" --embedding_dim 16 --vocab_size 69 --subsample_threshold 0.01 --negative_samples 15