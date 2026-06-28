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