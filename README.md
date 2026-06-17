# Mario Maker Procedural Generation

# Setting up
Run `pip install -r requirements.txt`
Run `pip install -r requirements_mario_diffusion.txt`

# Running Bat Files

**Extract real MM2 levels to ASCII** (bcd -> json -> images -> ascii)
bat\extract_levels_to_ascii.bat <name_filter> <count> <output_folder>

**Prepare a Mario Maker dataset end-to-end (simple presence captions):**
bat\prepare-mario-maker.bat <folder of ascii .txt> <seed>

**Prepare a Mario Maker dataset end-to-end (LLM captions via local Ollama):**
bat\prepare-mario-maker-llm.bat <folder of ascii .txt> <model> <seed>

**Run the full pipeline (prepare + train MLM + train diffusion + generate + evaluate):**
bat\run_full_pipeline.bat <input> <model> <type> <game> <seed>

**Generate samples from a trained model:**
bat\run_diffusion_multi.bat <model_path> <type> <game>

**Evaluate caption adherence of a trained model:**
bat\evaluate_caption_adherence_multi.bat <model_path> <type> <game>

# Running Python Scripts

**Run the dataset/level JSON viewer**
python mm2_viewer_json.py <level.json>

**Run the ASCII dataset browser**
python ascii_data_browser.py <dataset.json> <tileset.json>

**Build a dataset from ASCII level files (with captions):**
python build_dataset_with_ascii.py --input_file <input.txt or folder> --output <dataset_name>.json --tileset extended_tiles.json --sliding_window --stride 20 --convert_to_extended

**Converting a .json to BCD**
python json_to_bcd.py <exact .json file location> --toost-compat
--toost-compat is optional however it will make some custom levels appear in toost

**Converting a .json to SWE**
python json_to_swe.py <json file or folder> -o <output .swe path or folder> --user <username>