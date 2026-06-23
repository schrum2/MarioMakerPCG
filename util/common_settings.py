
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

# CHANGED FROM MM2 TO TRY TO GET THIS TO WORK!!!!! RESET!!!

MARIO_HEIGHT = 20
MARIO_WIDTH = 20

MARIO_TILE_PIXEL_DIM = 16
MARIO_TILE_COUNT = 17

MARIO_TILESET = 'extended_tiles.json'

MM_EXTENDED_TILE_COUNT = 17
MM_EXTENDED_TILESET = 'extended_tiles.json'

# Mario Maker 2 (the canonical training tileset; see memory/canonical-mm-tileset).
# Tiles are rendered from img/spritesheet.png using the per-object {x,y,w,h}
# rectangles in toost's LevelData.hpp ObjectLocation map (see mm2_tiles()).
MM2_TILESET = 'mm2_tileset_we.json'
MM2_TILE_PIXEL_DIM = 16
MM2_WIDTH = 20
MM2_HEIGHT = 20
# Default game style used to pick sprites (data is generated SMW by default in
# mm2_ascii_to_json.py). One of: SMB1, SMB3, SMW, NSMBU, SM3DW.
MM2_GAMESTYLE = 'SMW'
# MM2 sky-blue backdrop (toost's canvas background, #5C94FC) — composited behind
# each (often transparent) sprite so the tile grid reads correctly.
MM2_SKY_COLOR = (0x5C, 0x94, 0xFC)

