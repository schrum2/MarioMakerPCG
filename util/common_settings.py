
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

# Mario Maker uses the canonical MM2 tileset (mm2_tileset_we.json). The MM data is
# encoded as sorted(tileset['tiles']) + the appended '_' padding tile, giving 69
# tile ids (0-68): 68 real tiles plus '_'. The trained block2vec embeddings and the
# scene data both use this 69-id range (verified: datasets/MM_LevelsAndCaptions use
# ids 0-68; block_embeddings.pt is 69 rows), so the tile count must be 69. Using a
# smaller tileset (e.g. the 17-tile extended_tiles.json) makes the model emit tile
# ids outside the tileset and assign_caption raises KeyError: <id> on lookup.
MARIO_HEIGHT = 20
MARIO_WIDTH = 20

MARIO_TILE_PIXEL_DIM = 16
MARIO_TILE_COUNT = 69

MARIO_TILESET = 'mm2_tileset_we.json'

# Kept as aliases of the canonical MM2 tileset/count above so older "MM_EXTENDED"
# callers stay in sync.
MM_EXTENDED_TILE_COUNT = 69
MM_EXTENDED_TILESET = 'mm2_tileset_we.json'

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

# Old Mario
#MARIO_TILESET = 'datasets/smb.json'

# Not used

LR_HEIGHT = 32
LR_WIDTH = 32

LR_TILE_PIXEL_DIM = 8
LR_TILE_COUNT = 8

LR_TILESET = 'datasets/Loderunner.json'

MEGAMAN_HEIGHT = 14
MEGAMAN_WIDTH = 16

MM_TILE_PIXEL_DIM = 16
MM_SIMPLE_TILE_COUNT = 13
MM_FULL_TILE_COUNT = 41

MM_FULL_TILESET = 'datasets/MM.json'
MM_SIMPLE_TILESET = 'datasets/MM_Simple_Tileset.json'
