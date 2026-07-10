# Item handling in the MM2 -> SMM:WE pipeline

The pipeline has two stages that touch items. `mm2pipeline_data.extract` pulls
levels out of the HuggingFace dataset and can drop whole levels that contain
items we don't want. `mm2pipeline_data.swe` converts a single level JSON into a
`.swe` save, and that's where individual objects get remapped, approximated, or
dropped. Item names below are the canonical ones from `OBJ_ID` in
`mm2pipeline_data/bcd.py`; the SMM:WE targets are the `obj_*_res` names in
`OBJ_ID_MAP` (swe.py), written out in plain English.

## Levels skipped during extraction

These are opt-in filters, off by default. Pass the flag and the whole level is
dropped if it qualifies.

`--skip_items` drops a level if it contains any object in `SKIP_ITEM_NAMES`
(bcd.py). Tracks, snake blocks, and track blocks live in their own count fields
rather than the object array, so they're detected there:

- Fast Conveyor Belt, Conveyor Belt — no SMM:WE implementation, weird placement
  rules, and a load-bearing part of automation levels; easy to make a level
  impossible without them.
- Track, Track Block — not in SMM:WE, and usually the only way a level's
  automation or platforming works.
- Snake Block — not in SMM:WE, and often the sole means of progression.
- SMB2 Mushroom — a v3.0.0 style power-up with no SMM:WE equivalent; levels
  built around it fall apart without it. (If you *don't* pass `--skip_items`,
  swe.py still converts it, to `obj_SMB2_mushroom_res` — see the power-up notes.)

`--skip_3dworld` drops any level whose gamestyle is Super Mario 3D World. SMM:WE
has no 3D World style, so these can't convert cleanly; the filter is by gamestyle,
not by individual 3D-World-exclusive items.

`--skip_subworld_items` drops a level whose subworld contains any objects, snake
blocks, track blocks, or tracks.

## Objects with no SMM:WE equivalent (dropped in conversion)

swe.py maps each of these to `None` and emits them in the per-level "dropped"
warning count. Everything here is silently left out of the `.swe`:

Skipsqueak, Wiggler, Stingby, Cinobio, Cinobic, Ant Trooper, Warp Box, Clear
Pipe, Tree, Piranha Creeper, Blinking Block, Sound Effect marker, Crate,
Porcupuffer, Super Hammer, Bully, `!` Block (Exclamation Block), Cannon Box,
Propeller Box, Reel Camera, Snake Block, Track Block.

Snake Block and Track Block appear here too because a level can still reach swe.py
with them if it was extracted without `--skip_items`.

## Objects stored as level data, not as objects

These never become S4 objects — they carry level state that lands in the metadata
or terrain sections:

- Goal, Goal Ground — the flagpole goes into S1 (`goal_x`/`goal_y`); the goal
  ground under it is terrain.
- Player — start position, stored in S1 `start_y`.
- Water Marker — liquid level, stored in S1.
- Starting Brick — an editor-only marker, ignored.
- Slight Slope, Steep Slope — SMM:WE has no slopes, so they're filled in as
  stepped solid ground in S2 (Steep rises every column, Slight every two).

## Approximate substitutions

Direct 1:1 mappings aren't listed here (see `OBJ_ID_MAP`). These are the ones
where the target is a stand-in rather than the same object:

| MM2 item | SMM:WE object | Notes |
|---|---|---|
| Fish Bone | Cheep Cheep | |
| Spike Top | Spiny | |
| Lakitu's Cloud | Clown Car | nearest rideable vehicle |
| Koopa Car | Clown Car | |
| Shoe Goomba | Yoshi's Egg | green Yoshi; works in every gamestyle |
| Jumping Machine | Spring | |
| Mushroom Trampoline | Spring | |
| ON/OFF Trampoline | Spring | |
| Half-Collision Platform | Moving Platform | |
| Sprint Platform | Expanding Platform | |
| Charvaargh | Floruga | |
| Red Coin | Pink Coin | |
| Spike Block | Spikes | |
| Goomba Mask, Bullet Bill Mask | Cape power-up | |
| Red POW Box | POW Block | |
| Ground-as-object | Hard Block | terrain proper is in S2 |
| Lemmy, Morton, Larry, Wendy, Iggy, Roy, Ludwig | Ludwig | all seven Koopalings collapse to Ludwig, the only one in SMM:WE |

## Gamestyle-dependent power-ups

Two "style power-up" slots resolve differently per gamestyle:

- Big Mushroom (id 44, Style Power-up A) — SMB1 becomes a Super Mushroom (Mega
  Mario fallback), SMW becomes a Cape. SMB3 and NSMBU have no usable mapping and
  are dropped rather than shown wrong.
- SMB2 Mushroom (id 81, Style Power-up B) — every gamestyle maps to
  `obj_SMB2_mushroom_res` (Link / Frog Suit / Power Balloon / Super Acorn share
  the one SMM:WE slot).

Block contents (a Brick, Question, or Hidden block with an item inside) become a
sprout value on the block via `BLOCK_SPROUT_MAP`: coins turn the block into a
multi-coin block, Super Mushroom / Fire Flower / the two style power-ups set their
verified sprout ids, and anything else becomes an empty block.

## Objects routed to dedicated `.swe` sections

Not remapped so much as relocated — these get their own section builders instead
of the generic S4 object path:

- Pipe -> S5, with a Piranha Plant at the mouth of an upward pipe folded in as a
  masked plant instead of a separate object.
- Cannon -> S6, with mount and firing direction inferred from surrounding terrain.
- Door -> S8, paired two-per-entry by their in-level pairing order; a trailing
  unpaired door is dropped.
- Semisolid Platform, Bullet Bill Blaster, Mushroom Platform, Bridge, Castle
  Bridge, Lift, Seesaw -> S7 stretchy sprites. Seesaw and Lift become a moving
  platform; Castle Bridge reuses the Bridge sprite.

Flags on individual objects (wings, parachutes, and most direction bits) don't
carry over. Big-form enemies (flag bit 14) only have SMM:WE variants for Goomba,
Chain Chomp, Koopa, and Hammer Bro; every other enemy keeps its normal sprite,
since asking GameMaker for a sprite that doesn't exist crashes it.
