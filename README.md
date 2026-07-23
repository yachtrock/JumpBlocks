# JumpBlocks

A voxel platformer that crosses **Minecraft-style building** with the
**open-world platforming of Bowser's Fury**: an archipelago of islands rising
out of a shallow walkable sea, dotted with platforming challenges. Earn
trophies by clearing challenges, turn them in to unlock each island's locked
zone, and reshape the terrain with buildable blocks — including slopes, with
chamfered edges everywhere geometry meets.

## The loop

1. **Explore** the archipelago on foot (or ride the Sea Ferry between islands).
2. **Challenge**: step on a green **start pad** to activate a challenge — a
   golden **goal** appears somewhere on the island. Some challenges have a
   **time limit**; some have a **kill height** (fall below it and you're sent
   back to the start pad).
3. **Trophy**: reach the goal to earn that challenge's trophy (once; runs can
   be replayed for fun).
4. **Unlock**: every island has a **locked zone** (shown as a red translucent
   volume) where terrain can't be modified. Bring enough of that island's
   trophies to the zone's **pedestal** (purple crystal) to unlock it.
5. **Build**: place cubes and wedge ramps anywhere outside locked zones —
   except during an active challenge run.

Progress (trophies + unlocked zones) is saved per-world in `progress.json`.

## The world

| Island | Character | Challenges |
|---|---|---|
| **Haven Isle** | Big grassy spawn island | First Steps (untimed hop course), Ridge Runner (50s summit run) |
| **Ember Isle** | Dark basalt cone | Magma Hop (kill-height sea crossing), Ember Ascent (60s ramped spiral) |
| **Skyreach Spire** | Tall thin spire | Spire Spiral (90s floating spiral with a moving-platform gap) |
| **Step Cay** | Small sandy islet | — (free build space) |

Two kinematic **moving platforms** run on fixed oscillations: the *Sea Ferry*
(Haven ↔ Ember) and the *Spire Lift* (bridging the gap in Spire Spiral).

All of this is authored in
[`crates/voxel/src/world_def.rs`](crates/voxel/src/world_def.rs) as plain
data + deterministic stamping, so the same definition drives the game, the
tests, and the web exporter.

## Running

```sh
cargo run                      # play (hosts a listen server)
cargo run -- --connect IP:PORT # join a friend
cargo run -- --start-server    # dedicated headless server
cargo run -- --export-web world.json   # export the world for the web viewer
```

On a headless Linux box, `scripts/software-render.sh` runs the game under
Xvfb with Mesa's software Vulkan driver.

### Controls

| Action | Keyboard | Gamepad |
|---|---|---|
| Move / run | WASD + Shift | Left stick + East |
| Jump | Space | South |
| Inventory (enter build mode) | Tab | North |
| Place block / rotate | F / Q+arrows | ZR / ZL+d-pad |
| Debug UI / wireframe / chamfer / LOD tint | ` / F2 / F3 / F4 | — |

## Architecture

- **`crates/voxel`** — the voxel engine: 32³-cell chunks (0.5-unit voxels),
  multi-cell block shapes (cube, wedge ramp) with **chamfered-edge meshing**
  (cut & offset chamfer), LOD + dither crossfade, chunk clustering,
  streaming, persistence, archipelago worldgen (`worldgen.rs`), block
  stamping (`stamp.rs`), and the authored world (`world_def.rs`).
- **`src/`** — the game: avian3d physics + Tnua character controller,
  orbit camera, Rhai-scripted action states & HUD, block building
  (`building.rs`), and the challenge/trophy/zone systems (`challenge.rs`).
- **`crates/ui`** — threaded immediate-mode UI canvas driven by Rhai
  scripts in `assets/scripts/`.
- Multiplayer via lightyear (listen server by default).

## Tests

```sh
cargo test --workspace
```

Voxel mesh correctness (watertightness, chamfer behavior), worldgen,
stamping, world-definition consistency, challenge/zone logic, persistence
round-trips, and headless networking tests.
