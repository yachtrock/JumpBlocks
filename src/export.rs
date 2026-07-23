//! World export for the web viewer (`--export-web <file>`).
//!
//! Builds the authored world in memory and writes a JSON file containing:
//! - every *visible* block (at least one exposed face) as packed base64
//!   `Int16` data: `[x, y, z, shape, facing, texture]` per block, in global
//!   cell coordinates,
//! - the island / zone / challenge / moving-platform definitions in world
//!   coordinates.
//!
//! The web viewer renders this directly, so what you see in the browser is
//! exactly the data the game plays.

use std::path::Path;

use bevy::prelude::*;
use jumpblocks_voxel::chunk::{CHUNK_X, CHUNK_Y, CHUNK_Z, VOXEL_SIZE};
use jumpblocks_voxel::coords::{ChunkPos, RegionId};
use jumpblocks_voxel::region::Region;
use jumpblocks_voxel::world_def::{WorldDef, cell_base_world, zone_world_aabb};

use crate::world::REGION_ORIGIN;

pub fn export_world(out_path: &Path) {
    println!("Building world for export...");
    let def = WorldDef::standard();
    let mut region = Region::new(RegionId(0), REGION_ORIGIN);
    let chunk_count = def.build_into_region(&mut region);
    println!("  {} chunks generated", chunk_count);

    let blocks = collect_visible_blocks(&region);
    println!("  {} visible blocks collected", blocks.len() / 6);

    let json = build_json(&def, &blocks);
    std::fs::write(out_path, json).expect("failed to write export file");
    println!("Wrote {}", out_path.display());
}

/// Pack all blocks with at least one exposed face as flat i16 sextuples.
fn collect_visible_blocks(region: &Region) -> Vec<i16> {
    let mut out: Vec<i16> = Vec::new();

    // Occlusion check across chunk borders, in global cell coords. Only
    // full cubes occlude: slope caps mark partially-empty cells as occupied
    // (for gameplay), and treating those as solid would cull visible blocks
    // behind slopes.
    let occupied = |gx: i32, gy: i32, gz: i32| -> bool {
        let cx = gx.div_euclid(CHUNK_X as i32);
        let cy = gy.div_euclid(CHUNK_Y as i32);
        let cz = gz.div_euclid(CHUNK_Z as i32);
        let Some(slot) = region.get_chunk(ChunkPos::new(cx, cy, cz)) else {
            return false;
        };
        slot.data
            .resolve_block(
                gx.rem_euclid(CHUNK_X as i32) as usize,
                gy.rem_euclid(CHUNK_Y as i32) as usize,
                gz.rem_euclid(CHUNK_Z as i32) as usize,
            )
            .map(|b| b.shape == 0)
            .unwrap_or(false)
    };

    for (pos, slot) in region.iter_chunks() {
        let base_x = pos.x * CHUNK_X as i32;
        let base_y = pos.y * CHUNK_Y as i32;
        let base_z = pos.z * CHUNK_Z as i32;

        for block in slot.data.blocks.iter() {
            let (ox, oy, oz) = block.origin;
            let gx = base_x + ox as i32;
            let gy = base_y + oy as i32;
            let gz = base_z + oz as i32;

            // The flat sea floor (y=0) and everything under it is rendered
            // by the viewer as flat shallows discs — skip those blocks to
            // keep the export small.
            if gy <= 0 {
                continue;
            }

            // Footprint by shape: cube 2x1x2 (shape 0), wedge 2x2x2 (shape 1)
            let (sx, sy, sz) = if block.shape == 1 { (2, 2, 2) } else { (2, 1, 2) };

            // Wedges are always visible (sloped face); cubes only when a
            // face is exposed.
            let mut visible = block.shape != 0;
            if !visible {
                'faces: for dy in 0..sy {
                    for dx in 0..sx {
                        for dz in 0..sz {
                            let (cx, cy, cz) = (gx + dx, gy + dy, gz + dz);
                            for (nx, ny, nz) in
                                [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
                            {
                                let (tx, ty, tz) = (cx + nx, cy + ny, cz + nz);
                                // Skip cells inside this block
                                if tx >= gx && tx < gx + sx
                                    && ty >= gy && ty < gy + sy
                                    && tz >= gz && tz < gz + sz
                                {
                                    continue;
                                }
                                if !occupied(tx, ty, tz) {
                                    visible = true;
                                    break 'faces;
                                }
                            }
                        }
                    }
                }
            }

            if visible {
                // Center the viewer on the region midpoint: offset so
                // coordinates fit i16 (region is 8192 cells across).
                out.push((gx - 4096) as i16);
                out.push(gy as i16);
                out.push((gz - 4096) as i16);
                out.push(block.shape as i16);
                out.push(block.facing as i16);
                out.push(block.texture as i16);
            }
        }
    }
    out
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

/// World position relative to the viewer origin (region midpoint, which is
/// also the game's world origin since REGION_ORIGIN = -midpoint).
fn viewer_pos(cell: IVec3) -> [f32; 3] {
    let p = cell_base_world(REGION_ORIGIN, cell);
    [p.x, p.y, p.z]
}

fn build_json(def: &WorldDef, blocks: &[i16]) -> String {
    let mut bytes = Vec::with_capacity(blocks.len() * 2);
    for v in blocks {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let data_b64 = base64_encode(&bytes);

    let islands: Vec<serde_json::Value> = def
        .terrain
        .islands
        .iter()
        .map(|i| {
            serde_json::json!({
                "name": i.name,
                // chunk units -> world units, recentered on region midpoint
                "center": [
                    i.center.0 * CHUNK_X as f32 * VOXEL_SIZE - 2048.0,
                    i.center.1 * CHUNK_Z as f32 * VOXEL_SIZE - 2048.0
                ],
                "radius": i.radius_chunks * CHUNK_X as f32 * VOXEL_SIZE,
                "peak": i.peak_height as f32 * VOXEL_SIZE,
            })
        })
        .collect();

    let zones: Vec<serde_json::Value> = def
        .zones
        .iter()
        .map(|z| {
            let (min, max) = zone_world_aabb(REGION_ORIGIN, z);
            serde_json::json!({
                "name": z.name,
                "island": z.island,
                "min": [min.x, min.y, min.z],
                "max": [max.x, max.y, max.z],
                "required": z.trophies_required,
                "pedestal": viewer_pos(z.pedestal_cell),
            })
        })
        .collect();

    let challenges: Vec<serde_json::Value> = def
        .challenges
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "island": c.island,
                "start": viewer_pos(c.start_cell),
                "goal": viewer_pos(c.goal_cell),
                "timeLimit": c.time_limit,
                "killY": c.kill_y_cell.map(|y| y as f32 * VOXEL_SIZE),
            })
        })
        .collect();

    let platforms: Vec<serde_json::Value> = def
        .platforms
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.name,
                "from": viewer_pos(p.from_cell),
                "to": viewer_pos(p.to_cell),
                "halfExtents": [p.half_extents.x, p.half_extents.y, p.half_extents.z],
                "period": p.period,
            })
        })
        .collect();

    let doc = serde_json::json!({
        "voxelSize": VOXEL_SIZE,
        "blockCount": blocks.len() / 6,
        // The walkable sand shallows around each island (drawn as discs;
        // the actual sea-floor blocks are omitted from the block data).
        "shallowRadius": def.terrain.flat_extent * CHUNK_X as f32 * VOXEL_SIZE,
        "shallowHeight": def.terrain.flat_height as f32 * VOXEL_SIZE,
        "blocks": data_b64,
        "islands": islands,
        "zones": zones,
        "challenges": challenges,
        "platforms": platforms,
        "spawn": viewer_pos(def.spawn_cell),
    });
    doc.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_known_values() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn export_produces_blocks_and_defs() {
        let def = WorldDef::standard();
        let mut region = Region::new(RegionId(0), REGION_ORIGIN);
        def.build_into_region(&mut region);
        let blocks = collect_visible_blocks(&region);
        assert!(blocks.len() / 6 > 10_000, "expected many visible blocks, got {}", blocks.len() / 6);
        // Far fewer than total cells — hidden interior culled
        let json = build_json(&def, &blocks);
        assert!(json.contains("Haven Isle"));
        assert!(json.contains("challenges"));
    }
}
