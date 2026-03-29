//! Island terrain generator.
//!
//! Generates island terrain that spans multiple vertical chunks.
//! Uses a heightmap approach: for each XZ column, compute a terrain
//! height in cells, then fill blocks from y=0 up to that height,
//! distributing across as many Y-chunks as needed.

use crate::chunk::ChunkData;
use crate::coords::{ChunkPos, REGION_XZ, CHUNK_Y_MIN};
use crate::chunk::{CHUNK_X, CHUNK_Y, CHUNK_Z};
use crate::region::Region;
use crate::shape::{Facing, SHAPE_CUBE};

/// Configuration for island generation.
pub struct IslandGenConfig {
    /// Radius of the island in chunks (from center).
    pub radius_chunks: i32,
    /// Maximum terrain height in cells at the peak.
    pub peak_height: usize,
    /// Height at the island edge in cells.
    pub edge_height: usize,
    /// Number of sine bumps around the perimeter for coastline shape.
    pub perimeter_bumps: f32,
    /// Amplitude of perimeter bumps as fraction of radius.
    pub bump_amplitude: f32,
    /// Number of smaller sine bumps for surface ridges.
    pub ridge_count: f32,
    /// Height of ridges in cells.
    pub ridge_height: f32,
    /// Depth of underground fill below y=0 in cells.
    pub underground_depth: usize,
}

impl Default for IslandGenConfig {
    fn default() -> Self {
        Self {
            radius_chunks: 8,
            peak_height: 80,
            edge_height: 2,
            perimeter_bumps: 5.0,
            bump_amplitude: 0.25,
            ridge_count: 13.0,
            ridge_height: 8.0,
            underground_depth: 16,
        }
    }
}

/// Generate an island and insert chunks into the region.
/// Returns the number of chunks generated.
pub fn generate_island(region: &mut Region, config: &IslandGenConfig) -> usize {
    let center_x = REGION_XZ as f32 / 2.0;
    let center_z = REGION_XZ as f32 / 2.0;
    let radius = config.radius_chunks as f32;

    let min_cx = (center_x as i32 - config.radius_chunks - 1).max(0);
    let max_cx = (center_x as i32 + config.radius_chunks + 1).min(REGION_XZ - 1);
    let min_cz = (center_z as i32 - config.radius_chunks - 1).max(0);
    let max_cz = (center_z as i32 + config.radius_chunks + 1).min(REGION_XZ - 1);

    // First pass: compute the heightmap for every block column.
    // Stored as (cx, cz) → array of heights per block column in that chunk.
    // Block columns are at even cell coords (step 2) so 16x16 per chunk.
    let block_cols = CHUNK_X / 2;
    let mut chunk_heights: Vec<(i32, i32, Vec<Vec<i32>>)> = Vec::new();

    for cx in min_cx..=max_cx {
        for cz in min_cz..=max_cz {
            let mut heights: Vec<Vec<i32>> = vec![vec![0; block_cols]; block_cols];
            let mut any_nonzero = false;

            for bxi in 0..block_cols {
                for bzi in 0..block_cols {
                    let bx = bxi * 2;
                    let bz = bzi * 2;

                    let wx = cx as f32 + (bx as f32 / CHUNK_X as f32);
                    let wz = cz as f32 + (bz as f32 / CHUNK_Z as f32);

                    let h = compute_height(wx, wz, center_x, center_z, radius, config);
                    heights[bxi][bzi] = h;
                    if h > 0 {
                        any_nonzero = true;
                    }
                }
            }

            if any_nonzero {
                chunk_heights.push((cx, cz, heights));
            }
        }
    }

    // Second pass: for each XZ chunk column, create vertical chunks as needed.
    let mut count = 0;

    for (cx, cz, heights) in &chunk_heights {
        // Find the min and max cell Y across all block columns in this XZ chunk
        let min_cell_y: i32 = -(config.underground_depth as i32);
        let mut max_cell_y: i32 = 0;
        for col in heights {
            for &h in col {
                if h > max_cell_y {
                    max_cell_y = h;
                }
            }
        }

        if max_cell_y <= 0 && min_cell_y >= 0 {
            continue;
        }

        // Convert cell Y range to chunk Y range
        let min_chunk_y = cell_y_to_chunk_y(min_cell_y);
        let max_chunk_y = cell_y_to_chunk_y(max_cell_y);

        for cy in min_chunk_y..=max_chunk_y {
            let chunk_base_cell_y = cy * CHUNK_Y as i32;

            let mut data = ChunkData::new();
            let mut has_blocks = false;

            for bxi in 0..block_cols {
                for bzi in 0..block_cols {
                    let bx = bxi * 2;
                    let bz = bzi * 2;
                    let surface_h = heights[bxi][bzi];
                    let bottom = min_cell_y;

                    // Fill from bottom to surface_h
                    for local_y in 0..CHUNK_Y {
                        let global_cell_y = chunk_base_cell_y + local_y as i32;

                        if global_cell_y >= bottom && global_cell_y < surface_h {
                            data.place_std(bx, local_y, bz, SHAPE_CUBE, Facing::North, 1);
                            has_blocks = true;
                        }
                    }
                }
            }

            if has_blocks {
                let pos = ChunkPos::new(*cx, cy, *cz);
                if pos.in_bounds() {
                    region.set_chunk(pos, data);
                    count += 1;
                }
            }
        }
    }

    count
}

/// Compute terrain height at a world-space XZ position (in chunk units).
/// Returns height in cells above y=0.
fn compute_height(
    wx: f32,
    wz: f32,
    center_x: f32,
    center_z: f32,
    radius: f32,
    config: &IslandGenConfig,
) -> i32 {
    let dx = wx - center_x;
    let dz = wz - center_z;
    let dist = (dx * dx + dz * dz).sqrt();

    // Coastline shape: angular variation
    let angle = dz.atan2(dx);
    let bump = 1.0 + config.bump_amplitude * (angle * config.perimeter_bumps).sin();
    let effective_radius = radius * bump;

    if dist > effective_radius {
        return 0;
    }

    // Radial falloff: 1.0 at center, 0.0 at edge
    let t = 1.0 - (dist / effective_radius).clamp(0.0, 1.0);
    // Smooth falloff
    let t_smooth = t * t * (3.0 - 2.0 * t);

    // Base height from radial falloff
    let height_range = config.peak_height as f32 - config.edge_height as f32;
    let base_height = config.edge_height as f32 + t_smooth * height_range;

    // Surface ridges: two overlapping sine patterns for more interesting terrain
    let ridge1 = (wx * 1.7 + wz * 0.9).sin() * config.ridge_height * t;
    let ridge2 = (wx * 0.6 - wz * 2.3).sin() * config.ridge_height * 0.5 * t;

    let total = base_height + ridge1 + ridge2;
    total.max(0.0) as i32
}

/// Convert a global cell Y coordinate to the chunk Y index that contains it.
fn cell_y_to_chunk_y(cell_y: i32) -> i32 {
    if cell_y >= 0 {
        cell_y / CHUNK_Y as i32
    } else {
        (cell_y - CHUNK_Y as i32 + 1) / CHUNK_Y as i32
    }
}

/// Find the surface height at the center of the island (for spawn point).
/// Returns world-space Y coordinate of the surface.
pub fn island_spawn_height(config: &IslandGenConfig) -> f32 {
    let center_x = REGION_XZ as f32 / 2.0;
    let center_z = REGION_XZ as f32 / 2.0;
    let radius = config.radius_chunks as f32;

    // Sample a few points near center and take the max
    let mut max_h = 0i32;
    for dx in -1..=1 {
        for dz in -1..=1 {
            let wx = center_x + dx as f32 * 0.1;
            let wz = center_z + dz as f32 * 0.1;
            let h = compute_height(wx, wz, center_x, center_z, radius, config);
            if h > max_h {
                max_h = h;
            }
        }
    }

    // Convert cell height to world units (cells * VOXEL_SIZE)
    max_h as f32 * crate::chunk::VOXEL_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::Region;
    use crate::coords::RegionId;
    use bevy::prelude::*;

    #[test]
    fn generate_island_produces_chunks() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let config = IslandGenConfig::default();
        let count = generate_island(&mut region, &config);
        assert!(count > 0, "island should generate at least one chunk");
        // With 8-chunk radius and multiple Y levels, expect many chunks
        eprintln!("Generated {} chunks", count);
    }

    #[test]
    fn center_chunk_has_blocks() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let config = IslandGenConfig::default();
        generate_island(&mut region, &config);

        let center = ChunkPos::new(REGION_XZ / 2, 0, REGION_XZ / 2);
        let slot = region.get_chunk(center);
        assert!(slot.is_some(), "center chunk at y=0 should exist");
        assert!(!slot.unwrap().data.blocks.is_empty(), "center chunk should have blocks");
    }

    #[test]
    fn multi_y_chunks_generated() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let config = IslandGenConfig::default();
        generate_island(&mut region, &config);

        // With peak_height=80, we need at least 3 Y-chunks (80/32 = 2.5)
        let center_x = REGION_XZ / 2;
        let center_z = REGION_XZ / 2;
        let has_y1 = region.has_chunk(ChunkPos::new(center_x, 1, center_z));
        let has_y2 = region.has_chunk(ChunkPos::new(center_x, 2, center_z));
        assert!(has_y1, "should have chunk at y=1 for tall terrain");
        assert!(has_y2, "should have chunk at y=2 for tall terrain");
    }

    #[test]
    fn underground_chunks_generated() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let config = IslandGenConfig::default();
        generate_island(&mut region, &config);

        let center_x = REGION_XZ / 2;
        let center_z = REGION_XZ / 2;
        let has_neg = region.has_chunk(ChunkPos::new(center_x, -1, center_z));
        assert!(has_neg, "should have underground chunk at y=-1");
    }

    #[test]
    fn spawn_height_is_reasonable() {
        let config = IslandGenConfig::default();
        let h = island_spawn_height(&config);
        assert!(h > 5.0, "spawn height should be above ground, got {}", h);
        assert!(h < 100.0, "spawn height shouldn't be crazy high, got {}", h);
    }

    #[test]
    fn cell_y_to_chunk_y_correct() {
        assert_eq!(cell_y_to_chunk_y(0), 0);
        assert_eq!(cell_y_to_chunk_y(31), 0);
        assert_eq!(cell_y_to_chunk_y(32), 1);
        assert_eq!(cell_y_to_chunk_y(-1), -1);
        assert_eq!(cell_y_to_chunk_y(-32), -1);
        assert_eq!(cell_y_to_chunk_y(-33), -2);
    }
}
