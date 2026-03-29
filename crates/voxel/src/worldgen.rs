//! Simple island terrain generator.
//!
//! Generates a roughly island-shaped terrain using a radial falloff
//! from the center of the region, with some sine-based variation
//! for visual interest. Fills chunks with cube blocks.

use crate::chunk::ChunkData;
use crate::coords::{ChunkPos, CHUNK_WORLD_SIZE, REGION_XZ};
use crate::chunk::{CHUNK_X, CHUNK_Y, CHUNK_Z};
use crate::region::Region;
use crate::shape::{Facing, SHAPE_CUBE};

/// Configuration for island generation.
pub struct IslandGenConfig {
    /// Radius of the island in chunks (from center).
    pub radius_chunks: i32,
    /// Maximum height in cells above y=0.
    pub max_height: usize,
    /// Minimum height at the island edge in cells.
    pub edge_height: usize,
    /// Sea level in cells (below this is underwater/underground).
    pub sea_level: usize,
    /// Number of sine bumps around the perimeter for shape variation.
    pub perimeter_bumps: f32,
    /// Amplitude of perimeter bumps as fraction of radius.
    pub bump_amplitude: f32,
}

impl Default for IslandGenConfig {
    fn default() -> Self {
        Self {
            radius_chunks: 4,
            max_height: 12,
            edge_height: 1,
            sea_level: 0,
            perimeter_bumps: 5.0,
            bump_amplitude: 0.2,
        }
    }
}

/// Generate an island's worth of chunks and insert them into the region.
/// Returns the number of chunks generated.
pub fn generate_island(region: &mut Region, config: &IslandGenConfig) -> usize {
    let center_x = REGION_XZ as f32 / 2.0;
    let center_z = REGION_XZ as f32 / 2.0;
    let radius = config.radius_chunks as f32;

    let mut count = 0;

    // Iterate over chunks in the XZ plane around the center
    let min_cx = (center_x as i32 - config.radius_chunks - 1).max(0);
    let max_cx = (center_x as i32 + config.radius_chunks + 1).min(REGION_XZ - 1);
    let min_cz = (center_z as i32 - config.radius_chunks - 1).max(0);
    let max_cz = (center_z as i32 + config.radius_chunks + 1).min(REGION_XZ - 1);

    for cx in min_cx..=max_cx {
        for cz in min_cz..=max_cz {
            let chunk_data = generate_island_chunk(
                cx, cz, center_x, center_z, radius, config,
            );

            if chunk_data.blocks.is_empty() {
                continue;
            }

            // For now, all terrain goes in chunk y=0
            let pos = ChunkPos::new(cx, 0, cz);
            region.set_chunk(pos, chunk_data);
            count += 1;
        }
    }

    count
}

/// Generate a single chunk's worth of island terrain.
fn generate_island_chunk(
    cx: i32,
    cz: i32,
    center_x: f32,
    center_z: f32,
    radius: f32,
    config: &IslandGenConfig,
) -> ChunkData {
    let mut data = ChunkData::new();

    // For each 2x2 cell block position in this chunk (blocks are 2x1x2 cells)
    for bx in (0..CHUNK_X).step_by(2) {
        for bz in (0..CHUNK_Z).step_by(2) {
            // World-space position of this block (in chunk units)
            let wx = cx as f32 + (bx as f32 / CHUNK_X as f32);
            let wz = cz as f32 + (bz as f32 / CHUNK_Z as f32);

            // Distance from island center (in chunk units)
            let dx = wx - center_x;
            let dz = wz - center_z;
            let dist = (dx * dx + dz * dz).sqrt();

            // Angular variation for non-circular coastline
            let angle = dz.atan2(dx);
            let bump = 1.0 + config.bump_amplitude
                * (angle * config.perimeter_bumps).sin();
            let effective_radius = radius * bump;

            if dist > effective_radius {
                continue;
            }

            // Height falloff: 1.0 at center, 0.0 at edge
            let t = 1.0 - (dist / effective_radius).clamp(0.0, 1.0);
            // Smooth falloff (ease-in-out)
            let t_smooth = t * t * (3.0 - 2.0 * t);

            let height_range = config.max_height - config.edge_height;
            let height = config.edge_height + (t_smooth * height_range as f32) as usize;

            // Fill blocks from y=0 up to computed height
            for y in 0..height.min(CHUNK_Y) {
                data.place_std(bx, y, bz, SHAPE_CUBE, Facing::North, 1);
            }
        }
    }

    data
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
        assert!(count < 200, "island shouldn't be too many chunks, got {}", count);
    }

    #[test]
    fn center_chunk_has_blocks() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let config = IslandGenConfig::default();
        generate_island(&mut region, &config);

        let center = ChunkPos::new(REGION_XZ / 2, 0, REGION_XZ / 2);
        let slot = region.get_chunk(center);
        assert!(slot.is_some(), "center chunk should exist");
        assert!(!slot.unwrap().data.blocks.is_empty(), "center chunk should have blocks");
    }
}
