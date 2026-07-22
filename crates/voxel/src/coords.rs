use std::fmt;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Constants — tweak these to resize the world grid
// ---------------------------------------------------------------------------

/// Chunks per region along X and Z axes.
pub const REGION_XZ: i32 = 256;

/// Chunk range along Y axis: -REGION_Y_HALF..+REGION_Y_HALF.
pub const REGION_Y_HALF: i32 = 128;

/// Minimum chunk Y coordinate (inclusive).
pub const CHUNK_Y_MIN: i32 = -REGION_Y_HALF;

/// Maximum chunk Y coordinate (inclusive).
pub const CHUNK_Y_MAX: i32 = REGION_Y_HALF - 1;

// Re-export the voxel-level constants from chunk.rs for convenience.
pub use crate::chunk::{CHUNK_X, CHUNK_Y, CHUNK_Z, VOXEL_SIZE};

/// World-unit extent of a single chunk along each axis.
pub const CHUNK_WORLD_SIZE: f32 = CHUNK_X as f32 * VOXEL_SIZE; // 32 * 0.5 = 16.0

// ---------------------------------------------------------------------------
// RegionId
// ---------------------------------------------------------------------------

/// Unique identifier for a region (island).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RegionId(pub u32);

impl fmt::Display for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "region_{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ChunkPos — position of a chunk within a region
// ---------------------------------------------------------------------------

/// Integer position of a chunk within a region.
///
/// X and Z range from `0..REGION_XZ` (256).
/// Y ranges from `CHUNK_Y_MIN..=CHUNK_Y_MAX` (-128..=127).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Whether this position is within the valid region bounds.
    pub fn in_bounds(self) -> bool {
        self.x >= 0
            && self.x < REGION_XZ
            && self.z >= 0
            && self.z < REGION_XZ
            && self.y >= CHUNK_Y_MIN
            && self.y <= CHUNK_Y_MAX
    }

    /// Offset to a neighboring chunk position. Returns `None` if out of bounds.
    pub fn neighbor(self, dx: i32, dy: i32, dz: i32) -> Option<ChunkPos> {
        let pos = ChunkPos::new(self.x + dx, self.y + dy, self.z + dz);
        if pos.in_bounds() { Some(pos) } else { None }
    }

    /// Convert chunk position to world-space origin (local to the region).
    /// This is the minimum corner of the chunk in world units.
    pub fn to_world_offset(self) -> Vec3 {
        Vec3::new(
            self.x as f32 * CHUNK_WORLD_SIZE,
            self.y as f32 * CHUNK_WORLD_SIZE,
            self.z as f32 * CHUNK_WORLD_SIZE,
        )
    }

    /// Convert a world-space position (region-local) to the chunk that contains it.
    pub fn from_world(world_pos: Vec3) -> Self {
        Self {
            x: (world_pos.x / CHUNK_WORLD_SIZE).floor() as i32,
            y: (world_pos.y / CHUNK_WORLD_SIZE).floor() as i32,
            z: (world_pos.z / CHUNK_WORLD_SIZE).floor() as i32,
        }
    }

    /// Iterate all 26 neighbor offsets.
    pub fn neighbor_offsets() -> impl Iterator<Item = (i32, i32, i32)> {
        (-1..=1i32).flat_map(|dx| {
            (-1..=1i32).flat_map(move |dy| {
                (-1..=1i32).filter_map(move |dz| {
                    if dx == 0 && dy == 0 && dz == 0 {
                        None
                    } else {
                        Some((dx, dy, dz))
                    }
                })
            })
        })
    }
}

impl fmt::Display for ChunkPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------
// ColumnPos — XZ position of a vertical column of chunks
// ---------------------------------------------------------------------------

/// XZ position of a chunk column within a region.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ColumnPos {
    pub x: i32,
    pub z: i32,
}

impl ColumnPos {
    pub const fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    pub fn from_chunk(pos: ChunkPos) -> Self {
        Self { x: pos.x, z: pos.z }
    }

    pub fn in_bounds(self) -> bool {
        self.x >= 0 && self.x < REGION_XZ && self.z >= 0 && self.z < REGION_XZ
    }
}

// ---------------------------------------------------------------------------
// ECS component — attach to chunk entities for spatial lookup
// ---------------------------------------------------------------------------

/// ECS component that tags a chunk entity with its region and grid position.
#[derive(Component, Clone, Copy, Debug)]
pub struct ChunkCoord {
    pub region: RegionId,
    pub pos: ChunkPos,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pos_in_bounds() {
        assert!(ChunkPos::new(0, CHUNK_Y_MIN, 0).in_bounds());
        assert!(ChunkPos::new(255, CHUNK_Y_MAX, 255).in_bounds());
        assert!(!ChunkPos::new(256, 0, 0).in_bounds());
        assert!(!ChunkPos::new(0, CHUNK_Y_MAX + 1, 0).in_bounds());
        assert!(!ChunkPos::new(-1, 0, 0).in_bounds());
    }

    #[test]
    fn chunk_pos_world_roundtrip() {
        let pos = ChunkPos::new(10, -5, 20);
        let world = pos.to_world_offset();
        let back = ChunkPos::from_world(world);
        assert_eq!(pos, back);
    }

    #[test]
    fn chunk_pos_neighbor_out_of_bounds() {
        let edge = ChunkPos::new(255, CHUNK_Y_MAX, 255);
        assert!(edge.neighbor(1, 0, 0).is_none());
        assert!(edge.neighbor(0, 1, 0).is_none());
        assert!(edge.neighbor(-1, 0, 0).is_some());
    }

    #[test]
    fn chunk_world_size_correct() {
        // 32 cells * 0.5 units = 16.0
        assert!((CHUNK_WORLD_SIZE - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn neighbor_offsets_count() {
        assert_eq!(ChunkPos::neighbor_offsets().count(), 26);
    }
}
