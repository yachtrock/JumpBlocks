use bevy::prelude::*;

use crate::shape::Facing;

pub const CHUNK_X: usize = 16;
pub const CHUNK_Y: usize = 32; // voxel count vertically (half-height, so 16 world units)
pub const CHUNK_Z: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;

pub const VOXEL_WIDTH: f32 = 1.0;
pub const VOXEL_HEIGHT: f32 = 0.5;

/// Packed 32-bit voxel data.
///
/// Layout:
///   [1:0]   facing     - 2 bits (N/E/S/W)
///   [12:2]  shape      - 11 bits (index into ShapeTable, 0 = cube)
///   [23:13] texture    - 11 bits (index into texture table)
///   [31:24] reserved   - 8 bits
///
/// A value of 0 means empty (shape 0 = cube, but facing 0 + shape 0 + texture 0
/// is treated as empty via the `is_empty` check).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Voxel(pub u32);

impl Voxel {
    pub const EMPTY: Voxel = Voxel(0);

    /// Create a voxel with the given shape, facing, and texture.
    pub fn new(shape: u16, facing: Facing, texture: u16) -> Self {
        let mut bits = 0u32;
        bits |= (facing as u32) & 0x3;
        bits |= ((shape as u32) & 0x7FF) << 2;
        bits |= ((texture as u32) & 0x7FF) << 13;
        Voxel(bits)
    }

    /// Create a filled cube voxel (shape 0) with default facing and texture 1.
    /// Texture 1 so that it's distinguishable from EMPTY.
    pub fn filled() -> Self {
        Self::new(0, Facing::North, 1)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_filled(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub fn facing(self) -> Facing {
        Facing::from_bits((self.0 & 0x3) as u8)
    }

    #[inline]
    pub fn shape_index(self) -> u16 {
        ((self.0 >> 2) & 0x7FF) as u16
    }

    #[inline]
    pub fn texture_index(self) -> u16 {
        ((self.0 >> 13) & 0x7FF) as u16
    }

    #[inline]
    pub fn reserved(self) -> u8 {
        ((self.0 >> 24) & 0xFF) as u8
    }
}

impl std::fmt::Debug for Voxel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "Voxel(empty)")
        } else {
            write!(
                f,
                "Voxel(shape={}, facing={:?}, tex={})",
                self.shape_index(),
                self.facing(),
                self.texture_index()
            )
        }
    }
}

/// Storage for voxel data within a chunk.
#[derive(Clone)]
pub struct ChunkData {
    voxels: Box<[Voxel; CHUNK_VOLUME]>,
}

impl Default for ChunkData {
    fn default() -> Self {
        Self {
            voxels: Box::new([Voxel::EMPTY; CHUNK_VOLUME]),
        }
    }
}

impl ChunkData {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        y * CHUNK_X * CHUNK_Z + z * CHUNK_X + x
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.voxels[Self::index(x, y, z)]
        } else {
            Voxel::EMPTY
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.voxels[Self::index(x, y, z)] = voxel;
        }
    }

    /// Convenience: set a voxel to a filled cube or empty.
    pub fn set_filled(&mut self, x: usize, y: usize, z: usize, filled: bool) {
        self.set(
            x,
            y,
            z,
            if filled { Voxel::filled() } else { Voxel::EMPTY },
        );
    }

    pub fn is_filled(&self, x: usize, y: usize, z: usize) -> bool {
        self.get(x, y, z).is_filled()
    }

    /// Check if a neighbor position is filled. Out-of-bounds is treated as empty.
    pub fn is_neighbor_filled(&self, x: i32, y: i32, z: i32) -> bool {
        if x < 0 || y < 0 || z < 0 {
            return false;
        }
        self.is_filled(x as usize, y as usize, z as usize)
    }

    /// Get voxel at signed coordinates. Out-of-bounds returns EMPTY.
    pub fn get_signed(&self, x: i32, y: i32, z: i32) -> Voxel {
        if x < 0 || y < 0 || z < 0 {
            return Voxel::EMPTY;
        }
        self.get(x as usize, y as usize, z as usize)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkState {
    #[default]
    Dirty,
    Meshing,
    Ready,
}

/// ECS component representing a voxel chunk.
#[derive(Component)]
pub struct Chunk {
    pub data: ChunkData,
    pub state: ChunkState,
    pub world_aligned: bool,
    pub dynamic: bool,
}

impl Chunk {
    pub fn new(data: ChunkData) -> Self {
        Self {
            data,
            state: ChunkState::Dirty,
            world_aligned: true,
            dynamic: false,
        }
    }

    /// Mark the chunk as needing a new mesh.
    pub fn mark_dirty(&mut self) {
        self.state = ChunkState::Dirty;
    }
}
