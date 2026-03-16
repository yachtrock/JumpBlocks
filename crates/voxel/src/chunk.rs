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

// ---------------------------------------------------------------------------
// Chunk neighbors
// ---------------------------------------------------------------------------

/// All 26 neighbor directions of a chunk in a 3×3×3 grid (excluding center).
///
/// Each neighbor is identified by its offset `(dx, dy, dz)` where each component
/// is –1, 0, or +1. Stored as a flat array of 26 `Option<ChunkData>` slots.
///
/// `None` means the neighbor chunk doesn't exist (treated as all air).
#[derive(Clone)]
pub struct ChunkNeighbors {
    slots: [Option<ChunkData>; 26],
}

impl Default for ChunkNeighbors {
    fn default() -> Self {
        Self::empty()
    }
}

impl ChunkNeighbors {
    pub fn empty() -> Self {
        // const initializer not available for Option<ChunkData> (Box inside), so build at runtime
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }

    /// Convert a `(dx, dy, dz)` offset (each –1/0/+1, not all zero) to a flat index 0..25.
    #[inline]
    fn offset_to_index(dx: i32, dy: i32, dz: i32) -> usize {
        debug_assert!((-1..=1).contains(&dx) && (-1..=1).contains(&dy) && (-1..=1).contains(&dz));
        debug_assert!(dx != 0 || dy != 0 || dz != 0, "center is not a neighbor");
        let raw = ((dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) as usize;
        // raw 0..26, center = 13 → skip it
        if raw < 13 { raw } else { raw - 1 }
    }

    pub fn get(&self, dx: i32, dy: i32, dz: i32) -> Option<&ChunkData> {
        self.slots[Self::offset_to_index(dx, dy, dz)].as_ref()
    }

    pub fn set(&mut self, dx: i32, dy: i32, dz: i32, data: ChunkData) {
        self.slots[Self::offset_to_index(dx, dy, dz)] = Some(data);
    }
}

/// Look up a voxel at signed coordinates that may extend into neighbor chunks.
///
/// Coordinates are relative to the *current* chunk. If they fall outside
/// `[0..CHUNK_X) × [0..CHUNK_Y) × [0..CHUNK_Z)`, the appropriate neighbor is
/// consulted. Missing neighbors (or coords more than one chunk away) return
/// `Voxel::EMPTY`.
pub fn get_voxel(data: &ChunkData, neighbors: &ChunkNeighbors, x: i32, y: i32, z: i32) -> Voxel {
    let in_x = x >= 0 && x < CHUNK_X as i32;
    let in_y = y >= 0 && y < CHUNK_Y as i32;
    let in_z = z >= 0 && z < CHUNK_Z as i32;

    if in_x && in_y && in_z {
        return data.get(x as usize, y as usize, z as usize);
    }

    // Which neighbor chunk?
    let dx = if x < 0 { -1 } else if x >= CHUNK_X as i32 { 1 } else { 0 };
    let dy = if y < 0 { -1 } else if y >= CHUNK_Y as i32 { 1 } else { 0 };
    let dz = if z < 0 { -1 } else if z >= CHUNK_Z as i32 { 1 } else { 0 };

    // More than one chunk away → air
    let local_x = x - dx * CHUNK_X as i32;
    let local_y = y - dy * CHUNK_Y as i32;
    let local_z = z - dz * CHUNK_Z as i32;
    if local_x < 0 || local_x >= CHUNK_X as i32
        || local_y < 0 || local_y >= CHUNK_Y as i32
        || local_z < 0 || local_z >= CHUNK_Z as i32
    {
        return Voxel::EMPTY;
    }

    match neighbors.get(dx, dy, dz) {
        Some(neighbor) => neighbor.get(local_x as usize, local_y as usize, local_z as usize),
        None => Voxel::EMPTY,
    }
}

// ---------------------------------------------------------------------------
// ECS component
// ---------------------------------------------------------------------------

/// ECS component representing a voxel chunk.
#[derive(Component)]
pub struct Chunk {
    pub data: ChunkData,
    pub neighbors: ChunkNeighbors,
    pub state: ChunkState,
    pub world_aligned: bool,
    pub dynamic: bool,
}

impl Chunk {
    pub fn new(data: ChunkData) -> Self {
        Self {
            data,
            neighbors: ChunkNeighbors::empty(),
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
