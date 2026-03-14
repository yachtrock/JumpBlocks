use bevy::prelude::*;

pub const CHUNK_X: usize = 16;
pub const CHUNK_Y: usize = 32; // voxel count vertically (half-height, so 16 world units)
pub const CHUNK_Z: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;

pub const VOXEL_WIDTH: f32 = 1.0;
pub const VOXEL_HEIGHT: f32 = 0.5;

/// A single voxel in a chunk.
#[derive(Clone, Copy, Default)]
pub struct Voxel {
    pub filled: bool,
}

/// Storage for voxel data within a chunk.
#[derive(Clone)]
pub struct ChunkData {
    voxels: Box<[Voxel; CHUNK_VOLUME]>,
}

impl Default for ChunkData {
    fn default() -> Self {
        Self {
            voxels: Box::new([Voxel::default(); CHUNK_VOLUME]),
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

    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&Voxel> {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            Some(&self.voxels[Self::index(x, y, z)])
        } else {
            None
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.voxels[Self::index(x, y, z)] = voxel;
        }
    }

    pub fn set_filled(&mut self, x: usize, y: usize, z: usize, filled: bool) {
        self.set(x, y, z, Voxel { filled });
    }

    pub fn is_filled(&self, x: usize, y: usize, z: usize) -> bool {
        self.get(x, y, z).is_some_and(|v| v.filled)
    }

    /// Check if a neighbor position is filled. Out-of-bounds is treated as empty.
    pub fn is_neighbor_filled(&self, x: i32, y: i32, z: i32) -> bool {
        if x < 0 || y < 0 || z < 0 {
            return false;
        }
        self.is_filled(x as usize, y as usize, z as usize)
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
