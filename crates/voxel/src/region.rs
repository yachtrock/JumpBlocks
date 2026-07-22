use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;

use crate::chunk::ChunkData;
use crate::coords::*;

// ---------------------------------------------------------------------------
// Dirty tracking
// ---------------------------------------------------------------------------

/// Monotonically increasing generation counter.
/// Bumped every time chunk data changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Generation(pub u64);

impl Generation {
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Tracks what has been modified and what representations are stale.
#[derive(Clone, Debug)]
pub struct ChunkDirtyState {
    /// Current data generation (bumped on every block change).
    pub data_gen: Generation,
    /// Generation at which the chunk was last saved to disk.
    pub saved_gen: Generation,
    /// Generation at which the full-res mesh was last built.
    pub mesh_gen: Generation,
    /// Generation at which the LOD mesh was last built.
    pub lod_gen: Generation,
}

impl Default for ChunkDirtyState {
    fn default() -> Self {
        Self {
            data_gen: Generation(1), // starts dirty (never saved/meshed)
            saved_gen: Generation(0),
            mesh_gen: Generation(0),
            lod_gen: Generation(0),
        }
    }
}

impl ChunkDirtyState {
    /// Whether the chunk has unsaved changes.
    pub fn needs_save(&self) -> bool {
        self.saved_gen < self.data_gen
    }

    /// Whether the full-res mesh is stale.
    pub fn needs_mesh(&self) -> bool {
        self.mesh_gen < self.data_gen
    }

    /// Whether the LOD mesh is stale.
    pub fn needs_lod(&self) -> bool {
        self.lod_gen < self.data_gen
    }

    /// Bump data generation (call after any block modification).
    pub fn mark_data_changed(&mut self) {
        self.data_gen = self.data_gen.next();
    }

    pub fn mark_saved(&mut self) {
        self.saved_gen = self.data_gen;
    }

    pub fn mark_meshed(&mut self) {
        self.mesh_gen = self.data_gen;
    }

    pub fn mark_lod_built(&mut self) {
        self.lod_gen = self.data_gen;
    }
}

// ---------------------------------------------------------------------------
// ChunkSlot — what we store per chunk in the region
// ---------------------------------------------------------------------------

/// A chunk's data and metadata within a region.
#[derive(Clone, Debug)]
pub struct ChunkSlot {
    /// The voxel data. Wrapped in Arc so neighbors can share references
    /// without cloning the full 32K cell array.
    pub data: Arc<ChunkData>,

    /// Dirty / generation tracking.
    pub dirty: ChunkDirtyState,

    /// Entity handle if this chunk is currently spawned in the ECS world.
    pub entity: Option<Entity>,
}

impl ChunkSlot {
    pub fn new(data: ChunkData) -> Self {
        Self {
            data: Arc::new(data),
            dirty: ChunkDirtyState::default(),
            entity: None,
        }
    }

    /// Replace the chunk data (e.g. after a block placement).
    /// Bumps the data generation and returns a new Arc.
    pub fn update_data(&mut self, data: ChunkData) {
        self.data = Arc::new(data);
        self.dirty.mark_data_changed();
    }
}

// ---------------------------------------------------------------------------
// Region
// ---------------------------------------------------------------------------

/// Region-level dirty state for impostor invalidation.
#[derive(Clone, Debug)]
pub struct RegionDirtyState {
    /// Bumped whenever any chunk in the region changes.
    pub data_gen: Generation,
    /// Generation at which the region impostor was last built.
    pub impostor_gen: Generation,
}

impl Default for RegionDirtyState {
    fn default() -> Self {
        Self {
            data_gen: Generation(1),
            impostor_gen: Generation(0),
        }
    }
}

impl RegionDirtyState {
    pub fn needs_impostor(&self) -> bool {
        self.impostor_gen < self.data_gen
    }

    pub fn mark_chunk_changed(&mut self) {
        self.data_gen = self.data_gen.next();
    }

    pub fn mark_impostor_built(&mut self) {
        self.impostor_gen = self.data_gen;
    }
}

/// A region (island) — a bounded volume of chunks.
///
/// XZ: `0..REGION_XZ` (256 chunks).
/// Y: `CHUNK_Y_MIN..=CHUNK_Y_MAX` (-128..=127).
///
/// Chunks are stored sparsely — only populated chunks exist in the map.
pub struct Region {
    pub id: RegionId,

    /// World-space origin of this region. Chunk (0, 0, 0) sits at this position.
    pub world_origin: Vec3,

    /// Sparse chunk storage keyed by chunk position.
    chunks: HashMap<ChunkPos, ChunkSlot>,

    /// Region-level dirty tracking (for impostor invalidation).
    pub dirty: RegionDirtyState,
}

impl Region {
    pub fn new(id: RegionId, world_origin: Vec3) -> Self {
        Self {
            id,
            world_origin,
            chunks: HashMap::new(),
            dirty: RegionDirtyState::default(),
        }
    }

    /// Number of chunks currently stored.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get a chunk slot by position.
    pub fn get_chunk(&self, pos: ChunkPos) -> Option<&ChunkSlot> {
        self.chunks.get(&pos)
    }

    /// Get a mutable chunk slot by position.
    pub fn get_chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut ChunkSlot> {
        self.chunks.get_mut(&pos)
    }

    /// Get chunk data as an Arc (for neighbor sharing).
    pub fn get_chunk_data(&self, pos: ChunkPos) -> Option<Arc<ChunkData>> {
        self.chunks.get(&pos).map(|slot| Arc::clone(&slot.data))
    }

    /// Insert or replace a chunk at the given position.
    pub fn set_chunk(&mut self, pos: ChunkPos, data: ChunkData) {
        debug_assert!(pos.in_bounds(), "ChunkPos out of region bounds: {}", pos);
        let slot = ChunkSlot::new(data);
        self.chunks.insert(pos, slot);
        self.dirty.mark_chunk_changed();
    }

    /// Remove a chunk from the region. Returns the slot if it existed.
    pub fn remove_chunk(&mut self, pos: ChunkPos) -> Option<ChunkSlot> {
        self.chunks.remove(&pos)
    }

    /// Check if a chunk exists at the given position.
    pub fn has_chunk(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// Iterate over all chunk positions and slots.
    pub fn iter_chunks(&self) -> impl Iterator<Item = (ChunkPos, &ChunkSlot)> {
        self.chunks.iter().map(|(&pos, slot)| (pos, slot))
    }

    /// Iterate mutably over all chunk slots.
    pub fn iter_chunks_mut(&mut self) -> impl Iterator<Item = (ChunkPos, &mut ChunkSlot)> {
        self.chunks.iter_mut().map(|(&pos, slot)| (pos, slot))
    }

    /// Get all chunks that need saving.
    pub fn dirty_chunks(&self) -> impl Iterator<Item = (ChunkPos, &ChunkSlot)> {
        self.chunks
            .iter()
            .filter(|(_, slot)| slot.dirty.needs_save())
            .map(|(&pos, slot)| (pos, slot))
    }

    /// Convert a world-space position to this region's local ChunkPos.
    pub fn world_to_chunk(&self, world_pos: Vec3) -> ChunkPos {
        let local = world_pos - self.world_origin;
        ChunkPos::from_world(local)
    }

    /// Convert a ChunkPos to world-space origin.
    pub fn chunk_to_world(&self, pos: ChunkPos) -> Vec3 {
        self.world_origin + pos.to_world_offset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_insert_and_retrieve() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let pos = ChunkPos::new(10, 0, 20);
        let data = ChunkData::new();
        region.set_chunk(pos, data);

        assert!(region.has_chunk(pos));
        assert_eq!(region.chunk_count(), 1);
        assert!(region.get_chunk(pos).is_some());
    }

    #[test]
    fn region_remove() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let pos = ChunkPos::new(5, -3, 5);
        region.set_chunk(pos, ChunkData::new());
        assert!(region.has_chunk(pos));

        let removed = region.remove_chunk(pos);
        assert!(removed.is_some());
        assert!(!region.has_chunk(pos));
    }

    #[test]
    fn dirty_tracking() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let pos = ChunkPos::new(0, 0, 0);
        region.set_chunk(pos, ChunkData::new());

        let slot = region.get_chunk(pos).unwrap();
        assert!(slot.dirty.needs_save());
        assert!(slot.dirty.needs_mesh());

        let slot = region.get_chunk_mut(pos).unwrap();
        slot.dirty.mark_saved();
        assert!(!slot.dirty.needs_save());
        assert!(slot.dirty.needs_mesh()); // still needs mesh

        let slot = region.get_chunk_mut(pos).unwrap();
        slot.dirty.mark_meshed();
        assert!(!slot.dirty.needs_mesh());
    }

    #[test]
    fn region_dirty_state() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        assert!(region.dirty.needs_impostor());

        region.dirty.mark_impostor_built();
        assert!(!region.dirty.needs_impostor());

        // Inserting a chunk bumps the region generation
        region.set_chunk(ChunkPos::new(0, 0, 0), ChunkData::new());
        assert!(region.dirty.needs_impostor());
    }

    #[test]
    fn arc_chunk_data_sharing() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let pos = ChunkPos::new(1, 0, 1);
        region.set_chunk(pos, ChunkData::new());

        let arc1 = region.get_chunk_data(pos).unwrap();
        let arc2 = region.get_chunk_data(pos).unwrap();
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }

    #[test]
    fn world_to_chunk_conversion() {
        let region = Region::new(RegionId(0), Vec3::new(100.0, 0.0, 200.0));
        // Point at region origin → chunk (0,0,0)
        let pos = region.world_to_chunk(Vec3::new(100.0, 0.0, 200.0));
        assert_eq!(pos, ChunkPos::new(0, 0, 0));

        // Point 32 world units into the region in X → chunk (2,0,0) since CHUNK_WORLD_SIZE=16
        let pos = region.world_to_chunk(Vec3::new(132.0, 0.0, 200.0));
        assert_eq!(pos, ChunkPos::new(2, 0, 0));
    }
}
