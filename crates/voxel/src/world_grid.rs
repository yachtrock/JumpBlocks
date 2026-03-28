use std::collections::HashMap;

use bevy::prelude::*;

use crate::coords::*;
use crate::region::Region;

/// The top-level world resource. Holds all regions and global state.
#[derive(Resource)]
pub struct WorldGrid {
    /// All regions, keyed by RegionId.
    regions: HashMap<RegionId, Region>,

    /// Which region the player is currently in.
    pub active_region: RegionId,

    /// Next region ID to allocate.
    next_id: u32,
}

impl WorldGrid {
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
            active_region: RegionId(0),
            next_id: 0,
        }
    }

    /// Create a new region at the given world origin. Returns its ID.
    pub fn create_region(&mut self, world_origin: Vec3) -> RegionId {
        let id = RegionId(self.next_id);
        self.next_id += 1;
        let region = Region::new(id, world_origin);
        self.regions.insert(id, region);
        id
    }

    /// Get a region by ID.
    pub fn get_region(&self, id: RegionId) -> Option<&Region> {
        self.regions.get(&id)
    }

    /// Get a mutable region by ID.
    pub fn get_region_mut(&mut self, id: RegionId) -> Option<&mut Region> {
        self.regions.get_mut(&id)
    }

    /// Get the currently active region.
    pub fn active_region(&self) -> Option<&Region> {
        self.regions.get(&self.active_region)
    }

    /// Get the currently active region mutably.
    pub fn active_region_mut(&mut self) -> Option<&mut Region> {
        self.regions.get_mut(&self.active_region)
    }

    /// Iterate over all regions.
    pub fn iter_regions(&self) -> impl Iterator<Item = (RegionId, &Region)> {
        self.regions.iter().map(|(&id, region)| (id, region))
    }

    /// Number of regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Find which region a world-space point falls within, if any.
    /// Checks if the point is within the XZ footprint of each region.
    pub fn region_at_world_pos(&self, world_pos: Vec3) -> Option<RegionId> {
        for (&id, region) in &self.regions {
            let local = world_pos - region.world_origin;
            let extent = REGION_XZ as f32 * CHUNK_WORLD_SIZE;
            if local.x >= 0.0
                && local.x < extent
                && local.z >= 0.0
                && local.z < extent
                && local.y >= CHUNK_Y_MIN as f32 * CHUNK_WORLD_SIZE
                && local.y < (CHUNK_Y_MAX + 1) as f32 * CHUNK_WORLD_SIZE
            {
                return Some(id);
            }
        }
        None
    }
}

impl Default for WorldGrid {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_retrieve_region() {
        let mut world = WorldGrid::new();
        let id = world.create_region(Vec3::ZERO);
        assert_eq!(id, RegionId(0));
        assert!(world.get_region(id).is_some());
        assert_eq!(world.region_count(), 1);
    }

    #[test]
    fn multiple_regions() {
        let mut world = WorldGrid::new();
        let id0 = world.create_region(Vec3::ZERO);
        let id1 = world.create_region(Vec3::new(10000.0, 0.0, 0.0));
        assert_ne!(id0, id1);
        assert_eq!(world.region_count(), 2);
    }

    #[test]
    fn region_at_world_pos_hit() {
        let mut world = WorldGrid::new();
        let id = world.create_region(Vec3::new(100.0, 0.0, 100.0));
        // Inside the region
        let found = world.region_at_world_pos(Vec3::new(110.0, 5.0, 110.0));
        assert_eq!(found, Some(id));
    }

    #[test]
    fn region_at_world_pos_miss() {
        let mut world = WorldGrid::new();
        world.create_region(Vec3::ZERO);
        // Way outside (negative)
        let found = world.region_at_world_pos(Vec3::new(-100.0, 0.0, 0.0));
        assert_eq!(found, None);
    }
}
