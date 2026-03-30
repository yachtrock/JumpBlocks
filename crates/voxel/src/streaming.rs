//! Chunk streaming: loads/unloads chunks around the camera and auto-wires neighbors.
//!
//! The streaming system works per-frame:
//! 1. Determine which chunks should be loaded based on camera position
//! 2. Load new chunks (from region data or disk) up to a per-frame budget
//! 3. Unload distant chunks (despawn entities, optionally save if dirty)
//! 4. Auto-wire neighbor references for newly loaded chunks
//! 5. Write back building edits from ECS chunks to region storage

use std::collections::HashSet;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::camera::primitives::Aabb;

use std::path::PathBuf;

use crate::chunk::{Chunk, ChunkNeighbors};
use crate::chunk_lod::{ChunkDitherMaterial, LodTier};
use crate::coords::{ChunkCoord, ChunkPos, CHUNK_WORLD_SIZE};
use crate::world_grid::WorldGrid;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Streaming configuration resource.
#[derive(Resource, Debug, Clone)]
pub struct StreamingConfig {
    /// Horizontal (XZ) radius in chunks around the camera to keep loaded.
    pub load_radius_xz: i32,
    /// Vertical (Y) radius in chunks around the camera to keep loaded.
    pub load_radius_y: i32,
    /// Extra radius beyond load_radius before chunks get unloaded (hysteresis).
    pub unload_padding: i32,
    /// Maximum chunks to load (spawn) per frame.
    pub load_budget: usize,
    /// Maximum chunks to unload (despawn) per frame.
    pub unload_budget: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            load_radius_xz: 32,
            load_radius_y: 8,
            unload_padding: 4,
            load_budget: 16,
            unload_budget: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Tag component for the entity whose position drives chunk streaming.
/// Typically placed on the player or the camera.
#[derive(Component)]
pub struct StreamingAnchor;

/// Resource holding the default dither material for streamed chunk entities.
/// The game should insert this during setup.
#[derive(Resource)]
pub struct ChunkMaterial(pub Handle<ChunkDitherMaterial>);

/// Resource holding the world save directory path.
/// If not present, chunks won't be saved/loaded from disk.
#[derive(Resource)]
pub struct WorldSavePath(pub PathBuf);

// ---------------------------------------------------------------------------
// Chunk streaming system
// ---------------------------------------------------------------------------

/// Determine which chunks should be loaded, spawn new ones, despawn distant ones.
pub fn chunk_streaming_system(
    mut commands: Commands,
    config: Res<StreamingConfig>,
    mut world_grid: ResMut<WorldGrid>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
    loaded_chunks: Query<(Entity, &ChunkCoord)>,
    chunk_material: Option<Res<ChunkMaterial>>,
) {
    let Ok(anchor_transform) = anchor_query.single() else {
        return;
    };
    let anchor_pos = anchor_transform.translation();

    let active_id = world_grid.active_region;

    let Some(region) = world_grid.get_region(active_id) else {
        return;
    };

    let anchor_chunk = region.world_to_chunk(anchor_pos);

    // Build the set of desired chunk positions
    let mut desired: HashSet<ChunkPos> = HashSet::new();
    for dx in -config.load_radius_xz..=config.load_radius_xz {
        for dz in -config.load_radius_xz..=config.load_radius_xz {
            for dy in -config.load_radius_y..=config.load_radius_y {
                let pos = ChunkPos::new(
                    anchor_chunk.x + dx,
                    anchor_chunk.y + dy,
                    anchor_chunk.z + dz,
                );
                if !pos.in_bounds() {
                    continue;
                }
                desired.insert(pos);
            }
        }
    }

    // Collect currently loaded chunk positions and entities for the active region
    let mut currently_loaded: HashSet<ChunkPos> = HashSet::new();
    let mut loaded_entities: Vec<(Entity, ChunkPos)> = Vec::new();
    for (entity, coord) in loaded_chunks.iter() {
        if coord.region == active_id {
            currently_loaded.insert(coord.pos);
            loaded_entities.push((entity, coord.pos));
        }
    }

    // --- Load new chunks (up to budget) ---
    let mut loaded_this_frame = 0;

    // Sort by distance to anchor for priority loading (closest first)
    let mut to_load: Vec<ChunkPos> = desired
        .iter()
        .filter(|pos| !currently_loaded.contains(pos) && region.has_chunk(**pos))
        .copied()
        .collect();
    to_load.sort_by_key(|pos| {
        let dx = pos.x - anchor_chunk.x;
        let dy = pos.y - anchor_chunk.y;
        let dz = pos.z - anchor_chunk.z;
        dx * dx + dy * dy + dz * dz
    });

    // Collect spawn list before taking mutable borrow
    let mut spawn_list: Vec<(ChunkPos, Vec3)> = Vec::new();
    for pos in to_load {
        if loaded_this_frame >= config.load_budget {
            break;
        }
        let world_pos = region.chunk_to_world(pos);
        spawn_list.push((pos, world_pos));
        loaded_this_frame += 1;
    }

    // Spawn chunk entities
    let region = world_grid.get_region_mut(active_id).unwrap();
    for (pos, world_pos) in &spawn_list {
        let Some(slot) = region.get_chunk(*pos) else {
            continue;
        };
        let chunk_data = (*slot.data).clone();

        // Pre-insert an Aabb so Bevy's frustum culling works immediately,
        // even before the mesh is generated. The AABB is in local space
        // (centered at half chunk size).
        let chunk_aabb = Aabb::from_min_max(
            Vec3::ZERO,
            Vec3::splat(CHUNK_WORLD_SIZE),
        );

        let mut entity_commands = commands.spawn((
            Chunk::new(chunk_data),
            ChunkCoord { region: active_id, pos: *pos },
            LodTier::default(),
            Transform::from_translation(*world_pos),
            chunk_aabb,
        ));

        if let Some(ref mat) = chunk_material {
            entity_commands.insert(MeshMaterial3d(mat.0.clone()));
        }

        let entity = entity_commands.id();

        if let Some(slot) = region.get_chunk_mut(*pos) {
            slot.entity = Some(entity);
        }
    }

    // --- Unload distant chunks (up to budget) ---
    let unload_radius_xz = config.load_radius_xz + config.unload_padding;
    let unload_radius_y = config.load_radius_y + config.unload_padding;
    let mut unloaded_this_frame = 0;

    for (entity, pos) in &loaded_entities {
        if unloaded_this_frame >= config.unload_budget {
            break;
        }
        let dx = (pos.x - anchor_chunk.x).abs();
        let dy = (pos.y - anchor_chunk.y).abs();
        let dz = (pos.z - anchor_chunk.z).abs();

        if dx > unload_radius_xz || dy > unload_radius_y || dz > unload_radius_xz {
            commands.entity(*entity).despawn();
            if let Some(slot) = region.get_chunk_mut(*pos) {
                slot.entity = None;
            }
            unloaded_this_frame += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Neighbor wiring system
// ---------------------------------------------------------------------------

/// Auto-wire neighbor references for loaded chunks.
/// Runs after streaming so newly loaded chunks get their neighbors.
pub fn neighbor_wiring_system(
    world_grid: Res<WorldGrid>,
    mut chunks: Query<(&ChunkCoord, &mut Chunk)>,
) {
    let active_id = world_grid.active_region;
    let Some(region) = world_grid.get_region(active_id) else {
        return;
    };

    for (coord, mut chunk) in chunks.iter_mut() {
        if coord.region != active_id {
            continue;
        }

        // Only wire neighbors for chunks in Loaded state (not yet promoted to Dirty)
        if chunk.state != crate::chunk::ChunkState::Loaded {
            continue;
        }

        let mut neighbors = ChunkNeighbors::empty();
        for (dx, dy, dz) in ChunkPos::neighbor_offsets() {
            if let Some(neighbor_pos) = coord.pos.neighbor(dx, dy, dz) {
                if let Some(arc_data) = region.get_chunk_data(neighbor_pos) {
                    neighbors.set_arc(dx, dy, dz, arc_data);
                }
            }
        }
        chunk.neighbors = neighbors;
    }
}

// ---------------------------------------------------------------------------
// Chunk writeback system
// ---------------------------------------------------------------------------

/// Syncs modified chunk data from ECS entities back to region storage.
/// This ensures building edits are persisted in the region's canonical data.
pub fn chunk_writeback_system(
    mut world_grid: ResMut<WorldGrid>,
    chunks: Query<(&ChunkCoord, &Chunk), Changed<Chunk>>,
) {
    let active_id = world_grid.active_region;

    for (coord, chunk) in chunks.iter() {
        if coord.region != active_id {
            continue;
        }

        // Only write back if the chunk has pending modifications
        if chunk.pending_modifications.is_empty() {
            continue;
        }

        if let Some(region) = world_grid.get_region_mut(active_id) {
            if let Some(slot) = region.get_chunk_mut(coord.pos) {
                slot.update_data(chunk.data.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Disk save system
// ---------------------------------------------------------------------------

/// Timer for throttling disk saves.
#[derive(Resource)]
pub struct SaveTimer(pub Timer);

impl Default for SaveTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(5.0, TimerMode::Repeating))
    }
}

/// Periodically saves dirty chunks to disk (every 5 seconds).
pub fn chunk_save_system(
    mut world_grid: ResMut<WorldGrid>,
    save_path: Option<Res<WorldSavePath>>,
    time: Res<Time>,
    mut timer: ResMut<SaveTimer>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }

    let Some(save_path) = save_path else {
        return;
    };

    let active_id = world_grid.active_region;
    let Some(region) = world_grid.get_region_mut(active_id) else {
        return;
    };

    match crate::persistence::save_dirty_chunks(&save_path.0, region) {
        Ok(0) => {}
        Ok(n) => {
            info!("Saved {} dirty chunks to disk", n);
        }
        Err(e) => {
            error!("Failed to save chunks: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Plugin that adds chunk streaming, neighbor wiring, writeback, and save systems.
pub struct StreamingPlugin;

impl Plugin for StreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StreamingConfig>()
            .init_resource::<SaveTimer>()
            .add_systems(Update, (
                chunk_streaming_system,
                neighbor_wiring_system.after(chunk_streaming_system),
                chunk_writeback_system,
                chunk_save_system.after(chunk_writeback_system),
            ));
    }
}
