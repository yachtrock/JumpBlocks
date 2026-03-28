//! LOD tier management: assigns mesh detail levels based on camera distance.
//!
//! Tiers:
//! - Full:    chamfered mesh + physics collider (near the player)
//! - Reduced: LOD mesh (no chamfer), no collider (mid distance)
//! - Hidden:  no mesh, no collider (will be replaced by impostor system later)

use bevy::prelude::*;

use crate::chunk::Chunk;
use crate::coords::{ChunkCoord, CHUNK_WORLD_SIZE};
use crate::streaming::StreamingAnchor;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The detail tier a chunk is currently rendered at.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LodTier {
    /// Full chamfered mesh + collider.
    #[default]
    Full,
    /// Simplified LOD mesh, no collider.
    Reduced,
    /// No mesh rendered (beyond LOD range, awaiting impostor).
    Hidden,
}

/// LOD configuration resource.
#[derive(Resource, Debug, Clone)]
pub struct LodConfig {
    /// Chunks within this distance (in chunk units) get full detail.
    pub full_radius: i32,
    /// Chunks within this distance get reduced LOD. Beyond this → hidden.
    pub reduced_radius: i32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            full_radius: 4,
            reduced_radius: 8,
        }
    }
}

/// Holds the LOD mesh handle generated during meshing, so we can swap to it.
#[derive(Component)]
pub struct ChunkLodMesh {
    pub full_res: Option<Handle<Mesh>>,
    pub lod: Option<Handle<Mesh>>,
}

/// Optional debug materials for visualizing LOD tiers.
/// If present, chunks change color based on their tier.
#[derive(Resource)]
pub struct LodDebugMaterials {
    pub full: Handle<StandardMaterial>,
    pub reduced: Handle<StandardMaterial>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Assigns LodTier to chunk entities based on distance to the streaming anchor.
pub fn lod_tier_assignment_system(
    config: Res<LodConfig>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
    mut chunks: Query<(&ChunkCoord, &mut LodTier)>,
) {
    let Ok(anchor_transform) = anchor_query.single() else {
        return;
    };
    let anchor_pos = anchor_transform.translation();

    for (coord, mut tier) in chunks.iter_mut() {
        let chunk_center = coord.pos.to_world_offset()
            + Vec3::splat(CHUNK_WORLD_SIZE * 0.5);

        let dist_chunks = ((anchor_pos - chunk_center) / CHUNK_WORLD_SIZE).abs();
        let max_dist = dist_chunks.x.max(dist_chunks.y).max(dist_chunks.z) as i32;

        let new_tier = if max_dist <= config.full_radius {
            LodTier::Full
        } else if max_dist <= config.reduced_radius {
            LodTier::Reduced
        } else {
            LodTier::Hidden
        };

        if *tier != new_tier {
            *tier = new_tier;
        }
    }
}

/// Swaps the visible mesh and optionally material based on the current LodTier.
/// Requires ChunkLodMesh to have been populated by the meshing pipeline.
pub fn lod_mesh_swap_system(
    mut commands: Commands,
    chunks: Query<(Entity, &LodTier, &ChunkLodMesh), Changed<LodTier>>,
    debug_mats: Option<Res<LodDebugMaterials>>,
) {
    for (entity, tier, lod_mesh) in chunks.iter() {
        match tier {
            LodTier::Full => {
                if let Some(ref handle) = lod_mesh.full_res {
                    commands.entity(entity).insert(Mesh3d(handle.clone()));
                }
                if let Some(ref mats) = debug_mats {
                    commands.entity(entity).insert(MeshMaterial3d(mats.full.clone()));
                }
            }
            LodTier::Reduced => {
                if let Some(ref handle) = lod_mesh.lod {
                    commands.entity(entity).insert(Mesh3d(handle.clone()));
                }
                if let Some(ref mats) = debug_mats {
                    commands.entity(entity).insert(MeshMaterial3d(mats.reduced.clone()));
                }
            }
            LodTier::Hidden => {
                commands.entity(entity).remove::<Mesh3d>();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct LodPlugin;

impl Plugin for LodPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodConfig>()
            .add_systems(Update, (
                lod_tier_assignment_system,
                lod_mesh_swap_system.after(lod_tier_assignment_system),
            ));
    }
}
