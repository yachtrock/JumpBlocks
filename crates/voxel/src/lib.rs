pub mod chunk;
pub mod chunk_lod;
pub mod coords;
pub mod cut_offset_chamfer;
pub mod meshing;
pub mod persistence;
pub mod region;
pub mod shape;
pub mod streaming;
pub mod world_grid;
pub mod worldgen;

#[cfg(test)]
mod tests;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use chunk::*;
use chunk_lod::{ChunkLodMesh, LodConfig};
use coords::{ChunkCoord, CHUNK_WORLD_SIZE};
use meshing::*;
use shape::*;
use streaming::StreamingAnchor;

/// Which presentation algorithm to use for the full-res mesh.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PresentationMode {
    /// LOD mesh — no chamfer, no shared verts.
    Flat,
    /// Cut-and-offset chamfer: insert edge loops then offset original verts.
    #[default]
    CutAndOffset,
}

impl PresentationMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::Flat => Self::CutAndOffset,
            Self::CutAndOffset => Self::Flat,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Flat => "Flat (no chamfer)",
            Self::CutAndOffset => "Cut & Offset Chamfer",
        }
    }
}

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShapeTable>()
            .init_resource::<PresentationMode>()
            .add_systems(Update, (
                promote_loaded_chunks,
                start_chunk_meshing.after(promote_loaded_chunks),
                handle_chunk_mesh_results,
                cycle_presentation_mode,
            ));
    }
}

/// Attached to a chunk entity while its mesh is being generated.
#[derive(Component)]
struct ChunkMeshTask(Task<ChunkMeshResult>);

/// Tracks what mesh level has been generated for a chunk.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkMeshLevel {
    /// No mesh generated yet.
    #[default]
    None,
    /// Only LOD mesh generated (fast, no chamfer).
    LodOnly,
    /// Both full-res and LOD meshes generated.
    Full,
}

/// Marker for debug overlay child entities (so we can despawn them on remesh).
#[derive(Component)]
struct DebugOverlay;

/// System that promotes Loaded chunks to Dirty once all their neighbors are
/// either absent (None) or have data loaded. Currently neighbors are stored as
/// `Option<ChunkData>` so they're always ready if present — this gate will
/// become meaningful once neighbors are entity references with their own state.
fn promote_loaded_chunks(
    mut query: Query<&mut Chunk>,
) {
    for mut chunk in query.iter_mut() {
        if chunk.state == ChunkState::Loaded {
            chunk.state = ChunkState::Dirty;
        }
    }
}

/// System that kicks off background mesh generation for dirty chunks.
/// Decides mesh level based on distance: close chunks get Full (chamfer),
/// distant chunks get LodOnly (fast). Also upgrades LodOnly→Full when
/// a chunk enters close range.
fn start_chunk_meshing(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Chunk, &ChunkCoord, Option<&ChunkMeshLevel>), Without<ChunkMeshTask>>,
    shape_table: Res<ShapeTable>,
    presentation: Res<PresentationMode>,
    lod_config: Res<LodConfig>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
) {
    let pool = AsyncComputeTaskPool::get();
    let mode = *presentation;

    let anchor_pos = anchor_query
        .single()
        .map(|t| t.translation())
        .unwrap_or(Vec3::ZERO);

    for (entity, mut chunk, coord, mesh_level) in query.iter_mut() {
        let current_level = mesh_level.copied().unwrap_or(ChunkMeshLevel::None);

        // Compute distance to decide mesh level
        let chunk_center = coord.pos.to_world_offset() + Vec3::splat(CHUNK_WORLD_SIZE * 0.5);
        let dist = ((anchor_pos - chunk_center) / CHUNK_WORLD_SIZE).abs();
        let max_dist = dist.x.max(dist.y).max(dist.z) as i32;
        let needs_full = max_dist <= lod_config.full_radius + 1; // +1 for transition margin

        let desired_level = if needs_full {
            MeshLevel::Full
        } else {
            MeshLevel::LodOnly
        };

        // Decide whether to (re)mesh
        let should_mesh = if chunk.state == ChunkState::Dirty {
            true
        } else if current_level == ChunkMeshLevel::LodOnly && desired_level == MeshLevel::Full {
            // Upgrade: chunk is close now but only has LOD mesh
            true
        } else {
            false
        };

        if !should_mesh {
            continue;
        }

        chunk.state = ChunkState::Meshing;
        let data = chunk.data.clone();
        let neighbors = chunk.neighbors.clone();
        let shapes = shape_table.clone();

        let task = pool.spawn(async move {
            generate_chunk_mesh_at_level(&data, &neighbors, &shapes, mode, desired_level)
        });

        commands.entity(entity).insert(ChunkMeshTask(task));
    }
}

/// System that polls completed mesh tasks and inserts the mesh + collider.
fn handle_chunk_mesh_results(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Chunk, &mut ChunkMeshTask)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    debug_overlays: Query<Entity, With<DebugOverlay>>,
) {
    for (entity, mut chunk, mut task) in query.iter_mut() {
        let Some(result) = block_on(poll_once(&mut task.0)) else {
            continue;
        };

        // Despawn old debug overlays that are children of this chunk
        for dbg_entity in debug_overlays.iter() {
            commands.entity(dbg_entity).try_despawn();
        }

        // Skip if LOD mesh is empty (no filled voxels)
        if result.lod.positions.is_empty() {
            commands.entity(entity)
                .insert(ChunkMeshLevel::None)
                .remove::<ChunkMeshTask>();
            chunk.state = ChunkState::Ready;
            continue;
        }

        let lod_mesh = build_lod_mesh(&result.lod);
        let lod_handle = meshes.add(lod_mesh);

        let full_res_handle = result.full_res.as_ref().map(|fr| {
            meshes.add(build_full_res_mesh(fr))
        });

        let collider = Collider::trimesh(
            result.collider_data.vertices,
            result.collider_data.indices,
        );

        // Use full-res mesh if available, otherwise LOD mesh for display
        let display_mesh = full_res_handle.clone().unwrap_or_else(|| lod_handle.clone());

        let mesh_level = match result.level {
            MeshLevel::Full => ChunkMeshLevel::Full,
            MeshLevel::LodOnly => ChunkMeshLevel::LodOnly,
        };

        commands
            .entity(entity)
            .insert((
                Mesh3d(display_mesh),
                collider,
                RigidBody::Static,
                ChunkLodMesh {
                    full_res: full_res_handle,
                    lod: Some(lod_handle),
                },
                mesh_level,
            ))
            .remove::<ChunkMeshTask>();

        // Spawn debug overlay as child entity with red material
        if let Some(debug_data) = result.debug_overlay {
            if !debug_data.positions.is_empty() {
                let dbg_mesh = build_lod_mesh(&debug_data);
                let dbg_mesh_handle = meshes.add(dbg_mesh);
                let dbg_material = materials.add(StandardMaterial {
                    base_color: Color::srgba(1.0, 0.0, 0.0, 0.8),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    ..default()
                });
                commands.entity(entity).with_child((
                    DebugOverlay,
                    Mesh3d(dbg_mesh_handle),
                    MeshMaterial3d(dbg_material),
                ));
            }
        }

        chunk.state = ChunkState::Ready;
    }
}

/// F3 cycles the presentation mode and marks all chunks dirty for regeneration.
fn cycle_presentation_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut presentation: ResMut<PresentationMode>,
    mut chunks: Query<&mut Chunk>,
) {
    if keyboard.just_pressed(KeyCode::F3) {
        let old = *presentation;
        *presentation = old.cycle();
        info!("Presentation mode: {} → {}", old.label(), presentation.label());

        // Mark all chunks dirty to regenerate with new mode
        for mut chunk in chunks.iter_mut() {
            chunk.mark_dirty();
        }
    }
}
