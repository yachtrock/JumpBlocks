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

#[cfg(test)]
mod tests;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use chunk::*;
use chunk_lod::{ChunkLodMesh, LodTier};
use meshing::*;
use shape::*;

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
fn start_chunk_meshing(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Chunk), Without<ChunkMeshTask>>,
    shape_table: Res<ShapeTable>,
    presentation: Res<PresentationMode>,
) {
    let pool = AsyncComputeTaskPool::get();
    let mode = *presentation;

    for (entity, mut chunk) in query.iter_mut() {
        if chunk.state != ChunkState::Dirty {
            continue;
        }

        chunk.state = ChunkState::Meshing;
        let data = chunk.data.clone();
        let neighbors = chunk.neighbors.clone();
        let shapes = shape_table.clone();

        let task = pool.spawn(async move { generate_chunk_mesh(&data, &neighbors, &shapes, mode) });

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

        // Skip if mesh is empty (no filled voxels)
        if result.full_res.positions.is_empty() {
            commands.entity(entity).remove::<ChunkMeshTask>();
            chunk.state = ChunkState::Ready;
            continue;
        }

        let full_res_mesh = build_full_res_mesh(&result.full_res);
        let full_res_handle = meshes.add(full_res_mesh);

        let lod_mesh = build_lod_mesh(&result.lod);
        let lod_handle = meshes.add(lod_mesh);

        let collider = Collider::trimesh(
            result.collider_data.vertices,
            result.collider_data.indices,
        );

        commands
            .entity(entity)
            .insert((
                Mesh3d(full_res_handle.clone()),
                collider,
                RigidBody::Static,
                ChunkLodMesh {
                    full_res: Some(full_res_handle),
                    lod: Some(lod_handle),
                },
                LodTier::Full,
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
