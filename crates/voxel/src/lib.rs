pub mod chunk;
pub mod halfedge_chamfer;
pub mod meshing;
pub mod shape;

#[cfg(test)]
mod tests;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use chunk::*;
use meshing::*;
use shape::*;

/// Which presentation algorithm to use for the full-res mesh.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PresentationMode {
    /// LOD mesh — no chamfer, no shared verts.
    Flat,
    /// Edge-graph chamfer post-process.
    #[default]
    EdgeGraphChamfer,
    /// Half-edge mesh chamfer using procedural_modelling crate.
    HalfEdgeChamfer,
}

impl PresentationMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::Flat => Self::EdgeGraphChamfer,
            Self::EdgeGraphChamfer => Self::HalfEdgeChamfer,
            Self::HalfEdgeChamfer => Self::Flat,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Flat => "Flat (no chamfer)",
            Self::EdgeGraphChamfer => "Edge-Graph Chamfer",
            Self::HalfEdgeChamfer => "Half-Edge Chamfer",
        }
    }
}

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShapeTable>()
            .init_resource::<PresentationMode>()
            .add_systems(Update, (start_chunk_meshing, handle_chunk_mesh_results, cycle_presentation_mode));
    }
}

/// Attached to a chunk entity while its mesh is being generated.
#[derive(Component)]
struct ChunkMeshTask(Task<ChunkMeshResult>);

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
        let shapes = shape_table.clone();

        let task = pool.spawn(async move { generate_chunk_mesh(&data, &shapes, mode) });

        commands.entity(entity).insert(ChunkMeshTask(task));
    }
}

/// System that polls completed mesh tasks and inserts the mesh + collider.
fn handle_chunk_mesh_results(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Chunk, &mut ChunkMeshTask)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for (entity, mut chunk, mut task) in query.iter_mut() {
        let Some(result) = block_on(poll_once(&mut task.0)) else {
            continue;
        };

        // Skip if mesh is empty (no filled voxels)
        if result.full_res.positions.is_empty() {
            commands.entity(entity).remove::<ChunkMeshTask>();
            chunk.state = ChunkState::Ready;
            continue;
        }

        let mesh = build_full_res_mesh(&result.full_res);
        let mesh_handle = meshes.add(mesh);

        let collider = Collider::trimesh(
            result.collider_data.vertices,
            result.collider_data.indices,
        );

        commands
            .entity(entity)
            .insert((Mesh3d(mesh_handle), collider, RigidBody::Static))
            .remove::<ChunkMeshTask>();

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
