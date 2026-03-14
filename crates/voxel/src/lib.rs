pub mod chunk;
pub mod meshing;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use chunk::*;
use meshing::*;

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (start_chunk_meshing, handle_chunk_mesh_results));
    }
}

/// Attached to a chunk entity while its mesh is being generated.
#[derive(Component)]
struct ChunkMeshTask(Task<ChunkMeshResult>);

/// System that kicks off background mesh generation for dirty chunks.
fn start_chunk_meshing(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Chunk), Without<ChunkMeshTask>>,
) {
    let pool = AsyncComputeTaskPool::get();

    for (entity, mut chunk) in query.iter_mut() {
        if chunk.state != ChunkState::Dirty {
            continue;
        }

        chunk.state = ChunkState::Meshing;
        let data = chunk.data.clone();

        let task = pool.spawn(async move { generate_chunk_mesh(&data) });

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
        if result.mesh_data.positions.is_empty() {
            commands.entity(entity).remove::<ChunkMeshTask>();
            chunk.state = ChunkState::Ready;
            continue;
        }

        let mesh = build_mesh(&result.mesh_data);
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
