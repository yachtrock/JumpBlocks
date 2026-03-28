use avian3d::prelude::*;
use bevy::prelude::*;
use jumpblocks_voxel::chunk::ChunkData;
use jumpblocks_voxel::chunk_lod::LodDebugMaterials;
use jumpblocks_voxel::coords::ChunkPos;
use jumpblocks_voxel::shape::{Facing, ShapeTable, SHAPE_CUBE};
use jumpblocks_voxel::streaming::ChunkMaterial;
use jumpblocks_voxel::world_grid::WorldGrid;

use crate::layers::GameLayer;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldGrid>()
            .add_systems(Startup, setup_world)
            .add_systems(Update, attach_chunk_collision_layers);
    }
}

/// Attaches CollisionLayers to newly spawned chunk entities that don't have them yet.
fn attach_chunk_collision_layers(
    mut commands: Commands,
    chunks: Query<Entity, (With<jumpblocks_voxel::chunk::Chunk>, Without<CollisionLayers>)>,
) {
    for entity in chunks.iter() {
        commands.entity(entity).insert(
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        );
    }
}

fn setup_world(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    mut world_grid: ResMut<WorldGrid>,
) {
    let has_rendering = meshes.is_some() && materials.is_some();

    // Ground plane
    let ground = commands.spawn((
        RigidBody::Static,
        Collider::half_space(Vec3::Y),
        CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
    ));
    let ground_entity = ground.id();

    let platforms = [
        Vec3::new(5.0, 1.0, 0.0),
        Vec3::new(10.0, 2.5, 3.0),
        Vec3::new(15.0, 4.0, -1.0),
        Vec3::new(12.0, 5.5, -6.0),
        Vec3::new(7.0, 7.0, -8.0),
    ];

    let platform_entities: Vec<(Entity, Vec3)> = platforms
        .iter()
        .map(|&pos| {
            let e = commands
                .spawn((
                    Transform::from_translation(pos),
                    RigidBody::Static,
                    Collider::cuboid(4.0, 0.5, 4.0),
                    CollisionLayers::new(
                        [GameLayer::Default, GameLayer::CameraBlocking],
                        LayerMask::ALL,
                    ),
                ))
                .id();
            (e, pos)
        })
        .collect();

    if let (Some(mut meshes), Some(mut materials)) = (meshes, materials) {
        commands.entity(ground_entity).insert((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(50.0)))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.6, 0.3),
                ..default()
            })),
        ));

        let platform_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.6),
            ..default()
        });
        let platform_mesh = meshes.add(Cuboid::new(4.0, 0.5, 4.0));

        for (entity, _pos) in &platform_entities {
            commands.entity(*entity).insert((
                Mesh3d(platform_mesh.clone()),
                MeshMaterial3d(platform_material.clone()),
            ));
        }

        // Directional light — pointing straight down for clean platformer shadows
        commands.spawn((
            DirectionalLight {
                illuminance: 8_000.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        ));

        // Low ambient — IBL on the camera provides the real ambient coloring
        commands.insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.9, 0.9, 1.0),
            brightness: 50.0,
            ..default()
        });

        // Default material for streamed chunks
        let chunk_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.5, 0.4),
            ..default()
        });
        commands.insert_resource(ChunkMaterial(chunk_mat.clone()));

        // Debug materials for LOD tier visualization
        commands.insert_resource(LodDebugMaterials {
            full: chunk_mat,
            reduced: materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.7, 0.9),
                ..default()
            }),
        });
    }

    // --- Populate the region with demo chunk data ---
    // The streaming system will handle spawning chunk entities.
    let region_id = world_grid.create_region(Vec3::new(-5.0, 0.0, 5.0));
    world_grid.active_region = region_id;

    let region = world_grid.get_region_mut(region_id).unwrap();

    // Chunk (0,0,0): Main demo chunk
    let chunk_data = build_main_demo_chunk();
    let shapes = ShapeTable::default();
    let errors = chunk_data.validate(&shapes);
    for e in &errors {
        error!("[world] main: {}", e);
    }
    region.set_chunk(ChunkPos::new(0, 0, 0), chunk_data);

    // Chunk (1,0,0): Neighbor chunk (+X direction)
    let neighbor_data = build_neighbor_demo_chunk();
    let errors = neighbor_data.validate(&shapes);
    for e in &errors {
        error!("[world] neighbor: {}", e);
    }
    region.set_chunk(ChunkPos::new(1, 0, 0), neighbor_data);

    // Chunk (2,0,0): Floating test blocks (offset into chunk 2 space)
    let test_data = build_test_demo_chunk();
    let errors = test_data.validate(&shapes);
    for e in &errors {
        error!("[world] test: {}", e);
    }
    region.set_chunk(ChunkPos::new(2, 0, 0), test_data);

    info!(
        "[world] Region {} created with {} chunks, streaming will handle spawning",
        region_id,
        region.chunk_count()
    );
}

// ---------------------------------------------------------------------------
// Demo chunk builders (extracted from old manual setup)
// ---------------------------------------------------------------------------

fn build_main_demo_chunk() -> ChunkData {
    let mut data = ChunkData::new();

    // Ground layer: a 10x10 block platform
    for bx in 0..10 {
        for bz in 0..10 {
            data.place_std(bx * 2, 0, bz * 2, SHAPE_CUBE, Facing::North, 1);
        }
    }

    // Staircase: 8 steps
    for step in 0..8usize {
        let y_top = 1 + step;
        for bx in 0..2 {
            // bx=0 when step>0 shares its column with bx=1 of the previous step,
            // which already filled up to y=(step). Only place the new top layer.
            let y_start = if bx == 0 && step > 0 { y_top } else { 1 };
            for bz in 0..3 {
                for y in y_start..=y_top {
                    data.place_std(bx * 2 + step * 2, y, bz * 2 + 6, SHAPE_CUBE, Facing::North, 1);
                }
            }
        }
    }

    // Tower at the top
    for by in 0..5 {
        for bx in 0..3 {
            for bz in 0..3 {
                data.place_std(16 + bx * 2, 9 + by, 6 + bz * 2, SHAPE_CUBE, Facing::North, 1);
            }
        }
    }

    // Wedge ramp alongside the stairs
    for step in 0..8usize {
        let wedge_y = 1 + step;
        for bz in 0..3 {
            data.place_wedge(step * 2, wedge_y, 14 + bz * 2, Facing::East, 1);
        }
        for y in 1..wedge_y {
            for bz in 0..3 {
                data.place_std(step * 2, y, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
            }
        }
    }

    // Wedges facing each direction
    data.place_wedge(0, 2, 20, Facing::North, 1);
    data.place_wedge(4, 2, 20, Facing::East, 1);
    data.place_wedge(8, 2, 20, Facing::South, 1);
    data.place_wedge(12, 2, 20, Facing::West, 1);

    // Bridge platform reaching the +X boundary
    for bx in 0..6 {
        for bz in 0..3 {
            data.place_std(20 + bx * 2, 0, 6 + bz * 2, SHAPE_CUBE, Facing::North, 1);
        }
    }
    for by in 0..2 {
        data.place_std(30, 1 + by, 8, SHAPE_CUBE, Facing::North, 1);
    }
    data.place_wedge(30, 3, 8, Facing::West, 1);
    for bz in 0..3 {
        data.place_std(30, 0, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
    }

    data
}

fn build_neighbor_demo_chunk() -> ChunkData {
    let mut data = ChunkData::new();

    // Continuation of the bridge platform
    for bx in 0..6 {
        for bz in 0..3 {
            data.place_std(bx * 2, 0, 6 + bz * 2, SHAPE_CUBE, Facing::North, 1);
        }
    }
    for by in 0..2 {
        data.place_std(0, 1 + by, 8, SHAPE_CUBE, Facing::North, 1);
    }
    data.place_wedge(0, 3, 8, Facing::East, 1);
    for bz in 0..3 {
        data.place_std(0, 0, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
    }

    // Small structure
    for bx in 0..3 {
        for bz in 0..3 {
            data.place_std(6 + bx * 2, 0, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
        }
    }
    for bz in 0..3 {
        data.place_wedge(6, 1, 14 + bz * 2, Facing::West, 1);
    }

    data
}

fn build_test_demo_chunk() -> ChunkData {
    let mut data = ChunkData::new();

    // Single cube
    data.place_std(4, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 4, 4, SHAPE_CUBE, Facing::North, 1);

    // Single wedge (each facing)
    data.place_wedge(4, 4, 10, Facing::North, 1);
    data.place_wedge(10, 4, 10, Facing::East, 1);
    data.place_wedge(16, 4, 10, Facing::South, 1);
    data.place_wedge(22, 4, 10, Facing::West, 1);

    // Smooth wedge
    data.place_wedge(4, 4, 16, Facing::North, 1);
    data.place_wedge(10, 4, 16, Facing::East, 1);

    // Wedge on cube
    data.place_std(16, 2, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_wedge(16, 3, 4, Facing::East, 1);

    // Two adjacent cubes
    data.place_std(22, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_std(24, 4, 4, SHAPE_CUBE, Facing::North, 1);

    // Staircase (concave fillet test)
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 15, 8, SHAPE_CUBE, Facing::North, 1);

    data
}
