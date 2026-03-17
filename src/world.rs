use avian3d::prelude::*;
use bevy::prelude::*;
use jumpblocks_voxel::chunk::{Chunk, ChunkData, ChunkNeighbors};
use jumpblocks_voxel::shape::{Facing, ShapeTable, SHAPE_CUBE};

use crate::layers::GameLayer;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_world);
    }
}

fn setup_world(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let has_rendering = meshes.is_some() && materials.is_some();

    // Ground plane
    let ground = commands.spawn((
        RigidBody::Static,
        Collider::half_space(Vec3::Y),
        CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
    ));
    if has_rendering {
        // Visuals will be added below after we create handles
    }
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

        // Directional light (sun)
        commands.spawn((
            DirectionalLight {
                illuminance: 10_000.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
        ));

        // Ambient light
        commands.insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.9, 0.9, 1.0),
            brightness: 300.0,
            ..default()
        });

        // Demo voxel chunk
        // All blocks are 2x2x2 cells. Each block = 1x1x1 world units (VOXEL_SIZE=0.5).
        // Block origins must be at even-aligned coordinates (0, 2, 4, ...).
        let mut chunk_data = ChunkData::new();

        // Ground layer: a 10x10 block platform (each block = 2x2x2 cells)
        for bx in 0..10 {
            for bz in 0..10 {
                chunk_data.place_std(bx * 2, 0, bz * 2, SHAPE_CUBE, Facing::North, 1);
            }
        }

        // Staircase: 8 steps, each 1 cell tall, 2 blocks wide in X, 3 blocks deep in Z
        // Each step goes 1 cell higher and 1 block (2 cells) further in X
        for step in 0..8usize {
            let y_top = 1 + step; // top of this step
            for bx in 0..2 {
                for bz in 0..3 {
                    // Fill from ground (y=1) up to this step's height
                    for y in 1..=y_top {
                        chunk_data.place_std(bx * 2 + step * 2, y, bz * 2 + 6, SHAPE_CUBE, Facing::North, 1);
                    }
                }
            }
        }

        // A small tower at the top
        for by in 0..5 {
            for bx in 0..3 {
                for bz in 0..3 {
                    chunk_data.place_std(16 + bx * 2, 9 + by, 6 + bz * 2, SHAPE_CUBE, Facing::North, 1);
                }
            }
        }

        // Wedge ramp alongside the stairs
        // Each wedge is 2 cells tall (solid base + slope), steps up by 1 cell
        // Step 0: wedge at y=1, Step 1: wedge at y=2, etc.
        for step in 0..8usize {
            let wedge_y = 1 + step; // wedge occupies y..y+2
            for bz in 0..3 {
                chunk_data.place_wedge(step * 2, wedge_y, 14 + bz * 2, Facing::East, 1);
            }
            // Fill cubes underneath (each cube is 1 cell tall)
            for y in 1..wedge_y {
                for bz in 0..3 {
                    chunk_data.place_std(step * 2, y, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
                }
            }
        }

        // A row of wedges facing each direction for testing rotation
        chunk_data.place_wedge(0, 2, 20, Facing::North, 1);
        chunk_data.place_wedge(4, 2, 20, Facing::East, 1);
        chunk_data.place_wedge(8, 2, 20, Facing::South, 1);
        chunk_data.place_wedge(12, 2, 20, Facing::West, 1);

        // Bridge platform reaching the +X boundary
        for bx in 0..6 {
            for bz in 0..3 {
                chunk_data.place_std(20 + bx * 2, 0, 6 + bz * 2, SHAPE_CUBE, Facing::North, 1);
            }
        }
        // A column at the boundary edge (x=30)
        for by in 0..2 {
            chunk_data.place_std(30, 2 + by, 8, SHAPE_CUBE, Facing::North, 1);
        }
        // Wedge at boundary top
        chunk_data.place_wedge(30, 6, 8, Facing::West, 1);

        // Row of cubes along boundary at ground level
        for bz in 0..3 {
            chunk_data.place_std(30, 0, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
        }

        let chunk_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.5, 0.4),
            ..default()
        });

        // ---- Neighbor chunk (+X direction) ----
        // Positioned so its x=0 face meets chunk 1's x=31 face
        let neighbor_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.55, 0.7),
            ..default()
        });

        let mut neighbor_data = ChunkData::new();

        // Continuation of the bridge platform
        for bx in 0..6 {
            for bz in 0..3 {
                neighbor_data.place_std(bx * 2, 0, 6 + bz * 2, SHAPE_CUBE, Facing::North, 1);
            }
        }
        // Matching column on this side
        for by in 0..2 {
            neighbor_data.place_std(0, 2 + by, 8, SHAPE_CUBE, Facing::North, 1);
        }
        // Wedge pointing back
        neighbor_data.place_wedge(0, 6, 8, Facing::East, 1);

        // Matching row of cubes along boundary
        for bz in 0..3 {
            neighbor_data.place_std(0, 0, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
        }

        // A small structure only in the neighbor chunk
        for bx in 0..3 {
            for bz in 0..3 {
                neighbor_data.place_std(6 + bx * 2, 0, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
            }
        }
        // Wedge ramp on top
        for bz in 0..3 {
            neighbor_data.place_wedge(6, 2, 14 + bz * 2, Facing::West, 1);
        }

        // Wire up neighbor references
        let mut chunk1_neighbors = ChunkNeighbors::empty();
        chunk1_neighbors.set(1, 0, 0, neighbor_data.clone());

        let mut chunk2_neighbors = ChunkNeighbors::empty();
        chunk2_neighbors.set(-1, 0, 0, chunk_data.clone());

        // Validate chunk data before creating chunks
        let shapes = ShapeTable::default();
        for (label, data) in [("main", &chunk_data), ("neighbor", &neighbor_data)] {
            let errors = data.validate(&shapes);
            for e in &errors {
                error!("[world] {}: {}", label, e);
            }
        }

        let mut chunk1 = Chunk::new(chunk_data);
        chunk1.neighbors = chunk1_neighbors;

        let mut chunk2 = Chunk::new(neighbor_data);
        chunk2.neighbors = chunk2_neighbors;

        commands.spawn((
            chunk1,
            MeshMaterial3d(chunk_material.clone()),
            Transform::from_translation(Vec3::new(-5.0, 0.0, 5.0)),
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        ));

        commands.spawn((
            chunk2,
            MeshMaterial3d(neighbor_material),
            Transform::from_translation(Vec3::new(11.0, 0.0, 5.0)),
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        ));

        // ---- Floating test blocks ----
        let mut test_data = ChunkData::new();

        // Single cube
        test_data.place_std(4, 4, 4, SHAPE_CUBE, Facing::North, 1);

        // Single smooth cube
        test_data.place_std(10, 4, 4, SHAPE_CUBE, Facing::North, 1);

        // Single wedge (each facing)
        test_data.place_wedge(4, 4, 10, Facing::North, 1);
        test_data.place_wedge(10, 4, 10, Facing::East, 1);
        test_data.place_wedge(16, 4, 10, Facing::South, 1);
        test_data.place_wedge(22, 4, 10, Facing::West, 1);

        // Single smooth wedge
        test_data.place_wedge(4, 4, 16, Facing::North, 1);
        test_data.place_wedge(10, 4, 16, Facing::East, 1);

        // Wedge on cube
        test_data.place_std(16, 2, 4, SHAPE_CUBE, Facing::North, 1);
        test_data.place_wedge(16, 3, 4, Facing::East, 1);

        // Two adjacent cubes
        test_data.place_std(22, 4, 4, SHAPE_CUBE, Facing::North, 1);
        test_data.place_std(24, 4, 4, SHAPE_CUBE, Facing::North, 1);

        // Staircase (concave fillet test):  [B]
        //                                 [A][C]
        test_data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);  // A: bottom-left
        test_data.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1); // C: bottom-right
        test_data.place_std(10, 15, 8, SHAPE_CUBE, Facing::North, 1); // B: top-right

        let test_errors = test_data.validate(&shapes);
        for e in &test_errors {
            error!("[world] test: {}", e);
        }

        commands.spawn((
            Chunk::new(test_data),
            MeshMaterial3d(chunk_material),
            Transform::from_translation(Vec3::new(10.0, 0.0, -10.0)),
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        ));
    }
}
