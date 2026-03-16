use avian3d::prelude::*;
use bevy::prelude::*;
use jumpblocks_voxel::chunk::{Chunk, ChunkData, ChunkNeighbors, Voxel};
use jumpblocks_voxel::shape::{Facing, SHAPE_CUBE, SHAPE_WEDGE};

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
        // Add visual components only when rendering is available
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

        // Demo voxel chunk — a staircase the player can climb
        let mut chunk_data = ChunkData::new();

        // Ground layer: a 10x10 platform, single voxel tall (0.5 world units)
        for x in 0..10 {
            for z in 0..10 {
                chunk_data.set_filled(x, 0, z, true);
            }
        }

        // Staircase: each step is 1 voxel tall (0.5 world units), 3 wide — smooth cube
        let smooth_voxel = Voxel::new(SHAPE_CUBE, Facing::North, 1);
        for step in 0..8 {
            let y_base = 1 + step;
            for x in 0..3 {
                for z in 0..3 {
                    chunk_data.set(x + step, y_base, z + 3, smooth_voxel);
                }
            }
        }

        // A small tower at the top — smooth cube
        for y in 9..18 {
            for x in 8..11 {
                for z in 3..6 {
                    chunk_data.set(x, y, z, smooth_voxel);
                }
            }
        }

        // Wedge ramp alongside the stairs (z=7..9, 3 wide)
        // Facing East: tall wall at -X, slope descends toward +X
        // One wedge per step forms a continuous diagonal slope
        let wedge_e = Voxel::new(SHAPE_WEDGE, Facing::East, 1);
        for step in 0..8 {
            let y_base = 1 + step;
            for z_off in 0..3 {
                chunk_data.set(step, y_base, 7 + z_off, wedge_e);
            }
            // Fill underneath with smooth cubes
            for y in 1..y_base {
                for z_off in 0..3 {
                    chunk_data.set(step, y, 7 + z_off, smooth_voxel);
                }
            }
        }

        // A row of wedges facing each direction for testing rotation
        let wedge_n = Voxel::new(SHAPE_WEDGE, Facing::North, 1);
        let wedge_e2 = Voxel::new(SHAPE_WEDGE, Facing::East, 1);
        let wedge_s = Voxel::new(SHAPE_WEDGE, Facing::South, 1);
        let wedge_w = Voxel::new(SHAPE_WEDGE, Facing::West, 1);
        chunk_data.set(0, 2, 10, wedge_n);
        chunk_data.set(2, 2, 10, wedge_e2);
        chunk_data.set(4, 2, 10, wedge_s);
        chunk_data.set(6, 2, 10, wedge_w);

        // Bridge platform reaching the +X boundary (x=15)
        // Platform at y=1, z=3..6, from x=10 to x=15
        for x in 10..16 {
            for z in 3..6 {
                chunk_data.set(x, 1, z, smooth_voxel);
            }
        }
        // A column at the boundary edge
        for y in 2..6 {
            chunk_data.set(15, y, 4, smooth_voxel);
        }
        // Wedge at boundary top pointing into neighbor
        chunk_data.set(15, 6, 4, Voxel::new(SHAPE_WEDGE, Facing::West, 1));

        // Row of cubes along boundary at ground level (z=7..9)
        for z in 7..10 {
            chunk_data.set(15, 0, z, Voxel::filled());
            chunk_data.set(15, 1, z, Voxel::filled());
        }

        let chunk_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.5, 0.4),
            ..default()
        });

        // ---- Neighbor chunk (+X direction) ----
        // Positioned so its x=0 face meets chunk 1's x=15 face
        let neighbor_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.55, 0.7),
            ..default()
        });

        let mut neighbor_data = ChunkData::new();

        // Continuation of the bridge platform at x=0, y=1, z=3..6
        for x in 0..6 {
            for z in 3..6 {
                neighbor_data.set(x, 1, z, smooth_voxel);
            }
        }
        // Matching column on this side of the boundary
        for y in 2..6 {
            neighbor_data.set(0, y, 4, smooth_voxel);
        }
        // Wedge pointing back toward chunk 1
        neighbor_data.set(0, 6, 4, Voxel::new(SHAPE_WEDGE, Facing::East, 1));

        // Matching row of cubes along boundary at ground level (z=7..9)
        for z in 7..10 {
            neighbor_data.set(0, 0, z, Voxel::filled());
            neighbor_data.set(0, 1, z, Voxel::filled());
        }

        // A small structure only in the neighbor chunk
        for x in 3..6 {
            for z in 7..10 {
                neighbor_data.set(x, 0, z, smooth_voxel);
            }
        }
        // Wedge ramp on top
        for z in 7..10 {
            neighbor_data.set(3, 1, z, Voxel::new(SHAPE_WEDGE, Facing::West, 1));
        }

        // Wire up neighbor references between the two adjacent chunks (+X / -X)
        let mut chunk1_neighbors = ChunkNeighbors::empty();
        chunk1_neighbors.set(1, 0, 0, neighbor_data.clone());

        let mut chunk2_neighbors = ChunkNeighbors::empty();
        chunk2_neighbors.set(-1, 0, 0, chunk_data.clone());

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

        // ---- Floating test blocks — isolated shapes for inspecting chamfer ----
        let mut test_data = ChunkData::new();

        // Single cube
        test_data.set(2, 4, 2, Voxel::filled());

        // Single smooth cube
        test_data.set(5, 4, 2, smooth_voxel);

        // Single wedge (each facing)
        test_data.set(2, 4, 5, Voxel::new(SHAPE_WEDGE, Facing::North, 1));
        test_data.set(5, 4, 5, Voxel::new(SHAPE_WEDGE, Facing::East, 1));
        test_data.set(8, 4, 5, Voxel::new(SHAPE_WEDGE, Facing::South, 1));
        test_data.set(11, 4, 5, Voxel::new(SHAPE_WEDGE, Facing::West, 1));

        // Single smooth wedge
        test_data.set(2, 4, 8, Voxel::new(SHAPE_WEDGE, Facing::North, 1));
        test_data.set(5, 4, 8, Voxel::new(SHAPE_WEDGE, Facing::East, 1));

        // Wedge on cube
        test_data.set(8, 3, 2, smooth_voxel);
        test_data.set(8, 4, 2, Voxel::new(SHAPE_WEDGE, Facing::East, 1));

        // Two adjacent cubes
        test_data.set(11, 4, 2, Voxel::filled());
        test_data.set(12, 4, 2, Voxel::filled());

        commands.spawn((
            Chunk::new(test_data),
            MeshMaterial3d(chunk_material),
            Transform::from_translation(Vec3::new(10.0, 0.0, -10.0)),
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        ));
    }
}
