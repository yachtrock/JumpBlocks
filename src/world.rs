use avian3d::prelude::*;
use bevy::prelude::*;
use jumpblocks_voxel::chunk::{Chunk, ChunkData, Voxel};
use jumpblocks_voxel::shape::{Facing, SHAPE_SMOOTH_CUBE, SHAPE_SMOOTH_WEDGE, SHAPE_WEDGE};

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
        let smooth_voxel = Voxel::new(SHAPE_SMOOTH_CUBE, Facing::North, 1);
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
        let wedge_e = Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1);
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

        let chunk_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.5, 0.4),
            ..default()
        });

        commands.spawn((
            Chunk::new(chunk_data),
            MeshMaterial3d(chunk_material),
            Transform::from_translation(Vec3::new(-5.0, 0.0, 5.0)),
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        ));
    }
}
