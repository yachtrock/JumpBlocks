use avian3d::prelude::*;
use bevy::prelude::*;

use crate::layers::GameLayer;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_world);
    }
}

fn setup_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(50.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.6, 0.3),
            ..default()
        })),
        RigidBody::Static,
        Collider::half_space(Vec3::Y),
        CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
    ));

    // Some platforms to jump on
    let platform_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.6),
        ..default()
    });
    let platform_mesh = meshes.add(Cuboid::new(4.0, 0.5, 4.0));

    let platforms = [
        Vec3::new(5.0, 1.0, 0.0),
        Vec3::new(10.0, 2.5, 3.0),
        Vec3::new(15.0, 4.0, -1.0),
        Vec3::new(12.0, 5.5, -6.0),
        Vec3::new(7.0, 7.0, -8.0),
    ];

    for pos in platforms {
        commands.spawn((
            Mesh3d(platform_mesh.clone()),
            MeshMaterial3d(platform_material.clone()),
            Transform::from_translation(pos),
            RigidBody::Static,
            Collider::cuboid(4.0, 0.5, 4.0),
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
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
}
