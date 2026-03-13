use bevy::prelude::*;
use bevy_third_person_camera::*;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera);
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        ThirdPersonCamera {
            zoom: Zoom::new(5.0, 15.0),
            cursor_lock_toggle_enabled: true,
            cursor_lock_active: true,
            cursor_lock_key: KeyCode::Escape,
            sensitivity: Vec2::new(2.0, 2.0),
            gamepad_settings: CustomGamepadSettings {
                sensitivity: Vec2::new(4.0, 4.0),
                ..default()
            },
            ..default()
        },
        Camera3d::default(),
    ));
}
