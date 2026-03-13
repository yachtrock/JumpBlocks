use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::player::Player;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, (camera_input, camera_follow));
    }
}

#[derive(Component)]
pub struct OrbitCamera {
    pub pitch: f32,
    pub yaw: f32,
    pub target_pitch: f32,
    pub target_yaw: f32,
    pub distance: f32,
    pub target_distance: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub mouse_sensitivity: Vec2,
    pub gamepad_sensitivity: Vec2,
    pub zoom_sensitivity: f32,
    /// How fast the camera smoothly catches up (higher = snappier).
    pub smoothing: f32,
    pub cursor_locked: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            pitch: -0.3,
            yaw: 0.0,
            target_pitch: -0.3,
            target_yaw: 0.0,
            distance: 8.0,
            target_distance: 8.0,
            min_distance: 3.0,
            max_distance: 20.0,
            mouse_sensitivity: Vec2::new(0.003, 0.003),
            gamepad_sensitivity: Vec2::new(2.5, 1.5),
            zoom_sensitivity: 1.0,
            smoothing: 18.0,
            cursor_locked: true,
        }
    }
}

fn spawn_camera(
    mut commands: Commands,
    mut window_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    commands.spawn((OrbitCamera::default(), Camera3d::default()));

    // Lock cursor on startup
    if let Ok(mut cursor) = window_query.single_mut() {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

fn camera_input(
    mut mouse_motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    mut camera_query: Query<&mut OrbitCamera>,
) {
    let Ok(mut cam) = camera_query.single_mut() else {
        return;
    };

    if !cam.cursor_locked {
        // Drain events so they don't accumulate
        mouse_motion.clear();
        scroll.clear();
        return;
    }

    // Accumulate mouse motion
    let mut mouse_delta = Vec2::ZERO;
    for ev in mouse_motion.read() {
        mouse_delta += ev.delta;
    }

    cam.target_yaw -= mouse_delta.x * cam.mouse_sensitivity.x;
    cam.target_pitch -= mouse_delta.y * cam.mouse_sensitivity.y;

    // Gamepad right stick
    let dt = time.delta_secs();
    for gamepad in gamepads.iter() {
        let stick = Vec2::new(
            gamepad.get(GamepadAxis::RightStickX).unwrap_or(0.0),
            gamepad.get(GamepadAxis::RightStickY).unwrap_or(0.0),
        );
        if stick.length() > 0.15 {
            cam.target_yaw -= stick.x * cam.gamepad_sensitivity.x * dt;
            cam.target_pitch += stick.y * cam.gamepad_sensitivity.y * dt;
        }
    }

    // Clamp pitch to avoid flipping
    cam.target_pitch = cam.target_pitch.clamp(-1.4, 1.0);

    // Zoom from scroll wheel
    let mut scroll_amount: f32 = 0.0;
    for ev in scroll.read() {
        scroll_amount += ev.y;
    }
    if scroll_amount.abs() > 0.0 {
        cam.target_distance -= scroll_amount * cam.target_distance * 0.1 * cam.zoom_sensitivity;
        cam.target_distance = cam.target_distance.clamp(cam.min_distance, cam.max_distance);
    }
}

fn camera_follow(
    time: Res<Time>,
    player_query: Query<&Transform, With<Player>>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform), Without<Player>>,
) {
    let Ok(player_transform) = player_query.single() else {
        return;
    };
    let Ok((mut cam, mut cam_transform)) = camera_query.single_mut() else {
        return;
    };

    let dt = time.delta_secs();
    let t = (cam.smoothing * dt).min(1.0);

    // Smoothly interpolate toward target
    cam.yaw = cam.yaw + (cam.target_yaw - cam.yaw) * t;
    cam.pitch = cam.pitch + (cam.target_pitch - cam.pitch) * t;
    cam.distance = cam.distance + (cam.target_distance - cam.distance) * t;

    // Compute orbit position
    let rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let offset = rotation * Vec3::new(0.0, 0.0, cam.distance);

    cam_transform.translation = player_transform.translation + offset;
    cam_transform.look_at(player_transform.translation, Vec3::Y);
}
