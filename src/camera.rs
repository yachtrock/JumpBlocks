use avian3d::prelude::*;
use bevy::image::Image;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::EnvironmentMapLight;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::layers::GameLayer;
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
    /// The actual distance used for positioning (accounts for collision).
    pub effective_distance: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub mouse_sensitivity: Vec2,
    pub gamepad_sensitivity: Vec2,
    pub zoom_sensitivity: f32,
    /// How fast the camera smoothly catches up (higher = snappier).
    pub smoothing: f32,
    /// How fast the camera eases back out after a collision clears.
    pub collision_recovery_speed: f32,
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
            effective_distance: 8.0,
            min_distance: 3.0,
            max_distance: 20.0,
            mouse_sensitivity: Vec2::new(0.003, 0.003),
            gamepad_sensitivity: Vec2::new(2.5, 1.5),
            zoom_sensitivity: 1.0,
            smoothing: 18.0,
            collision_recovery_speed: 4.0,
            cursor_locked: true,
        }
    }
}

/// Create a procedural gradient cubemap for image-based lighting.
///
/// Produces a sky-to-ground gradient: blue sky above, warm horizon glow at
/// the equator, and earthy tones below.  The result is used for both the
/// diffuse irradiance and specular maps in `EnvironmentMapLight`.
fn create_ibl_cubemap(size: u32) -> Image {
    let texels_per_face = (size * size) as usize;
    // Rgba32Float = 16 bytes per texel
    let mut data = vec![0u8; texels_per_face * 16 * 6];

    for face in 0..6u32 {
        for y in 0..size {
            for x in 0..size {
                // Map texel to direction on the unit cube, then normalize.
                let u = (2.0 * x as f32 + 1.0) / size as f32 - 1.0;
                let v = (2.0 * y as f32 + 1.0) / size as f32 - 1.0;

                let dir = match face {
                    0 => Vec3::new(1.0, -v, -u),  // +X
                    1 => Vec3::new(-1.0, -v, u),   // -X
                    2 => Vec3::new(u, 1.0, v),     // +Y
                    3 => Vec3::new(u, -1.0, -v),   // -Y
                    4 => Vec3::new(u, -v, 1.0),    // +Z
                    _ => Vec3::new(-u, -v, -1.0),  // -Z
                }
                .normalize();

                // Vertical blend factor: 1 = straight up, -1 = straight down.
                let h = dir.y;

                // Horizontal direction weights (in xz plane) for cardinal
                // color variation.  Each is 0..1, peaking when looking
                // toward that direction.  Using saturated dot products
                // with a soft power gives wide, overlapping lobes.
                let horiz = Vec2::new(dir.x, dir.z).normalize_or_zero();
                let w_south = horiz.dot(Vec2::new(0.0, 1.0)).max(0.0).powf(0.8);  // +Z
                let w_north = horiz.dot(Vec2::new(0.0, -1.0)).max(0.0).powf(0.8); // -Z
                let w_east = horiz.dot(Vec2::new(1.0, 0.0)).max(0.0).powf(0.8);   // +X
                let w_west = horiz.dot(Vec2::new(-1.0, 0.0)).max(0.0).powf(0.8);  // -X

                // Cardinal sky tints — asymmetric to fake a sun somewhere
                // in the south-east and cooler fill from the north-west.
                //          base blue     + directional tint
                let sky_south = Vec3::new(0.50, 0.60, 0.95); // warm haze
                let sky_north = Vec3::new(0.25, 0.45, 1.05); // deep cool blue
                let sky_east  = Vec3::new(0.55, 0.58, 0.85); // golden-warm
                let sky_west  = Vec3::new(0.30, 0.50, 1.00); // neutral cool
                let sky_base  = Vec3::new(0.35, 0.55, 1.0);

                let sky_tint = sky_base
                    + (sky_south - sky_base) * w_south
                    + (sky_north - sky_base) * w_north
                    + (sky_east - sky_base) * w_east
                    + (sky_west - sky_base) * w_west;
                let sky = sky_tint * (1.0 + h * 0.3);

                // Horizon: warmer glow toward the south-east "sun",
                // cooler and dimmer toward the north-west.
                let hz_south = Vec3::new(1.10, 0.88, 0.65);
                let hz_north = Vec3::new(0.75, 0.78, 0.85);
                let hz_east  = Vec3::new(1.05, 0.85, 0.60);
                let hz_west  = Vec3::new(0.80, 0.80, 0.85);
                let hz_base  = Vec3::new(0.90, 0.82, 0.72);

                let horizon = hz_base
                    + (hz_south - hz_base) * w_south
                    + (hz_north - hz_base) * w_north
                    + (hz_east - hz_base) * w_east
                    + (hz_west - hz_base) * w_west;

                // Ground bounce — slightly warmer from the sunlit side.
                let gnd_base = Vec3::new(0.45, 0.38, 0.30) * 0.5;
                let gnd_warm = Vec3::new(0.50, 0.40, 0.30) * 0.55;
                let sun_weight = (w_south + w_east) * 0.5;
                let ground = gnd_base.lerp(gnd_warm, sun_weight);

                let color = if h > 0.0 {
                    // Above horizon: blend sky ↔ horizon.
                    let t = h.powf(0.6);
                    horizon.lerp(sky, t)
                } else {
                    // Below horizon: blend horizon ↔ ground.
                    let t = (-h).powf(0.4);
                    horizon.lerp(ground, t)
                };

                let offset = ((face * size * size + y * size + x) as usize) * 16;
                data[offset..offset + 4].copy_from_slice(&color.x.to_le_bytes());
                data[offset + 4..offset + 8].copy_from_slice(&color.y.to_le_bytes());
                data[offset + 8..offset + 12].copy_from_slice(&color.z.to_le_bytes());
                data[offset + 12..offset + 16].copy_from_slice(&1.0f32.to_le_bytes());
            }
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba32Float,
        default(),
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image
}

/// Small offset so the camera doesn't clip into geometry.
const CAMERA_CLIP_OFFSET: f32 = 0.2;

fn spawn_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut window_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let diffuse_map = images.add(create_ibl_cubemap(32));
    let specular_map = images.add(create_ibl_cubemap(64));

    commands.spawn((
        OrbitCamera::default(),
        Camera3d::default(),
        Camera {
            is_active: false, // disabled until curtain is ready
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map,
            specular_map,
            intensity: 800.0,
            rotation: Quat::IDENTITY,
            affects_lightmapped_mesh_diffuse: false,
        },
    ));

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
    spatial_query: SpatialQuery,
    player_query: Query<(Entity, &Transform), With<Player>>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform), Without<Player>>,
) {
    let Ok((player_entity, player_transform)) = player_query.single() else {
        return;
    };
    let Ok((mut cam, mut cam_transform)) = camera_query.single_mut() else {
        return;
    };

    let dt = time.delta_secs();
    let t = (cam.smoothing * dt).min(1.0);

    // Smoothly interpolate toward target (ideal distance ignoring collision)
    cam.yaw = cam.yaw + (cam.target_yaw - cam.yaw) * t;
    cam.pitch = cam.pitch + (cam.target_pitch - cam.pitch) * t;
    cam.distance = cam.distance + (cam.target_distance - cam.distance) * t;

    // Compute ideal orbit direction
    let rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let orbit_dir = rotation * Vec3::Z; // direction from player toward camera

    // Raycast from player toward ideal camera position to detect blocking geometry
    let look_at_pos = player_transform.translation;
    let filter = SpatialQueryFilter::from_mask(GameLayer::CameraBlocking)
        .with_excluded_entities([player_entity]);

    let max_allowed = if let Ok(ray_dir) = Dir3::new(orbit_dir) {
        if let Some(hit) =
            spatial_query.cast_ray(look_at_pos, ray_dir, cam.distance, true, &filter)
        {
            // Hit something — cap distance at hit point minus offset
            (hit.distance - CAMERA_CLIP_OFFSET).max(cam.min_distance * 0.5)
        } else {
            cam.distance
        }
    } else {
        cam.distance
    };

    // Snap in instantly when blocked, ease out slowly when clear
    if cam.effective_distance > max_allowed {
        // Blocked — instant snap closer
        cam.effective_distance = max_allowed;
    } else if cam.effective_distance < cam.distance {
        // Clear — ease back toward ideal distance
        let recovery_t = (cam.collision_recovery_speed * dt).min(1.0);
        cam.effective_distance =
            cam.effective_distance + (cam.distance - cam.effective_distance) * recovery_t;
        // Don't overshoot the max allowed
        cam.effective_distance = cam.effective_distance.min(max_allowed);
    } else {
        cam.effective_distance = cam.distance;
    }

    // Final camera position using effective distance
    let offset = orbit_dir * cam.effective_distance;
    cam_transform.translation = look_at_pos + offset;
    cam_transform.look_at(look_at_pos, Vec3::Y);
}
