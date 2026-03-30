use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};
use bevy::window::PrimaryWindow;
use jumpblocks_voxel::chunk::{Chunk, ChunkState};

pub struct CurtainPlugin;

impl Plugin for CurtainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Material2dPlugin::<CurtainMaterial>::default())
            .init_resource::<CurtainState>()
            .add_systems(Startup, setup_curtain)
            .add_systems(
                Update,
                (enable_3d_camera, check_level_ready, animate_curtain, cleanup_curtain).chain(),
            );
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct CurtainState {
    /// Whether all chunks have reached Ready state at least once.
    pub chunks_ready: bool,
    /// Whether the reveal animation has started.
    pub revealing: bool,
    /// Current progress of the reveal animation (0.0 = black, 1.0 = fully open).
    pub progress: f32,
    /// Duration of the reveal animation in seconds.
    pub reveal_duration: f32,
    /// Whether the curtain has been fully raised and cleaned up.
    pub raised: bool,
    /// Frames elapsed since the curtain was set up (used to delay enabling the 3D camera).
    frames_alive: u32,
    /// Whether the 3D camera has been enabled.
    camera_enabled: bool,
}

impl Default for CurtainState {
    fn default() -> Self {
        Self {
            chunks_ready: false,
            revealing: false,
            progress: 0.0,
            reveal_duration: 0.4,
            raised: false,
            frames_alive: 0,
            camera_enabled: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct CurtainMaterial {
    #[uniform(0)]
    pub progress: f32,
    #[uniform(0)]
    pub points: f32,
    #[uniform(0)]
    pub aspect: f32,
    #[uniform(0)]
    pub _pad: f32,
}

impl Material2d for CurtainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/curtain.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

// ---------------------------------------------------------------------------
// Marker components
// ---------------------------------------------------------------------------

#[derive(Component)]
struct CurtainOverlay;

#[derive(Component)]
struct CurtainCamera;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

fn setup_curtain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CurtainMaterial>>,
    window_query: Query<&Window, With<PrimaryWindow>>,
) {
    let (width, height) = window_query
        .single()
        .map(|w| (w.resolution.width(), w.resolution.height()))
        .unwrap_or((1920.0, 1080.0));

    let aspect = width / height;

    let material = materials.add(CurtainMaterial {
        progress: 0.0,
        points: 5.0, // 5-pointed star
        aspect,
        _pad: 0.0,
    });

    // Fullscreen quad
    let mesh = meshes.add(Rectangle::new(width, height));

    // Dedicated 2D camera that renders on top of everything
    commands.spawn((
        Camera2d,
        Camera {
            order: 99, // render after main 3D camera
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        CurtainCamera,
    ));

    commands.spawn((
        Mesh2d(mesh),
        MeshMaterial2d(material),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        CurtainOverlay,
    ));
}

/// Wait a couple of frames for the curtain material to compile, then enable the 3D camera.
fn enable_3d_camera(
    mut curtain: ResMut<CurtainState>,
    mut camera_query: Query<&mut Camera, With<Camera3d>>,
) {
    if curtain.camera_enabled {
        return;
    }

    curtain.frames_alive += 1;

    // Give the curtain shader 2 frames to compile so it's covering the screen
    if curtain.frames_alive >= 2 {
        for mut camera in camera_query.iter_mut() {
            camera.is_active = true;
        }
        curtain.camera_enabled = true;
    }
}

/// Check if all chunks in the world have finished meshing.
fn check_level_ready(
    mut curtain: ResMut<CurtainState>,
    chunks: Query<&Chunk>,
) {
    if curtain.chunks_ready {
        return;
    }

    // We need at least one chunk to exist before we can say "ready"
    // Wait for a minimum number of chunks to be meshed, not all of them.
    // Distant chunks will finish meshing while the player is already playing.
    let min_ready = 8;
    let mut ready_count = 0;

    for chunk in chunks.iter() {
        if chunk.state == ChunkState::Ready {
            ready_count += 1;
        }
    }

    if ready_count >= min_ready {
        info!("{} chunks ready — raising the curtain", ready_count);
        curtain.chunks_ready = true;
        curtain.revealing = true;
    }
}

/// Animate the star cutout expanding once chunks are ready.
fn animate_curtain(
    time: Res<Time>,
    mut curtain: ResMut<CurtainState>,
    mut materials: ResMut<Assets<CurtainMaterial>>,
    overlay_query: Query<&MeshMaterial2d<CurtainMaterial>, With<CurtainOverlay>>,
    window_query: Query<&Window, With<PrimaryWindow>>,
) {
    if !curtain.revealing || curtain.raised {
        return;
    }

    curtain.progress += time.delta_secs() / curtain.reveal_duration;

    if curtain.progress >= 1.0 {
        curtain.progress = 1.0;
        curtain.raised = true;
    }

    // Update material uniforms
    for mat_handle in overlay_query.iter() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.progress = curtain.progress;

            // Keep aspect ratio in sync in case of resize
            if let Ok(window) = window_query.single() {
                let w = window.resolution.width();
                let h = window.resolution.height();
                if h > 0.0 {
                    mat.aspect = w / h;
                }
            }
        }
    }
}

/// Once the animation is complete, remove the overlay entities.
fn cleanup_curtain(
    mut commands: Commands,
    curtain: Res<CurtainState>,
    overlay_query: Query<Entity, With<CurtainOverlay>>,
    camera_query: Query<Entity, With<CurtainCamera>>,
) {
    if !curtain.raised {
        return;
    }

    for entity in overlay_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in camera_query.iter() {
        commands.entity(entity).despawn();
    }
}
