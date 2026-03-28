use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use crate::camera::OrbitCamera;
use crate::edge_detection::{EdgeDetectionSettings, PrecariousEdge};
use crate::player::{ControlScheme, ControlSchemeConfig, LeanSettings, Player, PlayerSettings};
use bevy_tnua::prelude::*;
use jumpblocks_voxel::chunk::Chunk;
use jumpblocks_voxel::chunk_lod::{LodConfig, LodTier};
use jumpblocks_voxel::coords::ChunkCoord;
use jumpblocks_voxel::streaming::StreamingConfig;

pub struct DebugUiPlugin;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((EguiPlugin::default(), FrameTimeDiagnosticsPlugin::default()))
            .init_resource::<DebugUiState>()
            .add_systems(Update, toggle_debug_ui)
            .add_systems(EguiPrimaryContextPass, debug_ui_system);
    }
}

#[derive(Resource)]
struct DebugUiState {
    visible: bool,
}

impl Default for DebugUiState {
    fn default() -> Self {
        Self { visible: false }
    }
}

fn toggle_debug_ui(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<DebugUiState>,
    mut camera_query: Query<&mut OrbitCamera>,
    mut window_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if keyboard.just_pressed(KeyCode::Backquote) {
        state.visible = !state.visible;
        if let Ok(mut cam) = camera_query.single_mut() {
            cam.cursor_locked = !state.visible;
        }
        if let Ok(mut cursor) = window_query.single_mut() {
            if state.visible {
                cursor.grab_mode = CursorGrabMode::None;
                cursor.visible = true;
            } else {
                cursor.grab_mode = CursorGrabMode::Locked;
                cursor.visible = false;
            }
        }
    }
}

fn debug_ui_system(
    mut contexts: EguiContexts,
    state: Res<DebugUiState>,
    diagnostics: Res<DiagnosticsStore>,
    mut player_query: Query<(&Transform, &TnuaConfig<ControlScheme>, &mut PlayerSettings, &mut LeanSettings, &PrecariousEdge, &mut EdgeDetectionSettings), With<Player>>,
    mut camera_query: Query<&mut OrbitCamera>,
    mut configs: ResMut<Assets<ControlSchemeConfig>>,
    gamepads: Query<(Entity, &Gamepad)>,
    chunk_query: Query<(&Chunk, &Mesh3d)>,
    mesh_assets: Res<Assets<Mesh>>,
    mut lod_config: ResMut<LodConfig>,
    mut streaming_config: ResMut<StreamingConfig>,
    lod_chunks: Query<(&ChunkCoord, &LodTier)>,
) {
    if !state.visible {
        return;
    }

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("Debug").show(ctx, |ui| {
        // FPS
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                ui.label(format!("FPS: {value:.1}"));
            }
        }
        if let Some(frame_time) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
            if let Some(value) = frame_time.smoothed() {
                ui.label(format!("Frame time: {value:.2} ms"));
            }
        }

        // Chunk / triangle stats
        let mut chunk_count = 0u32;
        let mut total_tris = 0u32;
        let mut total_verts = 0u32;
        for (_chunk, mesh_handle) in chunk_query.iter() {
            chunk_count += 1;
            if let Some(mesh) = mesh_assets.get(&mesh_handle.0) {
                total_verts += mesh.count_vertices() as u32;
                if let Some(indices) = mesh.indices() {
                    total_tris += indices.len() as u32 / 3;
                }
            }
        }
        ui.label(format!("Chunks: {chunk_count}  Tris: {total_tris}  Verts: {total_verts}"));

        ui.separator();

        // Player info
        if let Ok((transform, tnua_config, mut settings, mut lean, edge, mut edge_settings)) = player_query.single_mut() {
            ui.label(format!(
                "Position: ({:.1}, {:.1}, {:.1})",
                transform.translation.x, transform.translation.y, transform.translation.z
            ));
            if edge.on_edge {
                ui.colored_label(egui::Color32::ORANGE, format!(
                    "ON EDGE  dist: {:.2}  dir: ({:.2}, {:.2})",
                    edge.distance_from_edge, edge.overhang_direction.x, edge.overhang_direction.z
                ));
            }

            ui.separator();

            if let Some(config) = configs.get_mut(&tnua_config.0) {
                // Walk config
                ui.collapsing("Walk", |ui| {
                    let walk = &mut config.basis;
                    ui.add(egui::Slider::new(&mut walk.speed, 1.0..=30.0).text("Speed"));
                    ui.add(egui::Slider::new(&mut settings.run_multiplier, 1.0..=5.0).text("Run multiplier"));
                    ui.add(egui::Slider::new(&mut walk.float_height, 0.1..=5.0).text("Float height"));
                    ui.add(egui::Slider::new(&mut walk.acceleration, 1.0..=200.0).text("Acceleration"));
                    ui.add(egui::Slider::new(&mut walk.air_acceleration, 0.0..=100.0).text("Air acceleration"));
                    ui.add(egui::Slider::new(&mut walk.coyote_time, 0.0..=0.5).text("Coyote time"));
                    ui.add(egui::Slider::new(&mut walk.spring_strength, 10.0..=2000.0).text("Spring strength"));
                    ui.add(egui::Slider::new(&mut walk.spring_dampening, 0.0..=2.0).text("Spring dampening"));
                    ui.add(egui::Slider::new(&mut walk.cling_distance, 0.0..=5.0).text("Cling distance"));
                    ui.add(egui::Slider::new(&mut walk.free_fall_extra_gravity, 0.0..=200.0).text("Free fall extra gravity"));
                    ui.add(egui::Slider::new(&mut walk.turning_angvel, 0.0..=50.0).text("Turning angular vel"));
                    ui.add(egui::Slider::new(&mut walk.max_slope, 0.0..=1.571).text("Max slope (rad)"));
                });

                // Lean config
                ui.collapsing("Lean", |ui| {
                    ui.add(egui::Slider::new(&mut lean.max_angle, 0.0..=45.0).text("Forward lean (deg)"));
                    ui.add(egui::Slider::new(&mut lean.turn_max_angle, 0.0..=45.0).text("Turn lean (deg)"));
                    ui.add(egui::Slider::new(&mut lean.lerp_speed, 1.0..=30.0).text("Lerp speed"));
                });

                // Camera config
                if let Ok(mut cam) = camera_query.single_mut() {
                    ui.collapsing("Camera", |ui| {
                        ui.add(egui::Slider::new(&mut cam.smoothing, 1.0..=50.0).text("Smoothing"));
                        ui.add(egui::Slider::new(&mut cam.mouse_sensitivity.x, 0.001..=0.01).text("Mouse sens X"));
                        ui.add(egui::Slider::new(&mut cam.mouse_sensitivity.y, 0.001..=0.01).text("Mouse sens Y"));
                        ui.add(egui::Slider::new(&mut cam.gamepad_sensitivity.x, 0.5..=8.0).text("Gamepad sens X"));
                        ui.add(egui::Slider::new(&mut cam.gamepad_sensitivity.y, 0.5..=8.0).text("Gamepad sens Y"));
                        ui.add(egui::Slider::new(&mut cam.collision_recovery_speed, 0.5..=20.0).text("Collision recovery"));
                    });
                }

                // Edge detection config
                ui.collapsing("Edge Detection", |ui| {
                    ui.add(egui::Slider::new(&mut edge_settings.ray_max_distance, 0.5..=10.0).text("Ray distance"));
                });

                // Jump config
                ui.collapsing("Jump", |ui| {
                    let jump = &mut config.jump;
                    ui.add(egui::Slider::new(&mut jump.height, 0.5..=20.0).text("Height"));
                    ui.add(egui::Slider::new(&mut jump.takeoff_extra_gravity, 0.0..=100.0).text("Takeoff extra gravity"));
                    ui.add(egui::Slider::new(&mut jump.takeoff_above_velocity, 0.0..=20.0).text("Takeoff above velocity"));
                    ui.add(egui::Slider::new(&mut jump.fall_extra_gravity, 0.0..=100.0).text("Fall extra gravity"));
                    ui.add(egui::Slider::new(&mut jump.shorten_extra_gravity, 0.0..=200.0).text("Shorten extra gravity"));
                    ui.add(egui::Slider::new(&mut jump.upslope_extra_gravity, 0.0..=100.0).text("Upslope extra gravity"));
                    ui.add(egui::Slider::new(&mut jump.peak_prevention_at_upward_velocity, 0.0..=10.0).text("Peak prevention velocity"));
                    ui.add(egui::Slider::new(&mut jump.peak_prevention_extra_gravity, 0.0..=100.0).text("Peak prevention gravity"));
                    ui.add(egui::Slider::new(&mut jump.input_buffer_time, 0.0..=0.5).text("Input buffer time"));
                });
            }
        }

        ui.separator();

        // World / LOD
        ui.collapsing("World / LOD", |ui| {
            ui.label("Streaming");
            ui.add(egui::Slider::new(&mut streaming_config.load_radius_xz, 1..=32).text("Load radius XZ"));
            ui.add(egui::Slider::new(&mut streaming_config.load_radius_y, 1..=16).text("Load radius Y"));
            ui.add(egui::Slider::new(&mut streaming_config.unload_padding, 0..=8).text("Unload padding"));
            ui.add(egui::Slider::new(&mut streaming_config.load_budget, 1..=32).text("Load budget/frame"));
            ui.add(egui::Slider::new(&mut streaming_config.unload_budget, 1..=32).text("Unload budget/frame"));

            ui.separator();
            ui.label("LOD Tiers");
            ui.add(egui::Slider::new(&mut lod_config.full_radius, 0..=16).text("Full radius"));
            ui.add(egui::Slider::new(&mut lod_config.reduced_radius, 1..=32).text("Reduced radius"));

            ui.separator();
            let mut full_count = 0;
            let mut reduced_count = 0;
            let mut hidden_count = 0;
            for (_, tier) in lod_chunks.iter() {
                match tier {
                    LodTier::Full => full_count += 1,
                    LodTier::Reduced => reduced_count += 1,
                    LodTier::Hidden => hidden_count += 1,
                }
            }
            let total = full_count + reduced_count + hidden_count;
            ui.label(format!("Chunks: {} total", total));
            ui.label(format!("  Full: {}  Reduced: {}  Hidden: {}", full_count, reduced_count, hidden_count));
        });

        ui.separator();

        // Gamepad debug
        let gamepad_count = gamepads.iter().count();
        ui.collapsing(format!("Gamepads ({gamepad_count})"), |ui| {
            if gamepad_count == 0 {
                ui.label("No gamepads detected");
            }
            for (entity, gamepad) in gamepads.iter() {
                let vid = gamepad.vendor_id().map(|v| format!("{v:#06x}")).unwrap_or("??".into());
                let pid = gamepad.product_id().map(|v| format!("{v:#06x}")).unwrap_or("??".into());
                ui.label(format!("Entity: {entity}  VID: {vid}  PID: {pid}"));

                let lx = gamepad.get(GamepadAxis::LeftStickX).unwrap_or(0.0);
                let ly = gamepad.get(GamepadAxis::LeftStickY).unwrap_or(0.0);
                let rx = gamepad.get(GamepadAxis::RightStickX).unwrap_or(0.0);
                let ry = gamepad.get(GamepadAxis::RightStickY).unwrap_or(0.0);
                ui.label(format!("L stick: ({lx:.3}, {ly:.3})"));
                ui.label(format!("R stick: ({rx:.3}, {ry:.3})"));

                let buttons = [
                    ("South", GamepadButton::South),
                    ("East", GamepadButton::East),
                    ("West", GamepadButton::West),
                    ("North", GamepadButton::North),
                    ("LTrigger", GamepadButton::LeftTrigger),
                    ("RTrigger", GamepadButton::RightTrigger),
                    ("LTrigger2", GamepadButton::LeftTrigger2),
                    ("RTrigger2", GamepadButton::RightTrigger2),
                    ("DPadUp", GamepadButton::DPadUp),
                    ("DPadDown", GamepadButton::DPadDown),
                    ("DPadLeft", GamepadButton::DPadLeft),
                    ("DPadRight", GamepadButton::DPadRight),
                ];
                let pressed: Vec<&str> = buttons
                    .iter()
                    .filter(|(_, btn)| gamepad.pressed(*btn))
                    .map(|(name, _)| *name)
                    .collect();
                if pressed.is_empty() {
                    ui.label("Buttons: (none)");
                } else {
                    ui.label(format!("Buttons: {}", pressed.join(", ")));
                }
                ui.separator();
            }
        });
    });
}
