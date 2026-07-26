//! The building interface: material hotbar, shape picker, tools, esc menu.
//!
//! The game side (this module) owns all interface state in [`BuildState`]:
//! which material slot and shape are selected, which tool is active, and
//! which menus are open. The UI thread renders it from the `build` submap of
//! the game data (see `sync_build_snapshot`) and sends interactions back as
//! `GameUiEvent::Build` events, which `main::handle_ui_events` forwards into
//! [`BuildEventQueue`].
//!
//! Buildable areas: the player only gets the hotbar (and the auto-entered
//! `building` action state) inside a [`BuildAreas`] AABB. One test area is
//! stamped next to the spawn point, snapped inside a single chunk so the
//! auto-shape tool's chunk-local reshaping always has room to work.

use bevy::prelude::*;
use jumpblocks_voxel::coords::CHUNK_WORLD_SIZE;
use jumpblocks_voxel::shape::{cap_corner_heights, ShapeTable, SHAPE_CUBE};
use jumpblocks_voxel::worldgen::{
    TEX_BLUE, TEX_LIME, TEX_ORANGE, TEX_PINK, TEX_PURPLE, TEX_RED, TEX_TEAL, TEX_WHITE,
    TEX_YELLOW,
};

use crate::action_state::{ActionState, EnterActionStateQueue, EnterActionStateRequest, StateValue};
use crate::building::BuildingApi;
use crate::camera::OrbitCamera;
use crate::challenge::{ActiveChallenge, GoalMarker, HudMessages, WorldContent};
use crate::player::Player;
use crate::world::SpawnPoint;
use crate::{BuildUiData, InventoryState, MaterialSlotData, ShapeInfoData, UiInputBlock};

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// One hotbar slot: a material (colored block) with a remaining count.
#[derive(Clone, Debug)]
pub struct MaterialSlot {
    pub name: String,
    pub texture: u16,
    /// UI display color (sRGB, matches the worldgen palette).
    pub ui_color: [f32; 4],
    pub count: u32,
}

/// The two building tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BuildTool {
    /// Place the selected shape directly.
    #[default]
    Direct,
    /// Place cubes; the surrounding terrain is re-capped to match the contour.
    AutoShape,
}

/// All interface state for the building UI. Game-side source of truth.
#[derive(Resource, Debug, Default)]
pub struct BuildState {
    pub in_area: bool,
    pub selected_slot: usize,
    pub materials: Vec<MaterialSlot>,
    pub shape_selected: usize,
    pub shape_menu_open: bool,
    pub esc_menu_open: bool,
    pub esc_selected: usize,
    pub tool: BuildTool,
}

/// A placeable shape with its material cost.
#[derive(Clone, Debug)]
pub struct ShapeInfo {
    pub id: u16,
    pub name: String,
    pub cost: u32,
}

/// Catalog of shapes the player can place (the player always has all of them).
#[derive(Resource, Debug, Default)]
pub struct ShapeCatalog(pub Vec<ShapeInfo>);

impl ShapeCatalog {
    pub fn cost_of(&self, shape: u16) -> u32 {
        self.0
            .iter()
            .find(|s| s.id == shape)
            .map(|s| s.cost)
            .unwrap_or(1)
    }
}

/// World-space AABBs where the building interface is active.
#[derive(Resource, Debug, Default)]
pub struct BuildAreas(pub Vec<(Vec3, Vec3)>);

impl BuildAreas {
    pub fn contains(&self, p: Vec3) -> bool {
        self.0.iter().any(|(min, max)| {
            p.x >= min.x && p.x <= max.x
                && p.y >= min.y && p.y <= max.y
                && p.z >= min.z && p.z <= max.z
        })
    }
}

/// Build UI interactions forwarded from the UI thread (kind, value).
#[derive(Resource, Debug, Default)]
pub struct BuildEventQueue(pub Vec<(String, i64)>);

/// Snapshot of [`BuildState`] shaped for the UI thread, refreshed each frame.
#[derive(Resource, Debug, Default)]
pub struct BuildUiSnapshot(pub BuildUiData);

// ---------------------------------------------------------------------------
// Shape catalog & material construction
// ---------------------------------------------------------------------------

/// Material volume of a shape in cells³ — cost is proportional to how much
/// "stuff" the shape contains, so bulkier shapes cost more.
fn shape_volume_cells(shape: u16) -> f32 {
    if shape == SHAPE_CUBE {
        // 2×1×2 cells.
        return 4.0;
    }
    match cap_corner_heights(shape) {
        Some([h00, h10, h01, h11]) => {
            // Volume under the top surface: two triangles folded along the
            // (0,0)→(2,2) diagonal, each covering half the 2×2 footprint.
            let (h00, h10, h01, h11) = (h00 as f32, h10 as f32, h01 as f32, h11 as f32);
            (h00 + h10 + h11) / 3.0 * 2.0 + (h00 + h11 + h01) / 3.0 * 2.0
        }
        None => 4.0,
    }
}

fn build_shape_catalog(shapes: &ShapeTable) -> ShapeCatalog {
    let display_name = |raw: &str| -> String {
        match raw {
            "cube" => "Cube",
            "wedge" => "Wedge",
            "wedge_outer" => "Corner (out)",
            "wedge_inner" => "Corner (in)",
            "wedge_steep" => "Steep Wedge",
            "wedge_steep_outer" => "Steep Out",
            "wedge_steep_inner" => "Steep In",
            "wedge_diag" => "Diagonal",
            "wedge_steep_diag" => "Steep Diag",
            other => other,
        }
        .to_string()
    };

    let infos = shapes
        .shapes
        .iter()
        .enumerate()
        .map(|(id, s)| ShapeInfo {
            id: id as u16,
            name: display_name(&s.name),
            cost: (shape_volume_cells(id as u16) / 2.0).ceil() as u32,
        })
        .collect();
    ShapeCatalog(infos)
}

fn build_materials() -> Vec<MaterialSlot> {
    // sRGB display colors matching `worldgen::texture_color`'s palette.
    let defs: [(&str, u16, [f32; 3]); 9] = [
        ("White", TEX_WHITE, [0.93, 0.93, 0.90]),
        ("Red", TEX_RED, [0.90, 0.26, 0.21]),
        ("Orange", TEX_ORANGE, [0.98, 0.58, 0.16]),
        ("Yellow", TEX_YELLOW, [0.99, 0.85, 0.21]),
        ("Lime", TEX_LIME, [0.55, 0.87, 0.24]),
        ("Teal", TEX_TEAL, [0.19, 0.78, 0.74]),
        ("Blue", TEX_BLUE, [0.26, 0.52, 0.95]),
        ("Purple", TEX_PURPLE, [0.61, 0.35, 0.90]),
        ("Pink", TEX_PINK, [0.96, 0.48, 0.78]),
    ];
    defs.iter()
        .map(|(name, tex, c)| MaterialSlot {
            name: name.to_string(),
            texture: *tex,
            ui_color: [c[0], c[1], c[2], 1.0],
            count: 250,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Startup: state + the buildable test area near spawn
// ---------------------------------------------------------------------------

fn setup_build_state(
    mut commands: Commands,
    spawn: Res<SpawnPoint>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let shapes = ShapeTable::default();
    commands.insert_resource(build_shape_catalog(&shapes));
    commands.insert_resource(BuildState {
        materials: build_materials(),
        ..Default::default()
    });

    // Buildable test area: the 2×2 block of chunks around the spawn point,
    // picked so spawn sits centrally — the player can build in any direction
    // the moment they load in. Chunk-aligned so the auto-shape tool (whose
    // contour pass is chunk-local) has room to work away from the seams.
    let s = spawn.0;
    let base_x = (s.x / CHUNK_WORLD_SIZE).floor() * CHUNK_WORLD_SIZE;
    let base_z = (s.z / CHUNK_WORLD_SIZE).floor() * CHUNK_WORLD_SIZE;
    let min_x = if s.x - base_x < CHUNK_WORLD_SIZE * 0.5 { base_x - CHUNK_WORLD_SIZE } else { base_x };
    let min_z = if s.z - base_z < CHUNK_WORLD_SIZE * 0.5 { base_z - CHUNK_WORLD_SIZE } else { base_z };
    let min = Vec3::new(min_x, s.y - 8.0, min_z);
    let max = Vec3::new(
        min_x + CHUNK_WORLD_SIZE * 2.0,
        s.y + 12.0,
        min_z + CHUNK_WORLD_SIZE * 2.0,
    );
    info!(
        "[build_ui] Buildable area {:?}..{:?} (spawn {:?})",
        min, max, s
    );
    commands.insert_resource(BuildAreas(vec![(min, max)]));

    // --- Visual markers: translucent ground sheet + corner posts ---------
    let cx = (min.x + max.x) * 0.5;
    let cz = (min.z + max.z) * 0.5;
    let w = max.x - min.x;
    let d = max.z - min.z;

    let sheet_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.25, 0.85, 0.75, 0.10),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(w, 0.04, d))),
        MeshMaterial3d(sheet_mat),
        Transform::from_translation(Vec3::new(cx, s.y + 0.06, cz)),
    ));

    let post_mesh = meshes.add(Cuboid::new(0.16, 2.4, 0.16));
    let post_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.95, 0.92),
        emissive: LinearRgba::new(0.08, 0.5, 0.42, 1.0),
        ..default()
    });
    for (px, pz) in [
        (min.x, min.z),
        (max.x, min.z),
        (min.x, max.z),
        (max.x, max.z),
    ] {
        commands.spawn((
            Mesh3d(post_mesh.clone()),
            MeshMaterial3d(post_mat.clone()),
            Transform::from_translation(Vec3::new(px, s.y + 1.2, pz)),
        ));
    }
}

// ---------------------------------------------------------------------------
// Per-frame: area detection + auto enter/exit of the building action state
// ---------------------------------------------------------------------------

fn update_build_area(
    areas: Option<Res<BuildAreas>>,
    mut state: ResMut<BuildState>,
    inv_state: Res<InventoryState>,
    building_api: Res<BuildingApi>,
    players: Query<(&Transform, &ActionState), With<Player>>,
    mut enter_queue: ResMut<EnterActionStateQueue>,
) {
    let Some(areas) = areas else { return };
    let Ok((tf, action)) = players.single() else { return };

    let was_in_area = state.in_area;
    state.in_area = areas.contains(tf.translation);
    if state.in_area != was_in_area {
        info!("[build_ui] in_area: {}", state.in_area);
    }

    let menus_open = state.esc_menu_open || state.shape_menu_open || inv_state.open;

    // Auto-enter the building action state inside the area, so the preview
    // and place bindings are live the moment the hotbar shows.
    if state.in_area && !menus_open && action.0.is_none() {
        let mut context = std::collections::HashMap::new();
        let name = state
            .materials
            .get(state.selected_slot)
            .map(|m| m.name.clone())
            .unwrap_or_default();
        context.insert("item_name".to_string(), StateValue::Str(name));
        context.insert(
            "selected_slot".to_string(),
            StateValue::Int(state.selected_slot as i64),
        );
        enter_queue.0.push(EnterActionStateRequest {
            state_name: "building".to_string(),
            context,
        });
    }

    // Push the current selection into the building API so the action-state
    // script (and placement/preview systems) see it this frame.
    let auto = state.tool == BuildTool::AutoShape;
    let texture = state
        .materials
        .get(state.selected_slot)
        .map(|m| m.texture)
        .unwrap_or(TEX_WHITE);
    let shape = state
        .shape_selected
        .min(8) as u16;
    building_api.set_selection(shape, texture, auto, state.in_area && !menus_open);
}

// ---------------------------------------------------------------------------
// Input: esc menu, hotbar slots, shape menu, tool toggle
// ---------------------------------------------------------------------------

const ESC_ITEMS: usize = 2; // 0 = Resume, 1 = Cancel Challenge
const SHAPE_MENU_COLS: usize = 3;

fn build_input_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut state: ResMut<BuildState>,
    inv_state: Res<InventoryState>,
    catalog: Option<Res<ShapeCatalog>>,
    mut queue: ResMut<BuildEventQueue>,
) {
    let mut gp_start = false;
    let mut gp_south = false;
    let mut gp_up = false;
    let mut gp_down = false;
    for gp in gamepads.iter() {
        gp_start |= gp.just_pressed(GamepadButton::Start);
        gp_south |= gp.just_pressed(GamepadButton::South);
        gp_up |= gp.just_pressed(GamepadButton::DPadUp);
        gp_down |= gp.just_pressed(GamepadButton::DPadDown);
    }

    // --- Esc: close shape menu first, otherwise toggle the pause menu ----
    // (When the inventory is open the inventory script owns Escape.)
    if (keyboard.just_pressed(KeyCode::Escape) || gp_start) && !inv_state.open {
        if state.shape_menu_open {
            state.shape_menu_open = false;
        } else {
            state.esc_menu_open = !state.esc_menu_open;
            state.esc_selected = 0;
        }
        return;
    }

    // --- Esc menu navigation --------------------------------------------
    if state.esc_menu_open {
        if keyboard.just_pressed(KeyCode::ArrowDown) || gp_down {
            state.esc_selected = (state.esc_selected + 1) % ESC_ITEMS;
        }
        if keyboard.just_pressed(KeyCode::ArrowUp) || gp_up {
            state.esc_selected = (state.esc_selected + ESC_ITEMS - 1) % ESC_ITEMS;
        }
        if keyboard.just_pressed(KeyCode::Enter)
            || keyboard.just_pressed(KeyCode::Space)
            || gp_south
        {
            let kind = if state.esc_selected == 0 {
                "esc_resume"
            } else {
                "esc_cancel"
            };
            queue.0.push((kind.to_string(), 0));
        }
        return;
    }

    // --- Shape menu navigation ------------------------------------------
    if state.shape_menu_open {
        let count = catalog.as_ref().map(|c| c.0.len()).unwrap_or(9).max(1);
        if keyboard.just_pressed(KeyCode::KeyR) {
            state.shape_menu_open = false;
        }
        if keyboard.just_pressed(KeyCode::ArrowRight) {
            state.shape_selected = (state.shape_selected + 1) % count;
        }
        if keyboard.just_pressed(KeyCode::ArrowLeft) {
            state.shape_selected = (state.shape_selected + count - 1) % count;
        }
        if keyboard.just_pressed(KeyCode::ArrowDown) {
            state.shape_selected = (state.shape_selected + SHAPE_MENU_COLS) % count;
        }
        if keyboard.just_pressed(KeyCode::ArrowUp) {
            state.shape_selected = (state.shape_selected + count - SHAPE_MENU_COLS) % count;
        }
        if keyboard.just_pressed(KeyCode::Enter) || gp_south {
            state.shape_menu_open = false;
        }
        return;
    }

    // --- In-area bindings (no menu open) --------------------------------
    if !state.in_area || inv_state.open {
        return;
    }

    let digits = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    for (i, key) in digits.iter().enumerate() {
        if keyboard.just_pressed(*key) && i < state.materials.len() {
            state.selected_slot = i;
        }
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        state.shape_menu_open = true;
    }
    if keyboard.just_pressed(KeyCode::KeyV) {
        state.tool = match state.tool {
            BuildTool::Direct => BuildTool::AutoShape,
            BuildTool::AutoShape => BuildTool::Direct,
        };
    }
}

// ---------------------------------------------------------------------------
// UI events from the UI thread
// ---------------------------------------------------------------------------

fn handle_build_events(
    mut commands: Commands,
    mut queue: ResMut<BuildEventQueue>,
    mut state: ResMut<BuildState>,
    mut active: ResMut<ActiveChallenge>,
    goals: Query<Entity, With<GoalMarker>>,
    mut messages: ResMut<HudMessages>,
    content: Option<Res<WorldContent>>,
) {
    for (kind, value) in queue.0.drain(..) {
        match kind.as_str() {
            "slot" => {
                let idx = value.max(0) as usize;
                if idx < state.materials.len() {
                    state.selected_slot = idx;
                }
            }
            "shape" => {
                state.shape_selected = value.max(0) as usize;
                state.shape_menu_open = false;
            }
            "tool" => {
                state.tool = if value == 1 {
                    BuildTool::AutoShape
                } else {
                    BuildTool::Direct
                };
            }
            "shape_menu_closed" => {
                state.shape_menu_open = false;
            }
            "esc_resume" => {
                state.esc_menu_open = false;
            }
            "esc_cancel" => {
                // Cancel in place (no teleport — warping onto the armed start
                // pad would instantly restart the run).
                if let Some(run) = active.0.take() {
                    for goal in goals.iter() {
                        commands.entity(goal).despawn();
                    }
                    if let Some(ref content) = content {
                        let def = &content.def.challenges[run.challenge];
                        messages.push(format!("Challenge abandoned: {}.", def.name));
                    } else {
                        let _ = run;
                        messages.push("Challenge abandoned.");
                    }
                }
                state.esc_menu_open = false;
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Menu focus: cursor + movement blocking (transition-driven)
// ---------------------------------------------------------------------------

fn apply_menu_focus(
    state: Res<BuildState>,
    inv_state: Res<InventoryState>,
    mut input_block: ResMut<UiInputBlock>,
    mut cursor_query: Query<
        &mut bevy::window::CursorOptions,
        With<bevy::window::PrimaryWindow>,
    >,
    mut cameras: Query<&mut OrbitCamera>,
    mut last: Local<Option<bool>>,
) {
    let menus = state.esc_menu_open || state.shape_menu_open || inv_state.open;
    if *last == Some(menus) {
        return;
    }
    *last = Some(menus);

    input_block.0 = menus;
    if let Ok(mut cursor) = cursor_query.single_mut() {
        if menus {
            cursor.grab_mode = bevy::window::CursorGrabMode::None;
            cursor.visible = true;
        } else {
            cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
            cursor.visible = false;
        }
    }
    if let Ok(mut cam) = cameras.single_mut() {
        cam.cursor_locked = !menus;
    }
}

// ---------------------------------------------------------------------------
// UI snapshot
// ---------------------------------------------------------------------------

fn sync_build_snapshot(
    state: Res<BuildState>,
    catalog: Option<Res<ShapeCatalog>>,
    active: Option<Res<ActiveChallenge>>,
    content: Option<Res<WorldContent>>,
    mut snapshot: ResMut<BuildUiSnapshot>,
) {
    let (challenge_active, challenge_name) = match (&active, &content) {
        (Some(active), Some(content)) => match &active.0 {
            Some(run) => (
                true,
                content.def.challenges[run.challenge].name.clone(),
            ),
            None => (false, String::new()),
        },
        _ => (false, String::new()),
    };

    snapshot.0 = BuildUiData {
        in_area: state.in_area,
        materials: state
            .materials
            .iter()
            .map(|m| MaterialSlotData {
                name: m.name.clone(),
                color: m.ui_color,
                count: m.count,
            })
            .collect(),
        selected_slot: state.selected_slot,
        shapes: catalog
            .as_ref()
            .map(|c| {
                c.0.iter()
                    .map(|s| ShapeInfoData {
                        id: s.id,
                        name: s.name.clone(),
                        cost: s.cost,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        shape_selected: state.shape_selected,
        shape_menu_open: state.shape_menu_open,
        esc_menu_open: state.esc_menu_open,
        esc_selected: state.esc_selected,
        tool: match state.tool {
            BuildTool::Direct => "direct".to_string(),
            BuildTool::AutoShape => "auto".to_string(),
        },
        challenge_active,
        challenge_name,
    };
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct BuildUiPlugin;

impl Plugin for BuildUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BuildEventQueue>()
            .init_resource::<BuildUiSnapshot>()
            .init_resource::<BuildState>()
            .add_systems(
                Startup,
                setup_build_state.after(crate::world::setup_world),
            )
            .add_systems(
                Update,
                (
                    build_input_system,
                    handle_build_events.after(build_input_system),
                    update_build_area
                        .after(handle_build_events)
                        .before(crate::action_state::action_state_update),
                    apply_menu_focus.after(handle_build_events),
                    sync_build_snapshot
                        .after(handle_build_events)
                        .before(crate::sync_game_ui_data),
                ),
            );
    }
}
