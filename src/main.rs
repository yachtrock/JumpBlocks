mod camera;
mod debug_ui;
mod edge_detection;
mod layers;
mod native_gamepad;
mod network;
mod player;
mod player_state;
mod world;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::TnuaAvian3dPlugin;
use clap::Parser;
use crossbeam_channel::{Receiver, Sender, unbounded};
use jumpblocks_ui::{Canvas, UiInputState, UiPlugin};
use jumpblocks_ui::thread::UiDrawFn;
use std::net::SocketAddr;

use network::{NetworkPlugin, NetworkRole, DEFAULT_PORT};
use player::ControlScheme;

// ---------------------------------------------------------------------------
// Game ↔ UI data types (plain Rust, no ECS)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct InventoryItem {
    name: String,
    color: [f32; 4],
    description: String,
}

#[derive(Clone, Debug, Default)]
struct GameUiData {
    inventory_open: bool,
    items: Vec<InventoryItem>,
    health: f32,
    max_health: f32,
}

#[derive(Clone, Debug)]
enum GameUiEvent {
    InventoryClosed,
    ItemSelected(usize),
}

// ---------------------------------------------------------------------------
// Bevy resources for the game ↔ UI channels
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct UiDataTx(Sender<GameUiData>);

#[derive(Resource)]
struct UiEventRx(Receiver<GameUiEvent>);

/// When true, player movement systems are blocked.
#[derive(Resource, Default)]
pub struct UiInputBlock(pub bool);

/// Tracks whether the inventory is currently open on the game side.
#[derive(Resource, Default)]
struct InventoryState {
    open: bool,
    items: Vec<InventoryItem>,
}

// ---------------------------------------------------------------------------
// GameUi — the UiDrawFn implementation with mutable state
// ---------------------------------------------------------------------------

struct GameUi {
    data_rx: Receiver<GameUiData>,
    event_tx: Sender<GameUiEvent>,
    data: GameUiData,
    // UI-thread-only state
    selected_slot: usize,
    cursor_pulse: f32,
}

impl UiDrawFn for GameUi {
    fn draw(&mut self, input: &UiInputState, canvas: &mut Canvas) {
        // Drain latest game data
        while let Ok(data) = self.data_rx.try_recv() {
            self.data = data;
        }

        let win = canvas.window_size();

        // --- Always draw HUD ---
        draw_hud(canvas, &self.data, win);

        // --- Inventory overlay ---
        if self.data.inventory_open {
            self.cursor_pulse += 0.08;
            let items = &self.data.items;
            let cols = 4usize;
            let rows = (items.len() + cols - 1) / cols;

            // Navigation
            if input.key_just_pressed(KeyCode::ArrowRight) {
                self.selected_slot = (self.selected_slot + 1).min(items.len().saturating_sub(1));
            }
            if input.key_just_pressed(KeyCode::ArrowLeft) {
                self.selected_slot = self.selected_slot.saturating_sub(1);
            }
            if input.key_just_pressed(KeyCode::ArrowDown) {
                let next = self.selected_slot + cols;
                if next < items.len() {
                    self.selected_slot = next;
                }
            }
            if input.key_just_pressed(KeyCode::ArrowUp) {
                if self.selected_slot >= cols {
                    self.selected_slot -= cols;
                }
            }
            if input.key_just_pressed(KeyCode::Enter) {
                let _ = self.event_tx.send(GameUiEvent::ItemSelected(self.selected_slot));
            }
            if input.key_just_pressed(KeyCode::Escape) {
                let _ = self.event_tx.send(GameUiEvent::InventoryClosed);
            }

            // Layout
            let panel_w = 480.0f32;
            let panel_h = 60.0 + rows as f32 * 80.0 + 80.0; // title + grid + detail
            let panel_x = (win.x - panel_w) * 0.5;
            let panel_y = (win.y - panel_h) * 0.5;

            // Dim background
            canvas.rect(0.0, 0.0, win.x, win.y, [0.0, 0.0, 0.0, 0.5]);

            // Panel background
            canvas.rect(panel_x, panel_y, panel_w, panel_h, [0.12, 0.12, 0.18, 0.95]);
            // Panel border
            canvas.rect(panel_x, panel_y, panel_w, 2.0, [0.4, 0.4, 0.6, 1.0]);
            canvas.rect(panel_x, panel_y + panel_h - 2.0, panel_w, 2.0, [0.4, 0.4, 0.6, 1.0]);
            canvas.rect(panel_x, panel_y, 2.0, panel_h, [0.4, 0.4, 0.6, 1.0]);
            canvas.rect(panel_x + panel_w - 2.0, panel_y, 2.0, panel_h, [0.4, 0.4, 0.6, 1.0]);

            // Title
            canvas.text(
                panel_x + 16.0, panel_y + 16.0,
                "INVENTORY", 22.0,
                [0.9, 0.85, 0.6, 1.0],
            );
            canvas.text(
                panel_x + panel_w - 160.0, panel_y + 20.0,
                "[ESC] Close", 13.0,
                [0.6, 0.6, 0.6, 1.0],
            );

            // Item grid
            let grid_x = panel_x + 16.0;
            let grid_y = panel_y + 52.0;
            let cell_w = (panel_w - 32.0 - (cols as f32 - 1.0) * 8.0) / cols as f32;
            let cell_h = 72.0;

            for (i, item) in items.iter().enumerate() {
                let col = i % cols;
                let row = i / cols;
                let cx = grid_x + col as f32 * (cell_w + 8.0);
                let cy = grid_y + row as f32 * (cell_h + 8.0);

                // Cell background
                let is_selected = i == self.selected_slot;
                let bg = if is_selected {
                    let pulse = (self.cursor_pulse.sin() * 0.15 + 0.35).clamp(0.2, 0.5);
                    [0.3, 0.3, pulse + 0.3, 0.9]
                } else {
                    [0.18, 0.18, 0.24, 0.8]
                };
                canvas.rect(cx, cy, cell_w, cell_h, bg);

                // Selection border
                if is_selected {
                    let b = (self.cursor_pulse.sin() * 0.3 + 0.7).clamp(0.4, 1.0);
                    canvas.rect(cx, cy, cell_w, 2.0, [0.5, 0.5, b, 1.0]);
                    canvas.rect(cx, cy + cell_h - 2.0, cell_w, 2.0, [0.5, 0.5, b, 1.0]);
                    canvas.rect(cx, cy, 2.0, cell_h, [0.5, 0.5, b, 1.0]);
                    canvas.rect(cx + cell_w - 2.0, cy, 2.0, cell_h, [0.5, 0.5, b, 1.0]);
                }

                // Item icon (colored rect)
                canvas.rect(cx + 8.0, cy + 8.0, 32.0, 32.0, item.color);

                // Item name
                canvas.text(cx + 8.0, cy + 46.0, &item.name, 12.0, [0.9, 0.9, 0.9, 1.0]);
            }

            // Detail panel for selected item
            if let Some(item) = items.get(self.selected_slot) {
                let detail_y = grid_y + rows as f32 * (cell_h + 8.0) + 8.0;
                canvas.rect(grid_x, detail_y, panel_w - 32.0, 1.0, [0.4, 0.4, 0.5, 0.6]);
                canvas.text(
                    grid_x, detail_y + 10.0,
                    &item.name, 16.0,
                    [1.0, 1.0, 1.0, 1.0],
                );
                canvas.text(
                    grid_x, detail_y + 32.0,
                    &item.description, 13.0,
                    [0.7, 0.7, 0.7, 1.0],
                );
            }
        }
    }
}

fn draw_hud(canvas: &mut Canvas, data: &GameUiData, win: bevy::math::Vec2) {
    let pct = if data.max_health > 0.0 { data.health / data.max_health } else { 0.0 };

    // Health bar background
    canvas.rect(20.0, 20.0, 200.0, 24.0, [0.1, 0.1, 0.1, 0.8]);
    // Health bar fill
    canvas.rect(22.0, 22.0, 196.0 * pct, 20.0, [0.2, 0.9, 0.3, 1.0]);
    // Health text
    let hp_text = format!("HP: {} / {}", data.health as i32, data.max_health as i32);
    canvas.text(28.0, 23.0, &hp_text, 16.0, [1.0, 1.0, 1.0, 1.0]);

    // Bottom-right label
    canvas.text(win.x - 140.0, win.y - 30.0, "JumpBlocks", 14.0, [1.0, 1.0, 1.0, 0.5]);

    // Inventory hint
    if !data.inventory_open {
        canvas.text(20.0, 54.0, "[TAB] Inventory", 12.0, [0.7, 0.7, 0.7, 0.6]);
    }
}

// ---------------------------------------------------------------------------
// Dummy inventory data
// ---------------------------------------------------------------------------

fn make_dummy_items() -> Vec<InventoryItem> {
    vec![
        InventoryItem { name: "Iron Sword".into(), color: [0.7, 0.7, 0.8, 1.0], description: "A sturdy blade forged in the mountain furnaces.".into() },
        InventoryItem { name: "Health Potion".into(), color: [0.9, 0.2, 0.2, 1.0], description: "Restores 50 health points. Tastes like cherries.".into() },
        InventoryItem { name: "Oak Shield".into(), color: [0.6, 0.4, 0.2, 1.0], description: "Blocks incoming attacks. Smells like the forest.".into() },
        InventoryItem { name: "Speed Boots".into(), color: [0.2, 0.8, 0.9, 1.0], description: "Move 30% faster. The laces glow faintly.".into() },
        InventoryItem { name: "Fire Gem".into(), color: [1.0, 0.5, 0.1, 1.0], description: "A warm gem that pulses with elemental energy.".into() },
        InventoryItem { name: "Shadow Cloak".into(), color: [0.2, 0.15, 0.3, 1.0], description: "Grants brief invisibility when standing still.".into() },
        InventoryItem { name: "Golden Key".into(), color: [0.95, 0.85, 0.2, 1.0], description: "Opens the locked door in the ancient ruins.".into() },
        InventoryItem { name: "Mana Crystal".into(), color: [0.4, 0.3, 0.9, 1.0], description: "Restores 30 mana. Hums when held.".into() },
        InventoryItem { name: "Bomb".into(), color: [0.3, 0.3, 0.3, 1.0], description: "Destroys weak walls. Handle with care.".into() },
        InventoryItem { name: "Lucky Charm".into(), color: [0.1, 0.9, 0.4, 1.0], description: "Increases rare drop chance by 10%.".into() },
    ]
}

// ---------------------------------------------------------------------------
// Game systems
// ---------------------------------------------------------------------------

fn toggle_inventory(
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut inv_state: ResMut<InventoryState>,
    mut input_block: ResMut<UiInputBlock>,
    data_tx: Res<UiDataTx>,
    mut cursor_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let mut toggle = keyboard.just_pressed(KeyCode::Tab);
    for gamepad in gamepads.iter() {
        if gamepad.just_pressed(GamepadButton::North) {
            toggle = true;
        }
    }

    if toggle && !inv_state.open {
        inv_state.open = true;
        input_block.0 = true;

        // Show cursor
        if let Ok(mut cursor) = cursor_query.single_mut() {
            cursor.grab_mode = CursorGrabMode::None;
            cursor.visible = true;
        }

        // Send updated data to UI thread
        let _ = data_tx.0.send(GameUiData {
            inventory_open: true,
            items: inv_state.items.clone(),
            health: 75.0,
            max_health: 100.0,
        });
    }
}

fn handle_ui_events(
    event_rx: Res<UiEventRx>,
    mut inv_state: ResMut<InventoryState>,
    mut input_block: ResMut<UiInputBlock>,
    data_tx: Res<UiDataTx>,
    mut cursor_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    while let Ok(event) = event_rx.0.try_recv() {
        match event {
            GameUiEvent::InventoryClosed => {
                inv_state.open = false;
                input_block.0 = false;

                // Hide cursor, lock for gameplay
                if let Ok(mut cursor) = cursor_query.single_mut() {
                    cursor.grab_mode = CursorGrabMode::Locked;
                    cursor.visible = false;
                }

                // Tell UI thread inventory is closed
                let _ = data_tx.0.send(GameUiData {
                    inventory_open: false,
                    items: inv_state.items.clone(),
                    health: 75.0,
                    max_health: 100.0,
                });
            }
            GameUiEvent::ItemSelected(slot) => {
                if let Some(item) = inv_state.items.get(slot) {
                    info!("Selected item [{}]: {}", slot, item.name);
                }
            }
        }
    }
}

/// Send initial data to UI thread on startup (after 1 frame so window exists).
fn send_initial_ui_data(data_tx: Res<UiDataTx>, inv_state: Res<InventoryState>) {
    let _ = data_tx.0.send(GameUiData {
        inventory_open: false,
        items: inv_state.items.clone(),
        health: 75.0,
        max_health: 100.0,
    });
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// JumpBlocks — a multiplayer 3D platformer.
#[derive(Parser, Debug)]
#[command(name = "JumpBlocks")]
struct Cli {
    /// Run as a dedicated server (headless, no window).
    #[arg(long)]
    start_server: bool,

    /// Connect to a server at this address (e.g. 192.168.1.5:5000).
    #[arg(long)]
    connect: Option<String>,

    /// Port for the server to listen on (default: 5000).
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,

    /// Run without a window (headless mode for testing).
    #[arg(long)]
    headless: bool,
}

// ---------------------------------------------------------------------------
// App setup
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let role = if cli.start_server {
        NetworkRole::DedicatedServer
    } else if let Some(ref addr_str) = cli.connect {
        let server_addr: SocketAddr = addr_str
            .parse()
            .unwrap_or_else(|_| {
                // Try adding default port if only IP was given
                format!("{}:{}", addr_str, DEFAULT_PORT)
                    .parse()
                    .expect("Invalid server address. Use format: IP:PORT (e.g. 192.168.1.5:5000)")
            });
        NetworkRole::Client { server_addr }
    } else {
        NetworkRole::ListenServer
    };

    let is_headless = role == NetworkRole::DedicatedServer || cli.headless;

    // Create game ↔ UI channels (even in headless, resources must exist for systems)
    let (data_tx, data_rx) = unbounded::<GameUiData>();
    let (event_tx, event_rx) = unbounded::<GameUiEvent>();

    let game_ui = GameUi {
        data_rx,
        event_tx,
        data: GameUiData::default(),
        selected_slot: 0,
        cursor_pulse: 0.0,
    };

    let items = make_dummy_items();

    let mut app = App::new();

    // Insert network role before plugins
    app.insert_resource(role.clone());

    if is_headless {
        // Headless: full plugins but no window, exit via ScheduleRunnerPlugin
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    ..default()
                })
                .build()
                .disable::<bevy::gilrs::GilrsPlugin>(),
        );
        app.add_plugins(
            PhysicsPlugins::default()
                .set(PhysicsInterpolationPlugin::interpolate_all()),
        );
        app.add_plugins(TnuaControllerPlugin::<ControlScheme>::new(PhysicsSchedule));
        app.add_plugins(TnuaAvian3dPlugin::new(PhysicsSchedule));
    } else {
        // Client or listen server: full rendering
        app.add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: match &role {
                            NetworkRole::ListenServer => "JumpBlocks (Host)".to_string(),
                            NetworkRole::Client { server_addr } => {
                                format!("JumpBlocks ({})", server_addr)
                            }
                            _ => "JumpBlocks".to_string(),
                        },
                        ..default()
                    }),
                    ..default()
                })
                .build()
                .disable::<bevy::gilrs::GilrsPlugin>(),
            PhysicsPlugins::default()
                .set(PhysicsInterpolationPlugin::interpolate_all()),
            TnuaControllerPlugin::<ControlScheme>::new(PhysicsSchedule),
            TnuaAvian3dPlugin::new(PhysicsSchedule),
        ));
    }

    // Insert UI resources (needed even in headless for system queries)
    app.insert_resource(UiDataTx(data_tx));
    app.insert_resource(UiEventRx(event_rx));
    app.insert_resource(UiInputBlock::default());
    app.insert_resource(InventoryState { open: false, items });

    // Always add world (platforms need to exist on server too for physics)
    app.add_plugins(world::WorldPlugin);

    // Add networking
    app.add_plugins(NetworkPlugin);

    if !is_headless {
        // Client-side plugins (rendering available)
        app.add_plugins((
            jumpblocks_voxel::VoxelPlugin,
            player::PlayerPlugin,
            player_state::PlayerStatePlugin,
            camera::CameraPlugin,
            debug_ui::DebugUiPlugin,
            edge_detection::EdgeDetectionPlugin,
            native_gamepad::NativeGamepadPlugin,
            UiPlugin::new(game_ui),
        ));
        app.add_systems(Startup, send_initial_ui_data);
        app.add_systems(PreUpdate, toggle_inventory);
        app.add_systems(PostUpdate, handle_ui_events);
    } else if role != NetworkRole::DedicatedServer {
        // Headless client: spawn a player without visuals for network testing
        app.add_plugins(player::HeadlessPlayerPlugin);
    }

    app.run();
}
