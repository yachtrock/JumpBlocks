mod action_state;
mod building;
mod camera;
mod curtain;
mod debug_ui;
mod edge_detection;
mod layers;
mod native_gamepad;
mod network;
mod player;
mod player_state;
mod scripting;
mod world;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::TnuaAvian3dPlugin;
use clap::{CommandFactory, FromArgMatches, Parser};
use crossbeam_channel::{Receiver, Sender, unbounded};
use jumpblocks_ui::{Canvas, UiInputState, UiPlugin};
use jumpblocks_ui::thread::UiDrawFn;
use std::net::SocketAddr;

use network::{NetworkPlugin, NetworkRole, ServerPort, DEFAULT_PORT};
use player::ControlScheme;

// ---------------------------------------------------------------------------
// Game ↔ UI data types (plain Rust, no ECS)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ButtonHintData {
    label: String,
    keyboard: String,
    gamepad: String,
    keyboard2: Option<String>,
    gamepad2: Option<String>,
}

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
    button_hints: Vec<ButtonHintData>,
    input_mode: String,
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
// GameUi — Rhai-scripted UI with hot reload
// ---------------------------------------------------------------------------

struct GameUi {
    data_rx: Receiver<GameUiData>,
    event_tx: Sender<GameUiEvent>,
    data: GameUiData,
    script: scripting::ScriptEngine,
}

impl UiDrawFn for GameUi {
    fn draw(&mut self, input: &UiInputState, canvas: &mut Canvas) {
        // Drain latest game data
        while let Ok(data) = self.data_rx.try_recv() {
            self.data = data;
        }

        self.script
            .run_frame(input, &self.data, canvas, &self.event_tx);
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

fn toggle_wireframe(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut wireframe_config: ResMut<WireframeConfig>,
) {
    if keyboard.just_pressed(KeyCode::F2) {
        wireframe_config.global = !wireframe_config.global;
        info!("Wireframe: {}", if wireframe_config.global { "ON" } else { "OFF" });
    }
}

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
            button_hints: Vec::new(),
            input_mode: "keyboard".to_string(),
        });
    }
}

fn handle_ui_events(
    event_rx: Res<UiEventRx>,
    mut inv_state: ResMut<InventoryState>,
    mut input_block: ResMut<UiInputBlock>,
    data_tx: Res<UiDataTx>,
    mut cursor_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut enter_state_queue: ResMut<action_state::EnterActionStateQueue>,
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
                    button_hints: Vec::new(),
                    input_mode: "keyboard".to_string(),
                });
            }
            GameUiEvent::ItemSelected(slot) => {
                if let Some(item) = inv_state.items.get(slot) {
                    info!("Selected item [{}]: {}", slot, item.name);

                    // Enter building state with item context
                    let mut context = std::collections::HashMap::new();
                    context.insert(
                        "item_name".to_string(),
                        action_state::StateValue::Str(item.name.clone()),
                    );
                    context.insert(
                        "selected_slot".to_string(),
                        action_state::StateValue::Int(slot as i64),
                    );
                    enter_state_queue.0.push(action_state::EnterActionStateRequest {
                        state_name: "building".to_string(),
                        context,
                    });
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
        button_hints: Vec::new(),
        input_mode: "keyboard".to_string(),
    });
}

/// Sync button hints + input mode to the UI thread each frame.
fn sync_game_ui_data(
    data_tx: Res<UiDataTx>,
    inv_state: Res<InventoryState>,
    button_hints: Res<action_state::ButtonHints>,
    input_mode: Res<action_state::InputMode>,
) {
    let hints: Vec<ButtonHintData> = button_hints
        .0
        .iter()
        .map(|h| ButtonHintData {
            label: h.label.clone(),
            keyboard: h.keyboard.clone(),
            gamepad: h.gamepad.clone(),
            keyboard2: h.keyboard2.clone(),
            gamepad2: h.gamepad2.clone(),
        })
        .collect();

    let mode = match *input_mode {
        action_state::InputMode::Keyboard => "keyboard",
        action_state::InputMode::Gamepad => "gamepad",
    };

    let _ = data_tx.0.send(GameUiData {
        inventory_open: inv_state.open,
        items: inv_state.items.clone(),
        health: 75.0,
        max_health: 100.0,
        button_hints: hints,
        input_mode: mode.to_string(),
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
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).expect("Failed to parse CLI arguments");
    let port_explicit = matches.value_source("port") == Some(clap::parser::ValueSource::CommandLine);

    let role = if cli.start_server {
        NetworkRole::DedicatedServer
    } else if let Some(ref addr_str) = cli.connect {
        let server_addr: SocketAddr = addr_str
            .parse::<SocketAddr>()
            .unwrap_or_else(|_| {
                // Try adding default port if only IP was given
                format!("{}:{}", addr_str, DEFAULT_PORT)
                    .parse::<SocketAddr>()
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
        event_tx: event_tx.clone(),
        data: GameUiData::default(),
        script: scripting::ScriptEngine::new("assets/scripts"),
    };

    let items = make_dummy_items();

    let mut app = App::new();

    // Insert network role and server port config before plugins
    app.insert_resource(role.clone());
    app.insert_resource(ServerPort {
        port: cli.port,
        explicit: port_explicit,
    });

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
            action_state::ActionStatePlugin,
            building::BuildingPlugin,
            camera::CameraPlugin,
            curtain::CurtainPlugin,
            debug_ui::DebugUiPlugin,
            edge_detection::EdgeDetectionPlugin,
            native_gamepad::NativeGamepadPlugin,
            UiPlugin::new(game_ui),
            WireframePlugin::default(),
        ));
        app.add_systems(Update, toggle_wireframe);
        app.add_systems(Startup, send_initial_ui_data);
        app.add_systems(PreUpdate, toggle_inventory);
        app.add_systems(Update, sync_game_ui_data);
        app.add_systems(PostUpdate, handle_ui_events);
    } else if role != NetworkRole::DedicatedServer {
        // Headless client: spawn a player without visuals for network testing
        app.add_plugins(player::HeadlessPlayerPlugin);
    }

    app.run();
}
