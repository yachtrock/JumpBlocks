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
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::TnuaAvian3dPlugin;
use clap::Parser;
use std::net::SocketAddr;

use network::{NetworkPlugin, NetworkRole, DEFAULT_PORT};
use player::ControlScheme;

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
}

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

    let is_headless = role == NetworkRole::DedicatedServer;

    let mut app = App::new();

    // Insert network role before plugins
    app.insert_resource(role.clone());

    if is_headless {
        // Headless server: minimal plugins, no rendering
        app.add_plugins(MinimalPlugins);
        app.add_plugins(PhysicsPlugins::default());
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

    // Always add world (platforms need to exist on server too for physics)
    app.add_plugins(world::WorldPlugin);

    // Add networking
    app.add_plugins(NetworkPlugin);

    if !is_headless {
        // Client-side plugins
        app.add_plugins((
            jumpblocks_voxel::VoxelPlugin,
            player::PlayerPlugin,
            player_state::PlayerStatePlugin,
            camera::CameraPlugin,
            debug_ui::DebugUiPlugin,
            edge_detection::EdgeDetectionPlugin,
            native_gamepad::NativeGamepadPlugin,
        ));
    }

    app.run();
}
