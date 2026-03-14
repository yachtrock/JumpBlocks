use bevy::prelude::*;
use lightyear::prelude::*;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

use crate::player_state::PlayerState;

/// Default port for the game server.
pub const DEFAULT_PORT: u16 = 5000;
/// Protocol ID shared between client and server.
pub const PROTOCOL_ID: u64 = 0x4A554D50424C4B; // "JUMPBLK" in hex
/// Private key for netcode (zeroed for dev; replace for production).
pub const PRIVATE_KEY: [u8; 32] = [0u8; 32];
/// Fixed timestep rate.
pub const TICK_RATE: f64 = 60.0;

/// What role this instance plays in the network.
#[derive(Clone, Debug, PartialEq, Eq, Resource)]
pub enum NetworkRole {
    /// Listen server: runs both server and client, accepts remote connections.
    ListenServer,
    /// Dedicated server: headless, no local player.
    DedicatedServer,
    /// Client connecting to a remote server.
    Client { server_addr: SocketAddr },
}

/// Replicated component: the position of a networked player.
#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetworkedPosition(pub Vec3);

/// Replicated component: the rotation of a networked player's visual.
#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetworkedRotation(pub Quat);

/// Replicated component: the movement state of a networked player.
#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetworkedState(pub PlayerState);

/// Replicated component: associates a player entity with its owning client.
#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OwnedByClient(pub u64);

/// Marker for the local player (the one this instance controls).
#[derive(Component)]
pub struct LocalPlayer;

/// Marker for a remote player (replicated from another client).
#[derive(Component)]
pub struct RemotePlayer;

/// Plugin that registers the network protocol (components, channels, messages).
/// Must be added AFTER ClientPlugins/ServerPlugins.
pub struct ProtocolPlugin;

impl Plugin for ProtocolPlugin {
    fn build(&self, app: &mut App) {
        // Register replicated components
        app.register_component::<NetworkedPosition>();
        app.register_component::<NetworkedRotation>();
        app.register_component::<NetworkedState>();
        app.register_component::<OwnedByClient>();
    }
}

/// Resource to hold the server address for client startup.
#[derive(Resource)]
struct ClientServerAddr(SocketAddr);

/// Main networking plugin that sets up everything based on the NetworkRole.
pub struct NetworkPlugin;

impl Plugin for NetworkPlugin {
    fn build(&self, app: &mut App) {
        let role = app
            .world()
            .get_resource::<NetworkRole>()
            .expect("NetworkRole resource must be inserted before NetworkPlugin")
            .clone();

        let tick_duration = Duration::from_secs_f64(1.0 / TICK_RATE);

        match &role {
            NetworkRole::ListenServer => {
                app.add_plugins(server::ServerPlugins { tick_duration });
                app.add_plugins(client::ClientPlugins { tick_duration });
                app.add_plugins(ProtocolPlugin);
                app.add_systems(Startup, setup_listen_server);
                // When a remote client connects, lightyear spawns a LinkOf entity.
                // We must add ReplicationSender + MessageManager so replication works.
                app.add_observer(add_replication_sender_on_link);
                app.add_systems(Update, handle_new_client_connections);
                // ReplicationSender MUST exist before Replicate::to_server() is inserted,
                // because lightyear's on_insert hook queries for it and has no fallback
                // for SingleClient mode. Chain with ApplyDeferred to flush commands.
                app.add_systems(
                    Update,
                    (
                        add_replication_sender_on_client_connected,
                        ApplyDeferred,
                        init_local_player_networking,
                        sync_local_player_to_network,
                    )
                        .chain()
                        .run_if(any_with_component::<Connected>),
                );
                app.add_systems(Update, sync_network_to_remote_players);
                app.add_systems(Update, server_replicate_received_entities);
                #[cfg(debug_assertions)]
                app.add_systems(Update, debug_replication_state);
            }
            NetworkRole::DedicatedServer => {
                app.add_plugins(server::ServerPlugins { tick_duration });
                app.add_plugins(ProtocolPlugin);
                app.add_systems(Startup, setup_dedicated_server);
                app.add_observer(add_replication_sender_on_link);
                app.add_systems(Update, handle_new_client_connections);
                app.add_systems(Update, server_replicate_received_entities);
                #[cfg(debug_assertions)]
                app.add_systems(Update, debug_replication_state);
            }
            NetworkRole::Client { server_addr } => {
                app.insert_resource(ClientServerAddr(*server_addr));
                app.add_plugins(client::ClientPlugins { tick_duration });
                app.add_plugins(ProtocolPlugin);
                app.add_systems(Startup, setup_client);
                // ReplicationSender MUST exist before Replicate::to_server() is inserted,
                // because lightyear's on_insert hook queries for it and has no fallback
                // for SingleClient mode. Chain with ApplyDeferred to flush commands.
                app.add_systems(
                    Update,
                    (
                        add_replication_sender_on_client_connected,
                        ApplyDeferred,
                        init_local_player_networking,
                        sync_local_player_to_network,
                    )
                        .chain()
                        .run_if(any_with_component::<Connected>),
                );
                app.add_systems(Update, sync_network_to_remote_players);
                #[cfg(debug_assertions)]
                app.add_systems(Update, debug_replication_state);
            }
        }
    }
}

/// Spawns the server entity with netcode + UDP.
fn spawn_server(commands: &mut Commands) -> Entity {
    let server_addr = SocketAddr::new("0.0.0.0".parse().unwrap(), DEFAULT_PORT);

    let netcode = server::NetcodeServer::new(
        server::NetcodeConfig::default()
            .with_protocol_id(PROTOCOL_ID)
            .with_key(PRIVATE_KEY),
    );

    let server_entity = commands
        .spawn((
            Name::new("Server"),
            netcode,
            server::ServerUdpIo::default(),
            LocalAddr(server_addr),
        ))
        .id();

    // Start the server
    commands.trigger(server::Start {
        entity: server_entity,
    });

    server_entity
}

/// Sets up a listen server: server with UDP + a host client connected locally.
fn setup_listen_server(mut commands: Commands) {
    let server_entity = spawn_server(&mut commands);

    // Spawn a host client that connects to the local server.
    // The `Client` marker is required so lightyear's HostPlugin observer
    // can detect this entity and add `HostClient` + `Connected`.
    let client_entity = commands
        .spawn((
            Name::new("HostClient"),
            client::Client::default(),
            LinkOf {
                server: server_entity,
            },
        ))
        .id();

    // Connect the host client
    commands.trigger(client::Connect {
        entity: client_entity,
    });

    print_local_ip(DEFAULT_PORT);
}

/// Sets up a dedicated server (headless, no local client).
fn setup_dedicated_server(mut commands: Commands) {
    spawn_server(&mut commands);

    info!("Dedicated server started on port {}", DEFAULT_PORT);
    print_local_ip(DEFAULT_PORT);
}

/// Sets up a client connecting to a remote server.
fn setup_client(mut commands: Commands, addr: Res<ClientServerAddr>) {
    let server_addr = addr.0;

    let auth = Authentication::Manual {
        server_addr,
        client_id: rand_client_id(),
        private_key: PRIVATE_KEY,
        protocol_id: PROTOCOL_ID,
    };

    let netcode_client = client::NetcodeClient::new(auth, client::NetcodeConfig::default())
        .expect("Failed to create netcode client");

    let client_entity = commands
        .spawn((
            Name::new("Client"),
            client::Client::default(),
            netcode_client,
            UdpIo::default(),
            LocalAddr(SocketAddr::new("0.0.0.0".parse().unwrap(), 0)),
        ))
        .id();

    commands.trigger(client::Connect {
        entity: client_entity,
    });

    info!("Connecting to server at {}", server_addr);
}

/// Observer: when lightyear spawns a LinkOf entity for a new client connection on the server,
/// add ReplicationSender + ReplicationReceiver + MessageManager so the server can
/// send and receive replicated data to/from that client.
fn add_replication_sender_on_link(trigger: On<Add, LinkOf>, mut commands: Commands) {
    let entity = trigger.entity;
    info!(
        "Adding ReplicationSender + ReplicationReceiver to server link entity {:?}",
        entity
    );
    commands.entity(entity).insert((
        ReplicationSender::new(
            Duration::from_millis(100),
            SendUpdatesMode::SinceLastAck,
            false,
        ),
        ReplicationReceiver::default(),
        MessageManager::default(),
    ));
}

/// When the client entity becomes Connected, add ReplicationSender + ReplicationReceiver
/// + MessageManager so the client can send and receive replicated data.
fn add_replication_sender_on_client_connected(
    mut commands: Commands,
    query: Query<
        Entity,
        (
            With<Connected>,
            With<client::Client>,
            Without<ReplicationSender>,
        ),
    >,
) {
    for entity in query.iter() {
        info!(
            "Adding ReplicationSender + ReplicationReceiver to client entity {:?}",
            entity
        );
        commands.entity(entity).insert((
            ReplicationSender::new(
                Duration::from_millis(100),
                SendUpdatesMode::SinceLastAck,
                false,
            ),
            ReplicationReceiver::default(),
            MessageManager::default(),
        ));
    }
}

/// Log when clients connect.
fn handle_new_client_connections(query: Query<Entity, Added<Connected>>) {
    for entity in query.iter() {
        info!("Client connected: {:?}", entity);
    }
}

/// Sync the local player's transform and state to the networked components.
/// Split into two systems: one to initialize networking on the player, one to update.
fn init_local_player_networking(
    mut commands: Commands,
    local_players: Query<
        (Entity, &Transform, &PlayerState),
        (With<LocalPlayer>, Without<NetworkedPosition>),
    >,
    role: Res<NetworkRole>,
) {
    for (entity, transform, state) in local_players.iter() {
        // Choose the right replication mode based on our role
        let replicate = match *role {
            NetworkRole::ListenServer | NetworkRole::DedicatedServer => {
                Replicate::to_clients(NetworkTarget::All)
            }
            NetworkRole::Client { .. } => Replicate::to_server(),
        };
        info!(
            "Initializing network replication for local player {:?} at {:?} (role={:?})",
            entity, transform.translation, *role
        );
        commands.entity(entity).insert((
            NetworkedPosition(transform.translation),
            NetworkedRotation(Quat::IDENTITY),
            NetworkedState(*state),
            replicate,
        ));
    }
}

fn sync_local_player_to_network(
    mut commands: Commands,
    local_players: Query<
        (Entity, &Transform, &PlayerState),
        (With<LocalPlayer>, With<NetworkedPosition>, Changed<Transform>),
    >,
) {
    for (entity, transform, state) in local_players.iter() {
        commands.entity(entity).insert((
            NetworkedPosition(transform.translation),
            NetworkedState(*state),
        ));
    }
}

/// On the server, when we receive a replicated entity from a client, re-replicate it
/// to all other clients so everyone can see each other.
fn server_replicate_received_entities(
    mut commands: Commands,
    new_entities: Query<
        (Entity, Option<&NetworkedPosition>),
        (
            Added<Replicated>,
            Without<Replicate>,
        ),
    >,
) {
    for (entity, net_pos) in new_entities.iter() {
        info!(
            "Server received replicated entity {:?}, has NetworkedPosition: {}, pos: {:?}",
            entity,
            net_pos.is_some(),
            net_pos.map(|p| p.0),
        );
        commands
            .entity(entity)
            .insert(Replicate::to_clients(NetworkTarget::All));
    }
}

/// Periodic diagnostic system to log replication state (debug builds only).
#[cfg(debug_assertions)]
fn debug_replication_state(
    local_players: Query<(Entity, Option<&NetworkedPosition>, Option<&Replicate>, Has<Replicating>, Has<HasAuthority>), With<LocalPlayer>>,
    replicated: Query<(Entity, Option<&NetworkedPosition>), With<Replicated>>,
    remote_players: Query<Entity, With<RemotePlayer>>,
    connected: Query<Entity, With<Connected>>,
    senders: Query<(Entity, Has<ReplicationSender>, Has<MessageManager>), With<Connected>>,
    mut last_log: Local<Option<f64>>,
    time: Res<Time>,
) {
    let now = time.elapsed_secs_f64();
    let last = last_log.unwrap_or(0.0);
    if now - last < 3.0 {
        return;
    }
    *last_log = Some(now);

    let connected_count = connected.iter().count();
    let local_count = local_players.iter().count();
    let replicated_count = replicated.iter().count();
    let remote_count = remote_players.iter().count();

    debug!(
        "[NET DIAG] connected={}, local_players={}, replicated_entities={}, remote_players={}",
        connected_count, local_count, replicated_count, remote_count,
    );

    for (entity, has_sender, has_msg_mgr) in senders.iter() {
        debug!(
            "[NET DIAG]   Sender {:?}: has_replication_sender={}, has_message_manager={}",
            entity, has_sender, has_msg_mgr,
        );
    }

    for (entity, net_pos, replicate, replicating, has_authority) in local_players.iter() {
        debug!(
            "[NET DIAG]   LocalPlayer {:?}: has_net_pos={}, has_replicate={}, replicating={}, has_authority={}, pos={:?}",
            entity,
            net_pos.is_some(),
            replicate.is_some(),
            replicating,
            has_authority,
            net_pos.map(|p| p.0),
        );
    }

    for (entity, net_pos) in replicated.iter() {
        debug!(
            "[NET DIAG]   Replicated {:?}: pos={:?}",
            entity,
            net_pos.map(|p| p.0),
        );
    }
}

/// When we receive replicated NetworkedPosition from remote clients, update their visual.
fn sync_network_to_remote_players(
    mut commands: Commands,
    mut remote_query: Query<
        (
            Entity,
            &NetworkedPosition,
            &NetworkedState,
            Option<&mut Transform>,
        ),
        (With<Replicated>, Without<LocalPlayer>),
    >,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
    has_remote: Query<(), With<RemotePlayer>>,
) {
    for (entity, net_pos, net_state, transform) in remote_query.iter_mut() {
        if has_remote.get(entity).is_err() {
            info!(
                "First time seeing remote player {:?} at {:?} (state={:?})",
                entity, net_pos.0, net_state.0
            );
            // First time seeing this remote player
            commands.entity(entity).insert((
                RemotePlayer,
                Transform::from_translation(net_pos.0),
            ));
            // Add visuals only when rendering is available
            if let (Some(meshes), Some(materials)) = (meshes.as_mut(), materials.as_mut()) {
                commands.entity(entity).insert(Visibility::default());
                let mesh = meshes.add(Capsule3d::new(0.35, 1.0));
                let mat = materials.add(StandardMaterial {
                    base_color: net_state.0.color(),
                    ..default()
                });
                commands.entity(entity).with_children(|parent| {
                    parent.spawn((Mesh3d(mesh), MeshMaterial3d(mat)));
                });
            }
        } else if let Some(mut t) = transform {
            // Update existing remote player transform
            t.translation = t.translation.lerp(net_pos.0, 0.3);
        }
    }
}

/// Prints the machine's external IP address for others to connect.
fn print_local_ip(port: u16) {
    // Try to find a non-loopback IP
    if let Ok(socket) = std::net::UdpSocket::bind("0.0.0.0:0") {
        // Connect to a public address to determine which interface would be used
        if socket.connect("8.8.8.8:80").is_ok() {
            if let Ok(local_addr) = socket.local_addr() {
                info!(
                    "Server listening — connect with: --connect {}:{}",
                    local_addr.ip(),
                    port
                );
                println!("========================================");
                println!("  Server IP: {}:{}", local_addr.ip(), port);
                println!("========================================");
                return;
            }
        }
    }
    info!("Server listening on port {}", port);
    println!("========================================");
    println!("  Server listening on port {}", port);
    println!("========================================");
}

/// Generate a pseudo-random client ID from the current time.
fn rand_client_id() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
