//! Headless integration tests for lightyear multiplayer replication.
//!
//! These tests run a server and one or more clients in the same process using
//! `MinimalPlugins` (no window, no renderer). Each test gets a unique port so
//! tests can run in parallel.

use bevy::prelude::*;
use lightyear::prelude::*;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;

const PROTOCOL_ID: u64 = 0x4A554D50424C4B;
const PRIVATE_KEY: [u8; 32] = [0u8; 32];
const TICK_RATE: f64 = 60.0;
const TIMEOUT: Duration = Duration::from_secs(10);

/// Atomic counter so each test gets a unique port range.
static PORT_COUNTER: AtomicU16 = AtomicU16::new(16000);

fn next_port() -> u16 {
    PORT_COUNTER.fetch_add(10, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Replicated components (mirrors the real game's NetworkedPosition etc.)
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TestPosition(Vec3);

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TestState(u32);

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TestRotation(Quat);

/// Marker for an entity spawned locally on a client.
#[derive(Component)]
struct LocalEntity;

/// Marker tag to identify which client spawned an entity (for multi-client tests).
#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ClientTag(u64);

// ---------------------------------------------------------------------------
// App builders
// ---------------------------------------------------------------------------

/// Build a headless server app bound to the given port.
fn build_server_app(port: u16) -> App {
    let mut app = App::new();
    let tick_duration = Duration::from_secs_f64(1.0 / TICK_RATE);

    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::log::LogPlugin::default());
    app.add_plugins(server::ServerPlugins { tick_duration });

    // Register replicated components BEFORE finishing the app
    app.register_component::<TestPosition>();
    app.register_component::<TestState>();
    app.register_component::<TestRotation>();
    app.register_component::<ClientTag>();

    // Finalize all plugins (especially MessagePlugin which builds its
    // send/recv systems in finish())
    app.finish();

    let addr: SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();

    app.add_systems(Startup, move |mut commands: Commands| {
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
                LocalAddr(addr),
            ))
            .id();
        commands.trigger(server::Start {
            entity: server_entity,
        });
        println!("[SERVER] Started on {port}");
    });

    // Add ReplicationSender + ReplicationReceiver when a client link is created.
    // NOTE: Do NOT insert MessageManager here — lightyear adds it automatically via
    // required_components on LinkOf/ClientOf. Manually inserting a default would
    // overwrite lightyear's own MessageManager and break message routing.
    app.add_observer(
        |trigger: On<Add, LinkOf>, mut commands: Commands| {
            let entity = trigger.entity;
            commands.entity(entity).insert((
                ReplicationSender::new(
                    Duration::from_millis(100),
                    SendUpdatesMode::SinceLastAck,
                    false,
                ),
                ReplicationReceiver::default(),
            ));
        },
    );

    // Fallback: also add on Connected
    app.add_systems(
        Update,
        |mut commands: Commands,
         query: Query<Entity, (Added<Connected>, Without<ReplicationSender>)>| {
            for entity in query.iter() {
                commands.entity(entity).insert((
                    ReplicationSender::new(
                        Duration::from_millis(100),
                        SendUpdatesMode::SinceLastAck,
                        false,
                    ),
                    ReplicationReceiver::default(),
                ));
            }
        },
    );

    // Re-replicate entities received from clients to all clients
    app.add_systems(
        Update,
        |mut commands: Commands,
         new_entities: Query<
            (Entity, Option<&TestPosition>),
            (Added<Replicated>, Without<Replicate>),
         >| {
            for (entity, pos) in new_entities.iter() {
                println!(
                    "[SERVER] Re-replicating entity {:?}, pos={:?}",
                    entity,
                    pos.map(|p| p.0)
                );
                commands
                    .entity(entity)
                    .insert(Replicate::to_clients(NetworkTarget::All));
            }
        },
    );

    app
}

/// Build a headless client app that connects to 127.0.0.1:`port` with the given client_id.
fn build_client_app(port: u16, client_id: u64) -> App {
    let mut app = App::new();
    let tick_duration = Duration::from_secs_f64(1.0 / TICK_RATE);

    app.add_plugins(MinimalPlugins);
    app.add_plugins(client::ClientPlugins { tick_duration });

    // Register replicated components BEFORE finishing the app
    app.register_component::<TestPosition>();
    app.register_component::<TestState>();
    app.register_component::<TestRotation>();
    app.register_component::<ClientTag>();

    // Finalize all plugins
    app.finish();

    let server_addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    app.add_systems(Startup, move |mut commands: Commands| {
        let auth = Authentication::Manual {
            server_addr,
            client_id,
            private_key: PRIVATE_KEY,
            protocol_id: PROTOCOL_ID,
        };
        let netcode_client =
            client::NetcodeClient::new(auth, client::NetcodeConfig::default())
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
        println!("[CLIENT {client_id}] Connecting to 127.0.0.1:{port}");
    });

    // Add ReplicationSender + ReplicationReceiver when connected.
    // NOTE: Do NOT insert MessageManager — lightyear adds it automatically via
    // required_components on client::Client.
    app.add_systems(
        Update,
        |mut commands: Commands,
         query: Query<
            Entity,
            (
                With<Connected>,
                With<client::Client>,
                Without<ReplicationSender>,
            ),
         >| {
            for entity in query.iter() {
                commands.entity(entity).insert((
                    ReplicationSender::new(
                        Duration::from_millis(100),
                        SendUpdatesMode::SinceLastAck,
                        false,
                    ),
                    ReplicationReceiver::default(),
                ));
            }
        },
    );

    app
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Tick both apps once with a 16ms sleep (one frame at ~60fps).
fn tick(server: &mut App, client: &mut App) {
    server.update();
    client.update();
    std::thread::sleep(Duration::from_millis(16));
}

/// Tick server + two clients once.
fn tick3(server: &mut App, client_a: &mut App, client_b: &mut App) {
    server.update();
    client_a.update();
    client_b.update();
    std::thread::sleep(Duration::from_millis(16));
}

/// Let the server warm up for a few frames before clients connect.
fn warm_up_server(server: &mut App, frames: usize) {
    for _ in 0..frames {
        server.update();
        std::thread::sleep(Duration::from_millis(16));
    }
}

/// Wait until `predicate` returns true, ticking both apps each frame.
/// Panics with `msg` if the timeout expires.
fn wait_until(
    server: &mut App,
    client: &mut App,
    msg: &str,
    mut predicate: impl FnMut(&mut App, &mut App) -> bool,
) {
    let start = std::time::Instant::now();
    while start.elapsed() < TIMEOUT {
        tick(server, client);
        if predicate(server, client) {
            return;
        }
    }
    panic!("Timed out: {msg}");
}

/// Wait with three apps (server + two clients).
fn wait_until3(
    server: &mut App,
    client_a: &mut App,
    client_b: &mut App,
    msg: &str,
    mut predicate: impl FnMut(&mut App, &mut App, &mut App) -> bool,
) {
    let start = std::time::Instant::now();
    while start.elapsed() < TIMEOUT {
        tick3(server, client_a, client_b);
        if predicate(server, client_a, client_b) {
            return;
        }
    }
    panic!("Timed out: {msg}");
}

/// Returns true if the given app has at least `n` entities with the `Replicated` component.
fn has_replicated_count(app: &mut App, n: usize) -> bool {
    let world = app.world_mut();
    let mut query = world.query_filtered::<Entity, With<Replicated>>();
    query.iter(world).count() >= n
}

/// Returns true if the given app has a `Connected` entity.
fn is_connected(app: &mut App) -> bool {
    let world = app.world_mut();
    let mut query = world.query_filtered::<Entity, With<Connected>>();
    query.iter(world).count() > 0
}

// ===========================================================================
// Tests
// ===========================================================================

/// Basic smoke test: one client connects, spawns an entity, server receives it.
#[test]
fn test_single_client_replication() {
    let port = next_port();
    let mut server = build_server_app(port);
    let mut client = build_client_app(port, 42);

    warm_up_server(&mut server, 10);

    // Add system: client spawns entity once connected + ReplicationSender exists
    client.add_systems(
        Update,
        |mut commands: Commands,
         connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
         existing: Query<(), With<LocalEntity>>| {
            if !connected.is_empty() && existing.is_empty() {
                println!("[CLIENT] Spawning entity with Replicate::to_server()");
                commands.spawn((
                    LocalEntity,
                    TestPosition(Vec3::new(1.0, 2.0, 3.0)),
                    TestState(99),
                    Replicate::to_server(),
                ));
            }
        },
    );

    wait_until(&mut server, &mut client, "server receives replicated entity", |s, _c| {
        has_replicated_count(s, 1)
    });

    println!("[TEST] test_single_client_replication PASSED");
}

/// Verify that replicated component VALUES arrive correctly, not just the entity.
#[test]
fn test_component_value_accuracy() {
    let port = next_port();
    let mut server = build_server_app(port);
    let mut client = build_client_app(port, 43);

    let expected_pos = Vec3::new(42.5, -7.0, 100.25);
    let expected_state = 12345u32;
    let expected_rot = Quat::from_euler(EulerRot::YXZ, 1.0, 0.5, 0.25);

    warm_up_server(&mut server, 10);

    client.add_systems(
        Update,
        move |mut commands: Commands,
              connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
              existing: Query<(), With<LocalEntity>>| {
            if !connected.is_empty() && existing.is_empty() {
                commands.spawn((
                    LocalEntity,
                    TestPosition(expected_pos),
                    TestState(expected_state),
                    TestRotation(expected_rot),
                    Replicate::to_server(),
                ));
            }
        },
    );

    // Wait for server to receive the entity
    wait_until(
        &mut server,
        &mut client,
        "server receives entity with correct values",
        |s, _c| has_replicated_count(s, 1),
    );

    // Let a few more frames run to ensure components are fully synced
    for _ in 0..10 {
        tick(&mut server, &mut client);
    }

    // Verify values
    let world = server.world_mut();
    let mut query = world.query::<(&TestPosition, &TestState, &TestRotation)>();
    let results: Vec<_> = query.iter(world).collect();
    assert!(!results.is_empty(), "Should have at least one replicated entity");

    let (pos, state, rot) = results[0];
    assert_eq!(pos.0, expected_pos, "Position should match exactly");
    assert_eq!(state.0, expected_state, "State should match exactly");
    // Quaternions: check approximate equality due to floating point
    let dot = rot.0.dot(expected_rot).abs();
    assert!(
        dot > 0.999,
        "Rotation should match (dot={dot}, expected ~1.0)"
    );

    println!("[TEST] test_component_value_accuracy PASSED");
}

/// Two clients connect, each spawns an entity. Both entities should replicate
/// to the server, and each client should see the OTHER client's entity.
#[test]
fn test_two_client_mutual_visibility() {
    let port = next_port();
    let mut server = build_server_app(port);
    let mut client_a = build_client_app(port, 100);
    let mut client_b = build_client_app(port, 200);

    warm_up_server(&mut server, 10);

    // Client A spawns at position (10, 0, 0)
    client_a.add_systems(
        Update,
        |mut commands: Commands,
         connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
         existing: Query<(), With<LocalEntity>>| {
            if !connected.is_empty() && existing.is_empty() {
                println!("[CLIENT A] Spawning entity");
                commands.spawn((
                    LocalEntity,
                    TestPosition(Vec3::new(10.0, 0.0, 0.0)),
                    ClientTag(100),
                    Replicate::to_server(),
                ));
            }
        },
    );

    // Client B spawns at position (20, 0, 0)
    client_b.add_systems(
        Update,
        |mut commands: Commands,
         connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
         existing: Query<(), With<LocalEntity>>| {
            if !connected.is_empty() && existing.is_empty() {
                println!("[CLIENT B] Spawning entity");
                commands.spawn((
                    LocalEntity,
                    TestPosition(Vec3::new(20.0, 0.0, 0.0)),
                    ClientTag(200),
                    Replicate::to_server(),
                ));
            }
        },
    );

    // Wait for server to have both entities
    wait_until3(
        &mut server,
        &mut client_a,
        &mut client_b,
        "server receives both client entities",
        |s, _a, _b| has_replicated_count(s, 2),
    );

    println!("[TEST] Server has both entities");

    // Let replication propagate back to clients
    for _ in 0..30 {
        tick3(&mut server, &mut client_a, &mut client_b);
    }

    // Client A should see client B's entity (replicated from server)
    let world_a = client_a.world_mut();
    let mut q = world_a.query::<(&TestPosition, &ClientTag)>();
    let a_sees: Vec<_> = q.iter(world_a).collect();
    println!(
        "[TEST] Client A sees {} entities: {:?}",
        a_sees.len(),
        a_sees.iter().map(|(p, t)| (p.0, t.0)).collect::<Vec<_>>()
    );

    // Client B should see client A's entity
    let world_b = client_b.world_mut();
    let mut q = world_b.query::<(&TestPosition, &ClientTag)>();
    let b_sees: Vec<_> = q.iter(world_b).collect();
    println!(
        "[TEST] Client B sees {} entities: {:?}",
        b_sees.len(),
        b_sees.iter().map(|(p, t)| (p.0, t.0)).collect::<Vec<_>>()
    );

    // Each client should see at least the other's entity via replication
    // (they also see their own LocalEntity)
    let a_sees_b = a_sees.iter().any(|(_, tag)| tag.0 == 200);
    let b_sees_a = b_sees.iter().any(|(_, tag)| tag.0 == 100);
    assert!(a_sees_b, "Client A should see Client B's entity");
    assert!(b_sees_a, "Client B should see Client A's entity");

    println!("[TEST] test_two_client_mutual_visibility PASSED");
}

/// Verify that position UPDATES (not just initial values) propagate through the server.
#[test]
fn test_position_update_propagation() {
    let port = next_port();
    let mut server = build_server_app(port);
    let mut client = build_client_app(port, 44);

    warm_up_server(&mut server, 10);

    // Spawn entity at origin
    client.add_systems(
        Update,
        |mut commands: Commands,
         connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
         existing: Query<(), With<LocalEntity>>| {
            if !connected.is_empty() && existing.is_empty() {
                commands.spawn((
                    LocalEntity,
                    TestPosition(Vec3::ZERO),
                    Replicate::to_server(),
                ));
            }
        },
    );

    // Wait for initial replication
    wait_until(
        &mut server,
        &mut client,
        "server receives initial entity",
        |s, _c| has_replicated_count(s, 1),
    );

    // Now update the position on the client
    let new_pos = Vec3::new(50.0, 25.0, -10.0);
    {
        let world = client.world_mut();
        let mut query = world.query::<&mut TestPosition>();
        for mut pos in query.iter_mut(world) {
            pos.0 = new_pos;
        }
    }

    // Tick until the server sees the updated position
    let start = std::time::Instant::now();
    let mut updated = false;
    while start.elapsed() < TIMEOUT {
        tick(&mut server, &mut client);

        let world = server.world_mut();
        let mut query = world.query::<&TestPosition>();
        for pos in query.iter(world) {
            if pos.0 == new_pos {
                updated = true;
                break;
            }
        }
        if updated {
            break;
        }
    }

    assert!(
        updated,
        "Server should see the updated position {new_pos:?}"
    );

    println!("[TEST] test_position_update_propagation PASSED");
}

/// When a client disconnects, the server should not crash and should continue
/// serving remaining clients normally.
#[test]
fn test_client_disconnect_resilience() {
    let port = next_port();
    let mut server = build_server_app(port);
    let mut client_a = build_client_app(port, 300);
    let mut client_b = build_client_app(port, 301);

    warm_up_server(&mut server, 10);

    // Both clients spawn entities
    client_a.add_systems(
        Update,
        |mut commands: Commands,
         connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
         existing: Query<(), With<LocalEntity>>| {
            if !connected.is_empty() && existing.is_empty() {
                commands.spawn((
                    LocalEntity,
                    TestPosition(Vec3::X),
                    ClientTag(300),
                    Replicate::to_server(),
                ));
            }
        },
    );

    client_b.add_systems(
        Update,
        |mut commands: Commands,
         connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
         existing: Query<(), With<LocalEntity>>| {
            if !connected.is_empty() && existing.is_empty() {
                commands.spawn((
                    LocalEntity,
                    TestPosition(Vec3::Y),
                    ClientTag(301),
                    Replicate::to_server(),
                ));
            }
        },
    );

    // Wait for server to have both
    wait_until3(
        &mut server,
        &mut client_a,
        &mut client_b,
        "server receives both entities",
        |s, _a, _b| has_replicated_count(s, 2),
    );

    println!("[TEST] Both clients connected, dropping client A");

    // Drop client A — simulates disconnect
    drop(client_a);

    // Server should keep running without panicking. Tick for a while.
    let mut dummy = App::new();
    dummy.add_plugins(MinimalPlugins);
    for _ in 0..30 {
        server.update();
        client_b.update();
        std::thread::sleep(Duration::from_millis(16));
    }

    // Server should still be operational — client B should still be connected
    assert!(
        is_connected(&mut client_b),
        "Client B should still be connected after Client A disconnects"
    );

    println!("[TEST] test_client_disconnect_resilience PASSED");
}
