//! Headless integration test for lightyear replication.

use bevy::prelude::*;
use lightyear::prelude::*;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

const PROTOCOL_ID: u64 = 0x4A554D50424C4B;
const PRIVATE_KEY: [u8; 32] = [0u8; 32];
const TICK_RATE: f64 = 60.0;

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TestPosition(Vec3);

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TestState(u32);

#[derive(Component)]
struct LocalEntity;

fn build_server_app() -> App {
    let mut app = App::new();
    let tick_duration = Duration::from_secs_f64(1.0 / TICK_RATE);

    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::log::LogPlugin::default());
    app.add_plugins(server::ServerPlugins { tick_duration });
    app.register_component::<TestPosition>();
    app.register_component::<TestState>();

    app.add_systems(Startup, |mut commands: Commands| {
        let server_addr: SocketAddr = "0.0.0.0:15123".parse().unwrap();
        let netcode = server::NetcodeServer::new(
            server::NetcodeConfig::default()
                .with_protocol_id(PROTOCOL_ID)
                .with_key(PRIVATE_KEY),
        );
        let server_entity = commands
            .spawn((Name::new("Server"), netcode, server::ServerUdpIo::default(), LocalAddr(server_addr)))
            .id();
        commands.trigger(server::Start { entity: server_entity });
        println!("[SERVER] Started on 15123");
    });

    // Use observer (as lightyear docs recommend) to add ReplicationSender when LinkOf is added
    app.add_observer(|trigger: On<Add, LinkOf>, mut commands: Commands| {
        let entity = trigger.entity;
        println!("[SERVER] LinkOf added on {:?}, inserting ReplicationSender", entity);
        commands.entity(entity).insert((
            ReplicationSender::new(Duration::from_millis(100), SendUpdatesMode::SinceLastAck, false),
            MessageManager::default(),
        ));
    });

    // Also try with Connected
    app.add_systems(Update, |mut commands: Commands, query: Query<Entity, (Added<Connected>, Without<ReplicationSender>)>| {
        for entity in query.iter() {
            println!("[SERVER] Connected entity {:?} without ReplicationSender, adding it", entity);
            commands.entity(entity).insert((
                ReplicationSender::new(Duration::from_millis(100), SendUpdatesMode::SinceLastAck, false),
                MessageManager::default(),
            ));
        }
    });

    // Re-replicate entities received from clients
    app.add_systems(Update, |mut commands: Commands, new_entities: Query<(Entity, Option<&TestPosition>), (Added<Replicated>, Without<Replicate>)>| {
        for (entity, pos) in new_entities.iter() {
            println!("[SERVER] Received replicated entity {:?}, pos={:?}", entity, pos.map(|p| p.0));
            commands.entity(entity).insert(Replicate::to_clients(NetworkTarget::All));
        }
    });

    // Diagnostic: dump entities with relevant components every 2 seconds
    app.add_systems(Update, |
        connected: Query<(Entity, Has<ReplicationSender>, Has<LinkOf>), With<Connected>>,
        senders: Query<(Entity, Has<Connected>, Has<LinkOf>), With<ReplicationSender>>,
        replicated: Query<(Entity, Option<&TestPosition>), With<Replicated>>,
        mut last_log: Local<f64>,
        time: Res<Time>,
    | {
        let now = time.elapsed_secs_f64();
        if now - *last_log < 2.0 { return; }
        *last_log = now;
        println!("[SERVER DIAG] t={:.1}s: connected={}, senders={}, replicated={}",
            now, connected.iter().count(), senders.iter().count(), replicated.iter().count());
        for (e, has_sender, has_link) in connected.iter() {
            println!("[SERVER DIAG]   Connected {:?}: sender={}, link={}", e, has_sender, has_link);
        }
        for (e, has_conn, has_link) in senders.iter() {
            println!("[SERVER DIAG]   Sender {:?}: connected={}, link={}", e, has_conn, has_link);
        }
        for (e, pos) in replicated.iter() {
            println!("[SERVER DIAG]   Replicated {:?}: pos={:?}", e, pos.map(|p| p.0));
        }
    });

    app
}

fn build_client_app() -> App {
    let mut app = App::new();
    let tick_duration = Duration::from_secs_f64(1.0 / TICK_RATE);

    app.add_plugins(MinimalPlugins);
    // LogPlugin already registered via server app - don't add twice if in same process
    app.add_plugins(client::ClientPlugins { tick_duration });
    app.register_component::<TestPosition>();
    app.register_component::<TestState>();

    app.add_systems(Startup, |mut commands: Commands| {
        let server_addr: SocketAddr = "127.0.0.1:15123".parse().unwrap();
        let auth = Authentication::Manual {
            server_addr,
            client_id: 42,
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
        commands.trigger(client::Connect { entity: client_entity });
        println!("[CLIENT] Connecting to 127.0.0.1:15123");
    });

    // Add ReplicationSender to client entity when connected
    app.add_systems(Update, |mut commands: Commands, query: Query<Entity, (With<Connected>, With<client::Client>, Without<ReplicationSender>)>| {
        for entity in query.iter() {
            println!("[CLIENT] Adding ReplicationSender to client entity {:?}", entity);
            commands.entity(entity).insert((
                ReplicationSender::new(Duration::from_millis(100), SendUpdatesMode::SinceLastAck, false),
                MessageManager::default(),
            ));
        }
    });

    // Debug: check if PostUpdate runs
    app.add_systems(PostUpdate, |mut frame: Local<u32>| {
        *frame += 1;
        if *frame % 120 == 1 {
            println!("[CLIENT PostUpdate] Frame {}", *frame);
        }
    });

    // Spawn local entity with Replicate ONLY after ReplicationSender exists
    app.add_systems(Update, |mut commands: Commands,
         connected: Query<(), (With<Connected>, With<ReplicationSender>)>,
         existing: Query<(), With<LocalEntity>>| {
        if !connected.is_empty() && existing.is_empty() {
            println!("[CLIENT] Spawning local entity with Replicate::to_server()");
            commands.spawn((
                LocalEntity,
                TestPosition(Vec3::new(1.0, 2.0, 3.0)),
                TestState(99),
                Replicate::to_server(),
            ));
        }
    });

    // Client diagnostic
    app.add_systems(Update, |
        connected: Query<(Entity, Has<ReplicationSender>), With<Connected>>,
        replicated: Query<(Entity, Option<&TestPosition>), With<Replicated>>,
        local: Query<(Entity, Has<Replicate>), With<LocalEntity>>,
        mut last_log: Local<f64>,
        time: Res<Time>,
    | {
        let now = time.elapsed_secs_f64();
        if now - *last_log < 2.0 { return; }
        *last_log = now;
        println!("[CLIENT DIAG] t={:.1}s: connected={}, replicated={}, local={}",
            now, connected.iter().count(), replicated.iter().count(), local.iter().count());
        for (e, has_sender) in connected.iter() {
            println!("[CLIENT DIAG]   Connected {:?}: sender={}", e, has_sender);
        }
        for (e, has_repl) in local.iter() {
            println!("[CLIENT DIAG]   Local {:?}: has_replicate={}", e, has_repl);
        }
    });
    app.add_systems(Update, |
        local: Query<(Entity, Has<Replicate>, Has<HasAuthority>, Has<Replicating>, Has<ReplicationGroup>, Has<ReplicationState>), With<LocalEntity>>,
        mut last_log: Local<f64>,
        time: Res<Time>,
    | {
        let now = time.elapsed_secs_f64();
        if now - *last_log < 2.0 { return; }
        *last_log = now;
        for (e, has_repl, has_auth, has_replicating, has_group, has_state) in local.iter() {
            println!("[CLIENT DETAIL] {:?}: replicate={}, authority={}, replicating={}, group={}, state={}",
                e, has_repl, has_auth, has_replicating, has_group, has_state);
        }
    });

    app
}

#[test]
fn test_replication() {
    println!("\n=== Starting Network Replication Test ===\n");

    let mut server_app = build_server_app();
    let mut client_app = build_client_app();

    for _ in 0..10 {
        server_app.update();
        std::thread::sleep(Duration::from_millis(16));
    }

    let start = std::time::Instant::now();
    let mut server_received = false;

    while start.elapsed() < Duration::from_secs(10) {
        server_app.update();
        client_app.update();
        std::thread::sleep(Duration::from_millis(16));

        let server_world = server_app.world_mut();
        let mut query = server_world.query_filtered::<Entity, With<Replicated>>();
        if query.iter(server_world).count() > 0 {
            println!("\n[TEST] SUCCESS - Server received replicated entity from client!");
            server_received = true;
            for _ in 0..10 {
                server_app.update();
                client_app.update();
                std::thread::sleep(Duration::from_millis(16));
            }
            break;
        }
    }

    assert!(server_received, "Server should have received the client's replicated entity");
}
