//! Headless measurement of the player's real movement envelope.
//!
//! Runs the actual physics stack (avian3d + Tnua) with the same character
//! configuration as `src/player.rs` (keep the constants in sync!) and
//! empirically measures:
//!   - jump apex rise,
//!   - the tallest ledge the player can hop onto,
//!   - the widest flat gap the player can clear at a run,
//!   - the widest gap onto a ledge 1 wu higher.
//!
//! The measured envelope backs the authoring limits in
//! `crates/voxel/src/world_def.rs` (`PLAYER_MAX_RISE_WU` /
//! `PLAYER_MAX_GAP_WU`): the assertions here guarantee the real player can
//! clear every hop those limits allow. Run with `--nocapture` to see the
//! raw measurements.

use std::time::Duration;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_tnua::builtins::{TnuaBuiltinJumpConfig, TnuaBuiltinWalkConfig};
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::{TnuaAvian3dPlugin, TnuaAvian3dSensorShape};

// --- Mirror of src/player.rs (keep in sync) --------------------------------

const PLAYER_HEIGHT: f32 = 1.0;
const PLAYER_RADIUS: f32 = 0.35;
const RUN_MULTIPLIER: f32 = 1.8;

#[derive(TnuaScheme)]
#[scheme(basis = TnuaBuiltinWalk)]
enum Scheme {
    Jump(TnuaBuiltinJump),
}

fn player_config() -> SchemeConfig {
    SchemeConfig {
        basis: TnuaBuiltinWalkConfig {
            float_height: 1.5,
            spring_strength: 1600.0,
            cling_distance: 0.5,
            speed: 10.0,
            acceleration: 20.0,
            air_acceleration: 15.0,
            coyote_time: 0.12,
            ..default()
        },
        jump: TnuaBuiltinJumpConfig {
            height: 3.0,
            ..default()
        },
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const DT: f64 = 1.0 / 64.0;

/// Scripted input for the simulated player.
#[derive(Resource, Clone, Copy, Default)]
struct Drive {
    dir: Vec2,
    run: bool,
    /// Jump as soon as the player's X passes this value.
    jump_at_x: Option<f32>,
    /// How long to keep the jump held after triggering (seconds).
    jump_hold: f32,
}

#[derive(Resource, Default)]
struct DriveState {
    jump_started_at: Option<f64>,
    now: f64,
}

#[derive(Component)]
struct SimPlayer;

fn drive_system(
    drive: Res<Drive>,
    mut state: ResMut<DriveState>,
    mut players: Query<(&Transform, &mut TnuaController<Scheme>), With<SimPlayer>>,
) {
    state.now += DT;
    let Ok((tf, mut controller)) = players.single_mut() else {
        return;
    };

    let speed_mult = if drive.run { RUN_MULTIPLIER } else { 1.0 };
    let dir = Vec3::new(drive.dir.x, 0.0, drive.dir.y) * speed_mult;
    controller.basis = TnuaBuiltinWalk {
        desired_motion: dir,
        desired_forward: None,
    };

    let mut jump = false;
    if let Some(x) = drive.jump_at_x {
        if tf.translation.x >= x && state.jump_started_at.is_none() {
            state.jump_started_at = Some(state.now);
        }
    }
    if let Some(t0) = state.jump_started_at {
        if state.now - t0 <= drive.jump_hold as f64 {
            jump = true;
        }
    }

    controller.initiate_action_feeding();
    if jump {
        controller.action(Scheme::Jump(TnuaBuiltinJump::default()));
    }
}

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        bevy::transform::TransformPlugin,
        bevy::asset::AssetPlugin::default(),
        PhysicsPlugins::default(),
    ));
    // avian's collider systems expect mesh asset events + the scene spawner
    app.init_asset::<Mesh>();
    app.add_plugins((bevy::scene::ScenePlugin, bevy::diagnostic::DiagnosticsPlugin));
    app.add_plugins(TnuaControllerPlugin::<Scheme>::new(PhysicsSchedule));
    app.add_plugins(TnuaAvian3dPlugin::new(PhysicsSchedule));
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(DT),
    ));
    app.init_resource::<DriveState>();
    app.insert_resource(Drive::default());
    app.add_systems(Update, drive_system);
    app.finish();
    app.cleanup();
    app
}

fn spawn_player(app: &mut App, pos: Vec3) {
    let config = {
        let mut configs = app
            .world_mut()
            .resource_mut::<Assets<SchemeConfig>>();
        configs.add(player_config())
    };
    app.world_mut().spawn((
        SimPlayer,
        Transform::from_translation(pos),
        RigidBody::Dynamic,
        Collider::capsule(PLAYER_RADIUS, PLAYER_HEIGHT),
        LockedAxes::ROTATION_LOCKED,
        TnuaController::<Scheme>::default(),
        TnuaConfig::<Scheme>(config),
        TnuaAvian3dSensorShape(Collider::cylinder(PLAYER_RADIUS - 0.01, 0.0)),
    ));
}

/// Spawn a static box platform given min corner and size.
fn spawn_box(app: &mut App, min: Vec3, size: Vec3) {
    app.world_mut().spawn((
        Transform::from_translation(min + size * 0.5),
        RigidBody::Static,
        Collider::cuboid(size.x, size.y, size.z),
    ));
}

fn step(app: &mut App, seconds: f64) {
    for _ in 0..(seconds / DT).ceil() as usize {
        app.update();
    }
}

fn player_pos(app: &mut App) -> Vec3 {
    let mut q = app
        .world_mut()
        .query_filtered::<&Transform, With<SimPlayer>>();
    q.single(app.world()).unwrap().translation
}

/// Settle the player on ground and return the resting center height.
fn settle(app: &mut App) -> f32 {
    step(app, 2.0);
    player_pos(app).y
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

/// Max height of the player's center above its standing height during a jump.
fn measure_jump_apex() -> f32 {
    let mut app = build_app();
    spawn_box(&mut app, Vec3::new(-50.0, -1.0, -50.0), Vec3::new(100.0, 1.0, 100.0));
    spawn_player(&mut app, Vec3::new(0.0, 2.0, 0.0));
    let rest = settle(&mut app);

    app.insert_resource(Drive {
        dir: Vec2::ZERO,
        run: false,
        jump_at_x: Some(-1000.0), // immediately
        jump_hold: 0.4,
    });
    let mut max_y = rest;
    for _ in 0..(3.0 / DT) as usize {
        app.update();
        max_y = max_y.max(player_pos(&mut app).y);
    }
    max_y - rest
}

/// Whether the player can run off platform A and land on platform B whose top
/// is `rise` higher, across a horizontal edge-to-edge `gap`.
fn can_clear(gap: f32, rise: f32, run: bool) -> bool {
    let mut app = build_app();
    // Platform A: top at y=0, edge at x=0
    spawn_box(&mut app, Vec3::new(-30.0, -1.0, -10.0), Vec3::new(30.0, 1.0, 20.0));
    // Platform B: top at y=rise, near edge at x=gap (long, so the player
    // doesn't run off the far end before we check)
    spawn_box(&mut app, Vec3::new(gap, rise - 1.0, -10.0), Vec3::new(300.0, 1.0, 20.0));
    spawn_player(&mut app, Vec3::new(-8.0, 2.0, 0.0));
    let rest = settle(&mut app);

    app.insert_resource(Drive {
        dir: Vec2::new(1.0, 0.0),
        run,
        // Trigger the jump just before the edge (the capsule radius still
        // has support; mimics a player jumping at the lip).
        jump_at_x: Some(-PLAYER_RADIUS),
        jump_hold: 0.4,
    });
    step(&mut app, 4.0);
    let p = player_pos(&mut app);
    // Landed on B: past its edge, at B's standing height, not fallen through
    p.x > gap + PLAYER_RADIUS && (p.y - (rise + rest)).abs() < 0.3
}

/// Largest value in `candidates` for which `test` succeeds (assumes roughly
/// monotone behavior; scans all and returns the max success).
fn max_passing(candidates: &[f32], test: impl Fn(f32) -> bool) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for &c in candidates {
        if test(c) {
            best = best.max(c);
        }
    }
    best
}

#[test]
fn measure_player_envelope() {
    let apex = measure_jump_apex();
    println!("jump apex rise: {apex:.2} wu");

    // Tallest ledge (no gap) the player can hop onto, in half-voxel steps
    let rises: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();
    let max_step = max_passing(&rises, |r| can_clear(0.1, r, false));
    println!("max ledge hop-up (walk): {max_step:.1} wu");

    // Widest flat gap at a run
    let gaps: Vec<f32> = (2..=22).map(|i| i as f32 * 0.5).collect();
    let max_gap_flat = max_passing(&gaps, |g| can_clear(g, 0.0, true));
    println!("max flat gap (run): {max_gap_flat:.1} wu");

    // Widest gap onto a ledge 1 wu higher, at a run
    let max_gap_up = max_passing(&gaps, |g| can_clear(g, 1.0, true));
    println!("max gap onto +1.0 wu ledge (run): {max_gap_up:.1} wu");

    // Widest gap onto a ledge 2 wu higher, at a run
    let max_gap_up2 = max_passing(&gaps, |g| can_clear(g, 2.0, true));
    println!("max gap onto +2.0 wu ledge (run): {max_gap_up2:.1} wu");

    // --- The envelope the course authoring in world_def.rs relies on. ---
    // world_def uses PLAYER_MAX_RISE_WU = 2.0 and PLAYER_MAX_GAP_WU = 4.0
    // (with rises and gaps never maxed simultaneously beyond the +2wu row).
    assert!(apex >= 2.8, "jump apex {apex:.2} wu regressed below 2.8");
    assert!(max_step >= 2.0, "max ledge hop {max_step:.1} wu — authoring assumes >= 2.0");
    assert!(max_gap_flat >= 5.0, "max flat gap {max_gap_flat:.1} wu — authoring assumes >= 5.0");
    assert!(max_gap_up2 >= 4.0, "max gap onto +2wu {max_gap_up2:.1} wu — authoring assumes >= 4.0");
}
