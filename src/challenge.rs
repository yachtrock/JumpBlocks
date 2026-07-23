//! Platforming challenges, trophies, and locked zones.
//!
//! The authored [`WorldDef`] places a **start pad** per challenge in the
//! world. Standing on it activates the challenge: a golden **goal** appears
//! at the challenge's goal position and the run begins, optionally under a
//! time limit and/or above a kill height. Reaching the goal earns a
//! **trophy** (once per challenge; runs can be replayed).
//!
//! Each island has a **locked zone**: a volume whose terrain cannot be
//! modified. Bring enough of that island's trophies to the zone's pedestal
//! to unlock it (see [`BuildLocks`], consumed by the building system).
//!
//! Progress (trophies + unlocked zones) persists to `progress.json` in the
//! world save directory.

use std::collections::HashSet;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use serde::{Deserialize, Serialize};

use jumpblocks_voxel::streaming::WorldSavePath;
use jumpblocks_voxel::world_def::{
    WorldDef, cell_base_world, zone_world_aabb,
};
use jumpblocks_voxel::world_grid::WorldGrid;

use crate::layers::GameLayer;
use crate::player::Player;
use crate::world::SpawnPoint;

pub struct ChallengePlugin;

impl Plugin for ChallengePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveChallenge>()
            .init_resource::<PlayerProgress>()
            .init_resource::<BuildLocks>()
            .init_resource::<HudMessages>()
            .add_systems(
                Startup,
                setup_challenges.after(crate::world::setup_world),
            )
            .add_systems(
                Update,
                (
                    start_pad_system,
                    challenge_timer_system,
                    kill_volume_system,
                    goal_reach_system,
                    goal_spin_system,
                    pedestal_system,
                    moving_platform_system,
                    sync_build_locks,
                    tick_messages,
                    hold_player_until_world_ready,
                ),
            )
            .add_systems(EguiPrimaryContextPass, challenge_hud);
    }
}

// ---------------------------------------------------------------------------
// Resources & components
// ---------------------------------------------------------------------------

/// World content shared with gameplay systems.
#[derive(Resource)]
pub struct WorldContent {
    pub def: WorldDef,
    pub region_origin: Vec3,
}

/// Persistent player progress.
#[derive(Resource, Default, Serialize, Deserialize)]
pub struct PlayerProgress {
    /// Challenge indices whose trophy has been earned.
    pub trophies: HashSet<usize>,
    /// Zone indices that have been unlocked.
    pub unlocked_zones: HashSet<usize>,
}

impl PlayerProgress {
    /// Trophies earned on a given island.
    fn island_trophies(&self, def: &WorldDef, island: usize) -> u32 {
        self.trophies
            .iter()
            .filter(|&&c| def.challenges.get(c).map(|d| d.island) == Some(island))
            .count() as u32
    }
}

/// The currently running challenge, if any.
#[derive(Resource, Default)]
pub struct ActiveChallenge(pub Option<ActiveRun>);

pub struct ActiveRun {
    pub challenge: usize,
    pub timer: Option<Timer>,
}

/// Build restrictions consumed by the building system.
#[derive(Resource, Default)]
pub struct BuildLocks {
    /// World-space AABBs where building is currently prohibited.
    pub locked_zones: Vec<(Vec3, Vec3)>,
    /// True while a challenge run is active (no building mid-run).
    pub building_disabled: bool,
}

impl BuildLocks {
    /// Whether a world position may be modified.
    pub fn can_build_at(&self, pos: Vec3) -> bool {
        if self.building_disabled {
            return false;
        }
        !self.locked_zones.iter().any(|(min, max)| {
            pos.x >= min.x && pos.x <= max.x
                && pos.y >= min.y && pos.y <= max.y
                && pos.z >= min.z && pos.z <= max.z
        })
    }
}

/// Transient HUD messages.
#[derive(Resource, Default)]
pub struct HudMessages(Vec<(String, Timer)>);

impl HudMessages {
    pub(crate) fn push(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        info!("[challenge] {msg}");
        self.0.push((msg, Timer::from_seconds(5.0, TimerMode::Once)));
        if self.0.len() > 4 {
            self.0.remove(0);
        }
    }
}

/// Start pad for a challenge. `armed` prevents instant re-trigger until the
/// player has stepped away.
#[derive(Component)]
struct StartPad {
    challenge: usize,
    armed: bool,
}

/// The goal entity of the active challenge.
#[derive(Component)]
struct GoalMarker {
    challenge: usize,
}

/// Turn-in pedestal for a zone. `hinted` throttles the "not enough trophies"
/// message to once per approach.
#[derive(Component)]
struct Pedestal {
    zone: usize,
    hinted: bool,
}

/// Translucent barrier visualizing a locked zone.
#[derive(Component)]
struct ZoneBarrier {
    zone: usize,
}

/// Kinematic platform oscillating between two points.
#[derive(Component)]
struct MovingPlatform {
    from: Vec3,
    to: Vec3,
    period: f32,
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const PAD_RADIUS: f32 = 1.6;
const GOAL_RADIUS: f32 = 1.6;
const PEDESTAL_RADIUS: f32 = 2.5;

pub fn setup_challenges(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    world_grid: Res<WorldGrid>,
    save_path: Option<Res<WorldSavePath>>,
    mut progress: ResMut<PlayerProgress>,
) {
    let def = WorldDef::standard();
    let region_origin = world_grid
        .get_region(world_grid.active_region)
        .map(|r| r.world_origin)
        .unwrap_or(Vec3::new(-2048.0, 0.0, -2048.0));

    // Load saved progress
    if let Some(ref save) = save_path {
        match load_progress(&save.0) {
            Some(p) => {
                info!(
                    "[challenge] Loaded progress: {} trophies, {} zones unlocked",
                    p.trophies.len(),
                    p.unlocked_zones.len()
                );
                *progress = p;
            }
            None => info!("[challenge] No saved progress, starting fresh"),
        }
    }

    // --- Start pads ---
    let pad_mesh = meshes.add(Cylinder::new(1.3, 0.12));
    let pad_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.9, 0.4),
        emissive: LinearRgba::new(0.05, 0.6, 0.15, 1.0),
        ..default()
    });
    for (i, c) in def.challenges.iter().enumerate() {
        let pos = cell_base_world(region_origin, c.start_cell);
        commands.spawn((
            StartPad { challenge: i, armed: true },
            Mesh3d(pad_mesh.clone()),
            MeshMaterial3d(pad_material.clone()),
            Transform::from_translation(pos + Vec3::Y * 0.06),
        ));
    }

    // --- Pedestals (visual crystal above the stamped column) ---
    let crystal_mesh = meshes.add(Sphere::new(0.35).mesh().ico(1).unwrap());
    let crystal_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.4, 0.95),
        emissive: LinearRgba::new(0.4, 0.15, 0.7, 1.0),
        ..default()
    });
    for (i, z) in def.zones.iter().enumerate() {
        let stand = cell_base_world(region_origin, z.pedestal_cell);
        // Column is stamped at pedestal_cell + 2 cells in X, 3 cells tall
        let crystal = stand + Vec3::new(1.25, 2.0, 0.0);
        commands.spawn((
            Pedestal { zone: i, hinted: false },
            Mesh3d(crystal_mesh.clone()),
            MeshMaterial3d(crystal_material.clone()),
            Transform::from_translation(crystal),
        ));
    }

    // --- Zone barriers for still-locked zones ---
    let barrier_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.95, 0.3, 0.2, 0.12),
        emissive: LinearRgba::new(0.3, 0.05, 0.02, 1.0),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });
    for (i, z) in def.zones.iter().enumerate() {
        if progress.unlocked_zones.contains(&i) {
            continue;
        }
        let (min, max) = zone_world_aabb(region_origin, z);
        let size = max - min;
        let center = (min + max) * 0.5;
        commands.spawn((
            ZoneBarrier { zone: i },
            Mesh3d(meshes.add(Cuboid::new(size.x, size.y, size.z))),
            MeshMaterial3d(barrier_material.clone()),
            Transform::from_translation(center),
            bevy::light::NotShadowCaster,
        ));
    }

    // --- Moving platforms ---
    let platform_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.4, 0.25),
        ..default()
    });
    for p in &def.platforms {
        let from = cell_base_world(region_origin, p.from_cell);
        let to = cell_base_world(region_origin, p.to_cell);
        let size = p.half_extents * 2.0;
        commands.spawn((
            MovingPlatform { from, to, period: p.period },
            Mesh3d(meshes.add(Cuboid::new(size.x, size.y, size.z))),
            MeshMaterial3d(platform_material.clone()),
            Transform::from_translation(from),
            RigidBody::Kinematic,
            Collider::cuboid(size.x, size.y, size.z),
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        ));
    }

    info!(
        "[challenge] World ready: {} challenges, {} zones, {} moving platforms",
        def.challenges.len(),
        def.zones.len(),
        def.platforms.len()
    );

    commands.insert_resource(WorldContent { def, region_origin });
}

// ---------------------------------------------------------------------------
// Challenge flow
// ---------------------------------------------------------------------------

fn start_pad_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    content: Option<Res<WorldContent>>,
    mut active: ResMut<ActiveChallenge>,
    mut pads: Query<(&mut StartPad, &Transform)>,
    players: Query<&Transform, With<Player>>,
    mut messages: ResMut<HudMessages>,
    progress: Res<PlayerProgress>,
) {
    let Some(content) = content else { return };
    let Ok(player) = players.single() else { return };
    let p = player.translation;

    for (mut pad, pad_tf) in pads.iter_mut() {
        let d = (p - pad_tf.translation).length();
        if d > PAD_RADIUS + 1.0 {
            pad.armed = true;
            continue;
        }
        if d > PAD_RADIUS || !pad.armed || active.0.is_some() {
            continue;
        }
        pad.armed = false;

        let def = &content.def.challenges[pad.challenge];
        let goal_pos = cell_base_world(content.region_origin, def.goal_cell);
        spawn_goal(&mut commands, &mut meshes, &mut materials, pad.challenge, goal_pos);

        let mut announce = format!("Challenge started: {}", def.name);
        if let Some(t) = def.time_limit {
            announce.push_str(&format!("  ({}s limit)", t as u32));
        }
        if def.kill_y_cell.is_some() {
            announce.push_str("  [don't fall!]");
        }
        if progress.trophies.contains(&pad.challenge) {
            announce.push_str("  (trophy already earned)");
        }
        messages.push(announce);

        active.0 = Some(ActiveRun {
            challenge: pad.challenge,
            timer: def.time_limit.map(|t| Timer::from_seconds(t, TimerMode::Once)),
        });
    }
}

fn spawn_goal(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    challenge: usize,
    pos: Vec3,
) {
    let gold = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.85, 0.2),
        emissive: LinearRgba::new(1.0, 0.7, 0.1, 1.0),
        ..default()
    });
    commands
        .spawn((
            GoalMarker { challenge },
            Transform::from_translation(pos + Vec3::Y * 1.0),
            Visibility::default(),
        ))
        .with_children(|parent| {
            parent.spawn((
                Mesh3d(meshes.add(Torus::new(0.55, 0.75))),
                MeshMaterial3d(gold.clone()),
                Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
            ));
            parent.spawn((
                Mesh3d(meshes.add(Sphere::new(0.3))),
                MeshMaterial3d(gold),
            ));
            parent.spawn((
                PointLight {
                    color: Color::srgb(1.0, 0.8, 0.3),
                    intensity: 60_000.0,
                    range: 14.0,
                    shadows_enabled: false,
                    ..default()
                },
                Transform::from_translation(Vec3::Y * 0.5),
            ));
        });
}

fn goal_spin_system(time: Res<Time>, mut goals: Query<&mut Transform, With<GoalMarker>>) {
    for mut tf in goals.iter_mut() {
        tf.rotation = Quat::from_rotation_y(time.elapsed_secs() * 1.4);
        tf.translation.y += (time.elapsed_secs() * 2.0).sin() * 0.003;
    }
}

fn challenge_timer_system(
    mut commands: Commands,
    time: Res<Time>,
    content: Option<Res<WorldContent>>,
    mut active: ResMut<ActiveChallenge>,
    goals: Query<Entity, With<GoalMarker>>,
    mut players: Query<(&mut Transform, &mut LinearVelocity), With<Player>>,
    mut messages: ResMut<HudMessages>,
) {
    let Some(content) = content else { return };
    let Some(run) = active.0.as_mut() else { return };
    let Some(timer) = run.timer.as_mut() else { return };

    timer.tick(time.delta());
    if !timer.is_finished() {
        return;
    }

    let def = &content.def.challenges[run.challenge];
    messages.push(format!("Time's up! {} failed — back to the start.", def.name));

    for goal in goals.iter() {
        commands.entity(goal).despawn();
    }
    if let Ok((mut tf, mut vel)) = players.single_mut() {
        tf.translation = cell_base_world(content.region_origin, def.start_cell) + Vec3::Y * 1.0;
        vel.0 = Vec3::ZERO;
    }
    active.0 = None;
}

/// While the reveal curtain is still down the world's collision meshes may
/// not exist yet — without this the player falls through the terrain during
/// loading and ends up under the islands. Pin them to the spawn point until
/// the chunks are ready.
fn hold_player_until_world_ready(
    curtain: Option<Res<crate::curtain::CurtainState>>,
    spawn: Res<SpawnPoint>,
    mut players: Query<(&mut Transform, &mut LinearVelocity), With<Player>>,
) {
    let Some(curtain) = curtain else { return };
    if curtain.chunks_ready {
        return;
    }
    if let Ok((mut tf, mut vel)) = players.single_mut() {
        tf.translation = spawn.0;
        vel.0 = Vec3::ZERO;
    }
}

fn kill_volume_system(
    content: Option<Res<WorldContent>>,
    active: Res<ActiveChallenge>,
    spawn: Res<SpawnPoint>,
    mut players: Query<(&mut Transform, &mut LinearVelocity), With<Player>>,
    mut messages: ResMut<HudMessages>,
) {
    let Ok((mut tf, mut vel)) = players.single_mut() else { return };

    // Challenge kill volume: return to the start pad, run keeps going.
    if let (Some(content), Some(run)) = (content.as_ref(), active.0.as_ref()) {
        let def = &content.def.challenges[run.challenge];
        if let Some(kill_cell) = def.kill_y_cell {
            let kill_y = content.region_origin.y
                + kill_cell as f32 * jumpblocks_voxel::chunk::VOXEL_SIZE;
            if tf.translation.y < kill_y {
                messages.push("Fell out of the course! Back to the start.");
                tf.translation =
                    cell_base_world(content.region_origin, def.start_cell) + Vec3::Y * 1.0;
                vel.0 = Vec3::ZERO;
                return;
            }
        }
    }

    // Global void safety net
    if tf.translation.y < -15.0 {
        tf.translation = spawn.0;
        vel.0 = Vec3::ZERO;
    }
}

fn goal_reach_system(
    mut commands: Commands,
    content: Option<Res<WorldContent>>,
    mut active: ResMut<ActiveChallenge>,
    goals: Query<(Entity, &Transform, &GoalMarker)>,
    players: Query<&Transform, With<Player>>,
    mut progress: ResMut<PlayerProgress>,
    mut messages: ResMut<HudMessages>,
    save_path: Option<Res<WorldSavePath>>,
) {
    let Some(content) = content else { return };
    let Ok(player) = players.single() else { return };
    if active.0.is_none() {
        return;
    }

    for (entity, tf, marker) in goals.iter() {
        let delta = player.translation - tf.translation;
        let horizontal = Vec2::new(delta.x, delta.z).length();
        if horizontal > GOAL_RADIUS || delta.y.abs() > 2.5 {
            continue;
        }

        let def = &content.def.challenges[marker.challenge];
        let first_time = progress.trophies.insert(marker.challenge);
        if first_time {
            let island = def.island;
            let island_name = &content.def.terrain.islands[island].name;
            let earned = progress.island_trophies(&content.def, island);
            // Which zone does this island feed?
            let zone_hint = content
                .def
                .zones
                .iter()
                .enumerate()
                .find(|(zi, z)| z.island == island && !progress.unlocked_zones.contains(zi))
                .map(|(_, z)| format!("  ({island_name}: {earned}/{} for {})", z.trophies_required, z.name))
                .unwrap_or_default();
            messages.push(format!("Trophy earned: {}!{zone_hint}", def.name));
            if let Some(ref save) = save_path {
                save_progress(&save.0, &progress);
            }
        } else {
            messages.push(format!("{} cleared again — trophy already collected.", def.name));
        }

        commands.entity(entity).despawn();
        active.0 = None;
    }
}

// ---------------------------------------------------------------------------
// Zones
// ---------------------------------------------------------------------------

fn pedestal_system(
    mut commands: Commands,
    content: Option<Res<WorldContent>>,
    mut pedestals: Query<(&mut Pedestal, &Transform)>,
    barriers: Query<(Entity, &ZoneBarrier)>,
    players: Query<&Transform, With<Player>>,
    mut progress: ResMut<PlayerProgress>,
    mut messages: ResMut<HudMessages>,
    save_path: Option<Res<WorldSavePath>>,
) {
    let Some(content) = content else { return };
    let Ok(player) = players.single() else { return };

    for (mut pedestal, tf) in pedestals.iter_mut() {
        let d = (player.translation - tf.translation).length();
        if d > PEDESTAL_RADIUS + 1.5 {
            pedestal.hinted = false;
            continue;
        }
        if d > PEDESTAL_RADIUS {
            continue;
        }
        let zone_idx = pedestal.zone;
        if progress.unlocked_zones.contains(&zone_idx) {
            continue;
        }
        let zone = &content.def.zones[zone_idx];
        let earned = progress.island_trophies(&content.def, zone.island);

        if earned >= zone.trophies_required {
            progress.unlocked_zones.insert(zone_idx);
            messages.push(format!(
                "Turned in {} trophies — {} is unlocked! You can now build there.",
                zone.trophies_required, zone.name
            ));
            for (entity, barrier) in barriers.iter() {
                if barrier.zone == zone_idx {
                    commands.entity(entity).despawn();
                }
            }
            if let Some(ref save) = save_path {
                save_progress(&save.0, &progress);
            }
        } else if !pedestal.hinted {
            pedestal.hinted = true;
            messages.push(format!(
                "{} needs {} trophies from this island ({} so far).",
                zone.name, zone.trophies_required, earned
            ));
        }
    }
}

/// Keep the building system's view of locks in sync.
fn sync_build_locks(
    content: Option<Res<WorldContent>>,
    progress: Res<PlayerProgress>,
    active: Res<ActiveChallenge>,
    mut locks: ResMut<BuildLocks>,
) {
    let Some(content) = content else { return };
    locks.building_disabled = active.0.is_some();
    locks.locked_zones = content
        .def
        .zones
        .iter()
        .enumerate()
        .filter(|(i, _)| !progress.unlocked_zones.contains(i))
        .map(|(_, z)| zone_world_aabb(content.region_origin, z))
        .collect();
}

// ---------------------------------------------------------------------------
// Moving platforms
// ---------------------------------------------------------------------------

fn moving_platform_system(
    time: Res<Time>,
    mut platforms: Query<(&MovingPlatform, &Transform, &mut LinearVelocity)>,
) {
    let t = time.elapsed_secs();
    let dt = time.delta_secs().max(1e-4);
    for (platform, tf, mut vel) in platforms.iter_mut() {
        // Triangle wave 0→1→0 over `period`
        let phase = (t / platform.period).fract();
        let s = 1.0 - (2.0 * phase - 1.0).abs();
        let smooth = s * s * (3.0 - 2.0 * s);
        let target = platform.from.lerp(platform.to, smooth);
        vel.0 = (target - tf.translation) / dt;
    }
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

fn tick_messages(time: Res<Time>, mut messages: ResMut<HudMessages>) {
    for (_, timer) in messages.0.iter_mut() {
        timer.tick(time.delta());
    }
    messages.0.retain(|(_, timer)| !timer.is_finished());
}

fn challenge_hud(
    mut contexts: EguiContexts,
    content: Option<Res<WorldContent>>,
    active: Res<ActiveChallenge>,
    progress: Res<PlayerProgress>,
    messages: Res<HudMessages>,
) {
    let Some(content) = content else { return };
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::Area::new(egui::Id::new("challenge_hud"))
        .anchor(egui::Align2::CENTER_TOP, [0.0, 10.0])
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                let total = content.def.challenges.len();
                ui.colored_label(
                    egui::Color32::GOLD,
                    format!("Trophies: {}/{}", progress.trophies.len(), total),
                );

                if let Some(run) = active.0.as_ref() {
                    let def = &content.def.challenges[run.challenge];
                    let mut line = def.name.clone();
                    if let Some(timer) = run.timer.as_ref() {
                        line.push_str(&format!(
                            "  —  {:.1}s",
                            (timer.duration().as_secs_f32() - timer.elapsed_secs()).max(0.0)
                        ));
                    }
                    ui.colored_label(egui::Color32::LIGHT_GREEN, line);
                }

                for (msg, timer) in messages.0.iter() {
                    let remaining = (timer.duration().as_secs_f32() - timer.elapsed_secs())
                        .clamp(0.0, 1.0);
                    let alpha = (remaining * 255.0) as u8;
                    ui.colored_label(
                        egui::Color32::from_white_alpha(alpha.max(60)),
                        msg,
                    );
                }
            });
        });
}

// ---------------------------------------------------------------------------
// Progress persistence
// ---------------------------------------------------------------------------

fn progress_path(world_dir: &std::path::Path) -> std::path::PathBuf {
    world_dir.join("progress.json")
}

fn load_progress(world_dir: &std::path::Path) -> Option<PlayerProgress> {
    let text = std::fs::read_to_string(progress_path(world_dir)).ok()?;
    match serde_json::from_str(&text) {
        Ok(p) => Some(p),
        Err(e) => {
            warn!("[challenge] Failed to parse progress.json: {e}");
            None
        }
    }
}

fn save_progress(world_dir: &std::path::Path, progress: &PlayerProgress) {
    if let Err(e) = std::fs::create_dir_all(world_dir) {
        warn!("[challenge] Could not create world dir: {e}");
        return;
    }
    match serde_json::to_string_pretty(progress) {
        Ok(text) => {
            if let Err(e) = std::fs::write(progress_path(world_dir), text) {
                warn!("[challenge] Failed to save progress: {e}");
            }
        }
        Err(e) => warn!("[challenge] Failed to serialize progress: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_def() -> WorldDef {
        WorldDef::standard()
    }

    #[test]
    fn island_trophy_counting() {
        let def = test_def();
        let mut progress = PlayerProgress::default();
        progress.trophies.insert(0); // First Steps — island 0
        progress.trophies.insert(2); // Magma Hop — island 1
        assert_eq!(progress.island_trophies(&def, 0), 1);
        assert_eq!(progress.island_trophies(&def, 1), 1);
        assert_eq!(progress.island_trophies(&def, 2), 0);

        progress.trophies.insert(1); // Ridge Runner — island 0
        assert_eq!(progress.island_trophies(&def, 0), 2);
    }

    #[test]
    fn build_locks_respect_zones_and_active_run() {
        let def = test_def();
        let origin = Vec3::new(-2048.0, 0.0, -2048.0);
        let mut locks = BuildLocks::default();
        locks.locked_zones = def
            .zones
            .iter()
            .map(|z| zone_world_aabb(origin, z))
            .collect();

        // Center of the first locked zone is not buildable
        let (min, max) = locks.locked_zones[0];
        let inside = (min + max) * 0.5;
        assert!(!locks.can_build_at(inside));

        // Far away is buildable
        assert!(locks.can_build_at(Vec3::new(0.0, 5.0, 0.0)));

        // During a run nothing is buildable
        locks.building_disabled = true;
        assert!(!locks.can_build_at(Vec3::new(0.0, 5.0, 0.0)));
    }

    #[test]
    fn zone_unlock_threshold() {
        let def = test_def();
        let mut progress = PlayerProgress::default();
        // Earn both Haven trophies (challenges 0 and 1 are island 0)
        progress.trophies.insert(0);
        progress.trophies.insert(1);
        let zone = &def.zones[0];
        assert_eq!(zone.island, 0);
        assert!(progress.island_trophies(&def, 0) >= zone.trophies_required);
    }

    #[test]
    fn progress_roundtrip() {
        let dir = std::env::temp_dir().join("jumpblocks_progress_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut progress = PlayerProgress::default();
        progress.trophies.insert(3);
        progress.unlocked_zones.insert(1);
        save_progress(&dir, &progress);

        let loaded = load_progress(&dir).expect("progress should load");
        assert!(loaded.trophies.contains(&3));
        assert!(loaded.unlocked_zones.contains(&1));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
