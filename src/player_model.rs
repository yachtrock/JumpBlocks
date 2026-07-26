//! The player's rigged character model and its animation driver.
//!
//! The model is authored in Blender by `tools/build_player_model.py`
//! (source: `assets/models/player.blend`, game asset: `assets/models/player.glb`)
//! with five named clips: Idle, Walk, Run, Jump, Fall. This module swaps the
//! placeholder capsule for the skinned model once the glTF loads and
//! crossfades between clips as [`PlayerState`] changes.
//!
//! Everything meant for feel-tuning lives in [`PlayerAnimSettings`].

use std::time::Duration;

use bevy::gltf::Gltf;
use bevy::prelude::*;

use crate::player::{Player, PlayerVisual};
use crate::player_state::PlayerState;

pub struct PlayerModelPlugin;

impl Plugin for PlayerModelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerAnimSettings>()
            .add_systems(Startup, start_model_load)
            .add_systems(
                Update,
                (swap_in_model, bind_animation_player, drive_player_animation),
            );
    }
}

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/// Animation + attachment tuning knobs for the player model.
#[derive(Resource, Debug, Clone)]
pub struct PlayerAnimSettings {
    /// Uniform model scale (authored height ≈ 1.4 m).
    pub scale: f32,
    /// Model root offset below the physics origin — the controller floats at
    /// `float_height`, so the feet sit this far under the capsule center.
    pub y_offset: f32,
    /// Crossfade duration between states, seconds.
    pub fade: f32,
    /// Per-state playback speed multipliers.
    pub idle_speed: f32,
    pub walk_speed: f32,
    pub run_speed: f32,
    pub jump_speed: f32,
    pub fall_speed: f32,
}

impl Default for PlayerAnimSettings {
    fn default() -> Self {
        Self {
            scale: 1.1,
            y_offset: -1.5,
            fade: 0.15,
            idle_speed: 1.0,
            walk_speed: 1.5,
            run_speed: 1.6,
            jump_speed: 1.3,
            fall_speed: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Load state
// ---------------------------------------------------------------------------

/// The five clips, indexed in [`PlayerState`] order.
const CLIP_NAMES: [&str; 5] = ["Idle", "Walk", "Run", "Jump", "Fall"];

#[derive(Resource)]
struct PlayerModel {
    gltf: Handle<Gltf>,
    /// Set once the scene has been swapped in.
    graph: Option<Handle<AnimationGraph>>,
    nodes: [AnimationNodeIndex; 5],
}

fn start_model_load(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(PlayerModel {
        gltf: asset_server.load("models/player.glb"),
        graph: None,
        nodes: [AnimationNodeIndex::new(0); 5],
    });
}

// ---------------------------------------------------------------------------
// Swap the placeholder capsule for the skinned model
// ---------------------------------------------------------------------------

fn swap_in_model(
    mut commands: Commands,
    mut model: ResMut<PlayerModel>,
    settings: Res<PlayerAnimSettings>,
    gltfs: Res<Assets<Gltf>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    visuals: Query<(Entity, Option<&Children>), With<PlayerVisual>>,
) {
    if model.graph.is_some() {
        return;
    }
    let Some(gltf) = gltfs.get(&model.gltf) else {
        return;
    };
    let Ok((visual_entity, children)) = visuals.single() else {
        return;
    };

    // Build the animation graph from the named clips.
    let mut clips = Vec::with_capacity(CLIP_NAMES.len());
    for name in CLIP_NAMES {
        match gltf.named_animations.get(name) {
            Some(clip) => clips.push(clip.clone()),
            None => {
                error!("[player_model] player.glb is missing animation '{name}'");
                return;
            }
        }
    }
    let (graph, nodes) = AnimationGraph::from_clips(clips);
    model.nodes = [nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]];
    model.graph = Some(graphs.add(graph));

    // Out with the placeholder capsule + visor…
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }
    // …in with the rigged model.
    let scene = gltf.scenes[0].clone();
    commands.entity(visual_entity).with_children(|parent| {
        parent.spawn((
            SceneRoot(scene),
            // Blender's -Y forward exports facing +Z; the game's forward is
            // -Z, so spin the model half a turn.
            Transform::from_xyz(0.0, settings.y_offset, 0.0)
                .with_scale(Vec3::splat(settings.scale))
                .with_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
        ));
    });
    info!("[player_model] Skinned player model attached");
}

// ---------------------------------------------------------------------------
// Bind the scene's AnimationPlayer to our graph once it spawns
// ---------------------------------------------------------------------------

fn bind_animation_player(
    mut commands: Commands,
    model: Option<Res<PlayerModel>>,
    settings: Res<PlayerAnimSettings>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    let Some(model) = model else { return };
    let Some(ref graph) = model.graph else { return };

    for (entity, mut player) in players.iter_mut() {
        let mut transitions = AnimationTransitions::new();
        transitions
            .play(&mut player, model.nodes[0], Duration::ZERO)
            .set_speed(settings.idle_speed)
            .repeat();
        commands
            .entity(entity)
            .insert((AnimationGraphHandle(graph.clone()), transitions));
        info!("[player_model] AnimationPlayer bound (idle)");
    }
}

// ---------------------------------------------------------------------------
// Crossfade clips as the player state changes
// ---------------------------------------------------------------------------

fn drive_player_animation(
    model: Option<Res<PlayerModel>>,
    settings: Res<PlayerAnimSettings>,
    states: Query<&PlayerState, (With<Player>, Changed<PlayerState>)>,
    mut players: Query<(&mut AnimationPlayer, &mut AnimationTransitions)>,
) {
    let Some(model) = model else { return };
    if model.graph.is_none() {
        return;
    }
    let Ok(state) = states.single() else { return };

    let (index, speed, looping) = match state {
        PlayerState::Idle => (0, settings.idle_speed, true),
        PlayerState::Walk => (1, settings.walk_speed, true),
        PlayerState::Run => (2, settings.run_speed, true),
        PlayerState::Jump => (3, settings.jump_speed, false),
        PlayerState::Fall => (4, settings.fall_speed, true),
    };

    for (mut player, mut transitions) in players.iter_mut() {
        let anim = transitions.play(
            &mut player,
            model.nodes[index],
            Duration::from_secs_f32(settings.fade),
        );
        anim.set_speed(speed);
        if looping {
            anim.repeat();
        }
    }
}
