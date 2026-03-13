use bevy::prelude::*;
use bevy_tnua::prelude::*;
use serde::{Deserialize, Serialize};

use crate::player::{ControlScheme, Player, PlayerSettings};

pub struct PlayerStatePlugin;

impl Plugin for PlayerStatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (detect_player_state, apply_state_visuals));
    }
}

/// Represents the current movement state of a player character.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PlayerState {
    #[default]
    Idle,
    Walk,
    Run,
    Jump,
    Fall,
}

impl PlayerState {
    /// Returns the color associated with this state for visual feedback.
    pub fn color(&self) -> Color {
        match self {
            PlayerState::Idle => Color::srgb(0.2, 0.4, 0.9),  // Blue
            PlayerState::Walk => Color::srgb(0.2, 0.8, 0.3),  // Green
            PlayerState::Run => Color::srgb(0.9, 0.6, 0.1),   // Orange
            PlayerState::Jump => Color::srgb(0.9, 0.9, 0.2),  // Yellow
            PlayerState::Fall => Color::srgb(0.7, 0.2, 0.8),  // Purple
        }
    }
}

/// Detects the current player state from Tnua controller + input.
fn detect_player_state(
    mut player_query: Query<
        (
            &TnuaController<ControlScheme>,
            &PlayerSettings,
            &mut PlayerState,
        ),
        With<Player>,
    >,
) {
    for (controller, settings, mut state) in player_query.iter_mut() {
        let airborne = matches!(controller.is_airborne(), Ok(true));

        let new_state = if airborne {
            // If there's an active action (Jump is our only action), we're jumping
            if controller.current_action.is_some() {
                PlayerState::Jump
            } else {
                PlayerState::Fall
            }
        } else {
            // Grounded — check desired motion magnitude
            let motion = controller.basis.desired_motion;
            let speed = motion.length();
            if speed < 0.1 {
                PlayerState::Idle
            } else if speed > settings.run_multiplier * 0.5 {
                PlayerState::Run
            } else {
                PlayerState::Walk
            }
        };

        if *state != new_state {
            *state = new_state;
        }
    }
}

/// Changes the player body material color based on state.
fn apply_state_visuals(
    player_query: Query<(&PlayerState, &Children), (With<Player>, Changed<PlayerState>)>,
    visual_children: Query<&Children>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mesh_material_query: Query<&MeshMaterial3d<StandardMaterial>>,
) {
    for (state, children) in player_query.iter() {
        // The first child is PlayerVisual, its first child is the body mesh
        for child in children.iter() {
            if let Ok(grandchildren) = visual_children.get(child) {
                // First child of visual pivot is the body capsule
                if let Some(body_entity) = grandchildren.iter().next() {
                    if let Ok(mat_handle) = mesh_material_query.get(body_entity) {
                        if let Some(material) = materials.get_mut(&mat_handle.0) {
                            material.base_color = state.color();
                        }
                    }
                }
            }
        }
    }
}
