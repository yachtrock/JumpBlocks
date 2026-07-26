use bevy::prelude::*;
use bevy_tnua::prelude::*;
use serde::{Deserialize, Serialize};

use crate::player::{ControlScheme, Player, PlayerSettings};

pub struct PlayerStatePlugin;

impl Plugin for PlayerStatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, detect_player_state);
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

