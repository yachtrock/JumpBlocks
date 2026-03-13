#[cfg(target_os = "macos")]
mod macos;

use bevy::prelude::*;
use std::collections::HashMap;

pub struct NativeGamepadPlugin;

impl Plugin for NativeGamepadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NativeGamepads>();

        #[cfg(target_os = "macos")]
        {
            app.add_systems(PreStartup, macos::startup);
            app.add_systems(PreUpdate, macos::poll_gamepads);
        }
    }
}

/// Tracks native gamepad state for change detection.
#[derive(Resource, Default)]
pub struct NativeGamepads {
    /// Maps a platform controller ID to its tracked state.
    pub controllers: HashMap<usize, ControllerState>,
}

pub struct ControllerState {
    pub entity: Entity,
    pub name: String,
    /// True if this is a Nintendo controller (face buttons labeled differently).
    pub nintendo_layout: bool,
    /// Previous axis values: [LeftStickX, LeftStickY, RightStickX, RightStickY]
    pub prev_axes: [f32; 4],
    /// Previous button values indexed by ButtonIndex.
    pub prev_buttons: [f32; BUTTON_COUNT],
}

/// Button indices for our internal state tracking.
pub const BUTTON_COUNT: usize = 18;

#[repr(usize)]
#[derive(Clone, Copy)]
pub enum ButtonIndex {
    South = 0,
    East = 1,
    West = 2,
    North = 3,
    LeftTrigger = 4,
    LeftTrigger2 = 5,
    RightTrigger = 6,
    RightTrigger2 = 7,
    Select = 8,
    Start = 9,
    LeftThumb = 10,
    RightThumb = 11,
    DPadUp = 12,
    DPadDown = 13,
    DPadLeft = 14,
    DPadRight = 15,
    Mode = 16,
    C = 17,
}

impl ButtonIndex {
    pub fn to_gamepad_button(self) -> GamepadButton {
        match self {
            Self::South => GamepadButton::South,
            Self::East => GamepadButton::East,
            Self::West => GamepadButton::West,
            Self::North => GamepadButton::North,
            Self::LeftTrigger => GamepadButton::LeftTrigger,
            Self::LeftTrigger2 => GamepadButton::LeftTrigger2,
            Self::RightTrigger => GamepadButton::RightTrigger,
            Self::RightTrigger2 => GamepadButton::RightTrigger2,
            Self::Select => GamepadButton::Select,
            Self::Start => GamepadButton::Start,
            Self::LeftThumb => GamepadButton::LeftThumb,
            Self::RightThumb => GamepadButton::RightThumb,
            Self::DPadUp => GamepadButton::DPadUp,
            Self::DPadDown => GamepadButton::DPadDown,
            Self::DPadLeft => GamepadButton::DPadLeft,
            Self::DPadRight => GamepadButton::DPadRight,
            Self::Mode => GamepadButton::Mode,
            Self::C => GamepadButton::C,
        }
    }
}
