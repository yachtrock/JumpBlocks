use std::collections::HashSet;

use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::draw_cmd::DrawOp;

/// Dirty rect to upload to the atlas GPU texture.
#[derive(Clone, Debug)]
pub struct AtlasUpload {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA
}

/// One frame's worth of UI draw output.
#[derive(Debug)]
pub struct UiFrame {
    pub commands: Vec<DrawOp>,
    pub atlas_uploads: Vec<AtlasUpload>,
    pub atlas_size: (u32, u32),
    pub dpi_scale: f32,
}

/// Input events forwarded from the main thread to the UI thread.
#[derive(Clone, Debug)]
pub enum UiInput {
    WindowResized {
        width: f32,
        height: f32,
        dpi: f32,
    },
    MouseMoved {
        x: f32,
        y: f32,
    },
    MouseButton {
        button: MouseButton,
        pressed: bool,
    },
    Key {
        code: KeyCode,
        pressed: bool,
    },
    Scroll {
        dx: f32,
        dy: f32,
    },
    CharTyped(char),
    /// Signal the UI thread to shut down.
    Shutdown,
}

/// Accumulated input state for the current frame.
#[derive(Clone, Debug, Default)]
pub struct UiInputState {
    pub window_size: Vec2,
    pub dpi_scale: f32,
    pub mouse_pos: Vec2,
    pub mouse_buttons: [bool; 3], // left, right, middle
    pub mouse_just_pressed: [bool; 3],
    pub mouse_just_released: [bool; 3],
    pub scroll_delta: Vec2,
    pub keys_pressed: HashSet<KeyCode>,
    pub keys_just_pressed: HashSet<KeyCode>,
    pub keys_just_released: HashSet<KeyCode>,
}

impl UiInputState {
    pub fn has_window_size(&self) -> bool {
        self.window_size.x > 0.0 && self.window_size.y > 0.0
    }

    /// Reset per-frame transient state (just_pressed, scroll, etc.)
    pub fn begin_frame(&mut self) {
        self.mouse_just_pressed = [false; 3];
        self.mouse_just_released = [false; 3];
        self.scroll_delta = Vec2::ZERO;
        self.keys_just_pressed.clear();
        self.keys_just_released.clear();
    }

    /// Check if a key was just pressed this frame.
    pub fn key_just_pressed(&self, code: KeyCode) -> bool {
        self.keys_just_pressed.contains(&code)
    }

    /// Check if a key is currently held.
    pub fn key_pressed(&self, code: KeyCode) -> bool {
        self.keys_pressed.contains(&code)
    }

    /// Apply a single input event.
    pub fn apply(&mut self, input: UiInput) {
        match input {
            UiInput::WindowResized { width, height, dpi } => {
                self.window_size = Vec2::new(width, height);
                self.dpi_scale = dpi;
            }
            UiInput::MouseMoved { x, y } => {
                self.mouse_pos = Vec2::new(x, y);
            }
            UiInput::MouseButton { button, pressed } => {
                let idx = mouse_button_index(button);
                if let Some(idx) = idx {
                    if pressed && !self.mouse_buttons[idx] {
                        self.mouse_just_pressed[idx] = true;
                    }
                    if !pressed && self.mouse_buttons[idx] {
                        self.mouse_just_released[idx] = true;
                    }
                    self.mouse_buttons[idx] = pressed;
                }
            }
            UiInput::Scroll { dx, dy } => {
                self.scroll_delta += Vec2::new(dx, dy);
            }
            UiInput::Key { code, pressed } => {
                if pressed {
                    if self.keys_pressed.insert(code) {
                        self.keys_just_pressed.insert(code);
                    }
                } else {
                    if self.keys_pressed.remove(&code) {
                        self.keys_just_released.insert(code);
                    }
                }
            }
            UiInput::CharTyped(_) | UiInput::Shutdown => {}
        }
    }
}

fn mouse_button_index(button: MouseButton) -> Option<usize> {
    match button {
        MouseButton::Left => Some(0),
        MouseButton::Right => Some(1),
        MouseButton::Middle => Some(2),
        _ => None,
    }
}

/// Events sent from the UI thread back to the game.
#[derive(Clone, Debug, bevy::ecs::message::Message)]
pub enum UiEvent {
    Clicked(u32),
    Hovered(u32),
}

/// All channels for communicating with the UI thread.
#[derive(Resource)]
pub struct UiChannels {
    pub input_tx: Sender<UiInput>,
    pub frame_rx: Receiver<UiFrame>,
    pub event_rx: Receiver<UiEvent>,
}

/// Channels held by the UI thread side.
pub struct UiThreadChannels {
    pub input_rx: Receiver<UiInput>,
    pub frame_tx: Sender<UiFrame>,
    pub event_tx: Sender<UiEvent>,
}

/// Create all channel pairs.
pub fn create_channels() -> (UiChannels, UiThreadChannels) {
    let (input_tx, input_rx) = unbounded();
    let (frame_tx, frame_rx) = unbounded();
    let (event_tx, event_rx) = unbounded();

    (
        UiChannels { input_tx, frame_rx, event_rx },
        UiThreadChannels { input_rx, frame_tx, event_tx },
    )
}
