pub mod atlas;
pub mod bridge;
pub mod canvas;
pub mod draw_cmd;
pub mod glyph;
pub mod render;
pub mod thread;

use std::sync::{Arc, Mutex};

use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::window::CursorMoved;

use bridge::{UiChannels, UiEvent, UiInput, create_channels};
use glyph::GlyphCache;
use render::UiRenderAssetPlugin;
use thread::{UiDrawFn, spawn_ui_thread};

// Re-export key types for users
pub use bridge::UiInputState;
pub use canvas::Canvas;

/// Plugin that sets up the off-thread UI rendering system.
///
/// The draw function is stored in a Mutex so we can take it out in `finish()`.
pub struct UiPlugin<F: UiDrawFn> {
    draw_fn: Mutex<Option<F>>,
}

impl<F: UiDrawFn> UiPlugin<F> {
    pub fn new(draw_fn: F) -> Self {
        Self {
            draw_fn: Mutex::new(Some(draw_fn)),
        }
    }
}

impl<F: UiDrawFn> Plugin for UiPlugin<F> {
    fn build(&self, app: &mut App) {
        app.add_message::<UiEvent>();
        // Register shader as embedded asset (must happen during build phase)
        app.add_plugins(UiRenderAssetPlugin);
    }

    fn finish(&self, app: &mut App) {
        let (main_channels, thread_channels) = create_channels();

        // Create shared glyph cache
        let glyph_cache = Arc::new(Mutex::new(GlyphCache::new()));

        // Store the frame receiver before moving main_channels
        let frame_rx = main_channels.frame_rx.clone();

        // Insert main-world resources
        app.insert_resource(main_channels);

        // Take the draw function out of the mutex
        let draw_fn = self
            .draw_fn
            .lock()
            .unwrap()
            .take()
            .expect("UiPlugin::finish called twice");

        let _handle = spawn_ui_thread(thread_channels, glyph_cache, draw_fn);
        // Thread handle is leaked intentionally — the thread runs for the app's lifetime
        // and is signaled to stop via the Shutdown message when channels are dropped.

        // Add input forwarding and event draining systems
        app.add_systems(PreUpdate, forward_input_to_ui);
        app.add_systems(PostUpdate, drain_ui_events);

        // Set up render-world resources, systems, and graph node directly
        // (sub-plugins added in finish() don't get their own finish() called)
        render::setup_render_world(app, frame_rx);
    }
}

/// System that reads Bevy input events and sends them to the UI thread.
fn forward_input_to_ui(
    channels: Res<UiChannels>,
    windows: Query<&Window>,
    mut cursor_moved: MessageReader<CursorMoved>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel: MessageReader<MouseWheel>,
) {
    let tx = &channels.input_tx;

    // Window size + DPI
    if let Ok(window) = windows.single() {
        let _ = tx.send(UiInput::WindowResized {
            width: window.width(),
            height: window.height(),
            dpi: window.scale_factor(),
        });
    }

    // Cursor position
    for event in cursor_moved.read() {
        let _ = tx.send(UiInput::MouseMoved {
            x: event.position.x,
            y: event.position.y,
        });
    }

    // Mouse buttons
    for button in [MouseButton::Left, MouseButton::Right, MouseButton::Middle] {
        if mouse_buttons.just_pressed(button) {
            let _ = tx.send(UiInput::MouseButton {
                button,
                pressed: true,
            });
        }
        if mouse_buttons.just_released(button) {
            let _ = tx.send(UiInput::MouseButton {
                button,
                pressed: false,
            });
        }
    }

    // Scroll
    for event in mouse_wheel.read() {
        let _ = tx.send(UiInput::Scroll {
            dx: event.x,
            dy: event.y,
        });
    }
}

/// System that drains UI events from the channel and emits them as Bevy messages.
fn drain_ui_events(channels: Res<UiChannels>, mut writer: MessageWriter<UiEvent>) {
    while let Ok(event) = channels.event_rx.try_recv() {
        writer.write(event);
    }
}
