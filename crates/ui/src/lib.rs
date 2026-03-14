//! # jumpblocks-ui — Off-Thread Immediate-Mode UI
//!
//! A custom UI system that renders entirely off the main (ECS) thread using
//! GPU-batched textured quads. Designed for fast, code-driven iteration on
//! game UI without involving the ECS in layout or drawing.
//!
//! ## Philosophy
//!
//! - **No ECS in UI.** The UI crate knows nothing about your game's components
//!   or resources. The game sends plain Rust data to the UI thread via channels,
//!   and the UI thread sends events back. This keeps the UI decoupled and
//!   testable.
//!
//! - **Off-thread rendering.** A dedicated OS thread runs the draw loop at its
//!   own cadence (~120 fps). The Bevy render world extracts finished frames
//!   from a channel and draws them as a full-screen overlay. The game thread
//!   never blocks on UI work.
//!
//! - **Immediate mode with persistent state.** Each frame, your [`UiDrawFn`]
//!   implementation receives `&mut self`, the current [`UiInputState`], and a
//!   [`Canvas`]. You draw rects, text, and clipped regions imperatively. Because
//!   the trait takes `&mut self`, you can keep animation timers, scroll offsets,
//!   selection indices, and other UI-only state directly on your struct — no ECS
//!   needed.
//!
//! - **GPU batching.** Every primitive (solid rect, glyph quad) becomes a
//!   textured quad. Solid rects sample a 1×1 white pixel in the atlas. Glyphs
//!   are rasterized with `cosmic-text` and packed into the atlas with `etagere`.
//!   The draw list is sorted into batches and rendered in a single draw call per
//!   batch with alpha blending.
//!
//! ## Architecture
//!
//! ```text
//! Game thread (ECS)              UI thread                Render world
//! ─────────────────              ─────────                ────────────
//! forward_input_to_ui ──input──► UiInputState
//!   (keyboard, mouse,            draw_fn.draw()
//!    gamepad→KeyCode)            Canvas → Vec<DrawCmd>
//!                                GlyphCache rasterizes
//!                                  ──UiFrame──►          extract_ui_frame
//!                                                        prepare_ui_buffers
//!                                                        UiOverlayNode::run
//!                                  ◄──UiEvent──
//! drain_ui_events ◄──────────────
//! ```
//!
//! ### Data flow
//!
//! 1. **Input forwarding** (`PreUpdate`): Bevy keyboard, mouse, and gamepad
//!    events are translated into [`UiInput`](bridge::UiInput) messages and sent
//!    to the UI thread. Gamepad buttons are mapped to `KeyCode` equivalents so
//!    the UI thread receives a single unified input stream.
//!
//! 2. **UI thread loop**: Drains input, calls your `draw()`, collects draw
//!    commands and atlas uploads into a [`UiFrame`](bridge::UiFrame), and sends
//!    it to the render world via channel.
//!
//! 3. **Extract** (`ExtractSchedule`): Drains the frame channel, keeping the
//!    latest frame but accumulating *all* atlas uploads (the UI thread outpaces
//!    the render world, so dropped frames' glyph uploads must not be lost).
//!
//! 4. **Prepare**: Builds GPU vertex/index buffers, uploads atlas dirty rects,
//!    and specializes the render pipeline for the current swapchain format.
//!
//! 5. **Render**: A `ViewNode` draws the batched quads as a full-screen overlay
//!    with alpha blending on top of the 3D scene.
//!
//! ### Game ↔ UI communication
//!
//! The UI crate provides the *transport* (channels, input state, canvas, draw
//! commands) but not the *protocol*. Your game defines its own data types and
//! creates its own crossbeam channels:
//!
//! ```rust,ignore
//! // Game-side types (plain Rust, no ECS)
//! struct MyUiData { health: f32, inventory_open: bool, items: Vec<Item> }
//! enum MyUiEvent { ItemSelected(usize), MenuClosed }
//!
//! // Your UiDrawFn holds receivers/senders for these
//! struct MyGameUi {
//!     data_rx: Receiver<MyUiData>,
//!     event_tx: Sender<MyUiEvent>,
//!     data: MyUiData,
//!     // ... UI-thread-only mutable state
//! }
//!
//! impl UiDrawFn for MyGameUi {
//!     fn draw(&mut self, input: &UiInputState, canvas: &mut Canvas) {
//!         while let Ok(d) = self.data_rx.try_recv() { self.data = d; }
//!         canvas.rect(0.0, 0.0, 200.0, 24.0, [0.1, 0.1, 0.1, 0.8]);
//!         canvas.text(8.0, 4.0, "Hello", 16.0, [1.0; 4]);
//!     }
//! }
//! ```
//!
//! On the game side, wrap the senders/receivers in Bevy resources and use
//! systems to push data and drain events. Use a `UiInputBlock`-style resource
//! with run conditions to suppress player input while menus are open.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`atlas`] | CPU-side RGBA atlas with `etagere` rect packing |
//! | [`bridge`] | Channel types, `UiInputState`, `UiFrame`, `UiEvent` |
//! | [`canvas`] | Immediate-mode drawing API (`rect`, `text`, `push_clip`, `hit_test`) |
//! | [`draw_cmd`] | `DrawCmd` → `UiVertex` batch builder |
//! | [`glyph`] | `cosmic-text` glyph rasterization + atlas insertion |
//! | [`render`] | Bevy render-world extract/prepare/node pipeline |
//! | [`thread`] | `UiDrawFn` trait + UI thread spawn/loop |

pub mod atlas;
pub mod bridge;
pub mod canvas;
pub mod draw_cmd;
pub mod ffd;
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
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
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
            let _ = tx.send(UiInput::MouseButton { button, pressed: true });
        }
        if mouse_buttons.just_released(button) {
            let _ = tx.send(UiInput::MouseButton { button, pressed: false });
        }
    }

    // Keyboard
    for code in keyboard.get_just_pressed() {
        let _ = tx.send(UiInput::Key { code: *code, pressed: true });
    }
    for code in keyboard.get_just_released() {
        let _ = tx.send(UiInput::Key { code: *code, pressed: false });
    }

    // Gamepad buttons → mapped to KeyCode so UI thread gets unified input
    for gamepad in gamepads.iter() {
        const MAPPINGS: &[(GamepadButton, KeyCode)] = &[
            (GamepadButton::DPadUp, KeyCode::ArrowUp),
            (GamepadButton::DPadDown, KeyCode::ArrowDown),
            (GamepadButton::DPadLeft, KeyCode::ArrowLeft),
            (GamepadButton::DPadRight, KeyCode::ArrowRight),
            (GamepadButton::South, KeyCode::Enter),
            (GamepadButton::East, KeyCode::Escape),
            (GamepadButton::North, KeyCode::Tab),
        ];
        for &(gp_button, key_code) in MAPPINGS {
            if gamepad.just_pressed(gp_button) {
                let _ = tx.send(UiInput::Key { code: key_code, pressed: true });
            }
            if gamepad.just_released(gp_button) {
                let _ = tx.send(UiInput::Key { code: key_code, pressed: false });
            }
        }
    }

    // Scroll
    for event in mouse_wheel.read() {
        let _ = tx.send(UiInput::Scroll { dx: event.x, dy: event.y });
    }
}

/// System that drains UI events from the channel and emits them as Bevy messages.
fn drain_ui_events(channels: Res<UiChannels>, mut writer: MessageWriter<UiEvent>) {
    while let Ok(event) = channels.event_rx.try_recv() {
        writer.write(event);
    }
}
