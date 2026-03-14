use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::bridge::{AtlasUpload, UiFrame, UiInput, UiInputState, UiThreadChannels};
use crate::canvas::Canvas;
use crate::glyph::GlyphCache;

/// Trait for the user's UI draw function.
///
/// Takes `&mut self` so implementations can hold persistent UI-thread state
/// (animation timers, scroll offsets, toggle flags, etc.)
pub trait UiDrawFn: Send + 'static {
    fn draw(&mut self, input: &UiInputState, canvas: &mut Canvas);
}

impl<F: FnMut(&UiInputState, &mut Canvas) + Send + 'static> UiDrawFn for F {
    fn draw(&mut self, input: &UiInputState, canvas: &mut Canvas) {
        (self)(input, canvas);
    }
}

/// Spawn the UI thread. Returns the thread handle.
pub fn spawn_ui_thread(
    channels: UiThreadChannels,
    glyph_cache: Arc<Mutex<GlyphCache>>,
    draw_fn: impl UiDrawFn,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("ui-thread".into())
        .spawn(move || {
            ui_thread_loop(channels, glyph_cache, draw_fn);
        })
        .expect("failed to spawn UI thread")
}

fn ui_thread_loop(
    channels: UiThreadChannels,
    glyph_cache: Arc<Mutex<GlyphCache>>,
    mut draw_fn: impl UiDrawFn,
) {
    let UiThreadChannels { input_rx, frame_tx, event_tx } = channels;

    let mut input_state = UiInputState::default();

    loop {
        // Reset per-frame transient state
        input_state.begin_frame();

        // Drain all pending input events
        let mut got_input = false;
        while let Ok(input) = input_rx.try_recv() {
            if matches!(input, UiInput::Shutdown) {
                return;
            }
            input_state.apply(input);
            got_input = true;
        }

        // If we haven't received window size yet, wait
        if !input_state.has_window_size() {
            // Block briefly on the channel to avoid spinning
            match input_rx.recv_timeout(Duration::from_millis(16)) {
                Ok(input) => {
                    if matches!(input, UiInput::Shutdown) {
                        return;
                    }
                    input_state.apply(input);
                }
                Err(_) => {}
            }
            continue;
        }

        // Create canvas and run user's draw function
        let mut canvas = Canvas::new(&glyph_cache, &event_tx, &input_state);
        draw_fn.draw(&input_state, &mut canvas);
        let commands = canvas.finish();

        // Collect atlas state
        let (atlas_uploads, atlas_size, dpi_scale) = {
            let mut cache = glyph_cache.lock().unwrap();
            let uploads = if cache.atlas.needs_full_upload {
                cache.atlas.needs_full_upload = false;
                // Full upload: send entire atlas as one upload
                vec![AtlasUpload {
                    x: 0,
                    y: 0,
                    width: cache.atlas.width,
                    height: cache.atlas.height,
                    pixels: cache.atlas.pixels.clone(),
                }]
            } else {
                cache.atlas.take_uploads()
            };
            let size = (cache.atlas.width, cache.atlas.height);
            (uploads, size, input_state.dpi_scale)
        };

        let frame = UiFrame {
            commands,
            atlas_uploads,
            atlas_size,
            dpi_scale,
        };

        // Send frame; if the render world is behind, drop old frames
        let _ = frame_tx.send(frame);

        // Target ~120fps for the UI thread (smooth interactions)
        if !got_input {
            std::thread::sleep(Duration::from_millis(8));
        } else {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
