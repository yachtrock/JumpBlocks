use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;

use crate::bridge::{UiEvent, UiInputState};
use crate::draw_cmd::{DrawCmd, DrawMesh, DrawOp};
use crate::ffd::{self, FfdSim};
use crate::glyph::GlyphCache;

/// Result of a hit test for mouse interaction.
pub struct HitResult {
    pub hovered: bool,
    pub clicked: bool,
    pub id: u32,
}

/// Immediate-mode drawing context. Created fresh each frame on the UI thread.
pub struct Canvas<'a> {
    commands: Vec<DrawOp>,
    clip_stack: Vec<[f32; 4]>,
    glyph_cache: &'a Arc<Mutex<GlyphCache>>,
    event_tx: &'a Sender<UiEvent>,
    input: &'a UiInputState,
    next_id: u32,
    /// Active FFD for deforming subsequent draws. Set via `begin_ffd`/`end_ffd`.
    active_ffd: Option<*const FfdSim>,
}

// SAFETY: The FFD pointer is only valid during the draw call scope.
// Canvas is not Send/Sync and the pointer is only used within a single
// frame's draw() call where the FfdSim reference is guaranteed alive.
unsafe impl Send for Canvas<'_> {}

impl<'a> Canvas<'a> {
    pub(crate) fn new(
        glyph_cache: &'a Arc<Mutex<GlyphCache>>,
        event_tx: &'a Sender<UiEvent>,
        input: &'a UiInputState,
    ) -> Self {
        Self {
            commands: Vec::with_capacity(256),
            clip_stack: Vec::new(),
            glyph_cache,
            event_tx,
            input,
            next_id: 0,
            active_ffd: None,
        }
    }

    /// Begin drawing with FFD deformation. All subsequent `rect()` and `text()`
    /// calls will be tessellated and warped through the given FFD simulation
    /// until `end_ffd()` is called.
    ///
    /// The `FfdSim` must outlive this call (it's borrowed for the duration of
    /// the FFD scope).
    pub fn begin_ffd(&mut self, ffd: &FfdSim) {
        self.active_ffd = Some(ffd as *const FfdSim);
    }

    /// Stop FFD deformation. Subsequent draws are normal axis-aligned quads.
    pub fn end_ffd(&mut self) {
        self.active_ffd = None;
    }

    /// Draw a solid-color rectangle.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let uvs_rect = self.glyph_cache.lock().unwrap().atlas.white_pixel_uvs();

        if let Some(ffd_ptr) = self.active_ffd {
            // SAFETY: pointer is valid for the duration of the draw scope
            let ffd = unsafe { &*ffd_ptr };
            let (positions, tex_coords, indices) =
                ffd::tessellate_through_ffd(ffd, [x, y, w, h], uvs_rect);
            self.commands.push(DrawOp::Mesh(DrawMesh {
                positions,
                uvs: tex_coords,
                indices,
                color,
                atlas_page: 0,
                clip: self.current_clip(),
            }));
        } else {
            self.commands.push(DrawOp::Quad(DrawCmd {
                rect: [x, y, w, h],
                uvs: uvs_rect,
                color,
                atlas_page: 0,
                clip: self.current_clip(),
            }));
        }
    }

    /// Draw a text string at the given position.
    pub fn text(&mut self, x: f32, y: f32, text: &str, font_size: f32, color: [f32; 4]) {
        let mut cache = self.glyph_cache.lock().unwrap();
        let glyphs = cache.layout_text(text, font_size);
        let atlas_w = cache.atlas.width;
        let atlas_h = cache.atlas.height;

        let clip = self.current_clip();

        if let Some(ffd_ptr) = self.active_ffd {
            // SAFETY: pointer is valid for the duration of the draw scope
            let ffd = unsafe { &*ffd_ptr };
            for g in &glyphs {
                let region = &g.entry.region;
                let uvs = region.uvs(atlas_w, atlas_h);
                let glyph_rect = [x + g.x, y + g.y, region.width as f32, region.height as f32];
                let (positions, tex_coords, indices) =
                    ffd::tessellate_through_ffd(ffd, glyph_rect, uvs);
                self.commands.push(DrawOp::Mesh(DrawMesh {
                    positions,
                    uvs: tex_coords,
                    indices,
                    color,
                    atlas_page: 0,
                    clip,
                }));
            }
        } else {
            for g in &glyphs {
                let region = &g.entry.region;
                let uvs = region.uvs(atlas_w, atlas_h);
                self.commands.push(DrawOp::Quad(DrawCmd {
                    rect: [x + g.x, y + g.y, region.width as f32, region.height as f32],
                    uvs,
                    color,
                    atlas_page: 0,
                    clip,
                }));
            }
        }
    }

    /// Push a scissor clip rect. Subsequent draws are clipped to this region.
    pub fn push_clip(&mut self, x: f32, y: f32, w: f32, h: f32) {
        let new_clip = if let Some(parent) = self.clip_stack.last() {
            // Intersect with parent clip
            let x0 = x.max(parent[0]);
            let y0 = y.max(parent[1]);
            let x1 = (x + w).min(parent[0] + parent[2]);
            let y1 = (y + h).min(parent[1] + parent[3]);
            [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)]
        } else {
            [x, y, w, h]
        };
        self.clip_stack.push(new_clip);
    }

    /// Pop the most recent clip rect.
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    /// Test if the mouse is over a rectangular region. Returns interaction state.
    pub fn hit_test(&mut self, x: f32, y: f32, w: f32, h: f32) -> HitResult {
        let id = self.next_id;
        self.next_id += 1;

        let mx = self.input.mouse_pos.x;
        let my = self.input.mouse_pos.y;

        let hovered = mx >= x && mx < x + w && my >= y && my < y + h;
        let clicked = hovered && self.input.mouse_just_pressed[0];

        if hovered {
            let _ = self.event_tx.send(UiEvent::Hovered(id));
        }
        if clicked {
            let _ = self.event_tx.send(UiEvent::Clicked(id));
        }

        HitResult { hovered, clicked, id }
    }

    /// The window size in logical pixels.
    pub fn window_size(&self) -> bevy::math::Vec2 {
        self.input.window_size
    }

    /// Current input state.
    pub fn input(&self) -> &UiInputState {
        self.input
    }

    fn current_clip(&self) -> Option<[f32; 4]> {
        self.clip_stack.last().copied()
    }

    /// Consume the canvas and return the draw command list.
    pub(crate) fn finish(self) -> Vec<DrawOp> {
        self.commands
    }
}
