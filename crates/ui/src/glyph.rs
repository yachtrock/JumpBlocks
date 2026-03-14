use std::collections::HashMap;

use cosmic_text::{
    Attrs, Buffer, CacheKeyFlags, Family, FontSystem, Metrics, Shaping, SwashCache, SwashContent,
};

use crate::atlas::{Atlas, AtlasRegion};

const DEFAULT_FONT_DATA: &[u8] = include_bytes!("../assets/fonts/FiraSans-Regular.ttf");

/// Key for looking up a cached glyph in the atlas.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GlyphKey {
    pub cache_key_flags: CacheKeyFlags,
    pub glyph_id: u16,
    pub font_size_bits: u32, // f32 bits for hashing
    pub x_bin: cosmic_text::SubpixelBin,
    pub y_bin: cosmic_text::SubpixelBin,
}

/// Cached glyph atlas entry.
#[derive(Clone, Copy, Debug)]
pub struct GlyphEntry {
    pub region: AtlasRegion,
    /// Offset from the layout position to where the glyph image starts.
    pub offset_x: i32,
    pub offset_y: i32,
}

/// Manages font system, glyph rasterization, and atlas insertion.
pub struct GlyphCache {
    pub font_system: FontSystem,
    swash_cache: SwashCache,
    pub atlas: Atlas,
    glyph_map: HashMap<GlyphKey, GlyphEntry>,
}

impl GlyphCache {
    pub fn new() -> Self {
        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(DEFAULT_FONT_DATA.to_vec());

        Self {
            font_system,
            swash_cache: SwashCache::new(),
            atlas: Atlas::new(),
            glyph_map: HashMap::new(),
        }
    }

    /// Look up or rasterize a glyph, returning its atlas entry.
    pub fn get_or_insert(
        &mut self,
        key: GlyphKey,
        cache_key: cosmic_text::CacheKey,
    ) -> Option<GlyphEntry> {
        if let Some(entry) = self.glyph_map.get(&key) {
            return Some(*entry);
        }

        // Rasterize the glyph
        let image = self.swash_cache.get_image(&mut self.font_system, cache_key).as_ref()?;

        let width = image.placement.width;
        let height = image.placement.height;

        if width == 0 || height == 0 {
            return None;
        }

        // Convert to RGBA if needed
        let rgba_pixels = match image.content {
            SwashContent::Mask => {
                let mut rgba = vec![0u8; (width * height * 4) as usize];
                for (i, &alpha) in image.data.iter().enumerate() {
                    let offset = i * 4;
                    rgba[offset] = 255;
                    rgba[offset + 1] = 255;
                    rgba[offset + 2] = 255;
                    rgba[offset + 3] = alpha;
                }
                rgba
            }
            SwashContent::Color => {
                // Already RGBA
                image.data.clone()
            }
            SwashContent::SubpixelMask => {
                // Treat as grayscale mask (average channels)
                let pixel_count = (width * height) as usize;
                let mut rgba = vec![0u8; pixel_count * 4];
                for i in 0..pixel_count {
                    let r = image.data[i * 3];
                    let g = image.data[i * 3 + 1];
                    let b = image.data[i * 3 + 2];
                    let avg = ((r as u16 + g as u16 + b as u16) / 3) as u8;
                    rgba[i * 4] = 255;
                    rgba[i * 4 + 1] = 255;
                    rgba[i * 4 + 2] = 255;
                    rgba[i * 4 + 3] = avg;
                }
                rgba
            }
        };

        // Try to allocate in atlas, growing if needed
        let region = match self.atlas.allocate(width, height, &rgba_pixels) {
            Some(r) => r,
            None => {
                if !self.atlas.grow() {
                    return None; // Atlas maxed out
                }
                self.atlas.allocate(width, height, &rgba_pixels)?
            }
        };

        let entry = GlyphEntry {
            region,
            offset_x: image.placement.left,
            offset_y: image.placement.top,
        };

        self.glyph_map.insert(key, entry);
        Some(entry)
    }

    /// Lay out a line of text and return glyph positions + atlas entries.
    pub fn layout_text(
        &mut self,
        text: &str,
        font_size: f32,
    ) -> Vec<LayoutGlyph> {
        let metrics = Metrics::new(font_size, font_size * 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_size(&mut self.font_system, Some(10000.0), Some(font_size * 2.0));
        buffer.set_text(&mut self.font_system, text, Attrs::new().family(Family::SansSerif), Shaping::Advanced);
        buffer.shape_until_scroll(&mut self.font_system, false);

        let mut glyphs = Vec::new();

        for run in buffer.layout_runs() {
            for glyph in run.glyphs.iter() {
                let physical = glyph.physical((0.0, 0.0), 1.0);

                let key = GlyphKey {
                    cache_key_flags: physical.cache_key.flags,
                    glyph_id: physical.cache_key.glyph_id,
                    font_size_bits: physical.cache_key.font_size_bits,
                    x_bin: physical.cache_key.x_bin,
                    y_bin: physical.cache_key.y_bin,
                };

                if let Some(entry) = self.get_or_insert(key, physical.cache_key) {
                    glyphs.push(LayoutGlyph {
                        x: physical.x as f32 + entry.offset_x as f32,
                        y: run.line_y + physical.y as f32 - entry.offset_y as f32,
                        entry,
                    });
                }
            }
        }

        glyphs
    }
}

/// A positioned glyph ready for rendering.
pub struct LayoutGlyph {
    pub x: f32,
    pub y: f32,
    pub entry: GlyphEntry,
}
