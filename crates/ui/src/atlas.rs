use etagere::{AllocId, BucketedAtlasAllocator, Size, size2};

use crate::bridge::AtlasUpload;

const INITIAL_ATLAS_SIZE: u32 = 1024;

/// Manages a CPU-side RGBA atlas texture with rectangle packing.
pub struct Atlas {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    packer: BucketedAtlasAllocator,
    pub pending_uploads: Vec<AtlasUpload>,
    /// True if the atlas was resized and needs a full re-upload.
    pub needs_full_upload: bool,
    /// Position of the 2×2 white pixel block, as allocated by the packer.
    white_pixel_x: u32,
    white_pixel_y: u32,
}

/// A region allocated in the atlas.
#[derive(Copy, Clone, Debug)]
pub struct AtlasRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub alloc_id: AllocId,
}

impl AtlasRegion {
    /// Get UV coordinates normalized to atlas size.
    pub fn uvs(&self, atlas_width: u32, atlas_height: u32) -> [f32; 4] {
        let aw = atlas_width as f32;
        let ah = atlas_height as f32;
        [
            self.x as f32 / aw,
            self.y as f32 / ah,
            (self.x + self.width) as f32 / aw,
            (self.y + self.height) as f32 / ah,
        ]
    }
}

impl Atlas {
    pub fn new() -> Self {
        let width = INITIAL_ATLAS_SIZE;
        let height = INITIAL_ATLAS_SIZE;
        let pixels = vec![0u8; (width * height * 4) as usize];

        let packer = BucketedAtlasAllocator::new(size2(width as i32, height as i32));

        let mut atlas = Self {
            pixels,
            width,
            height,
            packer,
            pending_uploads: Vec::new(),
            needs_full_upload: true,
            white_pixel_x: 0,
            white_pixel_y: 0,
        };

        // Allocate a 2×2 block through the packer and write white pixels there.
        // Using the packer's returned position (not hardcoded 0,0) ensures the
        // white pixel region is actually protected from future glyph allocations.
        let alloc = atlas.packer.allocate(size2(2, 2))
            .expect("failed to allocate white pixel in fresh atlas");
        let wp_x = alloc.rectangle.min.x as u32;
        let wp_y = alloc.rectangle.min.y as u32;
        atlas.white_pixel_x = wp_x;
        atlas.white_pixel_y = wp_y;

        for y in 0..2u32 {
            for x in 0..2u32 {
                let offset = (((wp_y + y) * width + wp_x + x) * 4) as usize;
                atlas.pixels[offset] = 255;
                atlas.pixels[offset + 1] = 255;
                atlas.pixels[offset + 2] = 255;
                atlas.pixels[offset + 3] = 255;
            }
        }

        atlas
    }

    /// UV coordinates for the 1×1 white pixel (center of the 2×2 block).
    pub fn white_pixel_uvs(&self) -> [f32; 4] {
        let u = (self.white_pixel_x as f32 + 0.5) / self.width as f32;
        let v = (self.white_pixel_y as f32 + 0.5) / self.height as f32;
        [u, v, u, v]
    }

    /// Allocate a region and copy pixel data into it.
    /// Returns None if the atlas is full (caller should trigger resize).
    pub fn allocate(&mut self, width: u32, height: u32, rgba_pixels: &[u8]) -> Option<AtlasRegion> {
        let alloc = self.packer.allocate(size2(width as i32, height as i32))?;
        let rect = alloc.rectangle;
        let x = rect.min.x as u32;
        let y = rect.min.y as u32;

        // Copy pixels into atlas
        for row in 0..height {
            let src_start = (row * width * 4) as usize;
            let src_end = src_start + (width * 4) as usize;
            let dst_start = (((y + row) * self.width + x) * 4) as usize;
            let dst_end = dst_start + (width * 4) as usize;
            self.pixels[dst_start..dst_end].copy_from_slice(&rgba_pixels[src_start..src_end]);
        }

        // Track dirty rect for GPU upload
        self.pending_uploads.push(AtlasUpload {
            x,
            y,
            width,
            height,
            pixels: rgba_pixels[..(width * height * 4) as usize].to_vec(),
        });

        Some(AtlasRegion {
            x,
            y,
            width,
            height,
            alloc_id: alloc.id,
        })
    }

    /// Grow the atlas to double its size. Existing data is preserved.
    /// Returns true if successfully grown.
    pub fn grow(&mut self) -> bool {
        let new_width = self.width * 2;
        let new_height = self.height * 2;

        // Cap at a reasonable max
        if new_width > 8192 {
            return false;
        }

        let mut new_pixels = vec![0u8; (new_width * new_height * 4) as usize];

        // Copy old data
        for row in 0..self.height {
            let src_start = (row * self.width * 4) as usize;
            let src_end = src_start + (self.width * 4) as usize;
            let dst_start = (row * new_width * 4) as usize;
            let dst_end = dst_start + (self.width * 4) as usize;
            new_pixels[dst_start..dst_end].copy_from_slice(&self.pixels[src_start..src_end]);
        }

        self.pixels = new_pixels;
        self.width = new_width;
        self.height = new_height;
        self.packer.grow(Size::new(new_width as i32, new_height as i32));
        self.needs_full_upload = true;

        true
    }

    /// Drain pending uploads (for the render thread to consume).
    pub fn take_uploads(&mut self) -> Vec<AtlasUpload> {
        std::mem::take(&mut self.pending_uploads)
    }
}
