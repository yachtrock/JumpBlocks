//! Block stamping — author structures into a [`Region`] at generation time.
//!
//! Works in **global cell coordinates** (region-local): X/Z in
//! `0..REGION_XZ*32`, Y signed. One cell = `VOXEL_SIZE` (0.5) world units.
//! Standard blocks have a 2×2-cell footprint, so X/Z origins are snapped to
//! even cells; wedges additionally occupy 2 cells of height.

use std::sync::Arc;

use crate::chunk::{CHUNK_X, CHUNK_Y, CHUNK_Z, ChunkData};
use crate::coords::{ChunkPos, REGION_XZ, CHUNK_Y_MIN, CHUNK_Y_MAX};
use crate::region::Region;
use crate::shape::{Facing, SHAPE_CUBE};

/// Convert a global cell coordinate to (chunk index, local cell) on one axis.
#[inline]
fn split_axis(g: i32, chunk_size: i32) -> (i32, usize) {
    let c = g.div_euclid(chunk_size);
    let l = g.rem_euclid(chunk_size) as usize;
    (c, l)
}

/// Stamps blocks into a region, creating chunks on demand.
pub struct Stamper<'a> {
    pub region: &'a mut Region,
    /// Number of blocks placed so far.
    pub placed: usize,
    /// Number of placements skipped (occupied / out of bounds / straddling).
    pub skipped: usize,
}

impl<'a> Stamper<'a> {
    pub fn new(region: &'a mut Region) -> Self {
        Self { region, placed: 0, skipped: 0 }
    }

    /// Chunk position + local cell coords for a global cell, if in bounds.
    fn locate(&self, gx: i32, gy: i32, gz: i32) -> Option<(ChunkPos, (usize, usize, usize))> {
        if gx < 0 || gz < 0 || gx >= REGION_XZ * CHUNK_X as i32 || gz >= REGION_XZ * CHUNK_Z as i32
        {
            return None;
        }
        let (cx, lx) = split_axis(gx, CHUNK_X as i32);
        let (cy, ly) = split_axis(gy, CHUNK_Y as i32);
        let (cz, lz) = split_axis(gz, CHUNK_Z as i32);
        if cy < CHUNK_Y_MIN || cy > CHUNK_Y_MAX {
            return None;
        }
        Some((ChunkPos::new(cx, cy, cz), (lx, ly, lz)))
    }

    /// Whether the cell at global coords is occupied.
    pub fn is_occupied(&self, gx: i32, gy: i32, gz: i32) -> bool {
        let Some((pos, (lx, ly, lz))) = self.locate(gx, gy, gz) else {
            return false;
        };
        self.region
            .get_chunk(pos)
            .map(|slot| slot.data.is_occupied(lx, ly, lz))
            .unwrap_or(false)
    }

    /// Highest occupied cell Y in the column at (gx, gz), scanning
    /// `max_y` down to `min_y`. Returns the Y *above* the top occupied cell
    /// (i.e. the surface cell where something could stand), or `None`.
    pub fn surface_y(&self, gx: i32, gz: i32, min_y: i32, max_y: i32) -> Option<i32> {
        for gy in (min_y..=max_y).rev() {
            if self.is_occupied(gx, gy, gz) {
                return Some(gy + 1);
            }
        }
        None
    }

    /// Mutate (or create) the chunk containing a cell.
    fn with_chunk<R>(&mut self, pos: ChunkPos, f: impl FnOnce(&mut ChunkData) -> R) -> R {
        if self.region.get_chunk(pos).is_none() {
            self.region.set_chunk(pos, ChunkData::new());
        }
        let slot = self.region.get_chunk_mut(pos).expect("chunk just ensured");
        let data = Arc::make_mut(&mut slot.data);
        let r = f(data);
        slot.dirty.mark_data_changed();
        self.region.dirty.mark_chunk_changed();
        r
    }

    /// Place a standard 2×1×2 cube block with its origin at the given global
    /// cell (X/Z snapped down to even). Returns false if any footprint cell
    /// is occupied, out of bounds, or straddles a chunk boundary.
    pub fn place_cube(&mut self, gx: i32, gy: i32, gz: i32, texture: u16) -> bool {
        self.place_shape(gx, gy, gz, SHAPE_CUBE, Facing::North, texture, &[(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)])
    }

    /// Place a 2×2×2 wedge (ramp). The slope descends toward `facing`.
    pub fn place_wedge(&mut self, gx: i32, gy: i32, gz: i32, facing: Facing, texture: u16) -> bool {
        self.place_shape(
            gx,
            gy,
            gz,
            crate::shape::SHAPE_WEDGE,
            facing,
            texture,
            &[
                (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
                (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
            ],
        )
    }

    fn place_shape(
        &mut self,
        gx: i32,
        gy: i32,
        gz: i32,
        shape: u16,
        facing: Facing,
        texture: u16,
        occupied: &[(u8, u8, u8)],
    ) -> bool {
        let gx = gx & !1;
        let gz = gz & !1;

        let Some((pos, (lx, ly, lz))) = self.locate(gx, gy, gz) else {
            self.skipped += 1;
            return false;
        };

        // All occupied cells must land in the same chunk (no External-cell
        // support in the stamper) and be empty.
        for &(dx, dy, dz) in occupied {
            let Some((p2, (l2x, l2y, l2z))) =
                self.locate(gx + dx as i32, gy + dy as i32, gz + dz as i32)
            else {
                self.skipped += 1;
                return false;
            };
            if p2 != pos {
                self.skipped += 1;
                return false;
            }
            if self
                .region
                .get_chunk(p2)
                .map(|slot| slot.data.is_occupied(l2x, l2y, l2z))
                .unwrap_or(false)
            {
                self.skipped += 1;
                return false;
            }
        }

        self.with_chunk(pos, |data| {
            data.place_block(shape, facing, texture, lx, ly, lz, occupied);
        });
        self.placed += 1;
        true
    }

    /// Stamp a rectangular platform of cube blocks.
    /// `w`/`d` are in blocks (each block = 2 cells), `h` in cells (layers).
    pub fn platform(&mut self, gx: i32, gy: i32, gz: i32, w: i32, h: i32, d: i32, texture: u16) {
        let gx = gx & !1;
        let gz = gz & !1;
        for layer in 0..h {
            for bx in 0..w {
                for bz in 0..d {
                    self.place_cube(gx + bx * 2, gy + layer, gz + bz * 2, texture);
                }
            }
        }
    }

    /// Stamp a square platform centered (block-wise) on a cell.
    /// `size` is the width/depth in blocks.
    pub fn platform_centered(&mut self, gx: i32, gy: i32, gz: i32, size: i32, texture: u16) {
        let half = size; // blocks are 2 cells wide; offset = size blocks / 2 * 2 cells
        self.platform(gx - half, gy, gz - half, size, 1, size, texture);
    }

    /// Stamp a solid column of cube blocks from `gy0` (inclusive) up to
    /// `gy1` (exclusive) with a 1-block footprint.
    pub fn column(&mut self, gx: i32, gy0: i32, gy1: i32, gz: i32, texture: u16) {
        for gy in gy0..gy1 {
            self.place_cube(gx, gy, gz, texture);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::RegionId;
    use bevy::prelude::*;

    #[test]
    fn stamp_and_read_back() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let mut s = Stamper::new(&mut region);
        assert!(s.place_cube(100, 5, 200, 7));
        assert!(s.is_occupied(100, 5, 200));
        assert!(s.is_occupied(101, 5, 201));
        assert!(!s.is_occupied(102, 5, 200));
        assert_eq!(s.placed, 1);
    }

    #[test]
    fn stamp_rejects_occupied() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let mut s = Stamper::new(&mut region);
        assert!(s.place_cube(10, 0, 10, 1));
        assert!(!s.place_cube(10, 0, 10, 1));
        assert_eq!(s.skipped, 1);
    }

    #[test]
    fn wedge_rejects_chunk_straddle() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let mut s = Stamper::new(&mut region);
        // y=31 puts the wedge's upper cells in the next chunk up
        assert!(!s.place_wedge(10, 31, 10, Facing::North, 1));
        // y=30 fits
        assert!(s.place_wedge(10, 30, 10, Facing::North, 1));
    }

    #[test]
    fn surface_y_finds_top() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let mut s = Stamper::new(&mut region);
        s.platform(20, 4, 20, 2, 3, 2, 1); // cells y=4..6 filled
        assert_eq!(s.surface_y(20, 20, -4, 40), Some(7));
        assert_eq!(s.surface_y(40, 40, -4, 40), None);
    }

    #[test]
    fn negative_y_cells_work() {
        let mut region = Region::new(RegionId(0), Vec3::ZERO);
        let mut s = Stamper::new(&mut region);
        assert!(s.place_cube(50, -3, 50, 1));
        assert!(s.is_occupied(50, -3, 50));
        let pos = ChunkPos::new(1, -1, 1);
        assert!(region.has_chunk(pos));
    }
}
