use std::sync::Arc;

use bevy::prelude::*;

use crate::shape::Facing;

pub const CHUNK_X: usize = 32;
pub const CHUNK_Y: usize = 32;
pub const CHUNK_Z: usize = 32;
pub const CHUNK_VOLUME: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;

/// Uniform cube voxel size. Chunk physical size = 32 * 0.5 = 16.0 world units per axis.
pub const VOXEL_SIZE: f32 = 0.5;

/// Cells occupied by a standard cube block (2x1x2).
pub const BLOCK_CELLS: [(u8, u8, u8); 4] = [
    (0,0,0), (1,0,0), (0,0,1), (1,0,1),
];

/// Cells occupied by a wedge block (2x2x2 — solid base + slope).
pub const WEDGE_CELLS: [(u8, u8, u8); 8] = [
    (0,0,0), (1,0,0), (0,0,1), (1,0,1),
    (0,1,0), (1,1,0), (0,1,1), (1,1,1),
];

/// A handle into the chunk's block list.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockId(pub u16);

/// Offset to a neighboring chunk (each component is -1, 0, or +1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChunkOffset {
    pub dx: i8,
    pub dy: i8,
    pub dz: i8,
}

/// What occupies a single cell in the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum Cell {
    Empty = 0,
    /// Block owned by this chunk.
    Local(BlockId),
    /// Block whose origin is in a neighboring chunk.
    External { neighbor: ChunkOffset, block_id: BlockId },
}

impl Default for Cell {
    fn default() -> Self {
        Cell::Empty
    }
}

impl Cell {
    #[inline]
    pub fn is_empty(self) -> bool {
        matches!(self, Cell::Empty)
    }

    #[inline]
    pub fn is_occupied(self) -> bool {
        !self.is_empty()
    }
}

/// A block placed in the chunk. Indexed by `BlockId`.
#[derive(Clone, Debug)]
pub struct Block {
    pub shape: u16,
    pub facing: Facing,
    pub texture: u16,
    /// Cell position of the block's origin within the chunk.
    pub origin: (u8, u8, u8),
}

/// Storage for cell and block data within a chunk.
#[derive(Clone)]
pub struct ChunkData {
    cells: Box<[Cell; CHUNK_VOLUME]>,
    pub blocks: Vec<Block>,
}

impl std::fmt::Debug for ChunkData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkData")
            .field("blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

impl Default for ChunkData {
    fn default() -> Self {
        Self {
            cells: Box::new([Cell::Empty; CHUNK_VOLUME]),
            blocks: Vec::new(),
        }
    }
}

impl ChunkData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from raw parts (used by deserialization).
    pub fn from_raw(cells: Box<[Cell; CHUNK_VOLUME]>, blocks: Vec<Block>) -> Self {
        Self { cells, blocks }
    }

    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        y * CHUNK_X * CHUNK_Z + z * CHUNK_X + x
    }

    /// Get a cell by its flat index (0..CHUNK_VOLUME).
    #[inline]
    pub fn get_cell_by_index(&self, idx: usize) -> Cell {
        if idx < CHUNK_VOLUME {
            self.cells[idx]
        } else {
            Cell::Empty
        }
    }

    /// Get the cell at the given position.
    pub fn get_cell(&self, x: usize, y: usize, z: usize) -> Cell {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.cells[Self::index(x, y, z)]
        } else {
            Cell::Empty
        }
    }

    /// Set the cell at the given position.
    fn set_cell(&mut self, x: usize, y: usize, z: usize, cell: Cell) {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.cells[Self::index(x, y, z)] = cell;
        }
    }

    /// Get the block for a given BlockId.
    pub fn get_block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(id.0 as usize)
    }

    /// Resolve a cell to its block (if any). For External cells, returns None
    /// (caller must look up the neighbor chunk).
    pub fn resolve_block(&self, x: usize, y: usize, z: usize) -> Option<&Block> {
        match self.get_cell(x, y, z) {
            Cell::Local(id) => self.get_block(id),
            _ => None,
        }
    }

    /// Check if a cell position is occupied.
    pub fn is_occupied(&self, x: usize, y: usize, z: usize) -> bool {
        self.get_cell(x, y, z).is_occupied()
    }

    /// Check if a neighbor position is occupied. Out-of-bounds is treated as empty.
    pub fn is_neighbor_occupied(&self, x: i32, y: i32, z: i32) -> bool {
        if x < 0 || y < 0 || z < 0 {
            return false;
        }
        self.is_occupied(x as usize, y as usize, z as usize)
    }

    /// Place a block, writing its cells into the grid.
    /// Returns the BlockId of the placed block.
    pub fn place_block(&mut self, shape: u16, facing: Facing, texture: u16, ox: usize, oy: usize, oz: usize, occupied: &[(u8, u8, u8)]) -> BlockId {
        let id = BlockId(self.blocks.len() as u16);
        self.blocks.push(Block {
            shape,
            facing,
            texture,
            origin: (ox as u8, oy as u8, oz as u8),
        });
        for &(dx, dy, dz) in occupied {
            let cx = ox + dx as usize;
            let cy = oy + dy as usize;
            let cz = oz + dz as usize;
            self.set_cell(cx, cy, cz, Cell::Local(id));
        }
        id
    }

    /// Convenience: place a 1x1x1 block (single cell).
    pub fn place_1x1x1(&mut self, x: usize, y: usize, z: usize, shape: u16, facing: Facing, texture: u16) -> BlockId {
        self.place_block(shape, facing, texture, x, y, z, &[(0, 0, 0)])
    }

    /// Convenience: place a standard 2x1x2 cube block.
    pub fn place_std(&mut self, x: usize, y: usize, z: usize, shape: u16, facing: Facing, texture: u16) -> BlockId {
        self.place_block(shape, facing, texture, x, y, z, &BLOCK_CELLS)
    }

    /// Convenience: place a 2x2x2 wedge block (solid base + slope).
    pub fn place_wedge(&mut self, x: usize, y: usize, z: usize, facing: Facing, texture: u16) -> BlockId {
        self.place_block(crate::shape::SHAPE_WEDGE, facing, texture, x, y, z, &WEDGE_CELLS)
    }

    /// Remove a block by its BlockId, clearing its cells.
    /// Note: does not compact the blocks vec, so BlockIds remain stable.
    pub fn remove_block(&mut self, id: BlockId, occupied: &[(u8, u8, u8)]) {
        if let Some(block) = self.blocks.get(id.0 as usize) {
            let (ox, oy, oz) = block.origin;
            for &(dx, dy, dz) in occupied {
                let cx = ox as usize + dx as usize;
                let cy = oy as usize + dy as usize;
                let cz = oz as usize + dz as usize;
                if self.get_cell(cx, cy, cz) == Cell::Local(id) {
                    self.set_cell(cx, cy, cz, Cell::Empty);
                }
            }
        }
    }

    /// Check whether a standard 2x1x2 block can be placed at `(x, y, z)`.
    pub fn can_place_std(&self, x: usize, y: usize, z: usize) -> bool {
        for &(dx, dy, dz) in &BLOCK_CELLS {
            let cx = x + dx as usize;
            let cy = y + dy as usize;
            let cz = z + dz as usize;
            if cx >= CHUNK_X || cy >= CHUNK_Y || cz >= CHUNK_Z {
                return false;
            }
            if self.get_cell(cx, cy, cz).is_occupied() {
                return false;
            }
        }
        for &(dx, dy, dz) in &BLOCK_CELLS {
            let cx = (x + dx as usize) as i32;
            let cy = (y + dy as usize) as i32;
            let cz = (z + dz as usize) as i32;
            for &(nx, ny, nz) in &[(-1i32,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)] {
                let tx = cx + nx;
                let ty = cy + ny;
                let tz = cz + nz;
                // Skip if inside the block itself (x..x+2, y..y+1, z..z+2)
                if tx >= x as i32 && tx < (x + 2) as i32
                    && ty >= y as i32 && ty < (y + 1) as i32
                    && tz >= z as i32 && tz < (z + 2) as i32
                {
                    continue;
                }
                if self.is_neighbor_occupied(tx, ty, tz) {
                    return true;
                }
            }
        }
        false
    }

    /// Check whether a block can be placed at `(x, y, z)`.
    pub fn can_build_at(&self, x: usize, y: usize, z: usize) -> bool {
        if x >= CHUNK_X || y >= CHUNK_Y || z >= CHUNK_Z {
            return false;
        }
        if self.get_cell(x, y, z).is_occupied() {
            return false;
        }

        let ix = x as i32;
        let iy = y as i32;
        let iz = z as i32;

        self.is_neighbor_occupied(ix - 1, iy, iz)
            || self.is_neighbor_occupied(ix + 1, iy, iz)
            || self.is_neighbor_occupied(ix, iy - 1, iz)
            || self.is_neighbor_occupied(ix, iy + 1, iz)
            || self.is_neighbor_occupied(ix, iy, iz - 1)
            || self.is_neighbor_occupied(ix, iy, iz + 1)
    }

    /// Get cell at signed coordinates. Out-of-bounds returns Empty.
    pub fn get_cell_signed(&self, x: i32, y: i32, z: i32) -> Cell {
        if x < 0 || y < 0 || z < 0 {
            return Cell::Empty;
        }
        self.get_cell(x as usize, y as usize, z as usize)
    }

    /// Validate the chunk data, checking for common errors.
    /// Returns a list of error messages. Empty = valid.
    pub fn validate(&self, shapes: &crate::shape::ShapeTable) -> Vec<String> {
        let mut errors = Vec::new();

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let id = BlockId(block_idx as u16);
            let (ox, oy, oz) = block.origin;

            // Check shape exists
            let Some(shape) = shapes.get(block.shape) else {
                errors.push(format!(
                    "Block {} at ({},{},{}): shape {} not found in ShapeTable",
                    block_idx, ox, oy, oz, block.shape
                ));
                continue;
            };

            // Check each occupied cell
            for &(dx, dy, dz) in &shape.occupied_cells {
                let cx = ox as usize + dx as usize;
                let cy = oy as usize + dy as usize;
                let cz = oz as usize + dz as usize;

                // Bounds check
                if cx >= CHUNK_X || cy >= CHUNK_Y || cz >= CHUNK_Z {
                    errors.push(format!(
                        "Block {} at ({},{},{}): cell ({},{},{}) out of chunk bounds",
                        block_idx, ox, oy, oz, cx, cy, cz
                    ));
                    continue;
                }

                // Cell should point back to this block
                let cell = self.get_cell(cx, cy, cz);
                match cell {
                    Cell::Empty => {
                        errors.push(format!(
                            "Block {} at ({},{},{}): cell ({},{},{}) is Empty, expected Local({})",
                            block_idx, ox, oy, oz, cx, cy, cz, block_idx
                        ));
                    }
                    Cell::Local(cell_id) if cell_id != id => {
                        errors.push(format!(
                            "Block {} at ({},{},{}): cell ({},{},{}) owned by block {} (overlap!)",
                            block_idx, ox, oy, oz, cx, cy, cz, cell_id.0
                        ));
                    }
                    Cell::Local(_) => {} // correct
                    Cell::External { .. } => {
                        errors.push(format!(
                            "Block {} at ({},{},{}): cell ({},{},{}) is External, expected Local({})",
                            block_idx, ox, oy, oz, cx, cy, cz, block_idx
                        ));
                    }
                }
            }
        }

        // Check for orphaned cells (cells pointing to non-existent blocks)
        for y in 0..CHUNK_Y {
            for z in 0..CHUNK_Z {
                for x in 0..CHUNK_X {
                    if let Cell::Local(id) = self.get_cell(x, y, z) {
                        if id.0 as usize >= self.blocks.len() {
                            errors.push(format!(
                                "Cell ({},{},{}): points to block {} which doesn't exist",
                                x, y, z, id.0
                            ));
                        }
                    }
                }
            }
        }

        errors
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkState {
    #[default]
    Unloaded,
    Loaded,
    Dirty,
    Meshing,
    Ready,
}

// ---------------------------------------------------------------------------
// Chunk neighbors
// ---------------------------------------------------------------------------

/// All 26 neighbor directions of a chunk in a 3×3×3 grid (excluding center).
#[derive(Clone)]
pub struct ChunkNeighbors {
    slots: [Option<Arc<ChunkData>>; 26],
}

impl Default for ChunkNeighbors {
    fn default() -> Self {
        Self::empty()
    }
}

impl ChunkNeighbors {
    pub fn empty() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }

    #[inline]
    fn offset_to_index(dx: i32, dy: i32, dz: i32) -> usize {
        debug_assert!((-1..=1).contains(&dx) && (-1..=1).contains(&dy) && (-1..=1).contains(&dz));
        debug_assert!(dx != 0 || dy != 0 || dz != 0, "center is not a neighbor");
        let raw = ((dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) as usize;
        if raw < 13 { raw } else { raw - 1 }
    }

    pub fn get(&self, dx: i32, dy: i32, dz: i32) -> Option<&ChunkData> {
        self.slots[Self::offset_to_index(dx, dy, dz)].as_deref()
    }

    /// Set a neighbor slot from an Arc (shared reference, no clone of data).
    pub fn set_arc(&mut self, dx: i32, dy: i32, dz: i32, data: Arc<ChunkData>) {
        self.slots[Self::offset_to_index(dx, dy, dz)] = Some(data);
    }

    /// Set a neighbor slot from owned data (wraps in Arc).
    pub fn set(&mut self, dx: i32, dy: i32, dz: i32, data: ChunkData) {
        self.slots[Self::offset_to_index(dx, dy, dz)] = Some(Arc::new(data));
    }

    /// Clear a neighbor slot.
    pub fn clear(&mut self, dx: i32, dy: i32, dz: i32) {
        self.slots[Self::offset_to_index(dx, dy, dz)] = None;
    }
}

/// Look up whether a cell is occupied at signed coordinates that may extend
/// into neighbor chunks. Returns true if occupied, false if empty or out of range.
pub fn is_cell_occupied(data: &ChunkData, neighbors: &ChunkNeighbors, x: i32, y: i32, z: i32) -> bool {
    let in_x = x >= 0 && x < CHUNK_X as i32;
    let in_y = y >= 0 && y < CHUNK_Y as i32;
    let in_z = z >= 0 && z < CHUNK_Z as i32;

    if in_x && in_y && in_z {
        return data.get_cell(x as usize, y as usize, z as usize).is_occupied();
    }

    let dx = if x < 0 { -1 } else if x >= CHUNK_X as i32 { 1 } else { 0 };
    let dy = if y < 0 { -1 } else if y >= CHUNK_Y as i32 { 1 } else { 0 };
    let dz = if z < 0 { -1 } else if z >= CHUNK_Z as i32 { 1 } else { 0 };

    let local_x = x - dx * CHUNK_X as i32;
    let local_y = y - dy * CHUNK_Y as i32;
    let local_z = z - dz * CHUNK_Z as i32;
    if local_x < 0 || local_x >= CHUNK_X as i32
        || local_y < 0 || local_y >= CHUNK_Y as i32
        || local_z < 0 || local_z >= CHUNK_Z as i32
    {
        return false;
    }

    match neighbors.get(dx, dy, dz) {
        Some(neighbor) => neighbor.get_cell(local_x as usize, local_y as usize, local_z as usize).is_occupied(),
        None => false,
    }
}

/// Resolve the block at signed coordinates (possibly in a neighbor chunk).
/// Returns the block's shape, facing, and texture if found.
pub fn resolve_block_at(data: &ChunkData, neighbors: &ChunkNeighbors, x: i32, y: i32, z: i32) -> Option<(u16, Facing, u16)> {
    let in_x = x >= 0 && x < CHUNK_X as i32;
    let in_y = y >= 0 && y < CHUNK_Y as i32;
    let in_z = z >= 0 && z < CHUNK_Z as i32;

    if in_x && in_y && in_z {
        return data.resolve_block(x as usize, y as usize, z as usize)
            .map(|b| (b.shape, b.facing, b.texture));
    }

    let dx = if x < 0 { -1 } else if x >= CHUNK_X as i32 { 1 } else { 0 };
    let dy = if y < 0 { -1 } else if y >= CHUNK_Y as i32 { 1 } else { 0 };
    let dz = if z < 0 { -1 } else if z >= CHUNK_Z as i32 { 1 } else { 0 };

    let local_x = x - dx * CHUNK_X as i32;
    let local_y = y - dy * CHUNK_Y as i32;
    let local_z = z - dz * CHUNK_Z as i32;
    if local_x < 0 || local_x >= CHUNK_X as i32
        || local_y < 0 || local_y >= CHUNK_Y as i32
        || local_z < 0 || local_z >= CHUNK_Z as i32
    {
        return None;
    }

    match neighbors.get(dx, dy, dz) {
        Some(neighbor) => neighbor.resolve_block(local_x as usize, local_y as usize, local_z as usize)
            .map(|b| (b.shape, b.facing, b.texture)),
        None => None,
    }
}

/// Resolve a cell to its block's full info. Returns (shape, facing, origin) if found.
pub fn resolve_block_info_at(data: &ChunkData, neighbors: &ChunkNeighbors, x: i32, y: i32, z: i32) -> Option<(u16, Facing, (u8, u8, u8))> {
    let in_x = x >= 0 && x < CHUNK_X as i32;
    let in_y = y >= 0 && y < CHUNK_Y as i32;
    let in_z = z >= 0 && z < CHUNK_Z as i32;

    if in_x && in_y && in_z {
        let cell = data.get_cell(x as usize, y as usize, z as usize);
        if let Cell::Local(id) = cell {
            return data.get_block(id).map(|b| (b.shape, b.facing, b.origin));
        }
        return None;
    }

    let dx = if x < 0 { -1 } else if x >= CHUNK_X as i32 { 1 } else { 0 };
    let dy = if y < 0 { -1 } else if y >= CHUNK_Y as i32 { 1 } else { 0 };
    let dz = if z < 0 { -1 } else if z >= CHUNK_Z as i32 { 1 } else { 0 };

    let local_x = x - dx * CHUNK_X as i32;
    let local_y = y - dy * CHUNK_Y as i32;
    let local_z = z - dz * CHUNK_Z as i32;
    if local_x < 0 || local_x >= CHUNK_X as i32
        || local_y < 0 || local_y >= CHUNK_Y as i32
        || local_z < 0 || local_z >= CHUNK_Z as i32
    {
        return None;
    }

    match neighbors.get(dx, dy, dz) {
        Some(neighbor) => {
            let cell = neighbor.get_cell(local_x as usize, local_y as usize, local_z as usize);
            if let Cell::Local(id) = cell {
                return neighbor.get_block(id).map(|b| (b.shape, b.facing, b.origin));
            }
            None
        }
        None => None,
    }
}

/// A pending block modification to be incorporated into a chunk.
#[derive(Debug, Clone)]
pub struct BlockModification {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub shape: u16,
    pub facing: Facing,
    pub texture: u16,
}

// ---------------------------------------------------------------------------
// ECS component
// ---------------------------------------------------------------------------

/// ECS component representing a voxel chunk.
#[derive(Component)]
pub struct Chunk {
    pub data: ChunkData,
    pub neighbors: ChunkNeighbors,
    pub state: ChunkState,
    pub world_aligned: bool,
    pub dynamic: bool,
    pub pending_modifications: Vec<BlockModification>,
}

impl Chunk {
    pub fn new(data: ChunkData) -> Self {
        Self {
            data,
            neighbors: ChunkNeighbors::empty(),
            state: ChunkState::Loaded,
            world_aligned: true,
            dynamic: false,
            pending_modifications: Vec::new(),
        }
    }

    /// Mark the chunk as needing a new mesh.
    pub fn mark_dirty(&mut self) {
        if self.state != ChunkState::Unloaded {
            self.state = ChunkState::Dirty;
        }
    }
}
