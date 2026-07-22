//! Binary serialization for chunk data.
//!
//! ## Chunk file format (`.chunk`)
//!
//! ```text
//! [magic: 4 bytes]  "JBCK"
//! [version: u8]     1
//! [block_count: u16 LE]
//! [block_table: block_count × 8 bytes each]
//!     shape:   u16 LE
//!     facing:  u8
//!     texture: u16 LE
//!     origin:  (u8, u8, u8)
//! [cell_grid: RLE compressed]
//!     Cells are serialized in Y-Z-X order (matching the index layout).
//!     Each run: [count: u16 LE] [cell_type: u8] [payload: 0 or 2 bytes]
//!       cell_type 0 = Empty           (no payload)
//!       cell_type 1 = Local(BlockId)  (payload: block_id u16 LE)
//!       cell_type 2 = External        (payload: not stored on disk — reconstructed)
//!
//! External cells are NOT saved. They are reconstructed from neighbor data
//! at load time. On disk, External cells are written as Empty.
//! ```

use std::fs;
use std::io::{self, Read, Cursor};

use crate::chunk::*;
use crate::shape::Facing;

const MAGIC: &[u8; 4] = b"JBCK";
const VERSION: u8 = 1;

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serialize a `ChunkData` to bytes.
pub fn serialize_chunk(data: &ChunkData) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1024);

    // Header
    buf.extend_from_slice(MAGIC);
    buf.push(VERSION);

    // Block table
    let block_count = data.blocks.len() as u16;
    buf.extend_from_slice(&block_count.to_le_bytes());

    for block in &data.blocks {
        buf.extend_from_slice(&block.shape.to_le_bytes());
        buf.push(block.facing as u8);
        buf.extend_from_slice(&block.texture.to_le_bytes());
        buf.push(block.origin.0);
        buf.push(block.origin.1);
        buf.push(block.origin.2);
    }

    // Cell grid — RLE encode
    // We iterate in the same order as the index function: y * (X*Z) + z * X + x
    rle_encode_cells(data, &mut buf);

    buf
}

/// Deserialize a `ChunkData` from bytes.
pub fn deserialize_chunk(bytes: &[u8]) -> io::Result<ChunkData> {
    let mut cursor = Cursor::new(bytes);

    // Magic
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic: expected JBCK"));
    }

    // Version
    let mut ver = [0u8; 1];
    cursor.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported chunk version {}", ver[0]),
        ));
    }

    // Block table
    let block_count = read_u16_le(&mut cursor)?;
    let mut blocks = Vec::with_capacity(block_count as usize);
    for _ in 0..block_count {
        let shape = read_u16_le(&mut cursor)?;
        let mut facing_byte = [0u8; 1];
        cursor.read_exact(&mut facing_byte)?;
        let facing = Facing::from_bits(facing_byte[0]);
        let texture = read_u16_le(&mut cursor)?;
        let mut origin = [0u8; 3];
        cursor.read_exact(&mut origin)?;
        blocks.push(Block {
            shape,
            facing,
            texture,
            origin: (origin[0], origin[1], origin[2]),
        });
    }

    // Cell grid — RLE decode
    let mut cells = Box::new([Cell::Empty; CHUNK_VOLUME]);
    rle_decode_cells(&mut cursor, &mut cells)?;

    Ok(ChunkData::from_raw(cells, blocks))
}

// ---------------------------------------------------------------------------
// RLE encoding
// ---------------------------------------------------------------------------

/// Cell type tags for the RLE stream.
const CELL_EMPTY: u8 = 0;
const CELL_LOCAL: u8 = 1;

/// RLE-encode the cell grid into the buffer.
fn rle_encode_cells(data: &ChunkData, buf: &mut Vec<u8>) {
    let mut i = 0;
    while i < CHUNK_VOLUME {
        let cell = data.get_cell_by_index(i);
        let (tag, payload) = cell_to_tag_payload(cell);

        // Count consecutive identical cells
        let mut run_len: u16 = 1;
        loop {
            let idx = i + run_len as usize;
            if idx >= CHUNK_VOLUME || run_len >= u16::MAX {
                break;
            }
            let next = data.get_cell_by_index(idx);
            let (next_tag, next_payload) = cell_to_tag_payload(next);
            if next_tag != tag || next_payload != payload {
                break;
            }
            run_len += 1;
        }

        buf.extend_from_slice(&run_len.to_le_bytes());
        buf.push(tag);
        if tag == CELL_LOCAL {
            buf.extend_from_slice(&payload.to_le_bytes());
        }

        i += run_len as usize;
    }
}

fn cell_to_tag_payload(cell: Cell) -> (u8, u16) {
    match cell {
        Cell::Empty => (CELL_EMPTY, 0),
        Cell::Local(id) => (CELL_LOCAL, id.0),
        // External cells are not saved — treat as empty on disk
        Cell::External { .. } => (CELL_EMPTY, 0),
    }
}

/// RLE-decode the cell grid from the cursor.
fn rle_decode_cells(cursor: &mut Cursor<&[u8]>, cells: &mut Box<[Cell; CHUNK_VOLUME]>) -> io::Result<()> {
    let mut i = 0;
    while i < CHUNK_VOLUME {
        let run_len = read_u16_le(cursor)? as usize;
        let mut tag = [0u8; 1];
        cursor.read_exact(&mut tag)?;

        let cell = match tag[0] {
            CELL_EMPTY => Cell::Empty,
            CELL_LOCAL => {
                let block_id = read_u16_le(cursor)?;
                Cell::Local(BlockId(block_id))
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown cell tag {}", other),
                ));
            }
        };

        let end = (i + run_len).min(CHUNK_VOLUME);
        for j in i..end {
            cells[j] = cell;
        }
        i = end;
    }
    Ok(())
}

fn read_u16_le(cursor: &mut Cursor<&[u8]>) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Disk path helpers
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};

use crate::coords::{ChunkPos, RegionId};

/// Build the directory path for a region's chunks.
/// Layout: `{world_dir}/regions/{region_id}/chunks/`
pub fn region_chunks_dir(world_dir: &Path, region_id: RegionId) -> PathBuf {
    world_dir.join("regions").join(region_id.to_string()).join("chunks")
}

/// Build the file path for a single chunk.
/// Layout: `{world_dir}/regions/{region_id}/chunks/{x}_{y}_{z}.chunk`
pub fn chunk_file_path(world_dir: &Path, region_id: RegionId, pos: ChunkPos) -> PathBuf {
    region_chunks_dir(world_dir, region_id).join(format!("{}_{}_{}.chunk", pos.x, pos.y, pos.z))
}

/// Build the file path for a region's metadata.
/// Layout: `{world_dir}/regions/{region_id}/region.meta`
pub fn region_meta_path(world_dir: &Path, region_id: RegionId) -> PathBuf {
    world_dir.join("regions").join(region_id.to_string()).join("region.meta")
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Save a chunk to disk. Creates directories as needed.
pub fn save_chunk_to_disk(
    world_dir: &Path,
    region_id: RegionId,
    pos: ChunkPos,
    data: &ChunkData,
) -> io::Result<()> {
    let path = chunk_file_path(world_dir, region_id, pos);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serialize_chunk(data);
    fs::write(&path, bytes)
}

/// Load a chunk from disk. Returns None if the file doesn't exist.
pub fn load_chunk_from_disk(
    world_dir: &Path,
    region_id: RegionId,
    pos: ChunkPos,
) -> io::Result<Option<ChunkData>> {
    let path = chunk_file_path(world_dir, region_id, pos);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)?;
    let data = deserialize_chunk(&bytes)?;
    Ok(Some(data))
}

/// Save region metadata to disk.
pub fn save_region_meta_to_disk(
    world_dir: &Path,
    meta: &RegionMeta,
) -> io::Result<()> {
    let path = region_meta_path(world_dir, RegionId(meta.region_id));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serialize_region_meta(meta);
    fs::write(&path, bytes)
}

/// Load region metadata from disk.
pub fn load_region_meta_from_disk(
    world_dir: &Path,
    region_id: RegionId,
) -> io::Result<Option<RegionMeta>> {
    let path = region_meta_path(world_dir, region_id);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)?;
    let meta = deserialize_region_meta(&bytes)?;
    Ok(Some(meta))
}

/// Save all dirty chunks in a region to disk.
/// Returns the number of chunks saved.
pub fn save_dirty_chunks(
    world_dir: &Path,
    region: &mut crate::region::Region,
) -> io::Result<usize> {
    let mut count = 0;
    // Collect dirty positions first to avoid borrow issues
    let dirty_positions: Vec<ChunkPos> = region
        .dirty_chunks()
        .map(|(pos, _)| pos)
        .collect();

    for pos in dirty_positions {
        if let Some(slot) = region.get_chunk(pos) {
            save_chunk_to_disk(world_dir, region.id, pos, &slot.data)?;
            count += 1;
        }
        if let Some(slot) = region.get_chunk_mut(pos) {
            slot.dirty.mark_saved();
        }
    }
    Ok(count)
}

// ---------------------------------------------------------------------------
// Region metadata
// ---------------------------------------------------------------------------

const META_MAGIC: &[u8; 4] = b"JBRG";
const META_VERSION: u8 = 1;

/// Serializable region metadata (written to region.meta).
#[derive(Debug, Clone)]
pub struct RegionMeta {
    pub region_id: u32,
    /// World-space origin (x, y, z).
    pub world_origin: [f32; 3],
    /// Number of chunks stored on disk for this region.
    pub chunk_count: u32,
    /// Data generation counter (for impostor invalidation).
    pub data_generation: u64,
}

pub fn serialize_region_meta(meta: &RegionMeta) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(META_MAGIC);
    buf.push(META_VERSION);
    buf.extend_from_slice(&meta.region_id.to_le_bytes());
    for &v in &meta.world_origin {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&meta.chunk_count.to_le_bytes());
    buf.extend_from_slice(&meta.data_generation.to_le_bytes());
    buf
}

pub fn deserialize_region_meta(bytes: &[u8]) -> io::Result<RegionMeta> {
    let mut cursor = Cursor::new(bytes);

    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != META_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad region meta magic"));
    }

    let mut ver = [0u8; 1];
    cursor.read_exact(&mut ver)?;
    if ver[0] != META_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported region meta version {}", ver[0]),
        ));
    }

    let region_id = read_u32_le(&mut cursor)?;
    let world_origin = [
        read_f32_le(&mut cursor)?,
        read_f32_le(&mut cursor)?,
        read_f32_le(&mut cursor)?,
    ];
    let chunk_count = read_u32_le(&mut cursor)?;
    let data_generation = read_u64_le(&mut cursor)?;

    Ok(RegionMeta {
        region_id,
        world_origin,
        chunk_count,
        data_generation,
    })
}

fn read_u32_le(cursor: &mut Cursor<&[u8]>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_le(cursor: &mut Cursor<&[u8]>) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_u64_le(cursor: &mut Cursor<&[u8]>) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::SHAPE_CUBE;

    #[test]
    fn roundtrip_empty_chunk() {
        let data = ChunkData::new();
        let bytes = serialize_chunk(&data);
        let restored = deserialize_chunk(&bytes).unwrap();

        assert_eq!(restored.blocks.len(), 0);
        // All cells should be empty
        for i in 0..CHUNK_VOLUME {
            assert!(restored.get_cell_by_index(i).is_empty());
        }
    }

    #[test]
    fn roundtrip_with_blocks() {
        let mut data = ChunkData::new();
        data.place_std(0, 0, 0, SHAPE_CUBE, Facing::North, 1);
        data.place_std(4, 2, 6, SHAPE_CUBE, Facing::East, 42);
        data.place_wedge(10, 0, 10, Facing::South, 7);

        let bytes = serialize_chunk(&data);
        let restored = deserialize_chunk(&bytes).unwrap();

        assert_eq!(restored.blocks.len(), 3);

        // Check block properties
        assert_eq!(restored.blocks[0].shape, SHAPE_CUBE);
        assert_eq!(restored.blocks[0].facing, Facing::North);
        assert_eq!(restored.blocks[0].texture, 1);
        assert_eq!(restored.blocks[0].origin, (0, 0, 0));

        assert_eq!(restored.blocks[1].shape, SHAPE_CUBE);
        assert_eq!(restored.blocks[1].facing, Facing::East);
        assert_eq!(restored.blocks[1].texture, 42);
        assert_eq!(restored.blocks[1].origin, (4, 2, 6));

        // Check that cells are occupied
        assert!(restored.get_cell(0, 0, 0).is_occupied());
        assert!(restored.get_cell(1, 0, 1).is_occupied());
        assert!(restored.get_cell(4, 2, 6).is_occupied());

        // Check that empty cells are still empty
        assert!(restored.get_cell(15, 15, 15).is_empty());
    }

    #[test]
    fn empty_chunk_compresses_well() {
        let data = ChunkData::new();
        let bytes = serialize_chunk(&data);
        // Header (5) + block count (2) + single RLE run for 32768 empty cells (3 bytes)
        assert!(bytes.len() < 20, "empty chunk should be tiny, got {} bytes", bytes.len());
    }

    #[test]
    fn bad_magic_fails() {
        let mut bytes = serialize_chunk(&ChunkData::new());
        bytes[0] = b'X';
        assert!(deserialize_chunk(&bytes).is_err());
    }

    #[test]
    fn bad_version_fails() {
        let mut bytes = serialize_chunk(&ChunkData::new());
        bytes[4] = 99;
        assert!(deserialize_chunk(&bytes).is_err());
    }

    #[test]
    fn roundtrip_region_meta() {
        let meta = RegionMeta {
            region_id: 42,
            world_origin: [100.0, 0.0, -200.5],
            chunk_count: 1337,
            data_generation: 99,
        };
        let bytes = serialize_region_meta(&meta);
        let restored = deserialize_region_meta(&bytes).unwrap();
        assert_eq!(restored.region_id, 42);
        assert_eq!(restored.world_origin, [100.0, 0.0, -200.5]);
        assert_eq!(restored.chunk_count, 1337);
        assert_eq!(restored.data_generation, 99);
    }

    #[test]
    fn save_and_load_chunk_file() {
        let dir = std::env::temp_dir().join("jumpblocks_test_save_load");
        let _ = std::fs::remove_dir_all(&dir);

        let region_id = crate::coords::RegionId(0);
        let pos = crate::coords::ChunkPos::new(5, -2, 10);

        let mut data = ChunkData::new();
        data.place_std(0, 0, 0, SHAPE_CUBE, Facing::North, 1);
        data.place_wedge(4, 2, 4, Facing::East, 3);

        save_chunk_to_disk(&dir, region_id, pos, &data).unwrap();
        let loaded = load_chunk_from_disk(&dir, region_id, pos).unwrap().unwrap();

        assert_eq!(loaded.blocks.len(), 2);
        assert!(loaded.get_cell(0, 0, 0).is_occupied());
        assert!(loaded.get_cell(4, 2, 4).is_occupied());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let dir = std::env::temp_dir().join("jumpblocks_test_nonexistent");
        let region_id = crate::coords::RegionId(99);
        let pos = crate::coords::ChunkPos::new(0, 0, 0);
        let result = load_chunk_from_disk(&dir, region_id, pos).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn chunk_file_paths() {
        use std::path::Path;
        let world_dir = Path::new("/saves/my_world");
        let path = chunk_file_path(world_dir, crate::coords::RegionId(0), crate::coords::ChunkPos::new(12, -3, 45));
        assert_eq!(path.to_str().unwrap(), "/saves/my_world/regions/region_0/chunks/12_-3_45.chunk");
    }

    #[test]
    fn roundtrip_preserves_cell_identity() {
        let mut data = ChunkData::new();
        // Place two adjacent blocks
        let id0 = data.place_std(0, 0, 0, SHAPE_CUBE, Facing::North, 1);
        let id1 = data.place_std(2, 0, 0, SHAPE_CUBE, Facing::North, 2);

        let bytes = serialize_chunk(&data);
        let restored = deserialize_chunk(&bytes).unwrap();

        // Cell (0,0,0) should point to block 0
        match restored.get_cell(0, 0, 0) {
            Cell::Local(id) => assert_eq!(id, id0),
            other => panic!("expected Local, got {:?}", other),
        }

        // Cell (2,0,0) should point to block 1
        match restored.get_cell(2, 0, 0) {
            Cell::Local(id) => assert_eq!(id, id1),
            other => panic!("expected Local, got {:?}", other),
        }
    }
}
