use std::collections::HashMap;
use std::time::Instant;

use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};

use crate::chunk::*;
use crate::shape::*;

/// Custom vertex attribute for chamfer offset (direction to push vertex for chamfering).
pub const ATTRIBUTE_CHAMFER_OFFSET: bevy::mesh::MeshVertexAttribute =
    bevy::mesh::MeshVertexAttribute::new("ChamferOffset", 930_481_752, bevy::render::render_resource::VertexFormat::Float32x3);

pub struct ChunkMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub chamfer_offsets: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

pub struct ChunkColliderData {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
}

pub struct ChunkMeshResult {
    pub full_res: ChunkMeshData,
    pub lod: ChunkMeshData,
    pub collider_data: ChunkColliderData,
    /// Debug overlay mesh highlighting faces with "impacted corner" vertices
    /// (vertices that share a position with a chamfered edge but whose own
    /// edges at that vertex are not sharp).
    pub debug_overlay: Option<ChunkMeshData>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn to_world(v: Vec3, wx: f32, wy: f32, wz: f32) -> Vec3 {
    Vec3::new(
        wx + v.x * VOXEL_SIZE,
        wy + v.y * VOXEL_SIZE,
        wz + v.z * VOXEL_SIZE,
    )
}


fn compute_world_normal(world_verts: &[Vec3], triangles: &[[usize; 3]]) -> Vec3 {
    let tri = &triangles[0];
    let a = world_verts[tri[0]];
    let b = world_verts[tri[1]];
    let c = world_verts[tri[2]];
    -(b - a).cross(c - a).normalize_or_zero()
}

/// Check if a face should be culled based on CellCover occlusion.
///
/// For each CellCover entry on the face:
/// - Compute the world cell position (block origin + cell offset)
/// - Check the neighbor cell in that direction
/// - Resolve its block and check if it has a CellCover on the opposite side with full=true
///
/// Face is occluded only if ALL coverage entries are occluded.
fn face_is_occluded(
    data: &ChunkData,
    neighbors: &ChunkNeighbors,
    shapes: &ShapeTable,
    block: &Block,
    face: &BlockFace,
    facing: Facing,
    size: (u8, u8, u8),
) -> bool {
    if face.cell_coverage.is_empty() {
        return false; // diagonal faces are never culled
    }

    let (ox, oy, oz) = block.origin;
    for cover in &face.cell_coverage {
        // Rotate the cell offset by the block's facing
        let rotated_cell = rotate_cell_offset(cover.cell, facing, size);
        let cell_x = ox as i32 + rotated_cell.0 as i32;
        let cell_y = oy as i32 + rotated_cell.1 as i32;
        let cell_z = oz as i32 + rotated_cell.2 as i32;

        // Rotate the face side by the block's facing
        let world_side = cover.side.rotated_by(facing);
        let Some((dx, dy, dz)) = world_side.neighbor_offset() else {
            return false; // FaceSide::None
        };

        let nx = cell_x + dx;
        let ny = cell_y + dy;
        let nz = cell_z + dz;

        if !neighbor_cell_occludes(data, neighbors, shapes, nx, ny, nz, world_side.opposite(), cover.coverage) {
            return false; // at least one coverage entry is not occluded
        }
    }

    true // all coverage entries are occluded
}

const NEIGHBOR_OFFSETS_WITH_SIDE: [(i32, i32, i32, FaceSide); 6] = [
    (1, 0, 0, FaceSide::East),
    (-1, 0, 0, FaceSide::West),
    (0, 1, 0, FaceSide::Top),
    (0, -1, 0, FaceSide::Bottom),
    (0, 0, 1, FaceSide::North),
    (0, 0, -1, FaceSide::South),
];

/// Check if a block is fully interior (every occupied cell has all 6 neighbors
/// fully occluded). If true, all faces would be occluded so the block can be
/// skipped entirely. Uses coverage checks so that partial-coverage neighbors
/// (e.g. wedge slopes) don't incorrectly mark a block as interior.
fn block_is_fully_interior(
    data: &ChunkData,
    neighbors: &ChunkNeighbors,
    shapes: &ShapeTable,
    block: &Block,
    shape: &BlockShape,
    facing: Facing,
) -> bool {
    let (ox, oy, oz) = block.origin;
    for &cell in &shape.occupied_cells {
        let rotated = rotate_cell_offset(cell, facing, shape.size);
        let cx = ox as i32 + rotated.0 as i32;
        let cy = oy as i32 + rotated.1 as i32;
        let cz = oz as i32 + rotated.2 as i32;
        for (dx, dy, dz, side) in NEIGHBOR_OFFSETS_WITH_SIDE {
            let nx = cx + dx;
            let ny = cy + dy;
            let nz = cz + dz;
            if !neighbor_cell_occludes(data, neighbors, shapes, nx, ny, nz, side.opposite(), Coverage::Full) {
                return false;
            }
        }
    }
    true
}

/// Check if the block at (nx, ny, nz) occludes the given coverage on `side_facing_us`.
///
/// Full coverage is occluded by any neighbor coverage (Full or Partial).
/// Partial coverage is only occluded by Full coverage or a matching Partial ID.
fn neighbor_cell_occludes(
    data: &ChunkData,
    neighbors: &ChunkNeighbors,
    shapes: &ShapeTable,
    nx: i32, ny: i32, nz: i32,
    side_facing_us: FaceSide,
    our_coverage: Coverage,
) -> bool {
    let Some((shape_idx, facing, origin)) = resolve_block_info_at(data, neighbors, nx, ny, nz) else {
        return false;
    };
    let Some(shape) = shapes.get(shape_idx) else {
        return false;
    };

    // Compute which cell within the neighbor block this position corresponds to.
    // resolve_block_info_at returns the origin in the neighbor's local chunk coords.
    // We need the cell position in the same coord space as the origin.
    let in_x = nx >= 0 && nx < CHUNK_X as i32;
    let in_y = ny >= 0 && ny < CHUNK_Y as i32;
    let in_z = nz >= 0 && nz < CHUNK_Z as i32;

    let (lx, ly, lz) = if in_x && in_y && in_z {
        (nx, ny, nz)
    } else {
        let dx = if nx < 0 { -1 } else if nx >= CHUNK_X as i32 { 1 } else { 0 };
        let dy = if ny < 0 { -1 } else if ny >= CHUNK_Y as i32 { 1 } else { 0 };
        let dz = if nz < 0 { -1 } else if nz >= CHUNK_Z as i32 { 1 } else { 0 };
        (nx - dx * CHUNK_X as i32, ny - dy * CHUNK_Y as i32, nz - dz * CHUNK_Z as i32)
    };

    let world_offset = (
        (lx as u8).wrapping_sub(origin.0),
        (ly as u8).wrapping_sub(origin.1),
        (lz as u8).wrapping_sub(origin.2),
    );

    for face in &shape.faces {
        for cover in &face.cell_coverage {
            let rotated_cell = rotate_cell_offset(cover.cell, facing, shape.size);
            let rotated_side = cover.side.rotated_by(facing);
            if rotated_cell != world_offset || rotated_side != side_facing_us {
                continue;
            }
            // Full neighbor coverage always occludes (original behavior)
            if cover.coverage == Coverage::Full {
                return true;
            }
            // Matching partial IDs occlude each other (wedge-wedge adjacency)
            if let (Coverage::Partial(our_id), Coverage::Partial(their_id)) = (our_coverage, cover.coverage) {
                if our_id == their_id {
                    return true;
                }
            }
        }
    }
    false
}

/// Rotate a cell offset within a block by the block's facing.
/// For 1x1x1 blocks this is always (0,0,0) so it's a no-op.
fn rotate_cell_offset(cell: (u8, u8, u8), facing: Facing, size: (u8, u8, u8)) -> (u8, u8, u8) {
    if facing == Facing::North {
        return cell;
    }
    // Rotate the center of the cell using the block's rotation
    let center = Vec3::new(cell.0 as f32 + 0.5, cell.1 as f32 + 0.5, cell.2 as f32 + 0.5);
    let rotated = facing.rotate_block_point(center, size);
    (rotated.x.floor() as u8, rotated.y.floor() as u8, rotated.z.floor() as u8)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn generate_chunk_mesh(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable, mode: crate::PresentationMode) -> ChunkMeshResult {
    let t0 = Instant::now();
    let (full_res, debug_overlay) = match mode {
        crate::PresentationMode::Flat => (generate_lod_mesh(data, neighbors, shapes), None),
        crate::PresentationMode::CutAndOffset => (crate::cut_offset_chamfer::generate_cut_offset_chamfer(data, neighbors, shapes), None),
    };
    let t1 = Instant::now();
    let lod = generate_lod_mesh(data, neighbors, shapes);
    let t2 = Instant::now();

    let full_ms = (t1 - t0).as_secs_f64() * 1000.0;
    let lod_ms = (t2 - t1).as_secs_f64() * 1000.0;
    info!(
        "Chunk meshed [{}]: full={:.2}ms, lod={:.2}ms, total={:.2}ms (full: {} verts/{} tris, lod: {} verts/{} tris)",
        match mode { crate::PresentationMode::Flat => "flat", crate::PresentationMode::CutAndOffset => "cut-offset" },
        full_ms, lod_ms, full_ms + lod_ms,
        full_res.positions.len(), full_res.indices.len() / 3,
        lod.positions.len(), lod.indices.len() / 3,
    );

    let collider_vertices: Vec<Vec3> = lod.positions.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    let collider_indices: Vec<[u32; 3]> = lod.indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    ChunkMeshResult {
        full_res,
        lod,
        collider_data: ChunkColliderData {
            vertices: collider_vertices,
            indices: collider_indices,
        },
        debug_overlay,
    }
}

// ---------------------------------------------------------------------------
// LOD mesh (simple, no chamfer)
// ---------------------------------------------------------------------------

fn generate_lod_mesh(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable) -> ChunkMeshData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let mut stats_blocks = 0u32;
    let mut stats_blocks_no_cells = 0u32;
    let mut stats_blocks_interior = 0u32;
    let mut stats_faces_total = 0u32;
    let mut stats_faces_culled = 0u32;

    for (block_idx, block) in data.blocks.iter().enumerate() {
        let Some(shape) = shapes.get(block.shape) else { continue; };
        let facing = block.facing;
        let (ox, oy, oz) = block.origin;
        let wx = ox as f32 * VOXEL_SIZE;
        let wy = oy as f32 * VOXEL_SIZE;
        let wz = oz as f32 * VOXEL_SIZE;
        stats_blocks += 1;

        // Verify this block still owns at least one cell (not removed)
        let has_cells = shape.occupied_cells.iter().any(|&(dx, dy, dz)| {
            let cx = ox as usize + dx as usize;
            let cy = oy as usize + dy as usize;
            let cz = oz as usize + dz as usize;
            data.get_cell(cx, cy, cz) == Cell::Local(BlockId(block_idx as u16))
        });
        if !has_cells { stats_blocks_no_cells += 1; continue; }

        // Skip fully interior blocks — all faces would be occluded
        if block_is_fully_interior(data, neighbors, shapes, block, shape, facing) {
            stats_blocks_interior += 1;
            stats_faces_total += shape.faces.len() as u32;
            stats_faces_culled += shape.faces.len() as u32;
            continue;
        }

        for face in &shape.faces {
            stats_faces_total += 1;
            if face_is_occluded(data, neighbors, shapes, block, face, facing, shape.size) {
                stats_faces_culled += 1;
                continue;
            }

            let base_index = positions.len() as u32;
            let world_verts: Vec<Vec3> = face.vertices.iter()
                .map(|v| to_world(facing.rotate_block_point(*v, shape.size), wx, wy, wz))
                .collect();
            let world_normal = compute_world_normal(&world_verts, &face.triangles);

            for wv in &world_verts {
                positions.push(wv.to_array());
                normals.push(world_normal.to_array());
            }
            let uv_map = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            for i in 0..face.vertices.len() { uvs.push(uv_map[i % 4]); }
            for tri in &face.triangles {
                indices.push(base_index + tri[2] as u32);
                indices.push(base_index + tri[1] as u32);
                indices.push(base_index + tri[0] as u32);
            }
        }
    }

    let stats_faces_emitted = stats_faces_total - stats_faces_culled;
    info!(
        "[lod mesh] blocks: {} (no_cells: {}, interior: {}), faces: {} total, {} culled, {} emitted",
        stats_blocks, stats_blocks_no_cells, stats_blocks_interior,
        stats_faces_total, stats_faces_culled, stats_faces_emitted,
    );

    let n = positions.len();
    ChunkMeshData { positions, normals, uvs, chamfer_offsets: vec![[0.0; 3]; n], indices }
}

// ---------------------------------------------------------------------------
// Solid mesh with shared vertices (input for chamfer post-process)
// ---------------------------------------------------------------------------

/// Quantize a world-space position for vertex deduplication.
fn quantize(p: Vec3) -> (i32, i32, i32) {
    const SCALE: f32 = 10000.0;
    (
        (p.x * SCALE).round() as i32,
        (p.y * SCALE).round() as i32,
        (p.z * SCALE).round() as i32,
    )
}

pub struct SolidFace {
    /// Indices into SolidMesh.positions (3 or 4 verts).
    pub verts: Vec<u32>,
    pub normal: Vec3,
    /// Source block origin position in chunk coords.
    pub voxel: (usize, usize, usize),
    /// Block shape size in cells.
    pub block_size: (u8, u8, u8),
    /// Original shape face triangulation (grid-aligned).
    /// Indices are into this face's `verts` list (0..verts.len()).
    pub orig_triangles: Vec<[usize; 3]>,
}

pub struct SolidMesh {
    pub positions: Vec<Vec3>,
    pub faces: Vec<SolidFace>,
    vert_map: HashMap<(i32, i32, i32), u32>,
}

impl SolidMesh {
    fn new() -> Self {
        Self { positions: Vec::new(), faces: Vec::new(), vert_map: HashMap::new() }
    }

    fn add_vert(&mut self, pos: Vec3) -> u32 {
        let key = quantize(pos);
        *self.vert_map.entry(key).or_insert_with(|| {
            let idx = self.positions.len() as u32;
            self.positions.push(pos);
            idx
        })
    }
}

/// Public entry point for building the solid mesh (used by cut_offset_chamfer).
pub fn build_solid_mesh_public(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable) -> SolidMesh {
    build_solid_mesh(data, neighbors, shapes)
}

fn build_solid_mesh(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable) -> SolidMesh {
    let mut mesh = SolidMesh::new();

    let mut stats_blocks = 0u32;
    let mut stats_blocks_no_cells = 0u32;
    let mut stats_blocks_interior = 0u32;
    let mut stats_faces_total = 0u32;
    let mut stats_faces_culled = 0u32;

    for (block_idx, block) in data.blocks.iter().enumerate() {
        let Some(shape) = shapes.get(block.shape) else { continue; };
        let facing = block.facing;
        let (ox, oy, oz) = block.origin;
        let wx = ox as f32 * VOXEL_SIZE;
        let wy = oy as f32 * VOXEL_SIZE;
        let wz = oz as f32 * VOXEL_SIZE;
        stats_blocks += 1;

        // Verify this block still owns at least one cell
        let has_cells = shape.occupied_cells.iter().any(|&(dx, dy, dz)| {
            let cx = ox as usize + dx as usize;
            let cy = oy as usize + dy as usize;
            let cz = oz as usize + dz as usize;
            data.get_cell(cx, cy, cz) == Cell::Local(BlockId(block_idx as u16))
        });
        if !has_cells { stats_blocks_no_cells += 1; continue; }

        // Skip fully interior blocks — all faces would be occluded
        if block_is_fully_interior(data, neighbors, shapes, block, shape, facing) {
            stats_blocks_interior += 1;
            stats_faces_total += shape.faces.len() as u32;
            stats_faces_culled += shape.faces.len() as u32;
            continue;
        }

        for face in &shape.faces {
            stats_faces_total += 1;
            if face_is_occluded(data, neighbors, shapes, block, face, facing, shape.size) {
                stats_faces_culled += 1;
                continue;
            }

            let world_verts: Vec<Vec3> = face.vertices.iter()
                .map(|v| to_world(facing.rotate_block_point(*v, shape.size), wx, wy, wz))
                .collect();
            let normal = compute_world_normal(&world_verts, &face.triangles);
            let vert_indices: Vec<u32> = world_verts.iter()
                .map(|v| mesh.add_vert(*v))
                .collect();

            mesh.faces.push(SolidFace {
                verts: vert_indices,
                normal,
                voxel: (ox as usize, oy as usize, oz as usize),
                block_size: shape.size,
                orig_triangles: face.triangles.clone(),
            });
        }
    }

    let faces_before_btb = mesh.faces.len();

    // Remove back-to-back faces
    let mut face_groups: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
    for (fi, face) in mesh.faces.iter().enumerate() {
        let mut key = face.verts.clone();
        key.sort();
        face_groups.entry(key).or_default().push(fi);
    }
    let remove: Vec<bool> = (0..mesh.faces.len())
        .map(|fi| {
            let mut key = mesh.faces[fi].verts.clone();
            key.sort();
            face_groups.get(&key).map_or(false, |g| g.len() >= 2)
        })
        .collect();
    mesh.faces = mesh.faces
        .into_iter()
        .enumerate()
        .filter(|(fi, _)| !remove[*fi])
        .map(|(_, f)| f)
        .collect();

    let btb_removed = faces_before_btb - mesh.faces.len();
    let stats_faces_emitted = stats_faces_total - stats_faces_culled;
    info!(
        "[solid mesh] blocks: {} (no_cells: {}, interior: {}), faces: {} total, {} culled, {} emitted, {} back-to-back removed",
        stats_blocks, stats_blocks_no_cells, stats_blocks_interior,
        stats_faces_total, stats_faces_culled, stats_faces_emitted, btb_removed,
    );
    if btb_removed > 0 {
        warn!("[solid mesh] {} back-to-back faces removed — indicates CellCover gap", btb_removed);
    }

    mesh
}

// ---------------------------------------------------------------------------
// Edge graph
// ---------------------------------------------------------------------------

pub fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

pub struct EdgeInfo {
    pub faces: Vec<usize>,
}

pub const SHARP_DOT_THRESHOLD: f32 = 0.985;

/// Compute fillet offset for the center-line vertex of a chamfer strip.
/// Places the vertex on a circular arc of radius CHAMFER_WIDTH that is
/// tangent to both face planes at the chamfer inner vertices.
/// Compute the fillet push: how far to push the center-line outward from the
/// flat chamfer midpoint toward the edge, to approximate a circular arc.
/// Returns the push vector along avg_normal.
/// Compute the fillet push amount for a circular arc tangent to two face
/// planes with normals `na` and `nb`.
///
/// `chamfer_width` is the SETBACK distance (how far from the edge the cut
/// line sits along each face).  The fillet RADIUS varies by edge angle:
///     r = setback * tan(θ/2)
/// where θ is the dihedral angle.
///
/// The push (distance from original edge to the arc surface):
///     push = setback * (1 - sin(θ/2)) / cos(θ/2)
///
/// In terms of k = |na + nb|:  sin(θ/2) = k/2,  cos(θ/2) = √(1-(k/2)²)
///     push = setback * (1 - k/2) / √(1 - k²/4)
pub fn fillet_push_amount(na: Vec3, nb: Vec3, chamfer_width: f32) -> f32 {
    let k = (na + nb).length();
    let sin_half = k / 2.0; // sin(θ/2) where θ = dihedral angle
    let cos_half_sq = 1.0 - sin_half * sin_half;
    if cos_half_sq < 1e-6 {
        return 0.0; // Nearly parallel faces — no push needed.
    }
    let cos_half = cos_half_sq.sqrt();
    (chamfer_width * (1.0 - sin_half) / cos_half).clamp(0.0, chamfer_width)
}

pub fn build_edge_graph(mesh: &SolidMesh) -> HashMap<(u32, u32), EdgeInfo> {
    let mut edges: HashMap<(u32, u32), EdgeInfo> = HashMap::new();
    for (fi, face) in mesh.faces.iter().enumerate() {
        let n = face.verts.len();
        for i in 0..n {
            let key = edge_key(face.verts[i], face.verts[(i + 1) % n]);
            edges.entry(key).or_insert_with(|| EdgeInfo { faces: Vec::new() }).faces.push(fi);
        }
    }
    edges
}

pub fn is_sharp(edge: &EdgeInfo, mesh: &SolidMesh) -> bool {
    if edge.faces.len() < 2 {
        return true;
    }
    let n0 = mesh.faces[edge.faces[0]].normal;
    let n1 = mesh.faces[edge.faces[1]].normal;
    n0.dot(n1) < SHARP_DOT_THRESHOLD
}

/// Check if a boundary edge lies on a chunk face where a filled neighbor cell
/// would continue the surface.
pub fn is_boundary_edge_at_neighbor_seam(
    ev0: u32,
    ev1: u32,
    face: &SolidFace,
    solid: &SolidMesh,
    data: &ChunkData,
    neighbors: &ChunkNeighbors,
) -> bool {
    let p0 = solid.positions[ev0 as usize];
    let p1 = solid.positions[ev1 as usize];
    let (vx, vy, vz) = face.voxel;

    let chunk_size = CHUNK_X as f32 * VOXEL_SIZE;
    let chunk_height = CHUNK_Y as f32 * VOXEL_SIZE;

    let checks: [(f32, f32, i32, i32, i32); 6] = [
        (p0.x, 0.0,          -1, 0, 0),
        (p0.x, chunk_size,    1, 0, 0),
        (p0.y, 0.0,           0, -1, 0),
        (p0.y, chunk_height,  0, 1, 0),
        (p0.z, 0.0,           0, 0, -1),
        (p0.z, chunk_size,    0, 0,  1),
    ];

    let eps = 1e-4;
    for (_, boundary, dx, dy, dz) in checks {
        let (c0, c1) = match (dx != 0, dy != 0) {
            (true, _)     => (p0.x, p1.x),
            (_, true)     => (p0.y, p1.y),
            _             => (p0.z, p1.z),
        };
        if (c0 - boundary).abs() < eps && (c1 - boundary).abs() < eps {
            let nx = vx as i32 + dx;
            let ny = vy as i32 + dy;
            let nz = vz as i32 + dz;
            if is_cell_occupied(data, neighbors, nx, ny, nz) {
                return true;
            }
        }
    }
    false
}


// ---------------------------------------------------------------------------
// Mesh builders
// ---------------------------------------------------------------------------

pub fn build_full_res_mesh(data: &ChunkMeshData) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs.clone());
    mesh.insert_attribute(ATTRIBUTE_CHAMFER_OFFSET, data.chamfer_offsets.clone());
    mesh.insert_indices(Indices::U32(data.indices.clone()));
    mesh
}

pub fn build_lod_mesh(data: &ChunkMeshData) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs.clone());
    mesh.insert_indices(Indices::U32(data.indices.clone()));
    mesh
}
