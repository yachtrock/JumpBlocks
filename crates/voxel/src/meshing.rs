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

/// A face polygon in world space, before vertex dedup / contact clipping.
struct RawFace {
    verts: Vec<Vec3>,
    normal: Vec3,
    voxel: (usize, usize, usize),
    block_size: (u8, u8, u8),
    orig_triangles: Vec<[usize; 3]>,
}

// ---------------------------------------------------------------------------
// Coplanar contact clipping
// ---------------------------------------------------------------------------
// Blocks that touch with PARTIALLY overlapping faces (e.g. a wedge's back
// wall resting against another wedge's side pentagon) both emit their faces
// whole, leaving interpenetrating opposite-facing geometry in the contact
// region.  We clip every face polygon by the polygons of opposite-facing
// coplanar faces from other blocks: the shared region is interior surface
// and must not be emitted.  Because block geometry is cell-aligned, the
// clipped seams coincide exactly with matching edges on both sides.

const CONTACT_AREA_EPS: f32 = 1e-5;

fn poly_area_2d(poly: &[Vec2]) -> f32 {
    let mut a = 0.0;
    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        a += p.x * q.y - q.x * p.y;
    }
    a * 0.5
}

/// Clip a convex polygon against one side of the infinite line a→b.
/// `keep_left`: keep the region to the left of a→b (positive perp-dot).
fn clip_halfplane(poly: &[Vec2], a: Vec2, b: Vec2, keep_left: bool) -> Vec<Vec2> {
    let side = |p: Vec2| -> f32 {
        let s = (b - a).perp_dot(p - a);
        if keep_left { s } else { -s }
    };
    let n = poly.len();
    let mut out: Vec<Vec2> = Vec::with_capacity(n + 2);
    for i in 0..n {
        let cur = poly[i];
        let nxt = poly[(i + 1) % n];
        let sc = side(cur);
        let sn = side(nxt);
        if sc >= -1e-6 {
            out.push(cur);
        }
        if (sc > 1e-6 && sn < -1e-6) || (sc < -1e-6 && sn > 1e-6) {
            let t = sc / (sc - sn);
            out.push(cur + (nxt - cur) * t);
        }
    }
    // Drop near-duplicate consecutive points introduced by clipping.
    let mut dedup: Vec<Vec2> = Vec::with_capacity(out.len());
    for p in out {
        if dedup.last().map_or(true, |l| l.distance_squared(p) > 1e-10) {
            dedup.push(p);
        }
    }
    while dedup.len() >= 2
        && dedup[0].distance_squared(*dedup.last().unwrap()) <= 1e-10
    {
        dedup.pop();
    }
    dedup
}

/// Area of the intersection of two convex polygons (b must be CCW).
fn convex_intersection_area(a: &[Vec2], b: &[Vec2]) -> f32 {
    let mut r = a.to_vec();
    let n = b.len();
    for i in 0..n {
        r = clip_halfplane(&r, b[i], b[(i + 1) % n], true);
        if r.len() < 3 {
            return 0.0;
        }
    }
    poly_area_2d(&r).abs()
}

/// Convex pieces of `a` outside convex `b` (both CCW).  The union of the
/// returned pieces is a − b.
fn convex_difference(a: &[Vec2], b: &[Vec2]) -> Vec<Vec<Vec2>> {
    let mut pieces = Vec::new();
    let mut remaining = a.to_vec();
    let n = b.len();
    for i in 0..n {
        let e0 = b[i];
        let e1 = b[(i + 1) % n];
        let outside = clip_halfplane(&remaining, e0, e1, false);
        if outside.len() >= 3 && poly_area_2d(&outside).abs() > CONTACT_AREA_EPS {
            pieces.push(outside);
        }
        remaining = clip_halfplane(&remaining, e0, e1, true);
        if remaining.len() < 3 || poly_area_2d(&remaining).abs() < CONTACT_AREA_EPS {
            break;
        }
    }
    pieces
}

/// Triangulate a simple (possibly non-convex) planar polygon by ear
/// clipping.  Collinear (T-junction) vertices are preserved on triangle
/// boundaries.  Output triangles follow the solid-mesh winding convention
/// (normal = -(b-a)×(c-a)).
fn ear_clip_triangulate(verts: &[Vec3], normal: Vec3) -> Vec<[usize; 3]> {
    let u = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = (u - normal * u.dot(normal)).normalize_or_zero();
    let v = normal.cross(u);
    let pts: Vec<Vec2> = verts.iter().map(|p| Vec2::new(p.dot(u), p.dot(v))).collect();

    let mut idx: Vec<usize> = (0..verts.len()).collect();
    let ring_area: f32 = {
        let mut a = 0.0;
        for k in 0..idx.len() {
            let p = pts[idx[k]];
            let q = pts[idx[(k + 1) % idx.len()]];
            a += p.x * q.y - q.x * p.y;
        }
        a * 0.5
    };
    if ring_area < 0.0 {
        idx.reverse();
    }

    let strictly_inside = |p: Vec2, a: Vec2, b: Vec2, c: Vec2| -> bool {
        let d1 = (b - a).perp_dot(p - a);
        let d2 = (c - b).perp_dot(p - b);
        let d3 = (a - c).perp_dot(p - c);
        d1 > 1e-9 && d2 > 1e-9 && d3 > 1e-9
    };

    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(verts.len().saturating_sub(2));
    while idx.len() > 3 {
        let n = idx.len();
        let mut clipped = false;
        for k in 0..n {
            let a = idx[(k + n - 1) % n];
            let b = idx[k];
            let c = idx[(k + 1) % n];
            let cross = (pts[b] - pts[a]).perp_dot(pts[c] - pts[b]);
            if cross <= 1e-9 {
                continue; // reflex or collinear vertex — not an ear
            }
            if idx.iter().any(|&m| {
                m != a && m != b && m != c && strictly_inside(pts[m], pts[a], pts[b], pts[c])
            }) {
                continue;
            }
            // Reject ears whose new diagonal (a-c) passes through another
            // remaining vertex: clipping would orphan that vertex, bridging
            // over a collinear T-junction and cracking the boundary.
            let ac = pts[c] - pts[a];
            let ac_len_sq = ac.length_squared().max(1e-12);
            if idx.iter().any(|&m| {
                if m == a || m == b || m == c {
                    return false;
                }
                let am = pts[m] - pts[a];
                let t = am.dot(ac) / ac_len_sq;
                t > 1e-6 && t < 1.0 - 1e-6 && ac.perp_dot(am).abs() < 1e-6 * ac_len_sq.sqrt()
            }) {
                continue;
            }
            tris.push([a, b, c]);
            idx.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            break; // degenerate remainder
        }
    }
    if idx.len() == 3 {
        let (a, b, c) = (idx[0], idx[1], idx[2]);
        if (pts[b] - pts[a]).perp_dot(pts[c] - pts[b]).abs() > 1e-9 {
            tris.push([a, b, c]);
        }
    }

    for t in &mut tris {
        let cr = (verts[t[1]] - verts[t[0]]).cross(verts[t[2]] - verts[t[0]]);
        if (-cr).dot(normal) < 0.0 {
            t.swap(0, 2);
        }
    }
    tris
}

/// Try to merge a set of coplanar same-facing faces into one polygon.  If
/// the union's boundary is not a single ring (the contact region punches a
/// hole through the surface, or the union pinches), bisect the set spatially
/// and retry each half: the cut opens the hole so each side chains cleanly,
/// and the straight seam between the halves is a full-edge-matched seam the
/// chamfer already handles.
fn try_merge_face_set(
    faces: &[RawFace],
    set: &[usize],
    merged: &mut Vec<RawFace>,
    consumed: &mut Vec<bool>,
) {
    if set.len() < 2 {
        return;
    }
    let qpos = |p: Vec3| {
        (
            (p.x * 10000.0).round() as i32,
            (p.y * 10000.0).round() as i32,
            (p.z * 10000.0).round() as i32,
        )
    };

    // Edge ownership within this subset.
    let mut edge_owner: HashMap<((i32, i32, i32), (i32, i32, i32)), usize> = HashMap::new();
    for &fi in set {
        let vs = &faces[fi].verts;
        for i in 0..vs.len() {
            let a = qpos(vs[i]);
            let b = qpos(vs[(i + 1) % vs.len()]);
            let key = if a <= b { (a, b) } else { (b, a) };
            *edge_owner.entry(key).or_insert(0) += 1;
        }
    }

    // Directed boundary edges (edges owned once within the subset).
    let mut dir_edges: HashMap<(i32, i32, i32), Vec<((i32, i32, i32), Vec3)>> = HashMap::new();
    let mut boundary_count = 0usize;
    let mut ok = true;
    for &fi in set {
        let vs = &faces[fi].verts;
        for i in 0..vs.len() {
            let pa = vs[i];
            let pb = vs[(i + 1) % vs.len()];
            let (a, b) = (qpos(pa), qpos(pb));
            let key = if a <= b { (a, b) } else { (b, a) };
            match edge_owner[&key] {
                1 => {
                    dir_edges.entry(a).or_default().push((b, pa));
                    boundary_count += 1;
                }
                2 => {} // interior seam
                _ => ok = false,
            }
        }
    }

    let mut ring: Vec<Vec3> = Vec::new();
    if ok && !dir_edges.values().any(|v| v.len() != 1) {
        if let Some(&start) = dir_edges.keys().min() {
            let mut cur = start;
            loop {
                let Some(next) = dir_edges.get(&cur) else {
                    ok = false;
                    break;
                };
                let (nb, pa) = next[0];
                ring.push(pa);
                cur = nb;
                if cur == start {
                    break;
                }
                if ring.len() > boundary_count {
                    ok = false;
                    break;
                }
            }
        }
    } else {
        ok = false;
    }

    if ok && ring.len() == boundary_count && ring.len() >= 3 {
        let first = &faces[set[0]];
        let tris = ear_clip_triangulate(&ring, first.normal);
        if !tris.is_empty() {
            merged.push(RawFace {
                verts: ring,
                normal: first.normal,
                voxel: first.voxel,
                block_size: first.block_size,
                orig_triangles: tris,
            });
            for &fi in set {
                consumed[fi] = true;
            }
        }
        return;
    }

    // Multi-ring union (the contact punches a hole through the surface):
    // a single-ring polygon cannot represent it — leave the faces unmerged.
    // Known limitation: the unmerged coplanar seam can leave a hairline
    // sliver at its endpoints.
}

/// Merge connected groups of coplanar faces with the SAME orientation that
/// share boundary edges into single polygons, deleting the interior seams.
///
/// Partial block contacts (wedge against wedge/cube) split one geometric
/// surface into several faces with T-seams; the per-face chamfer logic then
/// makes inconsistent decisions at the seam endpoints and cracks the mesh.
/// After merging, the surface is one polygon and its corners get ordinary
/// treatment.  Groups whose union is not a single ring (holes, pinches)
/// are left unmerged.
///
/// Only faces of blocks in `merge_blocks` (blocks involved in a partial
/// contact) are considered — full-face contacts already work unmerged, and
/// merging them would perturb established behavior for no benefit.
fn merge_coplanar_faces(
    faces: Vec<RawFace>,
    merge_blocks: &std::collections::HashSet<(usize, usize, usize)>,
) -> Vec<RawFace> {
    let quant = |x: f32| (x * 1000.0) as i32;
    let qpos = |p: Vec3| {
        (
            (p.x * 10000.0).round() as i32,
            (p.y * 10000.0).round() as i32,
            (p.z * 10000.0).round() as i32,
        )
    };

    // Group faces by oriented plane.  Merging is decided per connected SET:
    // a set merges only if it contains a face from a contact-clipped block,
    // but the merge then covers every coplanar face edge-connected to it.
    let mut groups: HashMap<(i32, i32, i32, i32), Vec<usize>> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        let n = f.normal;
        let d = n.dot(f.verts[0]);
        groups
            .entry((quant(n.x), quant(n.y), quant(n.z), (d * 10000.0).round() as i32))
            .or_default()
            .push(fi);
    }

    let mut consumed: Vec<bool> = vec![false; faces.len()];
    let mut merged: Vec<RawFace> = Vec::new();

    let mut group_keys: Vec<(i32, i32, i32, i32)> = groups.keys().copied().collect();
    group_keys.sort_unstable();
    for gkey in group_keys {
        let idxs = &groups[&gkey];
        if idxs.len() < 2 {
            continue;
        }
        // Union-find faces sharing a boundary edge.
        let mut edge_owner: HashMap<((i32, i32, i32), (i32, i32, i32)), Vec<usize>> =
            HashMap::new();
        for &fi in idxs {
            let vs = &faces[fi].verts;
            for i in 0..vs.len() {
                let a = qpos(vs[i]);
                let b = qpos(vs[(i + 1) % vs.len()]);
                let key = if a <= b { (a, b) } else { (b, a) };
                edge_owner.entry(key).or_default().push(fi);
            }
        }
        let mut parent: HashMap<usize, usize> = idxs.iter().map(|&i| (i, i)).collect();
        fn find(parent: &mut HashMap<usize, usize>, mut i: usize) -> usize {
            while parent[&i] != i {
                let p = parent[&parent[&i]];
                parent.insert(i, p);
                i = p;
            }
            i
        }
        for owners in edge_owner.values() {
            if owners.len() == 2 {
                let (a, b) = (find(&mut parent, owners[0]), find(&mut parent, owners[1]));
                if a != b {
                    parent.insert(a, b);
                }
            }
        }
        let mut sets: HashMap<usize, Vec<usize>> = HashMap::new();
        for &fi in idxs {
            let root = find(&mut parent, fi);
            sets.entry(root).or_default().push(fi);
        }

        let mut set_list: Vec<Vec<usize>> = sets.into_values().collect();
        for set in &mut set_list {
            set.sort_unstable();
        }
        set_list.sort();
        for set in &set_list {
            if set.len() < 2 {
                continue;
            }
            // Only merge sets touched by a partial contact — full-face
            // seams already work unmerged.
            if !set.iter().any(|&fi| merge_blocks.contains(&faces[fi].voxel)) {
                continue;
            }
            try_merge_face_set(&faces, set, &mut merged, &mut consumed);
        }
    }

    let mut out: Vec<RawFace> = faces
        .into_iter()
        .enumerate()
        .filter(|(fi, _)| !consumed[*fi])
        .map(|(_, f)| f)
        .collect();
    out.extend(merged);
    out
}

/// Clip opposite-facing coplanar face polygons of touching blocks against
/// each other, removing the shared (interior) contact regions.  Also returns
/// the set of blocks involved in a partial contact (for coplanar merging).
fn clip_contact_faces(
    faces: Vec<RawFace>,
    clip_shapes: Vec<RawFace>,
) -> (Vec<RawFace>, std::collections::HashSet<(usize, usize, usize)>) {
    let mut clipped_blocks: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    // Real (emitted) faces first, then occluded faces that only act as clip
    // shapes: a culled face still covers the region it rests against, so the
    // opposing face must be clipped by it even though it is not emitted.
    let real_count = faces.len();
    let mut faces = faces;
    faces.extend(clip_shapes);
    let quant = |x: f32| (x * 1000.0).round() as i32;

    // Canonical plane per face: orient the normal so its first significant
    // component is positive.  Faces on the same geometric plane share a key;
    // `side` records whether the face points along (+1) or against (-1) it.
    let mut plane_key: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(faces.len());
    let mut plane_side: Vec<f32> = Vec::with_capacity(faces.len());
    let mut basis: Vec<(Vec3, Vec3, Vec3)> = Vec::with_capacity(faces.len()); // (origin, u, v)
    for f in &faces {
        let n = f.normal;
        let sign = if n.x.abs() > 0.5 {
            n.x.signum()
        } else if n.y.abs() > 0.5 {
            n.y.signum()
        } else {
            n.z.signum()
        };
        let cn = n * sign;
        let d = cn.dot(f.verts[0]);
        plane_key.push((quant(cn.x), quant(cn.y), quant(cn.z), (d * 10000.0).round() as i32));
        plane_side.push(sign);
        let u = if cn.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let u = (u - cn * u.dot(cn)).normalize_or_zero();
        let v = cn.cross(u);
        basis.push((cn * d, u, v));
    }

    let mut groups: HashMap<(i32, i32, i32, i32), Vec<usize>> = HashMap::new();
    for i in 0..faces.len() {
        groups.entry(plane_key[i]).or_default().push(i);
    }

    // Project a face's polygon into its plane basis, wound CCW.
    let project = |fi: usize| -> Vec<Vec2> {
        let (origin, u, v) = basis[fi];
        let mut poly: Vec<Vec2> = faces[fi]
            .verts
            .iter()
            .map(|p| Vec2::new((*p - origin).dot(u), (*p - origin).dot(v)))
            .collect();
        if poly_area_2d(&poly) < 0.0 {
            poly.reverse();
        }
        poly
    };

    // For clipped faces: the remaining convex pieces (in plane 2D coords).
    let mut pieces: Vec<Option<Vec<Vec<Vec2>>>> = (0..faces.len()).map(|_| None).collect();

    let mut clip_group_keys: Vec<(i32, i32, i32, i32)> = groups.keys().copied().collect();
    clip_group_keys.sort_unstable();
    for gkey in clip_group_keys {
        let idxs = &groups[&gkey];
        let front: Vec<usize> = idxs.iter().copied().filter(|&i| plane_side[i] > 0.0).collect();
        let back: Vec<usize> = idxs.iter().copied().filter(|&i| plane_side[i] < 0.0).collect();
        if front.is_empty() || back.is_empty() {
            continue;
        }
        let polys: HashMap<usize, Vec<Vec2>> =
            idxs.iter().map(|&i| (i, project(i))).collect();

        for &fi in &front {
            for &gi in &back {
                if faces[fi].voxel == faces[gi].voxel {
                    continue;
                }
                if fi >= real_count && gi >= real_count {
                    continue;
                }
                if convex_intersection_area(&polys[&fi], &polys[&gi]) <= CONTACT_AREA_EPS {
                    continue;
                }
                clipped_blocks.insert(faces[fi].voxel);
                clipped_blocks.insert(faces[gi].voxel);
                for (idx, other) in [(fi, gi), (gi, fi)] {
                    if idx >= real_count {
                        continue; // clip shapes are never emitted
                    }
                    let current = pieces[idx]
                        .take()
                        .unwrap_or_else(|| vec![polys[&idx].clone()]);
                    let mut next = Vec::new();
                    for piece in &current {
                        next.extend(convex_difference(piece, &polys[&other]));
                    }
                    pieces[idx] = Some(next);
                }
            }
        }
    }

    // Rebuild the face list: untouched faces pass through; clipped faces are
    // replaced by one face per remaining convex piece.
    let mut out: Vec<RawFace> = Vec::with_capacity(real_count);
    for (fi, face) in faces.into_iter().enumerate() {
        if fi >= real_count {
            break;
        }
        let Some(face_pieces) = pieces[fi].take() else {
            out.push(face);
            continue;
        };
        let (origin, u, v) = basis[fi];
        for piece in face_pieces {
            let mut verts: Vec<Vec3> = piece
                .iter()
                .map(|p| origin + u * p.x + v * p.y)
                .collect();
            if verts.len() < 3 {
                continue;
            }
            // Wind so that triangles reproduce the face normal under the
            // solid-mesh convention (normal = -(b-a)×(c-a)).
            let cross = (verts[1] - verts[0]).cross(verts[2] - verts[0]);
            if (-cross).dot(face.normal) < 0.0 {
                verts.reverse();
            }
            let tris = ear_clip_triangulate(&verts, face.normal);
            if tris.is_empty() {
                continue;
            }
            out.push(RawFace {
                verts,
                normal: face.normal,
                voxel: face.voxel,
                block_size: face.block_size,
                orig_triangles: tris,
            });
        }
    }
    (out, clipped_blocks)
}

fn build_solid_mesh(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable) -> SolidMesh {
    let mut mesh = SolidMesh::new();
    let mut raw_faces: Vec<RawFace> = Vec::new();
    let mut clip_shapes: Vec<RawFace> = Vec::new();

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
            let world_verts: Vec<Vec3> = face.vertices.iter()
                .map(|v| to_world(facing.rotate_block_point(*v, shape.size), wx, wy, wz))
                .collect();
            let normal = compute_world_normal(&world_verts, &face.triangles);
            let raw = RawFace {
                verts: world_verts,
                normal,
                voxel: (ox as usize, oy as usize, oz as usize),
                block_size: shape.size,
                orig_triangles: face.triangles.clone(),
            };

            if face_is_occluded(data, neighbors, shapes, block, face, facing, shape.size) {
                stats_faces_culled += 1;
                // Culled faces still cover the region they rest against —
                // keep them as clip shapes for partial-contact clipping.
                clip_shapes.push(raw);
                continue;
            }

            raw_faces.push(raw);
        }
    }

    let faces_before_btb = raw_faces.len();

    // Remove exactly coincident back-to-back faces (full-face contacts).
    let mut face_groups: HashMap<Vec<(i32, i32, i32)>, usize> = HashMap::new();
    for face in &raw_faces {
        let mut key: Vec<(i32, i32, i32)> = face.verts.iter().map(|v| quantize(*v)).collect();
        key.sort();
        *face_groups.entry(key).or_insert(0) += 1;
    }
    raw_faces.retain(|face| {
        let mut key: Vec<(i32, i32, i32)> = face.verts.iter().map(|v| quantize(*v)).collect();
        key.sort();
        face_groups.get(&key).map_or(true, |&count| count < 2)
    });
    let btb_removed = faces_before_btb - raw_faces.len();

    // Clip partial contacts (opposite-facing coplanar overlaps), then merge
    // coplanar same-facing surfaces split by contact seams.
    let faces_before_clip = raw_faces.len();
    let (raw_faces, clipped_blocks) = clip_contact_faces(raw_faces, clip_shapes);
    let raw_faces = merge_coplanar_faces(raw_faces, &clipped_blocks);
    let clipped_delta = raw_faces.len() as i64 - faces_before_clip as i64;

    for face in raw_faces {
        let vert_indices: Vec<u32> = face.verts.iter().map(|v| mesh.add_vert(*v)).collect();
        mesh.faces.push(SolidFace {
            verts: vert_indices,
            normal: face.normal,
            voxel: face.voxel,
            block_size: face.block_size,
            orig_triangles: face.orig_triangles,
        });
    }

    let stats_faces_emitted = stats_faces_total - stats_faces_culled;
    info!(
        "[solid mesh] blocks: {} (no_cells: {}, interior: {}), faces: {} total, {} culled, {} emitted, {} back-to-back removed, {:+} from contact clipping",
        stats_blocks, stats_blocks_no_cells, stats_blocks_interior,
        stats_faces_total, stats_faces_culled, stats_faces_emitted, btb_removed, clipped_delta,
    );

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

/// Compute the fillet push amount for a circular arc of UNIFORM radius
/// tangent to two face planes with normals `na` and `nb`.
///
/// `radius` is the fillet radius (CHAMFER_WIDTH).  The setback (how far from
/// the edge the cut line sits, perpendicular in each face) varies by edge
/// angle so that the radius stays constant:
///     setback = radius / tan(θ/2)          (see `fillet_setback_amount`)
/// where θ is the dihedral angle.
///
/// The push (distance from the original edge to the arc surface, along the
/// interior bisector) follows from the arc center sitting at radius/sin(θ/2)
/// from the edge:
///     push = radius * (1 - sin(θ/2)) / sin(θ/2)
///
/// In terms of k = |na + nb|:  sin(θ/2) = k/2
pub fn fillet_push_amount(na: Vec3, nb: Vec3, radius: f32) -> f32 {
    let k = (na + nb).length();
    let sin_half = (k / 2.0).clamp(0.05, 1.0); // sin(θ/2), θ = dihedral angle
    (radius * (1.0 - sin_half) / sin_half).clamp(0.0, radius * 2.0)
}

/// Compute the per-edge setback (perpendicular distance from the edge to the
/// cut line in each face) that yields a fillet of uniform `radius`:
///     setback = radius / tan(θ/2) = radius * cos(θ/2) / sin(θ/2)
/// For a 90° edge this equals `radius`.  Sharper edges get larger setbacks,
/// shallower edges smaller ones, so every rounded edge has the same radius.
pub fn fillet_setback_amount(na: Vec3, nb: Vec3, radius: f32) -> f32 {
    let k = (na + nb).length();
    let sin_half = (k / 2.0).clamp(0.05, 1.0);
    let cos_half = (1.0 - sin_half * sin_half).max(0.0).sqrt();
    (radius * cos_half / sin_half).clamp(0.0, radius * 2.5)
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

/// Returns true if two blocks share a full face — touching along exactly one
/// axis with positive 2D overlap on the other two (not just an edge or vertex).
pub fn blocks_are_face_adjacent(
    voxel_a: (usize, usize, usize),
    size_a: (u8, u8, u8),
    voxel_b: (usize, usize, usize),
    size_b: (u8, u8, u8),
) -> bool {
    let min_a = [voxel_a.0, voxel_a.1, voxel_a.2];
    let max_a = [voxel_a.0 + size_a.0 as usize, voxel_a.1 + size_a.1 as usize, voxel_a.2 + size_a.2 as usize];
    let min_b = [voxel_b.0, voxel_b.1, voxel_b.2];
    let max_b = [voxel_b.0 + size_b.0 as usize, voxel_b.1 + size_b.1 as usize, voxel_b.2 + size_b.2 as usize];

    for touch_axis in 0..3 {
        let touching = max_a[touch_axis] == min_b[touch_axis]
            || max_b[touch_axis] == min_a[touch_axis];
        if !touching {
            continue;
        }
        let mut has_area = true;
        for other in 0..3 {
            if other == touch_axis {
                continue;
            }
            let overlap = min_a[other].max(min_b[other]);
            let overlap_end = max_a[other].min(max_b[other]);
            if overlap_end <= overlap {
                has_area = false;
                break;
            }
        }
        if has_area {
            return true;
        }
    }
    false
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
    // Generate tangents so IBL specular reflections match the full-res mesh.
    let _ = mesh.generate_tangents();
    mesh
}

/// Test-only wrapper.
pub fn ear_clip_triangulate_public(verts: &[Vec3], normal: Vec3) -> Vec<[usize; 3]> {
    ear_clip_triangulate(verts, normal)
}
