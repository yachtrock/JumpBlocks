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

/// Triangulate a convex polygon using greedy ear-clipping that maximizes
/// the minimum angle (Delaunay-like quality). Avoids the sliver triangles
/// that simple fan triangulation creates for elongated polygons.
///
/// Emits triangle indices into `indices` with the reversed winding order
/// that the chamfer code expects: `[i+1, i, base]` style.
fn triangulate_convex_polygon(
    positions: &[[f32; 3]],
    base: u32,
    n: usize,
    indices: &mut Vec<u32>,
) {
    if n < 3 { return; }
    if n == 3 {
        indices.extend_from_slice(&[base + 2, base + 1, base]);
        return;
    }
    if n == 4 {
        // Two possible splits — pick the one with better minimum angle
        let p: Vec<Vec3> = (0..4).map(|i| Vec3::from_array(positions[(base as usize) + i])).collect();
        let min_a = min_angle_3d(p[0], p[1], p[2]).min(min_angle_3d(p[0], p[2], p[3]));
        let min_b = min_angle_3d(p[0], p[1], p[3]).min(min_angle_3d(p[1], p[2], p[3]));
        if min_a >= min_b {
            indices.extend_from_slice(&[base + 2, base + 1, base]);
            indices.extend_from_slice(&[base + 3, base + 2, base]);
        } else {
            indices.extend_from_slice(&[base + 3, base + 1, base]);
            indices.extend_from_slice(&[base + 3, base + 2, base + 1]);
        }
        return;
    }

    // For 5+ vertices: greedy ear-clipping — remove the ear with the
    // largest minimum angle at each step
    let mut remaining: Vec<u32> = (0..n as u32).map(|i| base + i).collect();

    while remaining.len() > 3 {
        let m = remaining.len();
        let mut best_ear = 0;
        let mut best_min_angle = -1.0f32;

        for i in 0..m {
            let pi = remaining[(i + m - 1) % m] as usize;
            let ci = remaining[i] as usize;
            let ni = remaining[(i + 1) % m] as usize;
            let pp = Vec3::from_array(positions[pi]);
            let pc = Vec3::from_array(positions[ci]);
            let pn = Vec3::from_array(positions[ni]);
            let angle = min_angle_3d(pp, pc, pn);
            if angle > best_min_angle {
                best_min_angle = angle;
                best_ear = i;
            }
        }

        let m = remaining.len();
        let prev = remaining[(best_ear + m - 1) % m];
        let curr = remaining[best_ear];
        let next = remaining[(best_ear + 1) % m];
        // Reversed winding for Bevy
        indices.extend_from_slice(&[next, curr, prev]);
        remaining.remove(best_ear);
    }

    // Last triangle
    indices.extend_from_slice(&[remaining[2], remaining[1], remaining[0]]);
}

/// Minimum angle (in radians) of a triangle defined by three 3D points.
fn min_angle_3d(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = (b - a).normalize_or_zero();
    let ac = (c - a).normalize_or_zero();
    let ba = (a - b).normalize_or_zero();
    let bc = (c - b).normalize_or_zero();
    let ca = (a - c).normalize_or_zero();
    let cb = (b - c).normalize_or_zero();
    let angle_a = ab.dot(ac).clamp(-1.0, 1.0).acos();
    let angle_b = ba.dot(bc).clamp(-1.0, 1.0).acos();
    let angle_c = ca.dot(cb).clamp(-1.0, 1.0).acos();
    angle_a.min(angle_b).min(angle_c)
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

        if !neighbor_cell_occludes(data, neighbors, shapes, nx, ny, nz, world_side.opposite()) {
            return false; // at least one coverage entry is not occluded
        }
    }

    true // all coverage entries are occluded
}

/// Check if the block at (nx, ny, nz) has a cell that fully covers `side_facing_us`.
///
/// This does per-cell checking: it resolves which specific cell within the neighbor
/// block is at position (nx, ny, nz) and checks only that cell's coverage.
fn neighbor_cell_occludes(
    data: &ChunkData,
    neighbors: &ChunkNeighbors,
    shapes: &ShapeTable,
    nx: i32, ny: i32, nz: i32,
    side_facing_us: FaceSide,
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
            if rotated_cell == world_offset && rotated_side == side_facing_us && cover.full {
                return true;
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
    let full_res = match mode {
        crate::PresentationMode::Flat => generate_lod_mesh(data, neighbors, shapes),
        crate::PresentationMode::EdgeGraphChamfer => generate_chamfered_mesh(data, neighbors, shapes),
        crate::PresentationMode::HalfEdgeChamfer => crate::halfedge_chamfer::generate_halfedge_chamfer(data, neighbors, shapes),
    };
    let t1 = Instant::now();
    let lod = generate_lod_mesh(data, neighbors, shapes);
    let t2 = Instant::now();

    let full_ms = (t1 - t0).as_secs_f64() * 1000.0;
    let lod_ms = (t2 - t1).as_secs_f64() * 1000.0;
    info!(
        "Chunk meshed [{}]: full={:.2}ms, lod={:.2}ms, total={:.2}ms (full: {} verts/{} tris, lod: {} verts/{} tris)",
        match mode { crate::PresentationMode::Flat => "flat", crate::PresentationMode::EdgeGraphChamfer => "edge-graph", crate::PresentationMode::HalfEdgeChamfer => "half-edge" },
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

    for (block_idx, block) in data.blocks.iter().enumerate() {
        let Some(shape) = shapes.get(block.shape) else { continue; };
        let facing = block.facing;
        let (ox, oy, oz) = block.origin;
        let wx = ox as f32 * VOXEL_SIZE;
        let wy = oy as f32 * VOXEL_SIZE;
        let wz = oz as f32 * VOXEL_SIZE;

        // Verify this block still owns at least one cell (not removed)
        let has_cells = shape.occupied_cells.iter().any(|&(dx, dy, dz)| {
            let cx = ox as usize + dx as usize;
            let cy = oy as usize + dy as usize;
            let cz = oz as usize + dz as usize;
            data.get_cell(cx, cy, cz) == Cell::Local(BlockId(block_idx as u16))
        });
        if !has_cells { continue; }

        for face in &shape.faces {
            if face_is_occluded(data, neighbors, shapes, block, face, facing, shape.size) { continue; }

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

/// Public entry point for building the solid mesh (used by halfedge_chamfer).
pub fn build_solid_mesh_public(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable) -> SolidMesh {
    build_solid_mesh(data, neighbors, shapes)
}

fn build_solid_mesh(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable) -> SolidMesh {
    let mut mesh = SolidMesh::new();

    for (block_idx, block) in data.blocks.iter().enumerate() {
        let Some(shape) = shapes.get(block.shape) else { continue; };
        let facing = block.facing;
        let (ox, oy, oz) = block.origin;
        let wx = ox as f32 * VOXEL_SIZE;
        let wy = oy as f32 * VOXEL_SIZE;
        let wz = oz as f32 * VOXEL_SIZE;

        // Verify this block still owns at least one cell
        let has_cells = shape.occupied_cells.iter().any(|&(dx, dy, dz)| {
            let cx = ox as usize + dx as usize;
            let cy = oy as usize + dy as usize;
            let cz = oz as usize + dz as usize;
            data.get_cell(cx, cy, cz) == Cell::Local(BlockId(block_idx as u16))
        });
        if !has_cells { continue; }

        for face in &shape.faces {
            if face_is_occluded(data, neighbors, shapes, block, face, facing, shape.size) { continue; }

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

    mesh
}

// ---------------------------------------------------------------------------
// Edge graph
// ---------------------------------------------------------------------------

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

struct EdgeInfo {
    faces: Vec<usize>,
}

const SHARP_DOT_THRESHOLD: f32 = 0.985;

fn build_edge_graph(mesh: &SolidMesh) -> HashMap<(u32, u32), EdgeInfo> {
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

fn is_sharp(edge: &EdgeInfo, mesh: &SolidMesh) -> bool {
    if edge.faces.len() < 2 {
        return true;
    }
    let n0 = mesh.faces[edge.faces[0]].normal;
    let n1 = mesh.faces[edge.faces[1]].normal;
    n0.dot(n1) < SHARP_DOT_THRESHOLD
}

/// Check if a boundary edge lies on a chunk face where a filled neighbor cell
/// would continue the surface.
fn is_boundary_edge_at_neighbor_seam(
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
// Sutherland-Hodgman polygon clipping
// ---------------------------------------------------------------------------

fn clip_polygon_by_plane(polygon: &[Vec3], plane_point: Vec3, plane_normal: Vec3) -> Vec<Vec3> {
    if polygon.len() < 3 { return polygon.to_vec(); }
    let mut output = Vec::new();
    let n = polygon.len();
    for i in 0..n {
        let s = polygon[i];
        let e = polygon[(i + 1) % n];
        let ds = (s - plane_point).dot(plane_normal);
        let de = (e - plane_point).dot(plane_normal);

        if de >= -1e-6 {
            if ds < -1e-6 {
                let t = ds / (ds - de);
                output.push(s.lerp(e, t));
            }
            output.push(e);
        } else if ds >= -1e-6 {
            let t = ds / (ds - de);
            output.push(s.lerp(e, t));
        }
    }
    output
}

struct ChamferClipPlane {
    point: Vec3,
    normal: Vec3,
    faces: [usize; 2],
    verts: [u32; 2],
}

// ---------------------------------------------------------------------------
// Chamfered mesh generation (edge-graph post-process)
// ---------------------------------------------------------------------------

fn generate_chamfered_mesh(data: &ChunkData, neighbors: &ChunkNeighbors, shapes: &ShapeTable) -> ChunkMeshData {
    let solid = build_solid_mesh(data, neighbors, shapes);
    let edge_graph = build_edge_graph(&solid);

    let sharp_set: HashMap<(u32, u32), &EdgeInfo> = edge_graph.iter()
        .filter(|(key, info)| {
            if !is_sharp(info, &solid) {
                return false;
            }
            if info.faces.len() == 1 {
                let (ev0, ev1) = **key;
                let face = &solid.faces[info.faces[0]];
                if is_boundary_edge_at_neighbor_seam(ev0, ev1, face, &solid, data, neighbors) {
                    return false;
                }
            }
            true
        })
        .map(|(&key, info)| (key, info))
        .collect();

    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); solid.positions.len()];
    for (fi, face) in solid.faces.iter().enumerate() {
        for &vi in &face.verts {
            vertex_faces[vi as usize].push(fi);
        }
    }

    // ---------------------------------------------------------------
    // Pass 1: compute inner vertices for every face
    // ---------------------------------------------------------------
    let all_inner: Vec<Vec<Vec3>> = solid.faces.iter().map(|face| {
        let n = face.verts.len();
        let normal = face.normal;

        // Block world-space bounds for clamping inner vertices
        let (ox, oy, oz) = face.voxel;
        let (sx, sy, sz) = face.block_size;
        let block_min = Vec3::new(
            ox as f32 * VOXEL_SIZE,
            oy as f32 * VOXEL_SIZE,
            oz as f32 * VOXEL_SIZE,
        );
        let block_max = Vec3::new(
            (ox + sx as usize) as f32 * VOXEL_SIZE,
            (oy + sy as usize) as f32 * VOXEL_SIZE,
            (oz + sz as usize) as f32 * VOXEL_SIZE,
        );

        let edge_sharp_flags: Vec<bool> = (0..n).map(|i| {
            let key = edge_key(face.verts[i], face.verts[(i + 1) % n]);
            sharp_set.contains_key(&key)
        }).collect();

        let edge_inward_dirs: Vec<Vec3> = (0..n).map(|i| {
            let p0 = solid.positions[face.verts[i] as usize];
            let p1 = solid.positions[face.verts[(i + 1) % n] as usize];
            (p1 - p0).normalize_or_zero().cross(normal).normalize_or_zero()
        }).collect();

        (0..n).map(|vi| {
            let mut offset = Vec3::ZERO;
            let prev = (vi + n - 1) % n;
            let prev_sharp = edge_sharp_flags[prev];
            let next_sharp = edge_sharp_flags[vi];
            if prev_sharp && next_sharp {
                let d = edge_inward_dirs[prev].dot(edge_inward_dirs[vi]);
                if d > 0.99 {
                    offset += edge_inward_dirs[vi] * CHAMFER_WIDTH;
                } else {
                    offset += edge_inward_dirs[prev] * CHAMFER_WIDTH;
                    offset += edge_inward_dirs[vi] * CHAMFER_WIDTH;
                }
            } else {
                if prev_sharp {
                    offset += edge_inward_dirs[prev] * CHAMFER_WIDTH;
                }
                if next_sharp {
                    offset += edge_inward_dirs[vi] * CHAMFER_WIDTH;
                }
            }
            // Boundary projection: if the original vertex sits on a block
            // boundary wall AND there is an adjacent occupied cell on the
            // other side of that wall, pin the inner vertex to stay on the
            // wall. This keeps chamfer edges aligned at block seams without
            // affecting floating blocks or exposed edges.
            let pos = solid.positions[face.verts[vi] as usize];
            let mut inner = pos + offset;
            let eps = 1e-4;
            let (bx, by, bz) = face.voxel;
            let (bsx, bsy, bsz) = face.block_size;
            // Check each axis: is the vertex on a block boundary AND
            // is there an adjacent block on the other side?
            let boundary_checks: [(usize, f32, f32, i32, i32, i32); 6] = [
                (0, block_min[0], block_min[0], bx as i32 - 1, by as i32, bz as i32),
                (0, block_max[0], block_max[0], bx as i32 + bsx as i32, by as i32, bz as i32),
                (1, block_min[1], block_min[1], bx as i32, by as i32 - 1, bz as i32),
                (1, block_max[1], block_max[1], bx as i32, by as i32 + bsy as i32, bz as i32),
                (2, block_min[2], block_min[2], bx as i32, by as i32, bz as i32 - 1),
                (2, block_max[2], block_max[2], bx as i32, by as i32, bz as i32 + bsz as i32),
            ];
            for &(axis, boundary_val, pin_val, nx, ny, nz) in &boundary_checks {
                if (pos[axis] - boundary_val).abs() < eps {
                    // Only pin if there's a filled neighbor on the other side
                    if is_cell_occupied(data, neighbors, nx, ny, nz) {
                        inner[axis] = pin_val;
                    }
                }
            }
            inner
        }).collect()
    }).collect();

    // ---------------------------------------------------------------
    // Pass 2b: collect chamfer strip planes per owning voxel
    // ---------------------------------------------------------------
    struct StripClip {
        point: Vec3,
        normal: Vec3,
        owning_voxels: [(usize, usize, usize); 2],
        owning_sizes: [(u8, u8, u8); 2],
        edge_verts: [u32; 2],
        aabb_min: Vec3,
        aabb_max: Vec3,
        face_normal_dot: f32,
    }
    let mut voxel_strip_clips: HashMap<(usize, usize, usize), Vec<usize>> = HashMap::new();
    let mut all_strip_clips: Vec<StripClip> = Vec::new();

    for (&(ev0, ev1), info) in &edge_graph {
        if !is_sharp(info, &solid) { continue; }
        if info.faces.len() < 2 { continue; }

        let fi_a = info.faces[0];
        let fi_b = info.faces[1];
        let face_a = &solid.faces[fi_a];
        let face_b = &solid.faces[fi_b];

        let a0 = find_face_inner_at_vert(face_a, &all_inner[fi_a], ev0).unwrap();
        let a1 = find_face_inner_at_vert(face_a, &all_inner[fi_a], ev1).unwrap();
        let b0 = find_face_inner_at_vert(face_b, &all_inner[fi_b], ev0).unwrap();
        let b1 = find_face_inner_at_vert(face_b, &all_inner[fi_b], ev1).unwrap();

        let strip_normal = (a1 - a0).cross(b0 - a0);
        if strip_normal.length_squared() < 1e-10 { continue; }
        let strip_normal = strip_normal.normalize();
        let expected_out = (face_a.normal + face_b.normal).normalize_or_zero();
        let strip_normal = if strip_normal.dot(expected_out) < 0.0 { -strip_normal } else { strip_normal };

        let margin = CHAMFER_WIDTH * 0.5;
        let strip_min = a0.min(a1).min(b0).min(b1) - Vec3::splat(margin);
        let strip_max = a0.max(a1).max(b0).max(b1) + Vec3::splat(margin);

        let clip_idx = all_strip_clips.len();
        let owning = [face_a.voxel, face_b.voxel];
        let sizes = [face_a.block_size, face_b.block_size];
        all_strip_clips.push(StripClip {
            point: a0, normal: strip_normal, owning_voxels: owning,
            owning_sizes: sizes,
            edge_verts: [ev0, ev1], aabb_min: strip_min, aabb_max: strip_max,
            face_normal_dot: face_a.normal.dot(face_b.normal),
        });

        for &v in &owning {
            voxel_strip_clips.entry(v).or_default().push(clip_idx);
        }
    }

    let mut voxel_faces_map: HashMap<(usize, usize, usize), Vec<usize>> = HashMap::new();
    for (fi, face) in solid.faces.iter().enumerate() {
        voxel_faces_map.entry(face.voxel).or_default().push(fi);
    }

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut chamfer_offsets: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // ---------------------------------------------------------------
    // Pass 3: emit inner faces, clipped against chamfer strips from neighbors
    // ---------------------------------------------------------------
    for (fi, face) in solid.faces.iter().enumerate() {
        let normal_arr = face.normal.to_array();
        let inner = &all_inner[fi];

        let (vx, vy, vz) = face.voxel;
        let mut clip_planes: Vec<(Vec3, Vec3)> = Vec::new();
        let (sx, sy, sz) = face.block_size;
        // Check neighbors at the block's outer boundary, not just ±1 from origin
        let neighbor_offsets: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (sx as i32, 0, 0),
            (0, -1, 0), (0, sy as i32, 0),
            (0, 0, -1), (0, 0, sz as i32),
        ];
        for &(dx, dy, dz) in &neighbor_offsets {
            let nx = vx as i32 + dx;
            let ny = vy as i32 + dy;
            let nz = vz as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 { continue; }
            let nkey = (nx as usize, ny as usize, nz as usize);
            if nkey == face.voxel { continue; }
            if let Some(clip_indices) = voxel_strip_clips.get(&nkey) {
                for &ci in clip_indices {
                    let sc = &all_strip_clips[ci];
                    if sc.owning_voxels.contains(&face.voxel) { continue; }
                    let mut owner_min = Vec3::splat(f32::MAX);
                    let mut owner_max = Vec3::splat(f32::MIN);
                    for (i, &(ovx, ovy, ovz)) in sc.owning_voxels.iter().enumerate() {
                        let sz = sc.owning_sizes[i];
                        let lo = Vec3::new(ovx as f32 * VOXEL_SIZE, ovy as f32 * VOXEL_SIZE, ovz as f32 * VOXEL_SIZE);
                        let hi = lo + Vec3::new(sz.0 as f32 * VOXEL_SIZE, sz.1 as f32 * VOXEL_SIZE, sz.2 as f32 * VOXEL_SIZE);
                        owner_min = owner_min.min(lo);
                        owner_max = owner_max.max(hi);
                    }
                    let margin = CHAMFER_WIDTH * 0.5;
                    let extends = sc.aabb_min.x < owner_min.x - margin
                        || sc.aabb_max.x > owner_max.x + margin
                        || sc.aabb_min.y < owner_min.y - margin
                        || sc.aabb_max.y > owner_max.y + margin
                        || sc.aabb_min.z < owner_min.z - margin
                        || sc.aabb_max.z > owner_max.z + margin;
                    if !extends { continue; }
                    let face_min = inner.iter().copied().reduce(|a, b| a.min(b)).unwrap();
                    let face_max = inner.iter().copied().reduce(|a, b| a.max(b)).unwrap();
                    let overlaps = face_min.x <= sc.aabb_max.x && face_max.x >= sc.aabb_min.x
                        && face_min.y <= sc.aabb_max.y && face_max.y >= sc.aabb_min.y
                        && face_min.z <= sc.aabb_max.z && face_max.z >= sc.aabb_min.z;
                    if !overlaps { continue; }
                    clip_planes.push((sc.point, -sc.normal));
                }
            }
        }

        if clip_planes.is_empty() {
            let n = inner.len();
            let inner_base = positions.len() as u32;
            for i in 0..n {
                positions.push(inner[i].to_array());
                normals.push(normal_arr);
                uvs.push([0.0, 0.0]);
                chamfer_offsets.push([0.0; 3]);
            }
            // Use the shape's original grid-aligned triangulation.
            // The inner polygon has the same vertex order as the original face,
            // so the shape's triangle indices map directly.
            if face.orig_triangles.len() >= n - 2 {
                for tri in &face.orig_triangles {
                    indices.push(inner_base + tri[2] as u32);
                    indices.push(inner_base + tri[1] as u32);
                    indices.push(inner_base + tri[0] as u32);
                }
            } else {
                // Fallback to Delaunay if shape triangles are missing/wrong
                triangulate_convex_polygon(&positions, inner_base, n, &mut indices);
            }
        } else {
            let mut polygon: Vec<Vec3> = inner.clone();

            for &(plane_point, plane_normal) in &clip_planes {
                let mut has_inside = false;
                let mut has_outside = false;
                for p in &polygon {
                    let d = (*p - plane_point).dot(plane_normal);
                    if d < -1e-5 { has_outside = true; }
                    else { has_inside = true; }
                }
                if !has_outside { continue; }
                if !has_inside {
                    warn!("Face {} entirely on clip side of strip plane (voxel={:?})", fi, face.voxel);
                    polygon.clear();
                    break;
                }
                polygon = clip_polygon_by_plane(&polygon, plane_point, plane_normal);
                if polygon.len() < 3 { break; }
            }

            if polygon.len() >= 3 {
                let mut clean_poly: Vec<Vec3> = Vec::new();
                for &p in &polygon {
                    if clean_poly.last().map_or(true, |prev: &Vec3| (*prev - p).length_squared() > 1e-8) {
                        clean_poly.push(p);
                    }
                }
                if clean_poly.len() >= 3 && (clean_poly[0] - *clean_poly.last().unwrap()).length_squared() < 1e-8 {
                    clean_poly.pop();
                }

                if clean_poly.len() >= 3 {
                    let inner_base = positions.len() as u32;
                    for p in &clean_poly {
                        positions.push(p.to_array());
                        normals.push(normal_arr);
                        uvs.push([0.0, 0.0]);
                        chamfer_offsets.push([0.0; 3]);
                    }
                    triangulate_convex_polygon(&positions, inner_base, clean_poly.len(), &mut indices);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Pass 3: emit chamfer strips per-edge
    // ---------------------------------------------------------------
    fn find_face_inner_at_vert(face: &SolidFace, inner: &[Vec3], vert: u32) -> Option<Vec3> {
        face.verts.iter().position(|&v| v == vert).map(|i| inner[i])
    }

    fn collect_neighbor_clip_faces(
        chamfer_voxels: &[(usize, usize, usize)],
        voxel_faces_map: &HashMap<(usize, usize, usize), Vec<usize>>,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        for &(vx, vy, vz) in chamfer_voxels {
            for &(dx, dy, dz) in &[
                (-1i32,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)
            ] {
                let nx = vx as i32 + dx;
                let ny = vy as i32 + dy;
                let nz = vz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0 { continue; }
                let nkey = (nx as usize, ny as usize, nz as usize);
                if chamfer_voxels.contains(&nkey) { continue; }
                if let Some(faces) = voxel_faces_map.get(&nkey) {
                    for &fi in faces {
                        if !result.contains(&fi) {
                            result.push(fi);
                        }
                    }
                }
            }
        }
        result
    }

    for (&(ev0, ev1), info) in &edge_graph {
        if !sharp_set.contains_key(&edge_key(ev0, ev1)) { continue; }

        if info.faces.len() >= 2 {
            let fi_a = info.faces[0];
            let fi_b = info.faces[1];
            let face_a = &solid.faces[fi_a];
            let face_b = &solid.faces[fi_b];
            let inner_a = &all_inner[fi_a];
            let inner_b = &all_inner[fi_b];

            let a0 = find_face_inner_at_vert(face_a, inner_a, ev0).unwrap();
            let a1 = find_face_inner_at_vert(face_a, inner_a, ev1).unwrap();
            let b0 = find_face_inner_at_vert(face_b, inner_b, ev0).unwrap();
            let b1 = find_face_inner_at_vert(face_b, inner_b, ev1).unwrap();

            let na = face_a.normal;
            let nb = face_b.normal;
            let expected_out = (na + nb).normalize_or_zero();

            let tri_normal = (a1 - a0).cross(b0 - a0);
            let flipped = tri_normal.dot(expected_out) > 0.0;
            let strip_poly = if flipped {
                vec![b0, b1, a1, a0]
            } else {
                vec![a0, a1, b1, b0]
            };

            if strip_poly.len() >= 3 {
                let base = positions.len() as u32;
                let (n_first, n_second) = if flipped { (nb, na) } else { (na, nb) };

                for (pi, p) in strip_poly.iter().enumerate() {
                    positions.push(p.to_array());
                    let n = if pi < strip_poly.len() / 2 {
                        n_first.to_array()
                    } else {
                        n_second.to_array()
                    };
                    normals.push(n);
                    uvs.push([0.0, 0.0]);
                    chamfer_offsets.push([0.0; 3]);
                }
                triangulate_convex_polygon(&positions, base, strip_poly.len(), &mut indices);
            }
        } else {
            let fi_a = info.faces[0];
            let face_a = &solid.faces[fi_a];
            let inner_a = &all_inner[fi_a];

            let a0 = find_face_inner_at_vert(face_a, inner_a, ev0).unwrap();
            let a1 = find_face_inner_at_vert(face_a, inner_a, ev1).unwrap();
            let outer0 = solid.positions[ev0 as usize];
            let outer1 = solid.positions[ev1 as usize];

            let na = face_a.normal;
            let edge_dir = (solid.positions[ev1 as usize] - solid.positions[ev0 as usize]).normalize_or_zero();
            let outward = edge_dir.cross(na).normalize_or_zero();

            let tri_normal = (a1 - a0).cross(outer0 - a0);
            let expected_out = (na + outward).normalize_or_zero();
            let flipped = tri_normal.dot(expected_out) > 0.0;

            let strip_poly = if flipped {
                vec![outer0, outer1, a1, a0]
            } else {
                vec![a0, a1, outer1, outer0]
            };

            if strip_poly.len() >= 3 {
                let strip_normal = expected_out.to_array();
                let base = positions.len() as u32;
                for p in &strip_poly {
                    positions.push(p.to_array());
                    normals.push(strip_normal);
                    uvs.push([0.0, 0.0]);
                    chamfer_offsets.push([0.0; 3]);
                }
                triangulate_convex_polygon(&positions, base, strip_poly.len(), &mut indices);
            }
        }
    }

    // ---------------------------------------------------------------
    // Pass 4: corner caps at vertices with 3+ sharp edges
    // ---------------------------------------------------------------
    for (vi, adj_faces) in vertex_faces.iter().enumerate() {
        if adj_faces.is_empty() { continue; }

        let mut sharp_edge_keys: Vec<(u32, u32)> = Vec::new();
        for &fi in adj_faces {
            let face = &solid.faces[fi];
            let n = face.verts.len();
            for i in 0..n {
                if face.verts[i] as usize == vi || face.verts[(i + 1) % n] as usize == vi {
                    let key = edge_key(face.verts[i], face.verts[(i + 1) % n]);
                    if sharp_set.contains_key(&key) && !sharp_edge_keys.contains(&key) {
                        sharp_edge_keys.push(key);
                    }
                }
            }
        }

        if sharp_edge_keys.len() < 3 { continue; }

        let outer = solid.positions[vi];
        let mut ring: Vec<(Vec3, Vec3)> = Vec::new();
        for &fi in adj_faces {
            let face = &solid.faces[fi];
            if let Some(inner_pos) = find_face_inner_at_vert(face, &all_inner[fi], vi as u32) {
                if (inner_pos - outer).length_squared() > 1e-8 {
                    let dup = ring.iter().any(|(r, _)| (*r - inner_pos).length_squared() < 1e-6);
                    if !dup {
                        ring.push((inner_pos, face.normal));
                    }
                }
            }
        }

        if ring.len() < 3 { continue; }

        let axis = adj_faces.iter()
            .map(|&fi| solid.faces[fi].normal)
            .sum::<Vec3>()
            .normalize_or_zero();

        let ref_dir = if axis.x.abs() < 0.9 {
            axis.cross(Vec3::X).normalize_or_zero()
        } else {
            axis.cross(Vec3::Y).normalize_or_zero()
        };
        let tangent = axis.cross(ref_dir).normalize_or_zero();

        ring.sort_by(|a, b| {
            let da = a.0 - outer;
            let db = b.0 - outer;
            let angle_a = da.dot(tangent).atan2(da.dot(ref_dir));
            let angle_b = db.dot(tangent).atan2(db.dot(ref_dir));
            angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let ring_start = positions.len() as u32;
        for (rp, rn) in &ring {
            positions.push(rp.to_array());
            normals.push(rn.to_array());
            uvs.push([0.0, 0.0]);
            chamfer_offsets.push([0.0; 3]);
        }

        let mut adjacent_normals: Vec<Vec3> = Vec::new();
        for &fi in adj_faces {
            let n = solid.faces[fi].normal;
            if !adjacent_normals.iter().any(|an| an.dot(n).abs() > 0.99) {
                adjacent_normals.push(n);
            }
        }
        for &ek in &sharp_edge_keys {
            if let Some(info) = sharp_set.get(&ek) {
                if info.faces.len() >= 2 {
                    let na = solid.faces[info.faces[0]].normal;
                    let nb = solid.faces[info.faces[1]].normal;
                    let sn = (na + nb).normalize_or_zero();
                    if !adjacent_normals.iter().any(|an| an.dot(sn).abs() > 0.95) {
                        adjacent_normals.push(sn);
                    }
                }
            }
        }

        let tri_normal = (ring[1].0 - ring[0].0).cross(ring[2].0 - ring[0].0);
        let flip = tri_normal.dot(axis) < 0.0;

        for i in 1..ring.len() - 1 {
            let va = ring[0].0;
            let vb = ring[i].0;
            let vc = ring[i + 1].0;
            let fan_normal = (vb - va).cross(vc - va).normalize_or_zero();

            let mut skip = false;
            for &fi in adj_faces {
                let face = &solid.faces[fi];
                let fn_normal = face.normal;
                if fan_normal.dot(fn_normal).abs() < 0.99 { continue; }
                let face_point = solid.positions[face.verts[0] as usize];
                let da = (va - face_point).dot(fn_normal).abs();
                let db = (vb - face_point).dot(fn_normal).abs();
                let dc = (vc - face_point).dot(fn_normal).abs();
                if da < 0.001 && db < 0.001 && dc < 0.001 {
                    skip = true;
                    break;
                }
            }
            if skip { continue; }

            let a = ring_start;
            let b = ring_start + i as u32;
            let c = ring_start + i as u32 + 1;
            if flip {
                indices.extend_from_slice(&[a, c, b]);
            } else {
                indices.extend_from_slice(&[a, b, c]);
            }
        }
    }

    ChunkMeshData { positions, normals, uvs, chamfer_offsets, indices }
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
