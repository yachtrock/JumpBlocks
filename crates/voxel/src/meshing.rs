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
        wx + v.x * VOXEL_WIDTH,
        wy + v.y * VOXEL_HEIGHT,
        wz + v.z * VOXEL_WIDTH,
    )
}

fn compute_world_normal(world_verts: &[Vec3], triangles: &[[usize; 3]]) -> Vec3 {
    let tri = &triangles[0];
    let a = world_verts[tri[0]];
    let b = world_verts[tri[1]];
    let c = world_verts[tri[2]];
    -(b - a).cross(c - a).normalize_or_zero()
}

fn neighbor_occludes(
    data: &ChunkData,
    shapes: &ShapeTable,
    nx: i32, ny: i32, nz: i32,
    side_facing_us: FaceSide,
) -> bool {
    let neighbor = data.get_signed(nx, ny, nz);
    if neighbor.is_empty() {
        return false;
    }
    let Some(neighbor_shape) = shapes.get(neighbor.shape_index()) else {
        return false;
    };
    neighbor_shape.occlusion.occludes_world_side(side_facing_us, neighbor.facing())
}

fn opposite_side(side: FaceSide) -> FaceSide {
    match side {
        FaceSide::Top => FaceSide::Bottom,
        FaceSide::Bottom => FaceSide::Top,
        FaceSide::North => FaceSide::South,
        FaceSide::South => FaceSide::North,
        FaceSide::East => FaceSide::West,
        FaceSide::West => FaceSide::East,
        FaceSide::None => FaceSide::None,
    }
}

/// Should this face be culled (hidden by neighbor)?
fn face_is_occluded(
    data: &ChunkData,
    shapes: &ShapeTable,
    x: usize, y: usize, z: usize,
    rotated_side: FaceSide,
) -> bool {
    if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nz = z as i32 + dz;
        neighbor_occludes(data, shapes, nx, ny, nz, opposite_side(rotated_side))
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn generate_chunk_mesh(data: &ChunkData, shapes: &ShapeTable, mode: crate::PresentationMode) -> ChunkMeshResult {
    let t0 = Instant::now();
    let full_res = match mode {
        crate::PresentationMode::Flat => generate_lod_mesh(data, shapes),
        crate::PresentationMode::EdgeGraphChamfer => generate_chamfered_mesh(data, shapes),
        crate::PresentationMode::HalfEdgeChamfer => crate::halfedge_chamfer::generate_halfedge_chamfer(data, shapes),
        crate::PresentationMode::OpenMeshChamfer => crate::openmesh_chamfer::generate_openmesh_chamfer(data, shapes),
    };
    let t1 = Instant::now();
    let lod = generate_lod_mesh(data, shapes);
    let t2 = Instant::now();

    let full_ms = (t1 - t0).as_secs_f64() * 1000.0;
    let lod_ms = (t2 - t1).as_secs_f64() * 1000.0;
    info!(
        "Chunk meshed [{}]: full={:.2}ms, lod={:.2}ms, total={:.2}ms (full: {} verts/{} tris, lod: {} verts/{} tris)",
        match mode { crate::PresentationMode::Flat => "flat", crate::PresentationMode::EdgeGraphChamfer => "edge-graph", crate::PresentationMode::HalfEdgeChamfer => "half-edge", crate::PresentationMode::OpenMeshChamfer => "openmesh" },
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
// LOD mesh (unchanged — simple, no chamfer)
// ---------------------------------------------------------------------------

fn generate_lod_mesh(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for y in 0..CHUNK_Y {
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let voxel = data.get(x, y, z);
                if voxel.is_empty() { continue; }
                let Some(shape) = shapes.get(voxel.shape_index()) else { continue; };
                let facing = voxel.facing();
                let wx = x as f32 * VOXEL_WIDTH;
                let wy = y as f32 * VOXEL_HEIGHT;
                let wz = z as f32 * VOXEL_WIDTH;

                for face in &shape.faces {
                    let rotated_side = face.side.rotated_by(facing);
                    if face_is_occluded(data, shapes, x, y, z, rotated_side) { continue; }

                    let base_index = positions.len() as u32;
                    let world_verts: Vec<Vec3> = face.vertices.iter()
                        .map(|v| to_world(facing.rotate_point(*v), wx, wy, wz))
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
    pub chamfer_mode: ChamferMode,
    /// Source voxel position in chunk coords.
    pub voxel: (usize, usize, usize),
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
pub fn build_solid_mesh_public(data: &ChunkData, shapes: &ShapeTable) -> SolidMesh {
    build_solid_mesh(data, shapes)
}

fn build_solid_mesh(data: &ChunkData, shapes: &ShapeTable) -> SolidMesh {
    let mut mesh = SolidMesh::new();

    for y in 0..CHUNK_Y {
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let voxel = data.get(x, y, z);
                if voxel.is_empty() { continue; }
                let Some(shape) = shapes.get(voxel.shape_index()) else { continue; };
                let facing = voxel.facing();
                let wx = x as f32 * VOXEL_WIDTH;
                let wy = y as f32 * VOXEL_HEIGHT;
                let wz = z as f32 * VOXEL_WIDTH;

                for face in &shape.faces {
                    let rotated_side = face.side.rotated_by(facing);
                    if face_is_occluded(data, shapes, x, y, z, rotated_side) { continue; }

                    let world_verts: Vec<Vec3> = face.vertices.iter()
                        .map(|v| to_world(facing.rotate_point(*v), wx, wy, wz))
                        .collect();
                    let normal = compute_world_normal(&world_verts, &face.triangles);
                    let vert_indices: Vec<u32> = world_verts.iter()
                        .map(|v| mesh.add_vert(*v))
                        .collect();

                    mesh.faces.push(SolidFace {
                        verts: vert_indices,
                        normal,
                        chamfer_mode: face.chamfer_mode,
                        voxel: (x, y, z),
                    });
                }
            }
        }
    }

    // Remove back-to-back faces: pairs of faces that share the exact same
    // vertex set (e.g., adjacent wedge triangles facing opposite directions).
    // These are internal surfaces hidden by the slope above.
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
    /// Which faces share this edge (indices into SolidMesh.faces).
    faces: Vec<usize>,
}

/// Dot-product threshold: edges with adjacent normals below this are "sharp".
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
        return true; // boundary edge — always sharp
    }
    let n0 = mesh.faces[edge.faces[0]].normal;
    let n1 = mesh.faces[edge.faces[1]].normal;
    n0.dot(n1) < SHARP_DOT_THRESHOLD
}

// ---------------------------------------------------------------------------
// Sutherland-Hodgman polygon clipping
// ---------------------------------------------------------------------------

/// Clip a convex polygon against a half-plane. Keeps the side where `plane_normal` points.
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
            // E is inside
            if ds < -1e-6 {
                // S is outside → add intersection
                let t = ds / (ds - de);
                output.push(s.lerp(e, t));
            }
            output.push(e);
        } else if ds >= -1e-6 {
            // E is outside, S is inside → add intersection only
            let t = ds / (ds - de);
            output.push(s.lerp(e, t));
        }
    }
    output
}

/// A clip plane from a chamfer strip, used to trim neighboring faces.
struct ChamferClipPlane {
    point: Vec3,
    normal: Vec3,
    /// The two faces forming this chamfer (don't clip these).
    faces: [usize; 2],
    /// The edge vertices (for lookup).
    verts: [u32; 2],
}

// ---------------------------------------------------------------------------
// Chamfered mesh generation (edge-graph post-process)
// ---------------------------------------------------------------------------

fn generate_chamfered_mesh(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
    let solid = build_solid_mesh(data, shapes);
    let edge_graph = build_edge_graph(&solid);

    // Classify edges
    let sharp_set: HashMap<(u32, u32), &EdgeInfo> = edge_graph.iter()
        .filter(|(_, info)| is_sharp(info, &solid))
        .map(|(&key, info)| (key, info))
        .collect();

    // Build vertex → faces map for clipping chamfer vertices against adjacent faces
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
            if edge_sharp_flags[prev] {
                offset += edge_inward_dirs[prev] * CHAMFER_WIDTH;
            }
            if edge_sharp_flags[vi] {
                offset += edge_inward_dirs[vi] * CHAMFER_WIDTH;
            }
            solid.positions[face.verts[vi] as usize] + offset
        }).collect()
    }).collect();

    // ---------------------------------------------------------------
    // Pass 2b: collect chamfer strip planes per owning voxel
    // ---------------------------------------------------------------
    // For each interior sharp edge, compute the strip plane and associate
    // it with the owning voxels. These planes will be used to clip inner
    // faces of NEIGHBOR voxels.
    struct StripClip {
        point: Vec3,
        normal: Vec3,
        owning_voxels: [(usize, usize, usize); 2],
        edge_verts: [u32; 2],
        aabb_min: Vec3,
        aabb_max: Vec3,
        /// Dot product of the two face normals forming this chamfer
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

        // Compute AABB of the strip quad, expanded by a small margin
        let margin = CHAMFER_WIDTH * 0.5;
        let strip_min = a0.min(a1).min(b0).min(b1) - Vec3::splat(margin);
        let strip_max = a0.max(a1).max(b0).max(b1) + Vec3::splat(margin);

        let clip_idx = all_strip_clips.len();
        let owning = [face_a.voxel, face_b.voxel];
        all_strip_clips.push(StripClip {
            point: a0, normal: strip_normal, owning_voxels: owning,
            edge_verts: [ev0, ev1], aabb_min: strip_min, aabb_max: strip_max,
            face_normal_dot: face_a.normal.dot(face_b.normal),
        });

        // Associate with owning voxels
        for &v in &owning {
            voxel_strip_clips.entry(v).or_default().push(clip_idx);
        }
    }

    // Build voxel → faces map
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

        // Collect strip clip planes from NEIGHBOR voxels that might cross this face.
        // For each axis-aligned neighbor of this face's voxel, gather their strip planes.
        let (vx, vy, vz) = face.voxel;
        let mut clip_planes: Vec<(Vec3, Vec3)> = Vec::new(); // (point, normal)
        for &(dx, dy, dz) in &[(-1i32,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)] {
            let nx = vx as i32 + dx;
            let ny = vy as i32 + dy;
            let nz = vz as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 { continue; }
            let nkey = (nx as usize, ny as usize, nz as usize);
            if nkey == face.voxel { continue; }
            if let Some(clip_indices) = voxel_strip_clips.get(&nkey) {
                for &ci in clip_indices {
                    let sc = &all_strip_clips[ci];
                    // Only use strips owned by the neighbor (not our own voxel)
                    if sc.owning_voxels.contains(&face.voxel) { continue; }
                    // Only clip against strips whose geometry extends past their
                    // owning voxels' bounds (diagonal chamfer strips at corners).
                    let mut owner_min = Vec3::splat(f32::MAX);
                    let mut owner_max = Vec3::splat(f32::MIN);
                    for &(ovx, ovy, ovz) in &sc.owning_voxels {
                        let lo = Vec3::new(ovx as f32 * VOXEL_WIDTH, ovy as f32 * VOXEL_HEIGHT, ovz as f32 * VOXEL_WIDTH);
                        let hi = lo + Vec3::new(VOXEL_WIDTH, VOXEL_HEIGHT, VOXEL_WIDTH);
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
                    // Only clip if this face's AABB overlaps the strip's AABB
                    let face_min = inner.iter().copied().reduce(|a, b| a.min(b)).unwrap();
                    let face_max = inner.iter().copied().reduce(|a, b| a.max(b)).unwrap();
                    let overlaps = face_min.x <= sc.aabb_max.x && face_max.x >= sc.aabb_min.x
                        && face_min.y <= sc.aabb_max.y && face_max.y >= sc.aabb_min.y
                        && face_min.z <= sc.aabb_max.z && face_max.z >= sc.aabb_min.z;
                    if !overlaps { continue; }
                    // Keep the side AWAY from the chamfer interior
                    clip_planes.push((sc.point, -sc.normal));
                }
            }
        }

        if clip_planes.is_empty() {
            // No clipping needed
            let n = inner.len();
            let inner_base = positions.len() as u32;
            for i in 0..n {
                positions.push(inner[i].to_array());
                normals.push(normal_arr);
                uvs.push([0.0, 0.0]);
                chamfer_offsets.push([0.0; 3]);
            }
            for i in 1..n - 1 {
                indices.push(inner_base + i as u32 + 1);
                indices.push(inner_base + i as u32);
                indices.push(inner_base);
            }
        } else {
            // Clip face polygon against neighbor chamfer strip planes.
            // The strip normal points outward from the chamfer surface.
            // Keep the side where the strip normal points (exterior).
            let mut polygon: Vec<Vec3> = inner.clone();

            for &(plane_point, plane_normal) in &clip_planes {
                // Only clip if the face actually straddles this plane
                // (some vertices inside, some outside). If all on keep side, skip.
                let mut has_inside = false;
                let mut has_outside = false;
                for p in &polygon {
                    let d = (*p - plane_point).dot(plane_normal);
                    if d < -1e-5 { has_outside = true; }
                    else { has_inside = true; }
                }
                if !has_outside { continue; } // all on keep side, no clip needed
                if !has_inside {
                    // All on clip side — entire face removed. This shouldn't happen
                    // for faces that share a vertex with the strip edge.
                    warn!("Face {} entirely on clip side of strip plane (voxel={:?})", fi, face.voxel);
                    polygon.clear();
                    break;
                }
                polygon = clip_polygon_by_plane(&polygon, plane_point, plane_normal);
                if polygon.len() < 3 { break; }
            }

            if polygon.len() >= 3 {
                // Remove near-duplicate vertices that create degenerate triangles
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
                    for i in 1..clean_poly.len() - 1 {
                        indices.push(inner_base + i as u32 + 1);
                        indices.push(inner_base + i as u32);
                        indices.push(inner_base);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Pass 3: emit chamfer strips per-edge (not per-face)
    // ---------------------------------------------------------------
    fn find_face_inner_at_vert(face: &SolidFace, inner: &[Vec3], vert: u32) -> Option<Vec3> {
        face.verts.iter().position(|&v| v == vert).map(|i| inner[i])
    }

    /// Collect face indices from axis-aligned neighbor voxels (excluding owning voxels).
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
        if !is_sharp(info, &solid) { continue; }

        let chamfer_mode = if info.faces.len() >= 2 {
            let m0 = solid.faces[info.faces[0]].chamfer_mode;
            let m1 = solid.faces[info.faces[1]].chamfer_mode;
            if m0 == ChamferMode::Smooth && m1 == ChamferMode::Smooth {
                ChamferMode::Smooth
            } else {
                ChamferMode::Hard
            }
        } else {
            solid.faces[info.faces[0]].chamfer_mode
        };

        if info.faces.len() >= 2 {
            // Interior sharp edge: chamfer strip connects both faces' inner vertices
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

            // Build strip polygon (quad)
            let tri_normal = (a1 - a0).cross(b0 - a0);
            let flipped = tri_normal.dot(expected_out) > 0.0;
            let mut strip_poly = if flipped {
                vec![b0, b1, a1, a0]
            } else {
                vec![a0, a1, b1, b0]
            };

            // Clip strip against faces of neighboring voxels that the strip
            // Strips extend past voxel bounds; neighbor faces are clipped to accommodate.

            if strip_poly.len() >= 3 {
                let base = positions.len() as u32;
                let (n_first, n_second) = if flipped { (nb, na) } else { (na, nb) };

                for (pi, p) in strip_poly.iter().enumerate() {
                    positions.push(p.to_array());
                    let n = match chamfer_mode {
                        ChamferMode::Hard => expected_out.to_array(),
                        ChamferMode::Smooth => {
                            // First half of polygon gets face A normal,
                            // second half gets face B normal
                            if pi < strip_poly.len() / 2 {
                                n_first.to_array()
                            } else {
                                n_second.to_array()
                            }
                        }
                    };
                    normals.push(n);
                    uvs.push([0.0, 0.0]);
                    chamfer_offsets.push([0.0; 3]);
                }
                // Fan triangulation
                for i in 1..strip_poly.len() - 1 {
                    indices.push(base + i as u32 + 1);
                    indices.push(base + i as u32);
                    indices.push(base);
                }
            }
        } else {
            // Boundary edge: strip from inner to outer (original position)
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

            let mut strip_poly = if flipped {
                vec![outer0, outer1, a1, a0]
            } else {
                vec![a0, a1, outer1, outer0]
            };

            // Boundary strips extend freely.

            if strip_poly.len() >= 3 {
                let strip_normal = expected_out.to_array();
                let base = positions.len() as u32;
                for p in &strip_poly {
                    positions.push(p.to_array());
                    normals.push(strip_normal);
                    uvs.push([0.0, 0.0]);
                    chamfer_offsets.push([0.0; 3]);
                }
                for i in 1..strip_poly.len() - 1 {
                    indices.push(base + i as u32 + 1);
                    indices.push(base + i as u32);
                    indices.push(base);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Pass 4: corner caps at vertices with 3+ sharp edges
    // ---------------------------------------------------------------
    for (vi, adj_faces) in vertex_faces.iter().enumerate() {
        if adj_faces.is_empty() { continue; }

        // Count unique sharp edges at this vertex
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

        // Collect unique inner vertex positions + face normals from each adjacent face
        let outer = solid.positions[vi];
        let mut ring: Vec<(Vec3, Vec3)> = Vec::new(); // (position, face_normal)
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

        // Sort ring by angle around the outer vertex.
        // Use average face normal as the sort axis.
        let axis = adj_faces.iter()
            .map(|&fi| solid.faces[fi].normal)
            .sum::<Vec3>()
            .normalize_or_zero();

        // Build a stable reference frame perpendicular to axis
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

        // Determine corner chamfer mode: Hard wins over Smooth
        let corner_mode = {
            let mut mode = ChamferMode::Smooth;
            for &ek in &sharp_edge_keys {
                if let Some(info) = sharp_set.get(&ek) {
                    for &fi in &info.faces {
                        if solid.faces[fi].chamfer_mode == ChamferMode::Hard {
                            mode = ChamferMode::Hard;
                        }
                    }
                }
            }
            mode
        };

        // Triangulate the ring polygon (fan from ring[0])
        let ring_start = positions.len() as u32;
        for (rp, rn) in &ring {
            positions.push(rp.to_array());
            // Hard chamfer: flat averaged normal. Smooth/fillet: per-face normal.
            let n = match corner_mode {
                ChamferMode::Hard => axis.to_array(),
                ChamferMode::Smooth => rn.to_array(),
            };
            normals.push(n);
            uvs.push([0.0, 0.0]);
            chamfer_offsets.push([0.0; 3]);
        }

        // Collect ALL adjacent face normals (inner faces + chamfer strips) for overlap check
        let mut adjacent_normals: Vec<Vec3> = Vec::new();
        // Add normals from inner faces at this vertex
        for &fi in adj_faces {
            let n = solid.faces[fi].normal;
            if !adjacent_normals.iter().any(|an| an.dot(n).abs() > 0.99) {
                adjacent_normals.push(n);
            }
        }
        // Add normals from chamfer strips at this vertex
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

        // Check winding of first triangle against axis
        let tri_normal = (ring[1].0 - ring[0].0).cross(ring[2].0 - ring[0].0);
        let flip = tri_normal.dot(axis) < 0.0;

        for i in 1..ring.len() - 1 {
            // Skip fan triangles that are truly coplanar with an existing face
            // or chamfer strip (same normal AND all vertices on the same plane).
            let va = ring[0].0;
            let vb = ring[i].0;
            let vc = ring[i + 1].0;
            let fan_normal = (vb - va).cross(vc - va).normalize_or_zero();

            let mut skip = false;
            // Check against inner faces at this vertex
            for &fi in adj_faces {
                let face = &solid.faces[fi];
                let fn_normal = face.normal;
                if fan_normal.dot(fn_normal).abs() < 0.99 { continue; }
                // Same normal — check if triangle is on the face's plane
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
