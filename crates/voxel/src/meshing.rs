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

pub fn generate_chunk_mesh(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshResult {
    let t0 = Instant::now();
    let full_res = generate_chamfered_mesh(data, shapes);
    let t1 = Instant::now();
    let lod = generate_lod_mesh(data, shapes);
    let t2 = Instant::now();

    let full_ms = (t1 - t0).as_secs_f64() * 1000.0;
    let lod_ms = (t2 - t1).as_secs_f64() * 1000.0;
    info!(
        "Chunk meshed: full={:.2}ms, lod={:.2}ms, total={:.2}ms (full: {} verts/{} tris, lod: {} verts/{} tris)",
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

struct SolidFace {
    /// Indices into SolidMesh.positions (3 or 4 verts).
    verts: Vec<u32>,
    normal: Vec3,
    chamfer_mode: ChamferMode,
    /// Source voxel position in chunk coords — used to avoid clipping against own faces.
    voxel: (usize, usize, usize),
}

struct SolidMesh {
    positions: Vec<Vec3>,
    faces: Vec<SolidFace>,
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

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut chamfer_offsets: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // ---------------------------------------------------------------
    // Pass 2: emit inner faces
    // ---------------------------------------------------------------
    for (fi, face) in solid.faces.iter().enumerate() {
        let n = face.verts.len();
        let normal_arr = face.normal.to_array();
        let inner = &all_inner[fi];

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
    }

    // ---------------------------------------------------------------
    // Pass 3: emit chamfer strips per-edge (not per-face)
    // ---------------------------------------------------------------
    // Helper: find a face's inner vertex index for a given shared vertex
    fn find_face_inner_at_vert(face: &SolidFace, inner: &[Vec3], vert: u32) -> Option<Vec3> {
        face.verts.iter().position(|&v| v == vert).map(|i| inner[i])
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

            // Check winding: the strip should face outward (along average of face normals)
            let na = face_a.normal;
            let nb = face_b.normal;
            let expected_out = (na + nb).normalize_or_zero();
            let tri_normal = (a1 - a0).cross(b0 - a0);
            let flipped = tri_normal.dot(expected_out) > 0.0;

            let (p0, p1, p2, p3) = if flipped {
                (b0, b1, a1, a0)
            } else {
                (a0, a1, b1, b0)
            };
            // After potential flip, n_first/n_second track which face normal
            // goes with p0,p1 vs p2,p3
            let (n_first, n_second) = if flipped { (nb, na) } else { (na, nb) };

            let base = positions.len() as u32;
            positions.extend_from_slice(&[
                p0.to_array(), p1.to_array(),
                p2.to_array(), p3.to_array(),
            ]);

            match chamfer_mode {
                ChamferMode::Hard => {
                    let cn = expected_out.to_array();
                    normals.extend_from_slice(&[cn, cn, cn, cn]);
                }
                ChamferMode::Smooth => {
                    let nf = n_first.to_array();
                    let ns = n_second.to_array();
                    normals.extend_from_slice(&[nf, nf, ns, ns]);
                }
            }

            uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
            chamfer_offsets.extend_from_slice(&[[0.0; 3]; 4]);

            indices.extend_from_slice(&[
                base + 2, base + 1, base,
                base + 3, base + 2, base,
            ]);
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

            // Check winding
            let tri_normal = (a1 - a0).cross(outer0 - a0);
            let expected_out = (na + outward).normalize_or_zero();
            let flipped = tri_normal.dot(expected_out) > 0.0;

            let (p0, p1, p2, p3) = if flipped {
                (outer0, outer1, a1, a0)
            } else {
                (a0, a1, outer1, outer0)
            };

            let base = positions.len() as u32;
            positions.extend_from_slice(&[
                p0.to_array(), p1.to_array(),
                p2.to_array(), p3.to_array(),
            ]);

            match chamfer_mode {
                ChamferMode::Hard => {
                    let cn = expected_out.to_array();
                    normals.extend_from_slice(&[cn, cn, cn, cn]);
                }
                ChamferMode::Smooth => {
                    let (n_inner, n_outer) = if flipped {
                        (outward.to_array(), na.to_array())
                    } else {
                        (na.to_array(), outward.to_array())
                    };
                    normals.extend_from_slice(&[n_inner, n_inner, n_outer, n_outer]);
                }
            }

            uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
            chamfer_offsets.extend_from_slice(&[[0.0; 3]; 4]);

            indices.extend_from_slice(&[
                base + 2, base + 1, base,
                base + 3, base + 2, base,
            ]);
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

        // Check winding of first triangle against axis
        let tri_normal = (ring[1].0 - ring[0].0).cross(ring[2].0 - ring[0].0);
        let flip = tri_normal.dot(axis) < 0.0;

        for i in 1..ring.len() - 1 {
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
