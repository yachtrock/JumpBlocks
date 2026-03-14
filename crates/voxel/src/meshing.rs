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

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut chamfer_offsets: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Per-face: compute inner vertices, emit inner face + chamfer strips
    for (_fi, face) in solid.faces.iter().enumerate() {
        let n = face.verts.len();
        let normal = face.normal;
        let normal_arr = normal.to_array();

        // For each edge of this face, check if sharp + compute inward direction
        let mut edge_sharp: Vec<bool> = Vec::with_capacity(n);
        let mut edge_inward: Vec<Vec3> = Vec::with_capacity(n);

        for i in 0..n {
            let v0 = face.verts[i];
            let v1 = face.verts[(i + 1) % n];
            let key = edge_key(v0, v1);
            let sharp = sharp_set.contains_key(&key);
            edge_sharp.push(sharp);

            let p0 = solid.positions[v0 as usize];
            let p1 = solid.positions[v1 as usize];
            let edge_dir = (p1 - p0).normalize_or_zero();
            edge_inward.push(edge_dir.cross(normal).normalize_or_zero());
        }

        // Compute inner position for each vertex of this face, then clip
        // against adjacent faces connected by SMOOTH edges to prevent z-fighting.
        // (Faces connected by sharp edges are the chamfer target — don't clip those.)
        let inner: Vec<Vec3> = (0..n).map(|vi| {
            let mut offset = Vec3::ZERO;
            let prev = (vi + n - 1) % n;
            if edge_sharp[prev] {
                offset += edge_inward[prev] * CHAMFER_WIDTH;
            }
            if edge_sharp[vi] {
                offset += edge_inward[vi] * CHAMFER_WIDTH;
            }

            let outer_vi = face.verts[vi] as usize;
            let pos = solid.positions[outer_vi] + offset;


            pos
        }).collect();

        // --- Emit inner face ---
        let inner_base = positions.len() as u32;
        for i in 0..n {
            positions.push(inner[i].to_array());
            normals.push(normal_arr);
            uvs.push([0.0, 0.0]);
            chamfer_offsets.push([0.0; 3]);
        }
        // Triangulate (fan from vertex 0 — works for convex 3 or 4 verts)
        for i in 1..n - 1 {
            indices.push(inner_base + i as u32 + 1);
            indices.push(inner_base + i as u32);
            indices.push(inner_base);
        }

        // --- Emit chamfer strips for sharp edges ---
        for i in 0..n {
            if !edge_sharp[i] { continue; }

            let vi = i;
            let vj = (i + 1) % n;
            let outer_i = solid.positions[face.verts[vi] as usize];
            let outer_j = solid.positions[face.verts[vj] as usize];
            let inner_i = inner[vi];
            let inner_j = inner[vj];

            // Chamfer strip: outer_i, outer_j, inner_j, inner_i
            let base = positions.len() as u32;
            positions.extend_from_slice(&[
                outer_i.to_array(), outer_j.to_array(),
                inner_j.to_array(), inner_i.to_array(),
            ]);

            let edge_outward = -edge_inward[i];

            // Determine chamfer mode for this edge
            let key = edge_key(face.verts[vi], face.verts[vj]);
            let chamfer_mode = if let Some(info) = sharp_set.get(&key) {
                // Use smooth only if both faces are smooth
                if info.faces.len() >= 2 {
                    let m0 = solid.faces[info.faces[0]].chamfer_mode;
                    let m1 = solid.faces[info.faces[1]].chamfer_mode;
                    if m0 == ChamferMode::Smooth && m1 == ChamferMode::Smooth {
                        ChamferMode::Smooth
                    } else {
                        ChamferMode::Hard
                    }
                } else {
                    face.chamfer_mode
                }
            } else {
                face.chamfer_mode
            };

            match chamfer_mode {
                ChamferMode::Hard => {
                    let cn = (normal + edge_outward).normalize_or_zero().to_array();
                    normals.extend_from_slice(&[cn, cn, cn, cn]);
                }
                ChamferMode::Smooth => {
                    let outer_n = (normal + edge_outward).normalize_or_zero().to_array();
                    normals.extend_from_slice(&[outer_n, outer_n, normal_arr, normal_arr]);
                }
            }

            uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);

            // Chamfer offset: outer verts can be pushed inward by shader
            let off_i = (inner_i - outer_i).to_array();
            let off_j = (inner_j - outer_j).to_array();
            chamfer_offsets.extend_from_slice(&[off_i, off_j, [0.0; 3], [0.0; 3]]);

            indices.extend_from_slice(&[
                base + 2, base + 1, base,
                base + 3, base + 2, base,
            ]);
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
