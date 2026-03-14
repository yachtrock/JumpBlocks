use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};

use crate::chunk::*;
use crate::shape::*;

/// Custom vertex attribute for chamfer offset (direction to push vertex for chamfering).
/// The vertex shader applies: position + chamfer_offset * chamfer_amount
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

/// Transform a normalized-space point to world-local space within a voxel.
#[inline]
fn to_world(v: Vec3, wx: f32, wy: f32, wz: f32) -> Vec3 {
    Vec3::new(
        wx + v.x * VOXEL_WIDTH,
        wy + v.y * VOXEL_HEIGHT,
        wz + v.z * VOXEL_WIDTH,
    )
}

/// Compute the face normal from world-space vertices using the first triangle.
/// The triangle winding in shape definitions is interior-facing, so we negate.
fn compute_world_normal(world_verts: &[Vec3], triangles: &[[usize; 3]]) -> Vec3 {
    let tri = &triangles[0];
    let a = world_verts[tri[0]];
    let b = world_verts[tri[1]];
    let c = world_verts[tri[2]];
    // Negate because shape triangles are defined with inward winding
    // (they get reversed during index emission for correct rendering)
    -(b - a).cross(c - a).normalize_or_zero()
}

/// Check if the neighbor at (nx, ny, nz) fully occludes the given world-side
/// (the side facing back toward the voxel we're rendering).
fn neighbor_occludes(
    data: &ChunkData,
    shapes: &ShapeTable,
    nx: i32,
    ny: i32,
    nz: i32,
    side_facing_us: FaceSide,
) -> bool {
    let neighbor = data.get_signed(nx, ny, nz);
    if neighbor.is_empty() {
        return false;
    }
    let Some(neighbor_shape) = shapes.get(neighbor.shape_index()) else {
        return false;
    };
    neighbor_shape
        .occlusion
        .occludes_world_side(side_facing_us, neighbor.facing())
}

/// Check if a world-space vertex is shared with a neighbor voxel's geometry.
/// Looks at all face vertices of the neighbor shape to find a match.
fn vertex_shared_with_neighbor(
    vertex_world: Vec3,
    data: &ChunkData,
    shapes: &ShapeTable,
    nx: i32,
    ny: i32,
    nz: i32,
) -> bool {
    let neighbor = data.get_signed(nx, ny, nz);
    if neighbor.is_empty() {
        return false;
    }
    let Some(shape) = shapes.get(neighbor.shape_index()) else {
        return false;
    };
    let nfacing = neighbor.facing();
    let nwx = nx as f32 * VOXEL_WIDTH;
    let nwy = ny as f32 * VOXEL_HEIGHT;
    let nwz = nz as f32 * VOXEL_WIDTH;

    for face in &shape.faces {
        for v in &face.vertices {
            let world_v = to_world(nfacing.rotate_point(*v), nwx, nwy, nwz);
            if (world_v - vertex_world).length_squared() < 0.0001 {
                return true;
            }
        }
    }
    false
}

/// Check if a vertex is snapped to any neighbor along the edge's neighbor sides.
fn vertex_snapped_for_edge(
    vertex_world: Vec3,
    edge: &VoxelEdge,
    facing: Facing,
    data: &ChunkData,
    shapes: &ShapeTable,
    vx: i32,
    vy: i32,
    vz: i32,
) -> bool {
    for side in &edge.neighbor_sides {
        let rotated_side = side.rotated_by(facing);
        if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
            if vertex_shared_with_neighbor(
                vertex_world,
                data,
                shapes,
                vx + dx,
                vy + dy,
                vz + dz,
            ) {
                return true;
            }
        }
    }
    false
}

/// The side of a neighbor that faces back toward us.
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

/// Generate both full-res (chamfered) and LOD (simple) meshes plus collider data.
pub fn generate_chunk_mesh(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshResult {
    let full_res = generate_full_res_mesh(data, shapes);
    let lod = generate_lod_mesh(data, shapes);

    // Collider uses the LOD mesh (no chamfer, simpler geometry)
    let collider_vertices: Vec<Vec3> = lod
        .positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let collider_indices: Vec<[u32; 3]> = lod
        .indices
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    ChunkMeshResult {
        full_res,
        lod,
        collider_data: ChunkColliderData {
            vertices: collider_vertices,
            indices: collider_indices,
        },
    }
}

/// LOD mesh: simple per-face geometry from shape definitions, no chamfering.
fn generate_lod_mesh(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for y in 0..CHUNK_Y {
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let voxel = data.get(x, y, z);
                if voxel.is_empty() {
                    continue;
                }

                let Some(shape) = shapes.get(voxel.shape_index()) else {
                    continue;
                };
                let facing = voxel.facing();

                let wx = x as f32 * VOXEL_WIDTH;
                let wy = y as f32 * VOXEL_HEIGHT;
                let wz = z as f32 * VOXEL_WIDTH;

                for face in &shape.faces {
                    let rotated_side = face.side.rotated_by(facing);

                    if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if neighbor_occludes(data, shapes, nx, ny, nz, opposite_side(rotated_side)) {
                            continue;
                        }
                    }

                    let base_index = positions.len() as u32;

                    // Transform vertices to world space
                    let world_verts: Vec<Vec3> = face
                        .vertices
                        .iter()
                        .map(|v| to_world(facing.rotate_point(*v), wx, wy, wz))
                        .collect();

                    // Compute normal from world-space geometry
                    let world_normal = compute_world_normal(&world_verts, &face.triangles);

                    for wv in &world_verts {
                        positions.push(wv.to_array());
                        normals.push(world_normal.to_array());
                    }

                    let uv_map = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                    for i in 0..face.vertices.len() {
                        uvs.push(uv_map[i % 4]);
                    }

                    for tri in &face.triangles {
                        indices.push(base_index + tri[2] as u32);
                        indices.push(base_index + tri[1] as u32);
                        indices.push(base_index + tri[0] as u32);
                    }
                }
            }
        }
    }

    let n_positions = positions.len();
    ChunkMeshData {
        positions,
        normals,
        uvs,
        chamfer_offsets: vec![[0.0; 3]; n_positions],
        indices,
    }
}

/// Full-res mesh: faces with chamfer border geometry on exposed edges.
fn generate_full_res_mesh(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut chamfer_offsets = Vec::new();
    let mut indices = Vec::new();

    for y in 0..CHUNK_Y {
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let voxel = data.get(x, y, z);
                if voxel.is_empty() {
                    continue;
                }

                let Some(shape) = shapes.get(voxel.shape_index()) else {
                    continue;
                };
                let facing = voxel.facing();

                let wx = x as f32 * VOXEL_WIDTH;
                let wy = y as f32 * VOXEL_HEIGHT;
                let wz = z as f32 * VOXEL_WIDTH;

                for face in &shape.faces {
                    let rotated_side = face.side.rotated_by(facing);

                    if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if neighbor_occludes(data, shapes, nx, ny, nz, opposite_side(rotated_side)) {
                            continue;
                        }
                    }

                    emit_chamfered_face(
                        face,
                        facing,
                        data,
                        shapes,
                        x, y, z,
                        wx, wy, wz,
                        &mut positions,
                        &mut normals,
                        &mut uvs,
                        &mut chamfer_offsets,
                        &mut indices,
                    );
                }
            }
        }
    }

    ChunkMeshData {
        positions,
        normals,
        uvs,
        chamfer_offsets,
        indices,
    }
}

/// Emit geometry for a single face with chamfer borders.
fn emit_chamfered_face(
    face: &VoxelFace,
    facing: Facing,
    data: &ChunkData,
    shapes: &ShapeTable,
    vx: usize,
    vy: usize,
    vz: usize,
    wx: f32,
    wy: f32,
    wz: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    chamfer_offsets: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let n_verts = face.vertices.len();

    // Compute world-space outer vertices (unchamfered positions)
    let outer_world: Vec<Vec3> = face
        .vertices
        .iter()
        .map(|v| to_world(facing.rotate_point(*v), wx, wy, wz))
        .collect();

    // Compute face normal from world-space geometry
    let world_normal = compute_world_normal(&outer_world, &face.triangles);
    let normal_arr = world_normal.to_array();

    let ivx = vx as i32;
    let ivy = vy as i32;
    let ivz = vz as i32;

    // Determine which edges need chamfering (have no filled neighbor on any side).
    // Empty neighbor_sides means always chamfer (internal/diagonal edges).
    let edge_chamfered: Vec<bool> = face
        .edges
        .iter()
        .map(|edge| {
            if edge.neighbor_sides.is_empty() {
                return true;
            }
            for side in &edge.neighbor_sides {
                let rotated_side = side.rotated_by(facing);
                if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
                    if data.is_neighbor_filled(ivx + dx, ivy + dy, ivz + dz) {
                        return false;
                    }
                }
            }
            true
        })
        .collect();

    // Compute per-edge inward perpendicular directions (on the face plane).
    let edge_inwards: Vec<Vec3> = face
        .edges
        .iter()
        .map(|edge| {
            let edge_dir = (outer_world[edge.v1] - outer_world[edge.v0]).normalize_or_zero();
            edge_dir.cross(world_normal).normalize_or_zero()
        })
        .collect();

    // Per-edge, per-vertex snap detection: check if the neighbor's geometry
    // shares this vertex. [v0_snapped, v1_snapped] per edge.
    let edge_snap: Vec<[bool; 2]> = face
        .edges
        .iter()
        .enumerate()
        .map(|(ei, edge)| {
            if !edge_chamfered[ei] {
                return [false, false]; // not chamfered, doesn't matter
            }
            [
                vertex_snapped_for_edge(outer_world[edge.v0], edge, facing, data, shapes, ivx, ivy, ivz),
                vertex_snapped_for_edge(outer_world[edge.v1], edge, facing, data, shapes, ivx, ivy, ivz),
            ]
        })
        .collect();

    // Compute inner vertices with per-vertex chamfer width.
    // A vertex snapped for a particular edge gets 0 inset from that edge.
    let inner_world: Vec<Vec3> = (0..n_verts)
        .map(|vi| {
            let mut offset = Vec3::ZERO;

            for (ei, edge) in face.edges.iter().enumerate() {
                if !edge_chamfered[ei] {
                    continue;
                }
                if edge.v0 == vi {
                    let width = if edge_snap[ei][0] { 0.0 } else { CHAMFER_WIDTH };
                    offset += edge_inwards[ei] * width;
                } else if edge.v1 == vi {
                    let width = if edge_snap[ei][1] { 0.0 } else { CHAMFER_WIDTH };
                    offset += edge_inwards[ei] * width;
                }
            }

            // Clamp the inner vertex to the voxel's world-space bounds
            // so chamfer geometry never pokes past voxel boundaries.
            let inner = outer_world[vi] + offset;
            let min_bound = Vec3::new(wx, wy, wz);
            let max_bound = Vec3::new(wx + VOXEL_WIDTH, wy + VOXEL_HEIGHT, wz + VOXEL_WIDTH);
            inner.clamp(min_bound, max_bound)
        })
        .collect();

    // Chamfer offset = inner - outer (the displacement the vertex shader will apply)
    let offsets: Vec<[f32; 3]> = (0..n_verts)
        .map(|i| (inner_world[i] - outer_world[i]).to_array())
        .collect();

    // --- Emit the inner face ---
    let inner_base = positions.len() as u32;
    for i in 0..n_verts {
        positions.push(inner_world[i].to_array());
        normals.push(normal_arr);
        uvs.push([0.0, 0.0]); // TODO: proper UVs
        chamfer_offsets.push([0.0; 3]); // Inner verts are already at chamfered position
    }
    for tri in &face.triangles {
        indices.push(inner_base + tri[2] as u32);
        indices.push(inner_base + tri[1] as u32);
        indices.push(inner_base + tri[0] as u32);
    }

    // --- Emit chamfer strips for each chamfered edge ---
    for (ei, edge) in face.edges.iter().enumerate() {
        if !edge_chamfered[ei] {
            continue;
        }

        let ov0 = outer_world[edge.v0].to_array();
        let ov1 = outer_world[edge.v1].to_array();
        let iv0 = inner_world[edge.v0].to_array();
        let iv1 = inner_world[edge.v1].to_array();

        // Compute edge outward direction for normals
        let edge_outward = -edge_inwards[ei];

        let base = positions.len() as u32;

        // 4 vertices: outer0, outer1, inner1, inner0
        positions.extend_from_slice(&[ov0, ov1, iv1, iv0]);

        match face.chamfer_mode {
            ChamferMode::Hard => {
                // Flat chamfer normal (average of face + outward)
                let cn = (world_normal + edge_outward).normalize_or_zero().to_array();
                normals.extend_from_slice(&[cn, cn, cn, cn]);
            }
            ChamferMode::Smooth => {
                // Outer verts get averaged normal (matches adjacent face's chamfer strip),
                // inner verts get face normal — GPU interpolates for a rounded look.
                let outer_n = (world_normal + edge_outward).normalize_or_zero().to_array();
                let inner_n = normal_arr;
                normals.extend_from_slice(&[outer_n, outer_n, inner_n, inner_n]);
            }
        }
        uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);

        // Outer verts have chamfer offset (shader pushes them inward at close range)
        chamfer_offsets.push(offsets[edge.v0]);
        chamfer_offsets.push(offsets[edge.v1]);
        // Inner verts have zero offset (already inset)
        chamfer_offsets.push([0.0; 3]);
        chamfer_offsets.push([0.0; 3]);

        // Two triangles
        indices.extend_from_slice(&[
            base + 2, base + 1, base,
            base + 3, base + 2, base,
        ]);
    }
}

/// Build a Bevy `Mesh` from generated mesh data (full-res version with chamfer attribute).
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

/// Build a Bevy `Mesh` from LOD mesh data (no chamfer attribute).
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
