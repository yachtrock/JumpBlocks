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

/// LOD mesh: simple per-face quads from shape definitions, no chamfering.
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

                    // Check neighbor occlusion
                    if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if data.is_neighbor_filled(nx, ny, nz) {
                            continue;
                        }
                    }

                    let base_index = positions.len() as u32;
                    let rotated_normal = facing.rotate_normal(face.normal);

                    for vert in &face.vertices {
                        let rotated = facing.rotate_point(*vert);
                        let world_pos = [
                            wx + rotated.x * VOXEL_WIDTH,
                            wy + rotated.y * VOXEL_HEIGHT,
                            wz + rotated.z * VOXEL_WIDTH,
                        ];
                        positions.push(world_pos);
                        normals.push(rotated_normal.to_array());
                    }

                    // Simple UVs based on vertex index
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
///
/// For each visible face, we generate:
/// - An inner face (inset by CHAMFER_WIDTH on chamfered edges)
/// - A chamfer strip for each exposed edge (connecting outer to inner vertices)
/// - Corner triangles where two chamfered edges meet
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

                    // Check neighbor occlusion for the whole face
                    if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if data.is_neighbor_filled(nx, ny, nz) {
                            continue;
                        }
                    }

                    emit_chamfered_face(
                        face,
                        facing,
                        data,
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
    let rotated_normal = facing.rotate_normal(face.normal);
    let normal_arr = rotated_normal.to_array();
    let n_verts = face.vertices.len();

    // Determine which edges are chamfered (neighbor on that side is empty)
    let edge_chamfered: Vec<bool> = face
        .edges
        .iter()
        .map(|edge| {
            let rotated_side = edge.neighbor_side.rotated_by(facing);
            if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
                let nx = vx as i32 + dx;
                let ny = vy as i32 + dy;
                let nz = vz as i32 + dz;
                !data.is_neighbor_filled(nx, ny, nz)
            } else {
                // FaceSide::None edges are always chamfered
                true
            }
        })
        .collect();

    // Compute world-space outer vertices (unchamfered positions)
    let outer_world: Vec<Vec3> = face
        .vertices
        .iter()
        .map(|v| {
            let rotated = facing.rotate_point(*v);
            Vec3::new(
                wx + rotated.x * VOXEL_WIDTH,
                wy + rotated.y * VOXEL_HEIGHT,
                wz + rotated.z * VOXEL_WIDTH,
            )
        })
        .collect();

    // Compute per-edge inward perpendicular directions (on the face plane).
    // For edge from v0 to v1: inward = cross(edge_dir, face_normal)
    let edge_inwards: Vec<Vec3> = face
        .edges
        .iter()
        .map(|edge| {
            let edge_dir = (outer_world[edge.v1] - outer_world[edge.v0]).normalize_or_zero();
            edge_dir.cross(rotated_normal).normalize_or_zero()
        })
        .collect();

    // Compute inner vertices (inset perpendicular to chamfered edges).
    // For each vertex, accumulate inward offsets from all chamfered edges it belongs to.
    let inner_world: Vec<Vec3> = (0..n_verts)
        .map(|vi| {
            let mut offset = Vec3::ZERO;

            for (ei, edge) in face.edges.iter().enumerate() {
                if !edge_chamfered[ei] {
                    continue;
                }
                if edge.v0 == vi || edge.v1 == vi {
                    offset += edge_inwards[ei] * CHAMFER_WIDTH;
                }
            }

            outer_world[vi] + offset
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

        // Outer vertices at unchamfered positions (with chamfer_offset to push inward)
        // Inner vertices already placed above
        let ov0 = outer_world[edge.v0].to_array();
        let ov1 = outer_world[edge.v1].to_array();
        let iv0 = inner_world[edge.v0].to_array();
        let iv1 = inner_world[edge.v1].to_array();

        // Chamfer strip normal: average of face normal and the edge's outward direction
        let edge_outward = {
            let rotated_side = edge.neighbor_side.rotated_by(facing);
            if let Some((dx, dy, dz)) = rotated_side.neighbor_offset() {
                Vec3::new(dx as f32, dy as f32, dz as f32).normalize()
            } else {
                rotated_normal
            }
        };
        let chamfer_normal = (rotated_normal + edge_outward).normalize_or_zero();
        let cn = chamfer_normal.to_array();

        let base = positions.len() as u32;

        // 4 vertices: outer0, outer1, inner1, inner0
        positions.extend_from_slice(&[ov0, ov1, iv1, iv0]);
        normals.extend_from_slice(&[cn, cn, cn, cn]);
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
