use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};

use crate::chunk::*;

pub struct ChunkMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

pub struct ChunkColliderData {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
}

pub struct ChunkMeshResult {
    pub mesh_data: ChunkMeshData,
    pub collider_data: ChunkColliderData,
}

#[derive(Clone, Copy)]
enum Face {
    Top,    // +Y
    Bottom, // -Y
    North,  // +Z
    South,  // -Z
    East,   // +X
    West,   // -X
}

impl Face {
    fn normal(&self) -> [f32; 3] {
        match self {
            Face::Top => [0.0, 1.0, 0.0],
            Face::Bottom => [0.0, -1.0, 0.0],
            Face::North => [0.0, 0.0, 1.0],
            Face::South => [0.0, 0.0, -1.0],
            Face::East => [1.0, 0.0, 0.0],
            Face::West => [-1.0, 0.0, 0.0],
        }
    }

    fn neighbor_offset(&self) -> (i32, i32, i32) {
        match self {
            Face::Top => (0, 1, 0),
            Face::Bottom => (0, -1, 0),
            Face::North => (0, 0, 1),
            Face::South => (0, 0, -1),
            Face::East => (1, 0, 0),
            Face::West => (-1, 0, 0),
        }
    }
}

const FACES: [Face; 6] = [
    Face::Top,
    Face::Bottom,
    Face::North,
    Face::South,
    Face::East,
    Face::West,
];

/// Generate mesh and collider data for a chunk. This is a pure function
/// suitable for running on a background thread.
pub fn generate_chunk_mesh(data: &ChunkData) -> ChunkMeshResult {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for y in 0..CHUNK_Y {
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                if !data.is_filled(x, y, z) {
                    continue;
                }

                // World-local position of this voxel's minimum corner
                let wx = x as f32 * VOXEL_WIDTH;
                let wy = y as f32 * VOXEL_HEIGHT;
                let wz = z as f32 * VOXEL_WIDTH;

                for face in &FACES {
                    let (dx, dy, dz) = face.neighbor_offset();
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;

                    // Skip face if neighbor is solid (hidden face)
                    if data.is_neighbor_filled(nx, ny, nz) {
                        continue;
                    }

                    let base_index = positions.len() as u32;
                    let normal = face.normal();
                    let quad = face_vertices(wx, wy, wz, face);

                    for pos in &quad {
                        positions.push(*pos);
                        normals.push(normal);
                    }
                    uvs.extend_from_slice(&[
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                    ]);

                    // Two triangles per quad (CCW winding for front face)
                    indices.extend_from_slice(&[
                        base_index + 2,
                        base_index + 1,
                        base_index,
                        base_index + 3,
                        base_index + 2,
                        base_index,
                    ]);
                }
            }
        }
    }

    let collider_vertices: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let collider_indices: Vec<[u32; 3]> = indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    ChunkMeshResult {
        mesh_data: ChunkMeshData {
            positions,
            normals,
            uvs,
            indices,
        },
        collider_data: ChunkColliderData {
            vertices: collider_vertices,
            indices: collider_indices,
        },
    }
}

/// Returns the 4 vertices for a face quad, wound CCW when viewed from outside.
fn face_vertices(x: f32, y: f32, z: f32, face: &Face) -> [[f32; 3]; 4] {
    let w = VOXEL_WIDTH;
    let h = VOXEL_HEIGHT;

    match face {
        Face::Top => [
            [x, y + h, z],
            [x + w, y + h, z],
            [x + w, y + h, z + w],
            [x, y + h, z + w],
        ],
        Face::Bottom => [
            [x, y, z + w],
            [x + w, y, z + w],
            [x + w, y, z],
            [x, y, z],
        ],
        Face::North => [
            [x + w, y, z + w],
            [x, y, z + w],
            [x, y + h, z + w],
            [x + w, y + h, z + w],
        ],
        Face::South => [
            [x, y, z],
            [x + w, y, z],
            [x + w, y + h, z],
            [x, y + h, z],
        ],
        Face::East => [
            [x + w, y, z],
            [x + w, y, z + w],
            [x + w, y + h, z + w],
            [x + w, y + h, z],
        ],
        Face::West => [
            [x, y, z + w],
            [x, y, z],
            [x, y + h, z],
            [x, y + h, z + w],
        ],
    }
}

/// Build a Bevy `Mesh` from generated mesh data.
pub fn build_mesh(data: &ChunkMeshData) -> Mesh {
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
