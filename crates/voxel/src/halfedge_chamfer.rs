//! Chamfer implementation using the procedural_modelling crate's half-edge mesh.
//!
//! Strategy:
//! 1. Build a half-edge mesh from the solid mesh (shared vertices + faces)
//! 2. Identify sharp edges (dihedral angle > threshold)
//! 3. For each sharp edge, split it: inset both adjacent faces and insert
//!    a chamfer face between them
//! 4. Export back to ChunkMeshData

use bevy::prelude::*;
use procedural_modelling::prelude::*;

use crate::chunk::*;
use crate::meshing::{ChunkMeshData, build_solid_mesh_public};
use crate::shape::*;

/// Dihedral angle threshold: edges with face normal dot below this are "sharp".
const SHARP_DOT: f32 = 0.985;

/// Generate a chamfered mesh using half-edge operations.
pub fn generate_halfedge_chamfer(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
    // Step 1: build our solid mesh
    let solid = build_solid_mesh_public(data, shapes);

    if solid.positions.is_empty() {
        return ChunkMeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            chamfer_offsets: Vec::new(),
            indices: Vec::new(),
        };
    }

    // Step 2: build a half-edge mesh from the solid mesh
    // For now, just emit the solid mesh directly as a starting point.
    // TODO: convert to half-edge, chamfer sharp edges, convert back.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for face in &solid.faces {
        let n = face.verts.len();
        let normal_arr = face.normal.to_array();
        let base = positions.len() as u32;

        for &vi in &face.verts {
            positions.push(solid.positions[vi as usize].to_array());
            normals.push(normal_arr);
        }
        let uv_map = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for i in 0..n { uvs.push(uv_map[i % 4]); }

        for i in 1..n - 1 {
            indices.push(base + i as u32 + 1);
            indices.push(base + i as u32);
            indices.push(base);
        }
    }

    let n = positions.len();
    ChunkMeshData {
        positions,
        normals,
        uvs,
        chamfer_offsets: vec![[0.0; 3]; n],
        indices,
    }
}
