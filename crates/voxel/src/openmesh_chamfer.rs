//! Chamfer implementation using the OpenMesh C++ library via FFI.
//!
//! Strategy:
//! 1. Build a solid mesh with shared vertices (reusing build_solid_mesh_public)
//! 2. Feed polygon faces (quads/tris as-is) into OpenMesh PolyMesh — no triangulation
//! 3. Let OpenMesh chamfer sharp edges based on dihedral angle threshold
//! 4. Export back to ChunkMeshData

use crate::chunk::*;
use crate::meshing::{ChunkMeshData, build_solid_mesh_public};
use crate::shape::*;

/// Dihedral angle threshold in degrees: edges with angle above this are "sharp".
/// 10 degrees ≈ cos(170°) ≈ dot product threshold of ~0.985
const SHARP_ANGLE_DEG: f32 = 10.0;

/// Generate a chamfered mesh using OpenMesh's PolyMesh.
pub fn generate_openmesh_chamfer(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
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

    // Build the OpenMesh PolyMesh from our solid mesh
    let mut mesh = jumpblocks_openmesh::PolyMesh::new();

    // Add all vertices
    let vert_handles: Vec<jumpblocks_openmesh::VertexHandle> = solid
        .positions
        .iter()
        .map(|p| mesh.add_vertex(p.x, p.y, p.z))
        .collect();

    // Add faces as polygons (quads stay quads, tris stay tris — no pre-triangulation)
    for face in &solid.faces {
        let face_verts: Vec<jumpblocks_openmesh::VertexHandle> =
            face.verts.iter().map(|&vi| vert_handles[vi as usize]).collect();
        mesh.add_face(&face_verts);
    }

    // Chamfer sharp edges
    mesh.chamfer(SHARP_ANGLE_DEG, CHAMFER_WIDTH);

    // Export the chamfered mesh
    let exported = mesh.export();

    let n = exported.positions.len();
    let uvs: Vec<[f32; 2]> = (0..n).map(|i| {
        let idx = i % 4;
        match idx {
            0 => [0.0, 0.0],
            1 => [1.0, 0.0],
            2 => [1.0, 1.0],
            _ => [0.0, 1.0],
        }
    }).collect();

    ChunkMeshData {
        positions: exported.positions,
        normals: exported.normals,
        uvs,
        chamfer_offsets: vec![[0.0; 3]; n],
        indices: exported.indices,
    }
}
