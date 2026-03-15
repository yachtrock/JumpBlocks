//! Half-edge mesh chamfer implementation.
//!
//! A minimal half-edge data structure + chamfer algorithm that operates
//! on the solid mesh. No external dependencies.
//!
//! Strategy:
//! 1. Build a half-edge mesh from the solid mesh
//! 2. Identify sharp edges (dihedral angle threshold)
//! 3. For each sharp edge, inset both adjacent faces and insert a bevel face
//! 4. Export to ChunkMeshData

use std::collections::HashMap;
use bevy::prelude::*;

use crate::chunk::*;
use crate::meshing::{ChunkMeshData, build_solid_mesh_public};
use crate::shape::*;

// ---------------------------------------------------------------------------
// Minimal half-edge data structure
// ---------------------------------------------------------------------------

type VertId = u32;
type HalfEdgeId = u32;
type FaceId = u32;

#[derive(Clone, Debug)]
struct HEVertex {
    pos: Vec3,
    edge: HalfEdgeId, // one outgoing half-edge
}

#[derive(Clone, Debug)]
struct HEHalfEdge {
    origin: VertId,
    twin: HalfEdgeId,
    next: HalfEdgeId,
    prev: HalfEdgeId,
    face: Option<FaceId>, // None = boundary
}

#[derive(Clone, Debug)]
struct HEFace {
    edge: HalfEdgeId, // one half-edge on this face
    normal: Vec3,
    chamfer_mode: ChamferMode,
}

#[derive(Clone, Debug)]
struct HEMesh {
    verts: Vec<HEVertex>,
    edges: Vec<HEHalfEdge>,
    faces: Vec<HEFace>,
}

const SHARP_DOT: f32 = 0.985;

impl HEMesh {
    fn new() -> Self {
        Self { verts: Vec::new(), edges: Vec::new(), faces: Vec::new() }
    }

    fn add_vert(&mut self, pos: Vec3) -> VertId {
        let id = self.verts.len() as VertId;
        self.verts.push(HEVertex { pos, edge: u32::MAX });
        id
    }

    fn edge_origin_pos(&self, e: HalfEdgeId) -> Vec3 {
        self.verts[self.edges[e as usize].origin as usize].pos
    }

    fn edge_target(&self, e: HalfEdgeId) -> VertId {
        self.edges[self.edges[e as usize].next as usize].origin
    }

    fn edge_target_pos(&self, e: HalfEdgeId) -> Vec3 {
        self.verts[self.edge_target(e) as usize].pos
    }

    /// Build from our solid mesh. Returns the HEMesh.
    fn from_solid(solid: &crate::meshing::SolidMesh) -> Self {
        let mut mesh = HEMesh::new();

        // Add vertices
        for &pos in &solid.positions {
            mesh.add_vert(pos);
        }

        // For shared edge matching: (min_v, max_v) → half-edge id of first half
        let mut edge_twins: HashMap<(VertId, VertId), HalfEdgeId> = HashMap::new();

        for solid_face in &solid.faces {
            let n = solid_face.verts.len();
            if n < 3 { continue; }

            let face_id = mesh.faces.len() as FaceId;
            mesh.faces.push(HEFace {
                edge: mesh.edges.len() as HalfEdgeId,
                normal: solid_face.normal,
                chamfer_mode: solid_face.chamfer_mode,
            });

            let first_edge = mesh.edges.len() as HalfEdgeId;

            // Create half-edges for this face
            for i in 0..n {
                let origin = solid_face.verts[i];
                let he_id = mesh.edges.len() as HalfEdgeId;

                mesh.edges.push(HEHalfEdge {
                    origin,
                    twin: u32::MAX, // set below
                    next: first_edge + ((i + 1) % n) as u32,
                    prev: first_edge + ((i + n - 1) % n) as u32,
                    face: Some(face_id),
                });

                // Update vertex's outgoing edge
                if mesh.verts[origin as usize].edge == u32::MAX {
                    mesh.verts[origin as usize].edge = he_id;
                }

                // Try to find twin
                let target = solid_face.verts[(i + 1) % n];
                let key = if origin < target { (origin, target) } else { (target, origin) };

                if let Some(&twin_id) = edge_twins.get(&key) {
                    mesh.edges[he_id as usize].twin = twin_id;
                    mesh.edges[twin_id as usize].twin = he_id;
                } else {
                    edge_twins.insert(key, he_id);
                }
            }
        }

        // Create boundary half-edges for unpaired edges
        let edge_count = mesh.edges.len();
        for i in 0..edge_count {
            if mesh.edges[i].twin == u32::MAX {
                // Create a boundary twin
                let origin = mesh.edge_target(i as HalfEdgeId);
                let twin_id = mesh.edges.len() as HalfEdgeId;
                mesh.edges.push(HEHalfEdge {
                    origin,
                    twin: i as HalfEdgeId,
                    next: u32::MAX, // link boundary edges later
                    prev: u32::MAX,
                    face: None,
                });
                mesh.edges[i].twin = twin_id;
            }
        }

        // TODO: link boundary edge next/prev chains

        mesh
    }

    /// Check if a half-edge is sharp (the two adjacent faces have different normals).
    fn is_edge_sharp(&self, e: HalfEdgeId) -> bool {
        let he = &self.edges[e as usize];
        let twin = &self.edges[he.twin as usize];

        match (he.face, twin.face) {
            (Some(f1), Some(f2)) => {
                let n1 = self.faces[f1 as usize].normal;
                let n2 = self.faces[f2 as usize].normal;
                n1.dot(n2) < SHARP_DOT
            }
            _ => true, // boundary = sharp
        }
    }

    /// Export to ChunkMeshData.
    fn to_chunk_mesh_data(&self) -> ChunkMeshData {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        for face in &self.faces {
            let normal_arr = face.normal.to_array();
            let base = positions.len() as u32;

            // Collect face vertices by following the half-edge loop
            let mut verts = Vec::new();
            let mut e = face.edge;
            loop {
                verts.push(self.edges[e as usize].origin);
                e = self.edges[e as usize].next;
                if e == face.edge { break; }
                if verts.len() > 20 { break; } // safety
            }

            for &vi in &verts {
                positions.push(self.verts[vi as usize].pos.to_array());
                normals.push(normal_arr);
            }
            let uv_map = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            for i in 0..verts.len() { uvs.push(uv_map[i % 4]); }

            // Fan triangulation
            for i in 1..verts.len() - 1 {
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
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Generate a chamfered mesh using the half-edge data structure.
/// Currently builds the half-edge mesh and exports it directly.
/// TODO: implement chamfer operations on sharp edges.
pub fn generate_halfedge_chamfer(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
    let solid = build_solid_mesh_public(data, shapes);

    if solid.positions.is_empty() || solid.faces.is_empty() {
        return ChunkMeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            chamfer_offsets: Vec::new(),
            indices: Vec::new(),
        };
    }

    let he_mesh = HEMesh::from_solid(&solid);

    // TODO: iterate sharp edges and chamfer them
    // For now, just export the solid mesh through the half-edge pipeline
    // to verify the data structure is correct.

    he_mesh.to_chunk_mesh_data()
}
