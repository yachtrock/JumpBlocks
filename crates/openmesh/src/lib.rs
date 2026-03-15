//! Safe Rust wrapper around OpenMesh's PolyMesh via the openmesh-sys FFI crate.

use openmesh_sys::*;

/// Opaque handle to a vertex in the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexHandle(pub u32);

/// A polygonal mesh backed by OpenMesh's PolyMesh.
///
/// Supports arbitrary polygon faces (not just triangles), which is important
/// for chamfering — we preserve quads through the pipeline rather than
/// pre-triangulating.
pub struct PolyMesh {
    handle: OmeshHandle,
}

// SAFETY: OpenMesh PolyMesh instances are self-contained and don't use
// thread-local state, so they can be sent between threads.
unsafe impl Send for PolyMesh {}

impl Drop for PolyMesh {
    fn drop(&mut self) {
        unsafe { omesh_free(self.handle) };
    }
}

/// Exported mesh data with per-face-vertex positions and normals.
pub struct ExportedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl PolyMesh {
    /// Create a new empty polygonal mesh.
    pub fn new() -> Self {
        Self {
            handle: unsafe { omesh_new() },
        }
    }

    /// Add a vertex at the given position. Returns its handle.
    pub fn add_vertex(&mut self, x: f32, y: f32, z: f32) -> VertexHandle {
        VertexHandle(unsafe { omesh_add_vertex(self.handle, x, y, z) })
    }

    /// Add a polygon face from vertex handles.
    /// Works with triangles, quads, or any polygon — no forced triangulation.
    /// Returns true if the face was added successfully.
    pub fn add_face(&mut self, verts: &[VertexHandle]) -> bool {
        let indices: Vec<u32> = verts.iter().map(|v| v.0).collect();
        let result = unsafe { omesh_add_face(self.handle, indices.as_ptr(), indices.len() as u32) };
        result >= 0
    }

    /// Chamfer sharp edges.
    ///
    /// - `angle_threshold_deg`: edges with dihedral angle above this are "sharp"
    ///   (e.g., 10.0 degrees for nearly-coplanar threshold)
    /// - `width`: chamfer inset distance in world units
    ///
    /// Returns the number of chamfer faces inserted.
    pub fn chamfer(&mut self, angle_threshold_deg: f32, width: f32) -> u32 {
        unsafe { omesh_chamfer(self.handle, angle_threshold_deg, width) }
    }

    /// Number of vertices in the mesh.
    pub fn n_vertices(&self) -> u32 {
        unsafe { omesh_n_vertices(self.handle) }
    }

    /// Number of faces in the mesh.
    pub fn n_faces(&self) -> u32 {
        unsafe { omesh_n_faces(self.handle) }
    }

    /// Export the mesh to flat arrays suitable for GPU upload.
    ///
    /// Vertices are duplicated per-face (no sharing) so each face-vertex gets
    /// the face normal. Polygon faces are fan-triangulated during export.
    pub fn export(&self) -> ExportedMesh {
        let mut n_verts: u32 = 0;
        let mut n_indices: u32 = 0;
        unsafe { omesh_export_sizes(self.handle, &mut n_verts, &mut n_indices) };

        if n_verts == 0 {
            return ExportedMesh {
                positions: Vec::new(),
                normals: Vec::new(),
                indices: Vec::new(),
            };
        }

        let mut positions = vec![0.0f32; n_verts as usize * 3];
        let mut normals = vec![0.0f32; n_verts as usize * 3];
        let mut indices = vec![0u32; n_indices as usize];

        unsafe {
            omesh_export(
                self.handle,
                positions.as_mut_ptr(),
                normals.as_mut_ptr(),
                indices.as_mut_ptr(),
            );
        }

        // Reshape to [f32; 3] arrays
        let positions: Vec<[f32; 3]> = positions
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let normals: Vec<[f32; 3]> = normals
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        ExportedMesh {
            positions,
            normals,
            indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_drop() {
        let _mesh = PolyMesh::new();
    }

    #[test]
    fn add_triangle() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex(0.0, 1.0, 0.0);
        assert!(mesh.add_face(&[v0, v1, v2]));
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn add_quad() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex(1.0, 1.0, 0.0);
        let v3 = mesh.add_vertex(0.0, 1.0, 0.0);
        assert!(mesh.add_face(&[v0, v1, v2, v3]));
        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn export_triangle() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex(0.0, 1.0, 0.0);
        mesh.add_face(&[v0, v1, v2]);

        let exported = mesh.export();
        assert_eq!(exported.positions.len(), 3);
        assert_eq!(exported.indices.len(), 3);
    }

    #[test]
    fn chamfer_cube() {
        let mut mesh = PolyMesh::new();
        // Build a unit cube with 8 vertices and 6 quad faces
        let v = [
            mesh.add_vertex(0.0, 0.0, 0.0),
            mesh.add_vertex(1.0, 0.0, 0.0),
            mesh.add_vertex(1.0, 1.0, 0.0),
            mesh.add_vertex(0.0, 1.0, 0.0),
            mesh.add_vertex(0.0, 0.0, 1.0),
            mesh.add_vertex(1.0, 0.0, 1.0),
            mesh.add_vertex(1.0, 1.0, 1.0),
            mesh.add_vertex(0.0, 1.0, 1.0),
        ];
        // Front face (+Z)
        mesh.add_face(&[v[4], v[5], v[6], v[7]]);
        // Back face (-Z)
        mesh.add_face(&[v[3], v[2], v[1], v[0]]);
        // Top face (+Y)
        mesh.add_face(&[v[3], v[7], v[6], v[2]]);
        // Bottom face (-Y)
        mesh.add_face(&[v[0], v[1], v[5], v[4]]);
        // Right face (+X)
        mesh.add_face(&[v[1], v[2], v[6], v[5]]);
        // Left face (-X)
        mesh.add_face(&[v[0], v[4], v[7], v[3]]);

        assert_eq!(mesh.n_faces(), 6);

        // All edges of a cube are 90 degrees
        let n_chamfered = mesh.chamfer(45.0, 0.05);
        assert!(n_chamfered > 0, "Should have chamfered some edges");
        assert!(mesh.n_faces() > 6, "Should have more faces after chamfering");

        let exported = mesh.export();
        assert!(!exported.positions.is_empty());
        assert!(!exported.indices.is_empty());
        assert_eq!(exported.indices.len() % 3, 0, "Indices should be triangles");
    }
}
