//! Raw FFI bindings to the OpenMesh C shim.

use std::os::raw::c_float;

pub type OmeshHandle = *mut std::ffi::c_void;

unsafe extern "C" {
    pub fn omesh_new() -> OmeshHandle;
    pub fn omesh_free(mesh: OmeshHandle);

    pub fn omesh_add_vertex(mesh: OmeshHandle, x: c_float, y: c_float, z: c_float) -> u32;
    pub fn omesh_add_face(mesh: OmeshHandle, vert_indices: *const u32, n_verts: u32) -> i32;

    pub fn omesh_chamfer(mesh: OmeshHandle, angle_threshold_deg: c_float, width: c_float) -> u32;

    pub fn omesh_n_vertices(mesh: OmeshHandle) -> u32;
    pub fn omesh_n_faces(mesh: OmeshHandle) -> u32;

    pub fn omesh_export_sizes(
        mesh: OmeshHandle,
        out_n_verts: *mut u32,
        out_n_indices: *mut u32,
    );
    pub fn omesh_export(
        mesh: OmeshHandle,
        out_positions: *mut c_float,
        out_normals: *mut c_float,
        out_indices: *mut u32,
    );
}
