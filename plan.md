# OpenMesh Chamfer Wrapper — Implementation Plan

## Overview
Create a new Rust crate that wraps the C++ OpenMesh library via FFI, and expose it as a new `PresentationMode::OpenMeshChamfer` variant selectable with F3.

## Architecture

### New crate: `crates/openmesh-sys/`
Low-level FFI bindings to OpenMesh's C++ API.

- **`build.rs`**: Uses `cc` crate to compile a small C++ shim (`src/shim.cpp`) that exposes a flat C API for the OpenMesh operations we need. Links against OpenMesh (downloaded/vendored or found via pkg-config).
- **`src/shim.cpp`**: C++ wrapper functions:
  - `omesh_new()` → create a TriMesh or PolyMesh
  - `omesh_add_vertex(mesh, x, y, z)` → add vertex, return handle
  - `omesh_add_face(mesh, verts_ptr, n_verts)` → add polygon face (quad-aware, NOT forced triangulation)
  - `omesh_chamfer_edges(mesh, angle_threshold, width)` → identify sharp edges and chamfer them
  - `omesh_export(mesh, out_positions, out_normals, out_indices, ...)` → extract final mesh data
  - `omesh_free(mesh)` → cleanup
- **`src/lib.rs`**: Raw `extern "C"` declarations matching the shim

### New crate: `crates/openmesh/`
Safe Rust wrapper around `openmesh-sys`.

- **`src/lib.rs`**: Safe `PolyMesh` type wrapping the raw pointer with Drop, and methods:
  - `PolyMesh::new()`
  - `add_vertex(pos: Vec3) -> VertexHandle`
  - `add_face(verts: &[VertexHandle])` — passes polygons directly, no pre-triangulation
  - `chamfer(threshold: f32, width: f32)` — runs the chamfer operation
  - `export() -> (Vec<Vec3>, Vec<Vec3>, Vec<u32>)` — positions, normals, indices

### Changes to `crates/voxel/`

1. **`Cargo.toml`**: Add `jumpblocks_openmesh = { path = "../openmesh" }` dependency
2. **`src/lib.rs`**: Add `OpenMeshChamfer` variant to `PresentationMode` enum, update `cycle()` and `label()`
3. **New file `src/openmesh_chamfer.rs`**:
   - `generate_openmesh_chamfer(data, shapes) -> ChunkMeshData`
   - Builds `SolidMesh` via existing `build_solid_mesh_public()`
   - Feeds polygon faces (quads/tris as-is) into the OpenMesh `PolyMesh` — **no triangulation before chamfer**
   - Runs chamfer operation
   - Exports back to `ChunkMeshData`
4. **`src/meshing.rs`**: Add match arm in `generate_chunk_mesh()` for the new mode

## Key Design Decisions

- **PolyMesh, not TriMesh**: OpenMesh supports both. We use `PolyMesh` specifically to preserve quad faces during chamfering, addressing the triangulation concern.
- **Vendored OpenMesh**: Bundle OpenMesh source in `crates/openmesh-sys/vendor/` so the build is self-contained (no system dependency). OpenMesh is BSD-licensed.
- **Minimal C++ shim**: Keep the C++ layer thin — just bridge the C++ API to C calling convention. All logic stays in Rust or uses OpenMesh's built-in algorithms.

## File Changes Summary

```
crates/openmesh-sys/          (NEW)
  Cargo.toml
  build.rs
  src/lib.rs
  src/shim.cpp
  src/shim.h

crates/openmesh/              (NEW)
  Cargo.toml
  src/lib.rs

crates/voxel/
  Cargo.toml                  (EDIT — add openmesh dep)
  src/lib.rs                  (EDIT — new enum variant)
  src/meshing.rs              (EDIT — new match arm)
  src/openmesh_chamfer.rs     (NEW)

Cargo.toml                    (EDIT — add workspace members)
```
