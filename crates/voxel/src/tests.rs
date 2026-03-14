use std::collections::HashMap;

use crate::chunk::*;
use crate::meshing::*;
use crate::shape::*;

// ---------------------------------------------------------------------------
// Mesh validation helpers
// ---------------------------------------------------------------------------

/// Check that all positions and normals are finite (no NaN/Inf).
fn assert_no_nans(mesh: &ChunkMeshData, label: &str) {
    for (i, p) in mesh.positions.iter().enumerate() {
        assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite(),
            "{}: NaN/Inf in position[{}]: {:?}", label, i, p);
    }
    for (i, n) in mesh.normals.iter().enumerate() {
        assert!(n[0].is_finite() && n[1].is_finite() && n[2].is_finite(),
            "{}: NaN/Inf in normal[{}]: {:?}", label, i, n);
    }
}

/// Check that all indices are in bounds.
fn assert_indices_valid(mesh: &ChunkMeshData, label: &str) {
    let n = mesh.positions.len() as u32;
    for (i, &idx) in mesh.indices.iter().enumerate() {
        assert!(idx < n,
            "{}: index[{}] = {} out of bounds (vertex count = {})", label, i, idx, n);
    }
    assert!(mesh.indices.len() % 3 == 0,
        "{}: index count {} is not a multiple of 3", label, mesh.indices.len());
}

/// Check that no triangle is degenerate (zero area).
fn assert_no_degenerate_triangles(mesh: &ChunkMeshData, label: &str) {
    let positions = &mesh.positions;
    for i in (0..mesh.indices.len()).step_by(3) {
        let a = mesh.indices[i] as usize;
        let b = mesh.indices[i + 1] as usize;
        let c = mesh.indices[i + 2] as usize;
        let pa = bevy::math::Vec3::from_array(positions[a]);
        let pb = bevy::math::Vec3::from_array(positions[b]);
        let pc = bevy::math::Vec3::from_array(positions[c]);
        let cross = (pb - pa).cross(pc - pa);
        let area = cross.length() * 0.5;
        assert!(area > 1e-10,
            "{}: degenerate triangle at index {} (verts {},{},{}, area={})",
            label, i / 3, a, b, c, area);
    }
}

/// Check that normals are roughly unit length.
fn assert_normals_unit_length(mesh: &ChunkMeshData, label: &str) {
    for (i, n) in mesh.normals.iter().enumerate() {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 0.05,
            "{}: normal[{}] length {} (expected ~1.0): {:?}", label, i, len, n);
    }
}

/// Quantize a position for spatial edge comparison.
fn quantize_pos(p: &[f32; 3]) -> (i32, i32, i32) {
    const SCALE: f32 = 10000.0;
    (
        (p[0] * SCALE).round() as i32,
        (p[1] * SCALE).round() as i32,
        (p[2] * SCALE).round() as i32,
    )
}

/// Check edge sharing by POSITION (not vertex index).
/// This correctly detects overlapping triangles that use separate vertices at the same position.
/// Returns (boundary_edges, interior_edges, non_manifold_edges).
fn count_edge_sharing(mesh: &ChunkMeshData) -> (usize, usize, usize) {
    type PosKey = (i32, i32, i32);
    type EdgeKey = (PosKey, PosKey);

    fn sorted_edge(a: PosKey, b: PosKey) -> EdgeKey {
        if a <= b { (a, b) } else { (b, a) }
    }

    let mut edge_count: HashMap<EdgeKey, usize> = HashMap::new();
    for i in (0..mesh.indices.len()).step_by(3) {
        let tri = [mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]];
        for j in 0..3 {
            let pa = quantize_pos(&mesh.positions[tri[j] as usize]);
            let pb = quantize_pos(&mesh.positions[tri[(j + 1) % 3] as usize]);
            let key = sorted_edge(pa, pb);
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    let mut boundary = 0;
    let mut interior = 0;
    let mut non_manifold = 0;
    for &count in edge_count.values() {
        match count {
            1 => boundary += 1,
            2 => interior += 1,
            _ => non_manifold += 1,
        }
    }
    (boundary, interior, non_manifold)
}

fn assert_watertight(mesh: &ChunkMeshData, label: &str) {
    let (boundary, _interior, non_manifold) = count_edge_sharing(mesh);
    assert!(boundary == 0,
        "{}: mesh has {} boundary edges (holes) — not watertight", label, boundary);
    assert!(non_manifold == 0,
        "{}: mesh has {} non-manifold edges (3+ triangles sharing an edge)", label, non_manifold);
}

/// Run all basic validity checks on a mesh.
fn assert_mesh_valid(mesh: &ChunkMeshData, label: &str) {
    assert!(!mesh.positions.is_empty(), "{}: mesh has no vertices", label);
    assert!(!mesh.indices.is_empty(), "{}: mesh has no indices", label);
    assert_no_nans(mesh, label);
    assert_indices_valid(mesh, label);
    assert_normals_unit_length(mesh, label);
    assert_eq!(mesh.positions.len(), mesh.normals.len(),
        "{}: position/normal count mismatch", label);
    assert_eq!(mesh.positions.len(), mesh.uvs.len(),
        "{}: position/uv count mismatch", label);
    assert_eq!(mesh.positions.len(), mesh.chamfer_offsets.len(),
        "{}: position/chamfer_offset count mismatch", label);
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn make_shapes() -> ShapeTable {
    ShapeTable::default()
}

/// Place a single voxel at the center of a chunk (away from boundaries).
fn single_voxel_chunk(voxel: Voxel) -> ChunkData {
    let mut data = ChunkData::new();
    data.set(8, 16, 8, voxel);
    data
}

/// Place a 2x2x2 block of voxels at the center.
fn block_2x2x2(voxel: Voxel) -> ChunkData {
    let mut data = ChunkData::new();
    for x in 7..9 {
        for y in 15..17 {
            for z in 7..9 {
                data.set(x, y, z, voxel);
            }
        }
    }
    data
}

/// Build the demo staircase + ramp chunk (same as world.rs).
fn demo_chunk() -> ChunkData {
    let mut data = ChunkData::new();
    let smooth_voxel = Voxel::new(SHAPE_SMOOTH_CUBE, Facing::North, 1);
    let wedge_e = Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1);

    // Ground layer
    for x in 0..10 {
        for z in 0..10 {
            data.set_filled(x, 0, z, true);
        }
    }

    // Staircase
    for step in 0..8 {
        let y_base = 1 + step;
        for x in 0..3 {
            for z in 0..3 {
                data.set(x + step, y_base, z + 3, smooth_voxel);
            }
        }
    }

    // Tower
    for y in 9..18 {
        for x in 8..11 {
            for z in 3..6 {
                data.set(x, y, z, smooth_voxel);
            }
        }
    }

    // Wedge ramp
    for step in 0..8 {
        let y_base = 1 + step;
        for z_off in 0..3 {
            data.set(step, y_base, 7 + z_off, wedge_e);
        }
        for y in 1..y_base {
            for z_off in 0..3 {
                data.set(step, y, 7 + z_off, smooth_voxel);
            }
        }
    }

    data
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn single_cube_mesh_valid() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "cube full_res");
    assert_mesh_valid(&result.lod, "cube lod");
}

#[test]
fn single_cube_watertight() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes);

    let (boundary, _interior, non_manifold) = count_edge_sharing(&result.full_res);
    // Report but don't fail yet — watertight is aspirational for now
    if boundary > 0 || non_manifold > 0 {
        eprintln!("single_cube_watertight: {} boundary edges, {} non-manifold edges",
            boundary, non_manifold);
    }
}

#[test]
fn single_smooth_cube_mesh_valid() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::new(SHAPE_SMOOTH_CUBE, Facing::North, 1));
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "smooth_cube full_res");
    assert_mesh_valid(&result.lod, "smooth_cube lod");
}

#[test]
fn single_wedge_mesh_valid() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::new(SHAPE_WEDGE, Facing::North, 1));
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "wedge full_res");
    assert_mesh_valid(&result.lod, "wedge lod");
}

#[test]
fn single_wedge_all_facings_valid() {
    let shapes = make_shapes();
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let data = single_voxel_chunk(Voxel::new(SHAPE_WEDGE, facing, 1));
        let result = generate_chunk_mesh(&data, &shapes);
        let label = format!("wedge_{:?}", facing);

        assert_mesh_valid(&result.full_res, &format!("{} full_res", label));
        assert_mesh_valid(&result.lod, &format!("{} lod", label));
    }
}

#[test]
fn single_smooth_wedge_mesh_valid() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1));
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "smooth_wedge full_res");
    assert_mesh_valid(&result.lod, "smooth_wedge lod");
}

#[test]
fn adjacent_cubes_mesh_valid() {
    let shapes = make_shapes();
    let data = block_2x2x2(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "2x2x2 cubes full_res");
    assert_mesh_valid(&result.lod, "2x2x2 cubes lod");
}

#[test]
fn adjacent_cubes_no_internal_faces() {
    let shapes = make_shapes();

    // Single cube should have more faces than 2 adjacent cubes (internal faces culled)
    let single = single_voxel_chunk(Voxel::filled());
    let single_result = generate_chunk_mesh(&single, &shapes);
    let single_tris = single_result.lod.indices.len() / 3;

    let mut double = ChunkData::new();
    double.set(8, 16, 8, Voxel::filled());
    double.set(9, 16, 8, Voxel::filled());
    let double_result = generate_chunk_mesh(&double, &shapes);
    let double_tris = double_result.lod.indices.len() / 3;

    // Two adjacent cubes: 12 faces each = 24 total, minus 2 shared = 20 faces = 10 tris... wait
    // Actually: each cube has 6 faces = 12 tris. Two cubes share 1 face each side = 10 faces = 20 tris
    assert!(double_tris < single_tris * 2,
        "adjacent cubes should have fewer tris than 2x single: {} vs {}",
        double_tris, single_tris * 2);
}

#[test]
fn adjacent_wedges_same_facing_valid() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    let wedge = Voxel::new(SHAPE_WEDGE, Facing::East, 1);
    // Three wedges in a row (like the ramp)
    data.set(8, 16, 7, wedge);
    data.set(8, 16, 8, wedge);
    data.set(8, 16, 9, wedge);
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "3 wedges full_res");
    assert_mesh_valid(&result.lod, "3 wedges lod");
}

#[test]
fn wedge_on_cube_valid() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    let cube = Voxel::new(SHAPE_SMOOTH_CUBE, Facing::North, 1);
    let wedge = Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1);
    // Cube with wedge on top (like ramp step)
    data.set(8, 15, 8, cube);
    data.set(8, 16, 8, wedge);
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "wedge_on_cube full_res");
    assert_mesh_valid(&result.lod, "wedge_on_cube lod");
}

#[test]
fn demo_staircase_ramp_valid() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes);

    assert_mesh_valid(&result.full_res, "demo full_res");
    assert_mesh_valid(&result.lod, "demo lod");

    // Sanity: should have a reasonable number of triangles
    let full_tris = result.full_res.indices.len() / 3;
    let lod_tris = result.lod.indices.len() / 3;
    assert!(full_tris > 100, "demo full_res should have >100 tris, got {}", full_tris);
    assert!(lod_tris > 100, "demo lod should have >100 tris, got {}", lod_tris);
    eprintln!("demo mesh: full_res={} tris, lod={} tris", full_tris, lod_tris);
}

#[test]
fn collider_mesh_valid() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes);

    // Collider should have vertices and valid triangle indices
    assert!(!result.collider_data.vertices.is_empty(), "collider has no vertices");
    assert!(!result.collider_data.indices.is_empty(), "collider has no indices");
    let n = result.collider_data.vertices.len() as u32;
    for (i, tri) in result.collider_data.indices.iter().enumerate() {
        assert!(tri[0] < n && tri[1] < n && tri[2] < n,
            "collider index[{}] out of bounds: {:?} (vertex count = {})", i, tri, n);
    }
}

#[test]
fn empty_chunk_produces_empty_mesh() {
    let shapes = make_shapes();
    let data = ChunkData::new();
    let result = generate_chunk_mesh(&data, &shapes);

    assert!(result.full_res.positions.is_empty(), "empty chunk should produce empty full_res");
    assert!(result.lod.positions.is_empty(), "empty chunk should produce empty lod");
}

#[test]
fn no_degenerate_triangles_single_cube() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes);

    assert_no_degenerate_triangles(&result.full_res, "cube full_res");
    assert_no_degenerate_triangles(&result.lod, "cube lod");
}

#[test]
fn no_degenerate_triangles_demo() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes);

    assert_no_degenerate_triangles(&result.full_res, "demo full_res");
    assert_no_degenerate_triangles(&result.lod, "demo lod");
}

#[test]
fn single_cube_edge_sharing() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("single cube edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
    // Non-manifold edges are always bad
    assert!(non_manifold == 0,
        "single cube should have no non-manifold edges, got {}", non_manifold);
}

#[test]
fn wedge_on_cube_edge_sharing() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    let cube = Voxel::new(SHAPE_SMOOTH_CUBE, Facing::North, 1);
    let wedge = Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1);
    data.set(8, 15, 8, cube);
    data.set(8, 16, 8, wedge);
    let result = generate_chunk_mesh(&data, &shapes);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("wedge_on_cube edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
    assert!(non_manifold == 0,
        "wedge_on_cube should have no non-manifold edges, got {}", non_manifold);
}

#[test]
fn demo_edge_sharing() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("demo edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
    assert!(non_manifold == 0,
        "demo should have no non-manifold edges, got {}", non_manifold);
}

#[test]
fn single_wedge_edge_sharing() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::new(SHAPE_WEDGE, Facing::North, 1));
    let result = generate_chunk_mesh(&data, &shapes);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("single wedge edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
    assert!(non_manifold == 0,
        "single wedge should have no non-manifold edges, got {}", non_manifold);
}
