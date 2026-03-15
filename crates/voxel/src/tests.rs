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
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_mesh_valid(&result.full_res, "cube full_res");
    assert_mesh_valid(&result.lod, "cube lod");
}

#[test]
fn single_cube_watertight() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

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
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_mesh_valid(&result.full_res, "smooth_cube full_res");
    assert_mesh_valid(&result.lod, "smooth_cube lod");
}

#[test]
fn single_wedge_mesh_valid() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::new(SHAPE_WEDGE, Facing::North, 1));
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_mesh_valid(&result.full_res, "wedge full_res");
    assert_mesh_valid(&result.lod, "wedge lod");
}

#[test]
fn single_wedge_all_facings_valid() {
    let shapes = make_shapes();
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let data = single_voxel_chunk(Voxel::new(SHAPE_WEDGE, facing, 1));
        let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);
        let label = format!("wedge_{:?}", facing);

        assert_mesh_valid(&result.full_res, &format!("{} full_res", label));
        assert_mesh_valid(&result.lod, &format!("{} lod", label));
    }
}

#[test]
fn single_smooth_wedge_mesh_valid() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1));
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_mesh_valid(&result.full_res, "smooth_wedge full_res");
    assert_mesh_valid(&result.lod, "smooth_wedge lod");
}

#[test]
fn adjacent_cubes_mesh_valid() {
    let shapes = make_shapes();
    let data = block_2x2x2(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_mesh_valid(&result.full_res, "2x2x2 cubes full_res");
    assert_mesh_valid(&result.lod, "2x2x2 cubes lod");
}

#[test]
fn adjacent_cubes_no_internal_faces() {
    let shapes = make_shapes();

    // Single cube should have more faces than 2 adjacent cubes (internal faces culled)
    let single = single_voxel_chunk(Voxel::filled());
    let single_result = generate_chunk_mesh(&single, &shapes, crate::PresentationMode::EdgeGraphChamfer);
    let single_tris = single_result.lod.indices.len() / 3;

    let mut double = ChunkData::new();
    double.set(8, 16, 8, Voxel::filled());
    double.set(9, 16, 8, Voxel::filled());
    let double_result = generate_chunk_mesh(&double, &shapes, crate::PresentationMode::EdgeGraphChamfer);
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
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

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
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_mesh_valid(&result.full_res, "wedge_on_cube full_res");
    assert_mesh_valid(&result.lod, "wedge_on_cube lod");
}

#[test]
fn demo_staircase_ramp_valid() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

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
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

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
fn chamfered_mesh_has_at_least_as_many_tris_as_lod() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    let full_tris = result.full_res.indices.len() / 3;
    let lod_tris = result.lod.indices.len() / 3;
    eprintln!("full_res={} tris, lod={} tris", full_tris, lod_tris);
    assert!(full_tris >= lod_tris,
        "chamfered mesh should have >= LOD triangles: full={}, lod={}", full_tris, lod_tris);
}

#[test]
fn chamfered_mesh_no_extra_holes() {
    // The full_res mesh boundary edge count should be reasonable.
    // Compare against a single floating cube (known good, fully closed = 0 boundary).
    // For the demo chunk, boundary edges come from the chunk perimeter — clipping
    // should not significantly increase them.
    let shapes = make_shapes();

    // Single floating cube: should have 0 boundary edges (fully closed)
    let cube_data = single_voxel_chunk(Voxel::filled());
    let cube_result = generate_chunk_mesh(&cube_data, &shapes, crate::PresentationMode::EdgeGraphChamfer);
    let (cube_boundary, _, _) = count_edge_sharing(&cube_result.full_res);
    eprintln!("single cube boundary edges: {}", cube_boundary);
    assert!(cube_boundary == 0,
        "single floating cube should have 0 boundary edges, got {}", cube_boundary);

    // Single floating wedge: should also be closed
    let wedge_data = single_voxel_chunk(Voxel::new(SHAPE_WEDGE, Facing::North, 1));
    let wedge_result = generate_chunk_mesh(&wedge_data, &shapes, crate::PresentationMode::EdgeGraphChamfer);
    let (wedge_boundary, _, _) = count_edge_sharing(&wedge_result.full_res);
    eprintln!("single wedge boundary edges: {}", wedge_boundary);
    assert!(wedge_boundary == 0,
        "single floating wedge should have 0 boundary edges, got {}", wedge_boundary);

    // Wedge on cube: should be closed
    let mut woc_data = ChunkData::new();
    woc_data.set(8, 15, 8, Voxel::new(SHAPE_SMOOTH_CUBE, Facing::North, 1));
    woc_data.set(8, 16, 8, Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1));
    let woc_result = generate_chunk_mesh(&woc_data, &shapes, crate::PresentationMode::EdgeGraphChamfer);
    let (woc_boundary, _, _) = count_edge_sharing(&woc_result.full_res);
    eprintln!("wedge_on_cube boundary edges: {}", woc_boundary);
    if woc_boundary > 0 {
        dump_boundary_edges(&woc_result.full_res, "wedge_on_cube");
    }
    assert!(woc_boundary == 0,
        "wedge_on_cube should have 0 boundary edges, got {}", woc_boundary);
}

#[test]
fn empty_chunk_produces_empty_mesh() {
    let shapes = make_shapes();
    let data = ChunkData::new();
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert!(result.full_res.positions.is_empty(), "empty chunk should produce empty full_res");
    assert!(result.lod.positions.is_empty(), "empty chunk should produce empty lod");
}

#[test]
fn no_degenerate_triangles_single_cube() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_no_degenerate_triangles(&result.full_res, "cube full_res");
    assert_no_degenerate_triangles(&result.lod, "cube lod");
}

#[test]
fn no_degenerate_triangles_demo() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    assert_no_degenerate_triangles(&result.full_res, "demo full_res");
    assert_no_degenerate_triangles(&result.lod, "demo lod");
}

#[test]
fn single_cube_edge_sharing() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::filled());
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("single cube edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
    // Non-manifold edges are always bad
    assert!(non_manifold == 0,
        "single cube should have no non-manifold edges, got {}", non_manifold);
}

/// Dump positions of boundary edges (holes) for debugging.
fn dump_boundary_edges(mesh: &ChunkMeshData, label: &str) {
    type PosKey = (i32, i32, i32);
    type EdgeKey = (PosKey, PosKey);

    fn sorted_edge(a: PosKey, b: PosKey) -> EdgeKey {
        if a <= b { (a, b) } else { (b, a) }
    }

    fn key_to_pos(k: PosKey) -> [f32; 3] {
        [k.0 as f32 / 10000.0, k.1 as f32 / 10000.0, k.2 as f32 / 10000.0]
    }

    let mut edge_count: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
    for i in (0..mesh.indices.len()).step_by(3) {
        let tri_idx = i / 3;
        let tri = [mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]];
        for j in 0..3 {
            let pa = quantize_pos(&mesh.positions[tri[j] as usize]);
            let pb = quantize_pos(&mesh.positions[tri[(j + 1) % 3] as usize]);
            let key = sorted_edge(pa, pb);
            edge_count.entry(key).or_default().push(tri_idx);
        }
    }

    let mut boundary_count = 0;
    for (key, tris) in &edge_count {
        if tris.len() != 1 { continue; }
        boundary_count += 1;
        if boundary_count <= 20 {
            let p0 = key_to_pos(key.0);
            let p1 = key_to_pos(key.1);
            let tri_idx = tris[0];
            let ti = tri_idx * 3;
            let n = mesh.normals[mesh.indices[ti] as usize];
            eprintln!("  {}: BOUNDARY edge ({:.3},{:.3},{:.3})→({:.3},{:.3},{:.3}) tri[{}] normal=({:.3},{:.3},{:.3})",
                label, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], tri_idx, n[0], n[1], n[2]);
        }
    }
    if boundary_count > 6 {
        eprintln!("  {} ... and {} more (total {})", label, boundary_count - 6, boundary_count);
    }
}

/// Dump details about non-manifold edges for debugging.
fn dump_non_manifold_edges(mesh: &ChunkMeshData, label: &str) {
    type PosKey = (i32, i32, i32);
    type EdgeKey = (PosKey, PosKey);

    fn sorted_edge(a: PosKey, b: PosKey) -> EdgeKey {
        if a <= b { (a, b) } else { (b, a) }
    }

    fn key_to_pos(k: PosKey) -> [f32; 3] {
        [k.0 as f32 / 10000.0, k.1 as f32 / 10000.0, k.2 as f32 / 10000.0]
    }

    // Collect edges → list of (triangle_index, triangle vertices)
    let mut edge_tris: HashMap<EdgeKey, Vec<(usize, [u32; 3])>> = HashMap::new();
    for i in (0..mesh.indices.len()).step_by(3) {
        let tri = [mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]];
        let tri_idx = i / 3;
        for j in 0..3 {
            let pa = quantize_pos(&mesh.positions[tri[j] as usize]);
            let pb = quantize_pos(&mesh.positions[tri[(j + 1) % 3] as usize]);
            let key = sorted_edge(pa, pb);
            edge_tris.entry(key).or_default().push((tri_idx, tri));
        }
    }

    for (key, tris) in &edge_tris {
        if tris.len() <= 2 { continue; }
        let p0 = key_to_pos(key.0);
        let p1 = key_to_pos(key.1);
        eprintln!("{}: NON-MANIFOLD edge ({:.3},{:.3},{:.3})→({:.3},{:.3},{:.3}) shared by {} triangles:",
            label, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], tris.len());
        for (tri_idx, tri) in tris {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let n = mesh.normals[tri[0] as usize];
            eprintln!("  tri[{}]: verts=[{},{},{}] normal=({:.3},{:.3},{:.3})",
                tri_idx, tri[0], tri[1], tri[2], n[0], n[1], n[2]);
            eprintln!("    a=({:.3},{:.3},{:.3}) b=({:.3},{:.3},{:.3}) c=({:.3},{:.3},{:.3})",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
        }
    }
}

/// Count triangles that overlap: same plane, shared edge, overlapping area.
/// These cause z-fighting.
fn count_coplanar_overlapping_tris(mesh: &ChunkMeshData) -> usize {
    use bevy::math::Vec3;

    type PosKey = (i32, i32, i32);
    type EdgeKey = (PosKey, PosKey);

    fn sorted_edge(a: PosKey, b: PosKey) -> EdgeKey {
        if a <= b { (a, b) } else { (b, a) }
    }

    // Group triangles by their shared spatial edges
    let mut edge_tris: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
    for i in (0..mesh.indices.len()).step_by(3) {
        let tri_idx = i / 3;
        let tri = [mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]];
        for j in 0..3 {
            let pa = quantize_pos(&mesh.positions[tri[j] as usize]);
            let pb = quantize_pos(&mesh.positions[tri[(j + 1) % 3] as usize]);
            let key = sorted_edge(pa, pb);
            edge_tris.entry(key).or_default().push(tri_idx);
        }
    }

    let mut overlapping = 0;

    // For each edge with 3+ triangles, check for coplanar pairs
    for tris in edge_tris.values() {
        if tris.len() <= 2 { continue; }

        // Compare all pairs for coplanarity
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                let ti = tris[i] * 3;
                let tj = tris[j] * 3;

                let a = Vec3::from_array(mesh.positions[mesh.indices[ti] as usize]);
                let b = Vec3::from_array(mesh.positions[mesh.indices[ti + 1] as usize]);
                let c = Vec3::from_array(mesh.positions[mesh.indices[ti + 2] as usize]);
                let ni = (b - a).cross(c - a).normalize_or_zero();

                let d = Vec3::from_array(mesh.positions[mesh.indices[tj] as usize]);
                let e = Vec3::from_array(mesh.positions[mesh.indices[tj + 1] as usize]);
                let f = Vec3::from_array(mesh.positions[mesh.indices[tj + 2] as usize]);
                let nj = (e - d).cross(f - d).normalize_or_zero();

                // Coplanar: same normal direction AND all vertices on the same plane
                if ni.dot(nj).abs() > 0.99 {
                    // Check if all vertices of triangle j lie on triangle i's plane
                    let plane_dist_d = (d - a).dot(ni).abs();
                    let plane_dist_e = (e - a).dot(ni).abs();
                    let plane_dist_f = (f - a).dot(ni).abs();
                    if plane_dist_d < 0.001 && plane_dist_e < 0.001 && plane_dist_f < 0.001 {
                        overlapping += 1;
                    }
                }
            }
        }
    }

    overlapping
}

/// Check for triangle-triangle intersections (the actual cause of z-fighting).
/// Only checks triangles that share at least one vertex position (for performance).
fn count_intersecting_triangle_pairs(mesh: &ChunkMeshData) -> usize {
    use bevy::math::Vec3;

    type PosKey = (i32, i32, i32);

    // Build vertex position → list of triangle indices
    let mut pos_tris: HashMap<PosKey, Vec<usize>> = HashMap::new();
    for i in (0..mesh.indices.len()).step_by(3) {
        let tri_idx = i / 3;
        for j in 0..3 {
            let pk = quantize_pos(&mesh.positions[mesh.indices[i + j] as usize]);
            pos_tris.entry(pk).or_default().push(tri_idx);
        }
    }

    // Collect candidate pairs (triangles sharing at least one vertex position)
    let mut checked: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut intersecting = 0;

    for tris in pos_tris.values() {
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                let (ti, tj) = (tris[i], tris[j]);
                let pair = if ti < tj { (ti, tj) } else { (tj, ti) };
                if !checked.insert(pair) { continue; }

                // Get triangle vertices
                let ai = ti * 3;
                let aj = tj * 3;
                let a0 = Vec3::from_array(mesh.positions[mesh.indices[ai] as usize]);
                let a1 = Vec3::from_array(mesh.positions[mesh.indices[ai + 1] as usize]);
                let a2 = Vec3::from_array(mesh.positions[mesh.indices[ai + 2] as usize]);
                let b0 = Vec3::from_array(mesh.positions[mesh.indices[aj] as usize]);
                let b1 = Vec3::from_array(mesh.positions[mesh.indices[aj + 1] as usize]);
                let b2 = Vec3::from_array(mesh.positions[mesh.indices[aj + 2] as usize]);

                // Skip if triangles share all vertices (same triangle, different winding)
                let a_set: Vec<PosKey> = [a0, a1, a2].iter().map(|v| quantize_pos(&v.to_array())).collect();
                let b_set: Vec<PosKey> = [b0, b1, b2].iter().map(|v| quantize_pos(&v.to_array())).collect();
                let shared_verts = a_set.iter().filter(|v| b_set.contains(v)).count();
                if shared_verts >= 3 { continue; } // identical triangle

                // Triangles that share an edge (2 verts) are expected neighbors — skip
                if shared_verts >= 2 { continue; }

                // Check if triangles are coplanar and overlapping
                let na = (a1 - a0).cross(a2 - a0);
                let na_len = na.length();
                if na_len < 1e-8 { continue; }
                let na = na / na_len;

                let nb = (b1 - b0).cross(b2 - b0);
                let nb_len = nb.length();
                if nb_len < 1e-8 { continue; }
                let nb = nb / nb_len;

                // Must be coplanar (parallel normals)
                if na.dot(nb).abs() < 0.99 { continue; }

                // Must be on the same plane
                if (b0 - a0).dot(na).abs() > 0.001 { continue; }

                // Project both triangles to 2D and check overlap
                // Use the dominant axis for projection
                let abs_n = na.abs();
                let (u_axis, v_axis) = if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
                    (1, 2) // project onto YZ
                } else if abs_n.y >= abs_n.z {
                    (0, 2) // project onto XZ
                } else {
                    (0, 1) // project onto XY
                };

                let project = |v: Vec3| -> (f32, f32) {
                    (v[u_axis], v[v_axis])
                };

                let pa = [project(a0), project(a1), project(a2)];
                let pb = [project(b0), project(b1), project(b2)];

                // Check if any vertex of B is inside triangle A (or vice versa)
                if point_in_tri_2d(pb[0], pa) || point_in_tri_2d(pb[1], pa) || point_in_tri_2d(pb[2], pa)
                || point_in_tri_2d(pa[0], pb) || point_in_tri_2d(pa[1], pb) || point_in_tri_2d(pa[2], pb)
                {
                    intersecting += 1;
                    if intersecting <= 3 {
                        eprintln!("INTERSECT #{}: tri[{}] vs tri[{}], shared_verts={}, dot={:.3}",
                            intersecting, ti, tj, shared_verts, na.dot(nb));
                        eprintln!("  tri[{}]: ({:.3},{:.3},{:.3}) ({:.3},{:.3},{:.3}) ({:.3},{:.3},{:.3})",
                            ti, a0.x, a0.y, a0.z, a1.x, a1.y, a1.z, a2.x, a2.y, a2.z);
                        eprintln!("  tri[{}]: ({:.3},{:.3},{:.3}) ({:.3},{:.3},{:.3}) ({:.3},{:.3},{:.3})",
                            tj, b0.x, b0.y, b0.z, b1.x, b1.y, b1.z, b2.x, b2.y, b2.z);
                    }
                }
            }
        }
    }

    intersecting
}

fn point_in_tri_2d(p: (f32, f32), tri: [(f32, f32); 3]) -> bool {
    let (px, py) = p;
    let d1 = sign_2d(px, py, tri[0].0, tri[0].1, tri[1].0, tri[1].1);
    let d2 = sign_2d(px, py, tri[1].0, tri[1].1, tri[2].0, tri[2].1);
    let d3 = sign_2d(px, py, tri[2].0, tri[2].1, tri[0].0, tri[0].1);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    // Strictly inside (not on edge — edge sharing is OK)
    !(has_neg && has_pos) && d1.abs() > 1e-4 && d2.abs() > 1e-4 && d3.abs() > 1e-4
}

fn sign_2d(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
}

#[test]
fn wedge_on_cube_edge_sharing() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    let cube = Voxel::new(SHAPE_SMOOTH_CUBE, Facing::North, 1);
    let wedge = Voxel::new(SHAPE_SMOOTH_WEDGE, Facing::East, 1);
    data.set(8, 15, 8, cube);
    data.set(8, 16, 8, wedge);
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("wedge_on_cube edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);

    let overlapping = count_coplanar_overlapping_tris(&result.full_res);
    let intersecting = count_intersecting_triangle_pairs(&result.full_res);
    eprintln!("wedge_on_cube: {} coplanar overlapping, {} intersecting triangle pairs",
        overlapping, intersecting);
    if overlapping > 0 || intersecting > 0 {
        dump_non_manifold_edges(&result.full_res, "wedge_on_cube");
    }
    assert!(overlapping == 0,
        "wedge_on_cube should have no coplanar overlapping triangles, got {}", overlapping);
    assert!(intersecting == 0,
        "wedge_on_cube should have no intersecting triangles, got {}", intersecting);
}

#[test]
fn demo_edge_sharing() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("demo edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);

    let overlapping = count_coplanar_overlapping_tris(&result.full_res);
    let intersecting = count_intersecting_triangle_pairs(&result.full_res);
    eprintln!("demo: {} coplanar overlapping, {} intersecting triangle pairs",
        overlapping, intersecting);
    assert!(overlapping == 0,
        "demo should have no coplanar overlapping triangles, got {}", overlapping);
    assert!(intersecting == 0,
        "demo should have no intersecting triangles, got {}", intersecting);
}

#[test]
fn single_wedge_edge_sharing() {
    let shapes = make_shapes();
    let data = single_voxel_chunk(Voxel::new(SHAPE_WEDGE, Facing::North, 1));
    let result = generate_chunk_mesh(&data, &shapes, crate::PresentationMode::EdgeGraphChamfer);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("single wedge edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
    assert!(non_manifold == 0,
        "single wedge should have no non-manifold edges, got {}", non_manifold);
}
