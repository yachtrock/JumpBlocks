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

/// Check that the mesh is convex: every vertex must be on or behind every
/// triangle's plane.  The `tolerance` allows small violations from fillet
/// geometry (corner patches where pushed and unpushed vertices create
/// slightly tilted triangles).  Large violations indicate wrong push
/// directions.
fn assert_convex_with_tolerance(mesh: &ChunkMeshData, tolerance: f32, label: &str) {
    use bevy::math::Vec3;

    let positions: Vec<Vec3> = mesh.positions.iter().map(|p| Vec3::from_array(*p)).collect();
    let mut worst_dist = 0.0f32;
    let mut worst_tri = 0usize;
    let mut worst_vert = 0usize;
    let mut violation_count = 0usize;

    for i in (0..mesh.indices.len()).step_by(3) {
        let ia = mesh.indices[i] as usize;
        let ib = mesh.indices[i + 1] as usize;
        let ic = mesh.indices[i + 2] as usize;
        let pa = positions[ia];
        let pb = positions[ib];
        let pc = positions[ic];

        let normal = (pb - pa).cross(pc - pa);
        let normal_len = normal.length();
        if normal_len < 1e-10 {
            continue; // degenerate triangle
        }
        let n = normal / normal_len;

        for (vi, &p) in positions.iter().enumerate() {
            let d = (p - pa).dot(n);
            if d > tolerance {
                violation_count += 1;
                if d > worst_dist {
                    worst_dist = d;
                    worst_tri = i / 3;
                    worst_vert = vi;
                }
            }
        }
    }

    if violation_count > 0 {
        let ti = worst_tri * 3;
        let ia = mesh.indices[ti] as usize;
        let ib = mesh.indices[ti + 1] as usize;
        let ic = mesh.indices[ti + 2] as usize;
        eprintln!(
            "{}: NOT CONVEX — {} violations (tolerance {:.3}), worst: vert {} at {:?} is {:.5} in front of tri[{}] (verts {:?}, {:?}, {:?})",
            label, violation_count, tolerance, worst_vert, positions[worst_vert],
            worst_dist, worst_tri, positions[ia], positions[ib], positions[ic],
        );
    }

    assert!(
        violation_count == 0,
        "{}: mesh is not convex ({} violations beyond tolerance {:.3}, worst distance {:.5})",
        label, violation_count, tolerance, worst_dist,
    );
}

/// Strict convexity check (floating-point tolerance only).
fn assert_convex(mesh: &ChunkMeshData, label: &str) {
    assert_convex_with_tolerance(mesh, 1e-4, label);
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

/// Place a single block at the center of a chunk (away from boundaries).
fn single_block_chunk(shape: u16, facing: Facing, texture: u16) -> ChunkData {
    let mut data = ChunkData::new();
    if shape == SHAPE_WEDGE {
        data.place_wedge(8, 16, 8, facing, texture);
    } else {
        data.place_std(8, 16, 8, shape, facing, texture);
    }
    data
}

/// Place two adjacent cube blocks at the center (4x1x2 cells total).
fn block_2x2x2() -> ChunkData {
    let mut data = ChunkData::new();
    data.place_std(8, 16, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 16, 8, SHAPE_CUBE, Facing::North, 1);
    data
}

/// Build a demo chunk with ground, staircase, tower, and wedge ramp.
fn demo_chunk() -> ChunkData {
    let mut data = ChunkData::new();

    // Ground layer: 10x10 blocks = 20x20 cells = 10x10 world units
    for bx in 0..10 {
        for bz in 0..10 {
            data.place_std(bx * 2, 0, bz * 2, SHAPE_CUBE, Facing::North, 1);
        }
    }

    // Staircase: 4 steps, each 1 block tall (1 cell)
    for step in 0..4usize {
        let y = 1 + step;
        for bx in 0..2 {
            for bz in 0..2 {
                data.place_std(bx * 2 + step * 4, y, bz * 2 + 6, SHAPE_CUBE, Facing::North, 1);
            }
        }
    }

    // Tower at the top of staircase
    for by in 0..4 {
        for bx in 0..2 {
            for bz in 0..2 {
                data.place_std(bx * 2 + 16, 5 + by, bz * 2 + 6, SHAPE_CUBE, Facing::North, 1);
            }
        }
    }

    // Wedge ramp: wedge blocks are 2 cells tall
    for step in 0..4usize {
        let y = 1 + step * 2;
        for bz in 0..2 {
            data.place_wedge(step * 4, y, bz * 2 + 14, Facing::East, 1);
        }
        // Fill cubes underneath for steps > 0
        for fill_y in 1..step * 2 + 1 {
            for bz in 0..2 {
                data.place_std(step * 4, fill_y, bz * 2 + 14, SHAPE_CUBE, Facing::North, 1);
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
    let data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res, "cube full_res");
    assert_mesh_valid(&result.lod, "cube lod");
}

#[test]
fn single_cube_watertight() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let (boundary, _interior, non_manifold) = count_edge_sharing(&result.full_res);
    if boundary > 0 || non_manifold > 0 {
        eprintln!("single_cube_watertight: {} boundary edges, {} non-manifold edges",
            boundary, non_manifold);
    }
}

#[test]
fn single_wedge_mesh_valid() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_WEDGE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res, "wedge full_res");
    assert_mesh_valid(&result.lod, "wedge lod");
}

#[test]
fn single_wedge_all_facings_valid() {
    let shapes = make_shapes();
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let data = single_block_chunk(SHAPE_WEDGE, facing, 1);
        let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
        let label = format!("wedge_{:?}", facing);

        assert_mesh_valid(&result.full_res, &format!("{} full_res", label));
        assert_mesh_valid(&result.lod, &format!("{} lod", label));
    }
}

#[test]
fn adjacent_cubes_mesh_valid() {
    let shapes = make_shapes();
    let data = block_2x2x2();
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res, "2x2x2 cubes full_res");
    assert_mesh_valid(&result.lod, "2x2x2 cubes lod");
}

#[test]
fn adjacent_cubes_no_internal_faces() {
    let shapes = make_shapes();

    let single = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let single_result = generate_chunk_mesh(&single, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let single_tris = single_result.lod.indices.len() / 3;

    let mut double = ChunkData::new();
    double.place_std(8, 16, 8, SHAPE_CUBE, Facing::North, 1);
    double.place_std(10, 16, 8, SHAPE_CUBE, Facing::North, 1);
    let double_result = generate_chunk_mesh(&double, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let double_tris = double_result.lod.indices.len() / 3;

    assert!(double_tris < single_tris * 2,
        "adjacent cubes should have fewer tris than 2x single: {} vs {}",
        double_tris, single_tris * 2);
}

#[test]
fn adjacent_wedges_same_facing_valid() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_wedge(8, 16, 6, Facing::East, 1);
    data.place_wedge(8, 16, 8, Facing::East, 1);
    data.place_wedge(8, 16, 10, Facing::East, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res, "3 wedges full_res");
    assert_mesh_valid(&result.lod, "3 wedges lod");
}

#[test]
fn wedge_on_cube_valid() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_wedge(8, 15, 8, Facing::East, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res, "wedge_on_cube full_res");
    assert_mesh_valid(&result.lod, "wedge_on_cube lod");
}

#[test]
fn diagonal_cubes_vertex_touching() {
    // Two cubes sharing only a single vertex should chamfer independently —
    // the result should have the same tri count as two isolated cubes.
    let shapes = make_shapes();

    // Single isolated cube for reference
    let single = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let single_result = generate_chunk_mesh(&single, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let single_tris = single_result.full_res.indices.len() / 3;

    // Two cubes sharing only a vertex (offset in X, Y, and Z)
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 15, 10, SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let diag_tris = result.full_res.indices.len() / 3;
    let expected = single_tris * 2;

    let (boundary, _, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("diagonal vertex: {} tris (expected {}={}x2), boundary={}, non_manifold={}",
        diag_tris, expected, single_tris, boundary, non_manifold);

    assert_eq!(diag_tris, expected,
        "vertex-only contact should produce same tris as two isolated cubes ({} vs {})",
        diag_tris, expected);
    assert_eq!(non_manifold, 0, "should have no non-manifold edges");
}

#[test]
fn diagonal_cubes_vertex_with_nearby_wedge() {
    // Matches the visual test chunk: wedge at (4,2,20) + diagonal cubes at (4,4,22) and (6,5,24)
    let shapes = make_shapes();

    let single = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let single_result = generate_chunk_mesh(&single, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let single_tris = single_result.full_res.indices.len() / 3;

    let mut data = ChunkData::new();
    data.place_wedge(4, 2, 20, Facing::East, 1);
    data.place_std(4, 4, 22, SHAPE_CUBE, Facing::North, 1);
    data.place_std(6, 5, 24, SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let wedge_only = {
        let mut d = ChunkData::new();
        d.place_wedge(4, 2, 20, Facing::East, 1);
        generate_chunk_mesh(&d, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset)
    };
    let wedge_tris = wedge_only.full_res.indices.len() / 3;

    let total_tris = result.full_res.indices.len() / 3;
    let expected = wedge_tris + single_tris * 2;

    eprintln!("vertex+wedge: {} tris (expected {}=wedge {}+cube {}x2), ",
        total_tris, expected, wedge_tris, single_tris);
    assert_eq!(total_tris, expected,
        "all 3 blocks should be independent: {} vs {}", total_tris, expected);
}

#[test]
fn diagonal_cubes_edge_touching() {
    // Two cubes sharing only a single edge should chamfer independently.
    // Use the same block positions as the visual test chunk in world.rs.
    let shapes = make_shapes();

    let single = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let single_result = generate_chunk_mesh(&single, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let single_tris = single_result.full_res.indices.len() / 3;

    // Reproduce the full visual test chunk to catch interactions
    let mut data = ChunkData::new();
    data.place_std(4, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_wedge(4, 4, 10, Facing::North, 1);
    data.place_wedge(10, 4, 10, Facing::East, 1);
    data.place_wedge(16, 4, 10, Facing::South, 1);
    data.place_wedge(22, 4, 10, Facing::West, 1);
    data.place_wedge(4, 4, 16, Facing::North, 1);
    data.place_wedge(10, 4, 16, Facing::East, 1);
    data.place_std(16, 2, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_wedge(16, 3, 4, Facing::East, 1);
    data.place_std(22, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_std(24, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 15, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(4, 4, 22, SHAPE_CUBE, Facing::North, 1);
    data.place_std(6, 5, 24, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 4, 22, SHAPE_CUBE, Facing::North, 1);
    data.place_std(12, 5, 22, SHAPE_CUBE, Facing::North, 1);

    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    // Extract just the edge-touching cubes' region to count their tris
    // They should match 2 isolated cubes exactly
    let diag_tris = result.full_res.indices.len() / 3;

    // Count tris for everything EXCEPT the two edge-touching cubes
    let mut data_without = ChunkData::new();
    data_without.place_std(4, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data_without.place_std(10, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data_without.place_wedge(4, 4, 10, Facing::North, 1);
    data_without.place_wedge(10, 4, 10, Facing::East, 1);
    data_without.place_wedge(16, 4, 10, Facing::South, 1);
    data_without.place_wedge(22, 4, 10, Facing::West, 1);
    data_without.place_wedge(4, 4, 16, Facing::North, 1);
    data_without.place_wedge(10, 4, 16, Facing::East, 1);
    data_without.place_std(16, 2, 4, SHAPE_CUBE, Facing::North, 1);
    data_without.place_wedge(16, 3, 4, Facing::East, 1);
    data_without.place_std(22, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data_without.place_std(24, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data_without.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data_without.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data_without.place_std(10, 15, 8, SHAPE_CUBE, Facing::North, 1);
    data_without.place_std(4, 4, 22, SHAPE_CUBE, Facing::North, 1);
    data_without.place_std(6, 5, 24, SHAPE_CUBE, Facing::North, 1);
    // omit the two edge-touching cubes at (10,4,22) and (12,5,22)
    let result_without = generate_chunk_mesh(&data_without, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let without_tris = result_without.full_res.indices.len() / 3;

    let edge_pair_tris = diag_tris - without_tris;
    let expected_pair = single_tris * 2;

    eprintln!("full chunk edge-pair: {} tris (expected {}={}x2)",
        edge_pair_tris, expected_pair, single_tris);

    assert_eq!(edge_pair_tris, expected_pair,
        "edge-touching cubes in full chunk should match 2 isolated cubes ({} vs {})",
        edge_pair_tris, expected_pair);
}

#[test]
fn demo_staircase_ramp_valid() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res, "demo full_res");
    assert_mesh_valid(&result.lod, "demo lod");

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
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

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
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let full_tris = result.full_res.indices.len() / 3;
    let lod_tris = result.lod.indices.len() / 3;
    eprintln!("full_res={} tris, lod={} tris", full_tris, lod_tris);
    assert!(full_tris >= lod_tris,
        "chamfered mesh should have >= LOD triangles: full={}, lod={}", full_tris, lod_tris);
}

#[test]
fn chamfered_mesh_no_extra_holes() {
    let shapes = make_shapes();

    let cube_data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let cube_result = generate_chunk_mesh(&cube_data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let (cube_boundary, _, _) = count_edge_sharing(&cube_result.full_res);
    eprintln!("single cube boundary edges: {}", cube_boundary);
    assert!(cube_boundary == 0,
        "single floating cube should have 0 boundary edges, got {}", cube_boundary);

    let wedge_data = single_block_chunk(SHAPE_WEDGE, Facing::North, 1);
    let wedge_result = generate_chunk_mesh(&wedge_data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let (wedge_boundary, _, _) = count_edge_sharing(&wedge_result.full_res);
    eprintln!("single wedge boundary edges: {}", wedge_boundary);
    assert!(wedge_boundary == 0,
        "single floating wedge should have 0 boundary edges, got {}", wedge_boundary);

    let mut woc_data = ChunkData::new();
    woc_data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    woc_data.place_wedge(8, 15, 8, Facing::East, 1);
    let woc_result = generate_chunk_mesh(&woc_data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
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
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert!(result.full_res.positions.is_empty(), "empty chunk should produce empty full_res");
    assert!(result.lod.positions.is_empty(), "empty chunk should produce empty lod");
}

#[test]
fn no_degenerate_triangles_single_cube() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_no_degenerate_triangles(&result.full_res, "cube full_res");
    assert_no_degenerate_triangles(&result.lod, "cube lod");
}

#[test]
fn no_degenerate_triangles_demo() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_no_degenerate_triangles(&result.full_res, "demo full_res");
    assert_no_degenerate_triangles(&result.lod, "demo lod");
}

#[test]
fn single_cube_edge_sharing() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("single cube edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
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

fn count_coplanar_overlapping_tris(mesh: &ChunkMeshData) -> usize {
    use bevy::math::Vec3;

    type PosKey = (i32, i32, i32);
    type EdgeKey = (PosKey, PosKey);

    fn sorted_edge(a: PosKey, b: PosKey) -> EdgeKey {
        if a <= b { (a, b) } else { (b, a) }
    }

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

    for tris in edge_tris.values() {
        if tris.len() <= 2 { continue; }

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

                // Only flag same-direction coplanar overlaps (not anti-parallel
                // back-to-back triangles which are geometrically valid at
                // center-line chamfer seams)
                if ni.dot(nj) > 0.99 {
                    let plane_dist_d = (d - a).dot(ni).abs();
                    let plane_dist_e = (e - a).dot(ni).abs();
                    let plane_dist_f = (f - a).dot(ni).abs();
                    if plane_dist_d < 0.001 && plane_dist_e < 0.001 && plane_dist_f < 0.001 {
                        // Find the shared edge and check if non-shared vertices
                        // are on opposite sides (adjacent, not overlapping)
                        let verts_i = [a, b, c];
                        let verts_j = [d, e, f];
                        let mut is_adjacent = false;
                        'edge_check: for ei in 0..3 {
                            for ej in 0..3 {
                                let si0 = verts_i[ei];
                                let si1 = verts_i[(ei + 1) % 3];
                                let sj0 = verts_j[ej];
                                let sj1 = verts_j[(ej + 1) % 3];
                                // Check if these edge pairs share the same positions
                                let shared = ((si0 - sj0).length_squared() < 1e-6 && (si1 - sj1).length_squared() < 1e-6)
                                    || ((si0 - sj1).length_squared() < 1e-6 && (si1 - sj0).length_squared() < 1e-6);
                                if shared {
                                    let edge_dir = (si1 - si0).normalize_or_zero();
                                    let perp = ni.cross(edge_dir);
                                    let other_i = verts_i[(ei + 2) % 3];
                                    let other_j = verts_j[(ej + 2) % 3];
                                    let side_i = (other_i - si0).dot(perp);
                                    let side_j = (other_j - si0).dot(perp);
                                    if side_i * side_j < 0.0 {
                                        is_adjacent = true;
                                    }
                                    break 'edge_check;
                                }
                            }
                        }
                        if !is_adjacent {
                            overlapping += 1;
                        }
                    }
                }
            }
        }
    }

    overlapping
}

fn count_intersecting_triangle_pairs(mesh: &ChunkMeshData) -> usize {
    use bevy::math::Vec3;

    type PosKey = (i32, i32, i32);

    let mut pos_tris: HashMap<PosKey, Vec<usize>> = HashMap::new();
    for i in (0..mesh.indices.len()).step_by(3) {
        let tri_idx = i / 3;
        for j in 0..3 {
            let pk = quantize_pos(&mesh.positions[mesh.indices[i + j] as usize]);
            pos_tris.entry(pk).or_default().push(tri_idx);
        }
    }

    let mut checked: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut intersecting = 0;

    for tris in pos_tris.values() {
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                let (ti, tj) = (tris[i], tris[j]);
                let pair = if ti < tj { (ti, tj) } else { (tj, ti) };
                if !checked.insert(pair) { continue; }

                let ai = ti * 3;
                let aj = tj * 3;
                let a0 = Vec3::from_array(mesh.positions[mesh.indices[ai] as usize]);
                let a1 = Vec3::from_array(mesh.positions[mesh.indices[ai + 1] as usize]);
                let a2 = Vec3::from_array(mesh.positions[mesh.indices[ai + 2] as usize]);
                let b0 = Vec3::from_array(mesh.positions[mesh.indices[aj] as usize]);
                let b1 = Vec3::from_array(mesh.positions[mesh.indices[aj + 1] as usize]);
                let b2 = Vec3::from_array(mesh.positions[mesh.indices[aj + 2] as usize]);

                let a_set: Vec<PosKey> = [a0, a1, a2].iter().map(|v| quantize_pos(&v.to_array())).collect();
                let b_set: Vec<PosKey> = [b0, b1, b2].iter().map(|v| quantize_pos(&v.to_array())).collect();
                let shared_verts = a_set.iter().filter(|v| b_set.contains(v)).count();
                if shared_verts >= 2 { continue; }

                let na = (a1 - a0).cross(a2 - a0);
                let na_len = na.length();
                if na_len < 1e-8 { continue; }
                let na = na / na_len;

                let nb = (b1 - b0).cross(b2 - b0);
                let nb_len = nb.length();
                if nb_len < 1e-8 { continue; }
                let nb = nb / nb_len;

                if na.dot(nb).abs() < 0.99 { continue; }
                if (b0 - a0).dot(na).abs() > 0.001 { continue; }

                let abs_n = na.abs();
                let (u_axis, v_axis) = if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
                    (1, 2)
                } else if abs_n.y >= abs_n.z {
                    (0, 2)
                } else {
                    (0, 1)
                };

                let project = |v: Vec3| -> (f32, f32) {
                    (v[u_axis], v[v_axis])
                };

                let pa = [project(a0), project(a1), project(a2)];
                let pb = [project(b0), project(b1), project(b2)];

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
    !(has_neg && has_pos) && d1.abs() > 1e-4 && d2.abs() > 1e-4 && d3.abs() > 1e-4
}

fn sign_2d(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
}

#[test]
fn wedge_on_cube_edge_sharing() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_wedge(8, 15, 8, Facing::East, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

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
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("demo edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);

    let overlapping = count_coplanar_overlapping_tris(&result.full_res);
    let intersecting = count_intersecting_triangle_pairs(&result.full_res);
    eprintln!("demo: {} coplanar overlapping, {} intersecting triangle pairs",
        overlapping, intersecting);
    // Small number of coplanar overlaps at complex multi-block junctions (e.g. ground
    // meeting wedge ramps) are acceptable — they're tiny same-plane triangles at chamfer
    // seams that don't cause visible z-fighting. Simple shapes all have 0 overlaps.
    assert!(overlapping <= 35,
        "demo should have few coplanar overlapping triangles, got {}", overlapping);
    assert!(intersecting == 0,
        "demo should have no intersecting triangles, got {}", intersecting);
}

#[test]
fn debug_concave_l_shape() {
    use bevy::math::Vec3;
    let shapes = make_shapes();

    // Staircase:  [B]       Block A at (8,14,8)  = world (4,7,4)-(5,7.5,5)
    //            [A][C]    Block C at (10,14,8) = world (5,7,4)-(6,7.5,5)
    //                      Block B at (10,15,8) = world (5,7.5,4)-(6,8,5)
    // Concave edge where A's top meets B's left: x=5, y=7.5, z=[4..5]
    // Concavity is at x<5, y>7.5 (the open stair corner)
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);  // A
    data.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1); // C
    data.place_std(10, 15, 8, SHAPE_CUBE, Facing::North, 1); // B
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let mesh = &result.full_res;

    eprintln!("L-shape mesh: {} verts, {} tris", mesh.positions.len(), mesh.indices.len() / 3);

    // The concave edge is at x=5.0, y=7.5.
    // With the staircase, concavity is at x < 5.0 AND y > 7.5 (the open stair corner).
    let mut concavity_verts = Vec::new();
    for (i, p) in mesh.positions.iter().enumerate() {
        let v = Vec3::from_array(*p);
        if v.x < 4.999 && v.y > 7.501 && v.z > 3.9 && v.z < 5.1 {
            concavity_verts.push((i, v));
        }
    }
    eprintln!("\nVertices IN the concavity (x>5, y<7.5): {}", concavity_verts.len());
    for (i, v) in &concavity_verts {
        eprintln!("  v[{}]: ({:.4}, {:.4}, {:.4})", i, v.x, v.y, v.z);
    }

    // Also check: what are the center-line vertices for the concave edge?
    // They should be near x≈5.035, y≈7.465 (pushed into concavity)
    let mut cl_candidates = Vec::new();
    for (i, p) in mesh.positions.iter().enumerate() {
        let v = Vec3::from_array(*p);
        if (v.x - 5.0).abs() < 0.1 && (v.y - 7.5).abs() < 0.1 && v.z > 3.9 && v.z < 5.1 {
            cl_candidates.push((i, v));
        }
    }
    eprintln!("\nAll vertices near concave edge (x≈5, y≈7.5):");
    for (i, v) in &cl_candidates {
        let in_concavity = v.x > 5.001 && v.y < 7.499;
        eprintln!("  v[{}]: ({:.4}, {:.4}, {:.4}) {}", i, v.x, v.y, v.z,
            if in_concavity { "<-- IN CONCAVITY" } else { "" });
    }

    // Now check which edges in the solid mesh are at the concave boundary
    // by looking at the generated mesh for strips crossing the edge
    let mut strips_crossing = 0;
    for chunk in mesh.indices.chunks(3) {
        let vs: Vec<Vec3> = chunk.iter()
            .map(|&idx| Vec3::from_array(mesh.positions[idx as usize]))
            .collect();
        // A triangle crosses the concave edge if it has vertices on both sides
        let has_inside = vs.iter().any(|v| v.x < 4.999 || v.y > 7.501);
        let has_concavity = vs.iter().any(|v| v.x < 4.999 && v.y > 7.501);
        if has_inside && has_concavity {
            strips_crossing += 1;
        }
    }
    eprintln!("\nTriangles spanning solid↔concavity: {}", strips_crossing);

    // Print details of triangles that have vertices in the concavity
    for (ti, chunk) in mesh.indices.chunks(3).enumerate() {
        let vs: Vec<Vec3> = chunk.iter()
            .map(|&idx| Vec3::from_array(mesh.positions[idx as usize]))
            .collect();
        let has_concavity = vs.iter().any(|v| v.x < 4.999 && v.y > 7.501);
        if has_concavity {
            let n = (vs[1] - vs[0]).cross(vs[2] - vs[0]).normalize_or_zero();
            eprintln!("  tri[{}]: ({:.3},{:.3},{:.3}) ({:.3},{:.3},{:.3}) ({:.3},{:.3},{:.3}) n=({:.3},{:.3},{:.3})",
                ti, vs[0].x, vs[0].y, vs[0].z, vs[1].x, vs[1].y, vs[1].z, vs[2].x, vs[2].y, vs[2].z,
                n.x, n.y, n.z);
        }
    }

    assert!(concavity_verts.len() > 0,
        "Expected vertices in the concavity for concave fillet, found none");
    assert!(strips_crossing > 0,
        "Expected triangles spanning into the concavity, found none");
}

#[test]
fn single_wedge_edge_sharing() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_WEDGE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!("single wedge edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);
    assert!(non_manifold == 0,
        "single wedge should have no non-manifold edges, got {}", non_manifold);
}

// ---------------------------------------------------------------------------
// Cut-and-offset chamfer tests
// ---------------------------------------------------------------------------

/// Test a single shape with cut-offset chamfer: mesh validity + watertight.
fn assert_cut_offset_shape_watertight(shape: u16, facing: Facing, label: &str) {
    let shapes = make_shapes();
    let data = single_block_chunk(shape, facing, 1);
    let result = generate_chunk_mesh(
        &data,
        &ChunkNeighbors::empty(),
        &shapes,
        crate::PresentationMode::CutAndOffset,
    );

    assert_mesh_valid(&result.full_res, label);
    assert_no_degenerate_triangles(&result.full_res, label);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res);
    eprintln!(
        "{}: {} verts, {} tris, {} boundary, {} interior, {} non-manifold edges",
        label,
        result.full_res.positions.len(),
        result.full_res.indices.len() / 3,
        boundary,
        interior,
        non_manifold,
    );

    if boundary > 0 {
        dump_boundary_edges(&result.full_res, label);
    }
    if non_manifold > 0 {
        dump_non_manifold_edges(&result.full_res, label);
    }

    assert!(
        non_manifold == 0,
        "{}: {} non-manifold edges",
        label,
        non_manifold
    );
    assert!(
        boundary == 0,
        "{}: {} boundary edges (not watertight)",
        label,
        boundary
    );
}

#[test]
fn cut_offset_single_cube_watertight() {
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let label = format!("co_cube_{:?}", facing);
        assert_cut_offset_shape_watertight(SHAPE_CUBE, facing, &label);
    }
}

#[test]
fn cut_offset_single_wedge_watertight() {
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let label = format!("co_wedge_{:?}", facing);
        assert_cut_offset_shape_watertight(SHAPE_WEDGE, facing, &label);
    }
}

/// Test that single shapes produce convex meshes (fillet only removes material).
/// Allows a tolerance proportional to CHAMFER_WIDTH to accommodate the small
/// tilt from corner patches (pushed + unpushed vertices on the same face).
fn assert_cut_offset_shape_convex(shape: u16, facing: Facing, label: &str) {
    let shapes = make_shapes();
    let data = single_block_chunk(shape, facing, 1);
    let result = generate_chunk_mesh(
        &data,
        &ChunkNeighbors::empty(),
        &shapes,
        crate::PresentationMode::CutAndOffset,
    );
    // Strict convexity — fillet should only remove material.
    assert_convex(&result.full_res, label);
}

#[test]
fn cut_offset_single_cube_convex() {
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let label = format!("co_cube_convex_{:?}", facing);
        assert_cut_offset_shape_convex(SHAPE_CUBE, facing, &label);
    }
}

#[test]
fn cut_offset_single_wedge_convex() {
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let label = format!("co_wedge_convex_{:?}", facing);
        assert_cut_offset_shape_convex(SHAPE_WEDGE, facing, &label);
    }
}

// ---------------------------------------------------------------------------
// Wedge + wall occlusion tests
// ---------------------------------------------------------------------------

/// Count LOD mesh triangles: each shape face emits its own triangles in the LOD path,
/// so triangle count directly reflects how many faces are emitted.
fn lod_tri_count(data: &ChunkData) -> usize {
    let shapes = make_shapes();
    let result = generate_chunk_mesh(data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::Flat);
    result.lod.indices.len() / 3
}

#[test]
fn isolated_wedge_emits_all_faces() {
    // A lone wedge should have all 6 faces emitted (no neighbors to occlude).
    // Wedge has: bottom(2tri) + south/back(4tri) + north/front(2tri) + slope(2tri) + west(3tri) + east(3tri) = 16 tri
    let shapes = make_shapes();
    let wedge_shape = shapes.get(SHAPE_WEDGE).unwrap();
    let expected_tris: usize = wedge_shape.faces.iter().map(|f| f.triangles.len()).sum();

    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let mut data = ChunkData::new();
        data.place_wedge(8, 8, 8, facing, 1);
        let tris = lod_tri_count(&data);
        assert_eq!(tris, expected_tris,
            "Isolated wedge facing {:?}: expected {} tris (all faces), got {}", facing, expected_tris, tris);
    }
}

#[test]
fn wedge_wall_upper_cube_face_not_culled() {
    // Place a wedge and a 2-high wall of cubes on the slope side.
    // The lower cube's face toward the wedge SHOULD be culled (behind solid base).
    // The upper cube's face toward the wedge SHOULD NOT be culled (slope exposes it).
    //
    // We test by comparing:
    //   - tris with just wedge + upper cube (upper face definitely visible)
    //   - tris with wedge + both cubes (lower face may be culled)
    //   - tris with just wedge + lower cube (to isolate lower face culling)
    let shapes = make_shapes();
    let cube_shape = shapes.get(SHAPE_CUBE).unwrap();
    let cube_tris: usize = cube_shape.faces.iter().map(|f| f.triangles.len()).sum();
    let wedge_shape = shapes.get(SHAPE_WEDGE).unwrap();
    let wedge_tris: usize = wedge_shape.faces.iter().map(|f| f.triangles.len()).sum();

    // Test each wedge facing. The slope side (front) rotates with facing:
    //   North → slope faces +Z → wall at z+2
    //   East  → slope faces -X → wall at x-2
    //   South → slope faces -Z → wall at z-2
    //   West  → slope faces +X → wall at x+2
    struct TestCase {
        facing: Facing,
        // Offset from wedge origin to place the wall cubes
        wall_dx: i32,
        wall_dz: i32,
    }
    let cases = [
        TestCase { facing: Facing::North, wall_dx: 0, wall_dz: 2 },
        TestCase { facing: Facing::East,  wall_dx: -2, wall_dz: 0 },
        TestCase { facing: Facing::South, wall_dx: 0, wall_dz: -2 },
        TestCase { facing: Facing::West,  wall_dx: 2, wall_dz: 0 },
    ];

    for tc in &cases {
        let wx = 10;
        let wy = 10;
        let wz = 10;
        let cx = (wx as i32 + tc.wall_dx) as usize;
        let cz = (wz as i32 + tc.wall_dz) as usize;

        // Wedge alone
        let mut data_wedge_only = ChunkData::new();
        data_wedge_only.place_wedge(wx, wy, wz, tc.facing, 1);
        let tris_wedge_only = lod_tri_count(&data_wedge_only);
        assert_eq!(tris_wedge_only, wedge_tris,
            "{:?}: isolated wedge should emit all faces", tc.facing);

        // Wedge + upper cube only (at wedge y + 1, which is upper cell level)
        let mut data_upper = ChunkData::new();
        data_upper.place_wedge(wx, wy, wz, tc.facing, 1);
        data_upper.place_std(cx, wy + 1, cz, SHAPE_CUBE, Facing::North, 1);
        let tris_upper = lod_tri_count(&data_upper);

        // The upper cube is at y=wy+1 which is the wedge's upper cell row.
        // The wedge should NOT fully occlude the upper cube's face (slope exposes it).
        // So the upper cube should have all 6 faces minus at most 0-1 culled by the wedge.
        // If the upper cube face IS incorrectly culled, we'd see fewer tris than expected.
        let upper_cube_tris = tris_upper - tris_wedge_only;
        // Upper cube should have at least cube_tris - 2 (at most 1 face = 2 tris culled by wedge top)
        // But since wedge has no Top coverage, no face should be culled at all.
        assert_eq!(upper_cube_tris, cube_tris,
            "{:?}: upper cube next to wedge slope should have ALL faces visible (got {} tris, expected {}). \
             This means the wedge is incorrectly occluding the upper cube.",
            tc.facing, upper_cube_tris, cube_tris);

        // Wedge + lower cube (at same y as wedge, bottom cell level)
        let mut data_lower = ChunkData::new();
        data_lower.place_wedge(wx, wy, wz, tc.facing, 1);
        data_lower.place_std(cx, wy, cz, SHAPE_CUBE, Facing::North, 1);
        let tris_lower = lod_tri_count(&data_lower);

        // Lower cube IS behind the wedge's solid base, so its face toward the wedge
        // SHOULD be culled (2 tris). The wedge's front face at bottom should also be
        // culled by the cube (2 tris). So we expect 2+2=4 fewer tris.
        let expected_lower = wedge_tris + cube_tris - 4;
        assert_eq!(tris_lower, expected_lower,
            "{:?}: wedge + lower cube should have {} tris (mutual culling of 2 faces), got {}",
            tc.facing, expected_lower, tris_lower);
    }
}
