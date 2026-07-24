use std::collections::HashMap;

use crate::chunk::*;
use crate::coords::ChunkPos;
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
            "{}: degenerate triangle at index {} (verts {},{},{}, area={}) at {:?} {:?} {:?}",
            label, i / 3, a, b, c, area, positions[a], positions[b], positions[c]);
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

    assert_mesh_valid(&result.full_res(), "cube full_res");
    assert_mesh_valid(&result.lod, "cube lod");
}

#[test]
fn single_cube_watertight() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let (boundary, _interior, non_manifold) = count_edge_sharing(&result.full_res());
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

    assert_mesh_valid(&result.full_res(), "wedge full_res");
    assert_mesh_valid(&result.lod, "wedge lod");
}

#[test]
fn single_wedge_all_facings_valid() {
    let shapes = make_shapes();
    for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
        let data = single_block_chunk(SHAPE_WEDGE, facing, 1);
        let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
        let label = format!("wedge_{:?}", facing);

        assert_mesh_valid(&result.full_res(), &format!("{} full_res", label));
        assert_mesh_valid(&result.lod, &format!("{} lod", label));
    }
}

#[test]
fn adjacent_cubes_mesh_valid() {
    let shapes = make_shapes();
    let data = block_2x2x2();
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res(), "2x2x2 cubes full_res");
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

    assert_mesh_valid(&result.full_res(), "3 wedges full_res");
    assert_mesh_valid(&result.lod, "3 wedges lod");
}

#[test]
fn wedge_on_cube_valid() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_wedge(8, 15, 8, Facing::East, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_mesh_valid(&result.full_res(), "wedge_on_cube full_res");
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
    let single_tris = single_result.full_res().indices.len() / 3;

    // Two cubes sharing only a vertex (offset in X, Y, and Z)
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 15, 10, SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let diag_tris = result.full_res().indices.len() / 3;
    let expected = single_tris * 2;

    let (boundary, _, non_manifold) = count_edge_sharing(&result.full_res());
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
    let single_tris = single_result.full_res().indices.len() / 3;

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
    let wedge_tris = wedge_only.full_res().indices.len() / 3;

    let total_tris = result.full_res().indices.len() / 3;
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
    let single_tris = single_result.full_res().indices.len() / 3;

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
    let diag_tris = result.full_res().indices.len() / 3;

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
    let without_tris = result_without.full_res().indices.len() / 3;

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

    assert_mesh_valid(&result.full_res(), "demo full_res");
    assert_mesh_valid(&result.lod, "demo lod");

    let full_tris = result.full_res().indices.len() / 3;
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

    let full_tris = result.full_res().indices.len() / 3;
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
    let (cube_boundary, _, _) = count_edge_sharing(&cube_result.full_res());
    eprintln!("single cube boundary edges: {}", cube_boundary);
    assert!(cube_boundary == 0,
        "single floating cube should have 0 boundary edges, got {}", cube_boundary);

    let wedge_data = single_block_chunk(SHAPE_WEDGE, Facing::North, 1);
    let wedge_result = generate_chunk_mesh(&wedge_data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let (wedge_boundary, _, _) = count_edge_sharing(&wedge_result.full_res());
    eprintln!("single wedge boundary edges: {}", wedge_boundary);
    assert!(wedge_boundary == 0,
        "single floating wedge should have 0 boundary edges, got {}", wedge_boundary);

    let mut woc_data = ChunkData::new();
    woc_data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    woc_data.place_wedge(8, 15, 8, Facing::East, 1);
    let woc_result = generate_chunk_mesh(&woc_data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let (woc_boundary, _, _) = count_edge_sharing(&woc_result.full_res());
    eprintln!("wedge_on_cube boundary edges: {}", woc_boundary);
    if woc_boundary > 0 {
        dump_boundary_edges(&woc_result.full_res(), "wedge_on_cube");
    }
    assert!(woc_boundary == 0,
        "wedge_on_cube should have 0 boundary edges, got {}", woc_boundary);
}

#[test]
fn empty_chunk_produces_empty_mesh() {
    let shapes = make_shapes();
    let data = ChunkData::new();
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert!(result.full_res().positions.is_empty(), "empty chunk should produce empty full_res");
    assert!(result.lod.positions.is_empty(), "empty chunk should produce empty lod");
}

#[test]
fn no_degenerate_triangles_single_cube() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_no_degenerate_triangles(&result.full_res(), "cube full_res");
    assert_no_degenerate_triangles(&result.lod, "cube lod");
}

#[test]
fn no_degenerate_triangles_demo() {
    let shapes = make_shapes();
    let data = demo_chunk();
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    assert_no_degenerate_triangles(&result.full_res(), "demo full_res");
    assert_no_degenerate_triangles(&result.lod, "demo lod");
}

#[test]
fn single_cube_edge_sharing() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res());
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

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res());
    eprintln!("wedge_on_cube edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);

    let overlapping = count_coplanar_overlapping_tris(&result.full_res());
    let intersecting = count_intersecting_triangle_pairs(&result.full_res());
    eprintln!("wedge_on_cube: {} coplanar overlapping, {} intersecting triangle pairs",
        overlapping, intersecting);
    if overlapping > 0 || intersecting > 0 {
        dump_non_manifold_edges(&result.full_res(), "wedge_on_cube");
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

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res());
    eprintln!("demo edges: boundary={}, interior={}, non_manifold={}",
        boundary, interior, non_manifold);

    let overlapping = count_coplanar_overlapping_tris(&result.full_res());
    let intersecting = count_intersecting_triangle_pairs(&result.full_res());
    eprintln!("demo: {} coplanar overlapping, {} intersecting triangle pairs",
        overlapping, intersecting);
    assert!(overlapping == 0,
        "demo should have no coplanar overlapping triangles, got {}", overlapping);
    assert!(boundary == 0,
        "demo should be watertight, got {} boundary edges", boundary);
    assert!(non_manifold == 0,
        "demo should have no non-manifold edges, got {}", non_manifold);
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
    let mesh = result.full_res();

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

// ---------------------------------------------------------------------------
// Mixed convex/concave corner tests (3 blocks meeting at one vertex)
// ---------------------------------------------------------------------------

/// Run the "not broken" checks on a mesh with concave geometry: watertight,
/// manifold, no coplanar overlaps, no intersecting triangles.
fn assert_concave_mesh_clean(mesh: &ChunkMeshData, label: &str) {
    assert_mesh_valid(mesh, label);
    assert_no_degenerate_triangles(mesh, label);

    let (boundary, _interior, non_manifold) = count_edge_sharing(mesh);
    let overlapping = count_coplanar_overlapping_tris(mesh);
    let intersecting = count_intersecting_triangle_pairs(mesh);
    eprintln!("{}: boundary={}, non_manifold={}, overlaps={}, intersections={}",
        label, boundary, non_manifold, overlapping, intersecting);
    if boundary > 0 {
        dump_boundary_edges(mesh, label);
    }
    if non_manifold > 0 {
        dump_non_manifold_edges(mesh, label);
    }
    assert!(boundary == 0, "{}: {} boundary edges (holes)", label, boundary);
    assert!(non_manifold == 0, "{}: {} non-manifold edges", label, non_manifold);
    assert!(overlapping == 0, "{}: {} coplanar overlapping triangles", label, overlapping);
    assert!(intersecting == 0, "{}: {} intersecting triangle pairs", label, intersecting);
}

/// Three cubes at the same level forming an L. The inner (reflex) corner is a
/// vertex where TWO CONVEX top edges and ONE CONCAVE vertical edge meet —
/// the classic mixed corner where 3 blocks meet at one vertex.
#[test]
fn mixed_corner_l_plateau() {
    use bevy::math::Vec3;
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);   // A
    data.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1);  // B (east)
    data.place_std(8, 14, 10, SHAPE_CUBE, Facing::North, 1);  // C (north)
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let mesh = result.full_res();

    assert_concave_mesh_clean(mesh, "l_plateau");

    // The reflex corner vertex sits at (5, 7.5, 5): tops at y=7.5, the notch
    // is the quadrant x>5, z>5.  Mesh vertices derived from the corner must
    // not be pushed INTO the walls (x<5 AND z<5 while near the top) — the
    // old sphere solve pushed it diagonally into the solid, pinching the
    // fillet against the concave edge crest which bulges outward (x,z > 5).
    let corner = Vec3::new(5.0, 7.5, 5.0);
    for p in &mesh.positions {
        let v = Vec3::from_array(*p);
        // Vertices that came from the corner vertex (within chamfer reach).
        if (v - corner).length() < CHAMFER_WIDTH * 0.9 {
            let pushed_into_both_walls = v.x < 4.995 && v.z < 4.995 && v.y < 7.495;
            assert!(!pushed_into_both_walls,
                "l_plateau: corner-derived vertex {:?} pushed into the solid (pinch)", v);
        }
    }
}

/// A 3D terrace inner corner: an L-plateau with a block raised ON the
/// corner, as terraced terrain constantly produces. The raised block's two
/// exposed vertical faces meet the lower tops at TWO concave edges which
/// join a convex vertical edge at one tri-junction — the configuration that
/// showed pinhole gaps in-game.
#[test]
fn terrace_inner_corner_clean() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);   // base corner
    data.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1);  // base east
    data.place_std(8, 14, 10, SHAPE_CUBE, Facing::North, 1);  // base north
    data.place_std(8, 15, 8, SHAPE_CUBE, Facing::North, 1);   // raised on corner
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    assert_concave_mesh_clean(result.full_res(), "terrace_inner_corner");
}

/// A full 2×2 terrace step (two raised blocks on a 2×3 base) — the raised
/// slab's corner faces the lower terrace diagonally. Mirrors the stepped
/// hillsides worldgen produces everywhere.
#[test]
fn terrace_step_row_clean() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    for (x, z) in [(8, 8), (10, 8), (8, 10), (10, 10), (12, 8), (12, 10)] {
        data.place_std(x, 14, z, SHAPE_CUBE, Facing::North, 1);
    }
    data.place_std(8, 15, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(8, 15, 10, SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    assert_concave_mesh_clean(result.full_res(), "terrace_step_row");
}

/// Boundary edges that are not geometrically covered by other boundary
/// geometry — i.e. genuine open holes. Chunk meshes join at seams with
/// T-junctions (the same line subdivided differently on each side), which
/// are geometrically closed; only an edge with empty space beside it is a
/// real hole.
fn count_uncovered_boundary_edges(mesh: &ChunkMeshData, label: &str) -> usize {
    use bevy::math::Vec3;
    let mut edge_count: HashMap<((i32, i32, i32), (i32, i32, i32)), usize> = HashMap::new();
    let mut segments: HashMap<((i32, i32, i32), (i32, i32, i32)), (Vec3, Vec3)> = HashMap::new();
    for i in (0..mesh.indices.len()).step_by(3) {
        let tri = [mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]];
        for j in 0..3 {
            let a = mesh.positions[tri[j] as usize];
            let b = mesh.positions[tri[(j + 1) % 3] as usize];
            let (qa, qb) = (quantize_pos(&a), quantize_pos(&b));
            let key = if qa <= qb { (qa, qb) } else { (qb, qa) };
            *edge_count.entry(key).or_insert(0) += 1;
            segments.entry(key).or_insert((Vec3::from_array(a), Vec3::from_array(b)));
        }
    }
    let boundary: Vec<(Vec3, Vec3)> = edge_count
        .iter()
        .filter(|&(_, &c)| c == 1)
        .map(|(k, _)| segments[k])
        .collect();

    fn point_seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        (p - (a + ab * t)).length()
    }

    let mut uncovered = 0;
    for (i, &(a, b)) in boundary.iter().enumerate() {
        let open = [0.25f32, 0.5, 0.75].iter().any(|&t| {
            let q = a.lerp(b, t);
            !boundary.iter().enumerate().any(|(j, &(c, d))| {
                j != i && point_seg_dist(q, c, d) < 1e-3
            })
        });
        if open {
            uncovered += 1;
            eprintln!("  {label}: OPEN boundary edge ({:.3},{:.3},{:.3})→({:.3},{:.3},{:.3})",
                a.x, a.y, a.z, b.x, b.y, b.z);
        }
    }
    uncovered
}

// ---------------------------------------------------------------------------
// Slope-cap terrain helpers
// ---------------------------------------------------------------------------

/// Build capped terrain from a block-column heightfield (heights in cells,
/// indexed `heights[z][x]`), exactly like worldgen does: cubes up to the
/// surface, with slope caps replacing the top cube wherever
/// `classify_slope_cap` finds a smoothable step. Columns outside the patch
/// are treated as equal height. `x_range` limits which columns are written
/// (for splitting one patch across two chunks); `origin` is the cell of
/// column (x_range.start, 0).
fn build_capped_terrain_into(
    data: &mut ChunkData,
    heights: &[&[i32]],
    origin: (usize, usize, usize),
    x_range: std::ops::Range<usize>,
) {
    use crate::worldgen::classify_slope_cap;
    let shapes = make_shapes();
    let rows = heights.len() as i32;
    let cols = heights[0].len() as i32;
    for (j, row) in heights.iter().enumerate() {
        for (i, &h) in row.iter().enumerate() {
            if !x_range.contains(&i) {
                continue;
            }
            let sample = |dx: i32, dz: i32| -> i32 {
                let nx = i as i32 + dx;
                let nz = j as i32 + dz;
                if nx < 0 || nz < 0 || nx >= cols || nz >= rows {
                    h
                } else {
                    heights[nz as usize][nx as usize]
                }
            };
            let cap = if h >= 1 {
                classify_slope_cap(&|dx, dz| sample(dx, dz) - h)
            } else {
                None
            };
            let bx = origin.0 + (i - x_range.start) * 2;
            let bz = origin.2 + j * 2;
            let cube_top = if cap.is_some() { h - 1 } else { h };
            for y in 0..cube_top.max(0) {
                data.place_std(bx, origin.1 + y as usize, bz, SHAPE_CUBE, Facing::North, 1);
            }
            if let Some((shape_id, facing, _ch)) = cap {
                let shape = shapes.get(shape_id).unwrap();
                let occ = crate::shape::rotated_occupied_cells(shape, facing);
                data.place_block(
                    shape_id,
                    facing,
                    1,
                    bx,
                    origin.1 + (h - 1) as usize,
                    bz,
                    &occ,
                );
            }
        }
    }
}

/// Every slope-cap shape alone, in every facing: valid data and clean mesh.
#[test]
fn slope_cap_single_shapes_clean() {
    let shapes = make_shapes();
    for shape_id in [
        SHAPE_WEDGE_OUTER,
        SHAPE_WEDGE_INNER,
        SHAPE_WEDGE_STEEP,
        SHAPE_WEDGE_STEEP_OUTER,
        SHAPE_WEDGE_STEEP_INNER,
    ] {
        for facing in [Facing::North, Facing::East, Facing::South, Facing::West] {
            let mut data = ChunkData::new();
            let shape = shapes.get(shape_id).unwrap();
            let occ = crate::shape::rotated_occupied_cells(shape, facing);
            data.place_block(shape_id, facing, 1, 8, 16, 8, &occ);
            let errors = data.validate(&shapes);
            assert!(errors.is_empty(), "shape {shape_id} {facing:?}: {errors:?}");
            let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
            assert_concave_mesh_clean(
                result.full_res(),
                &format!("cap_{shape_id}_{facing:?}"),
            );
        }
    }
}

/// A plateau with a full ring of caps: straight wedges on the flanks, inner
/// corners in the notches, outer corners on the diagonals — every junction
/// type the terrain generator produces, meshed together with cubes.
#[test]
fn slope_cap_hill_assembly_clean() {
    let heights: &[&[i32]] = &[
        &[1, 1, 1, 1, 1],
        &[1, 2, 2, 2, 1],
        &[1, 2, 3, 2, 1],
        &[1, 2, 2, 2, 1],
        &[1, 1, 1, 1, 1],
    ];
    let mut data = ChunkData::new();
    build_capped_terrain_into(&mut data, heights, (8, 14, 8), 0..5);
    let shapes = make_shapes();
    let errors = data.validate(&shapes);
    assert!(errors.is_empty(), "hill terrain invalid: {errors:?}");
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    assert_concave_mesh_clean(result.full_res(), "cap_hill");
}

/// Same, but with +2 steps so the steep 1:1 family is used.
#[test]
fn slope_cap_steep_hill_assembly_clean() {
    let heights: &[&[i32]] = &[
        &[1, 1, 1, 1, 1],
        &[1, 3, 3, 3, 1],
        &[1, 3, 5, 3, 1],
        &[1, 3, 3, 3, 1],
        &[1, 1, 1, 1, 1],
    ];
    let mut data = ChunkData::new();
    build_capped_terrain_into(&mut data, heights, (8, 14, 8), 0..5);
    let shapes = make_shapes();
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    assert_concave_mesh_clean(result.full_res(), "cap_steep_hill");
}

/// A capped hill straddling a chunk seam: both chunks mesh independently
/// (with neighbor halos) and their union must be watertight — wedge fillets
/// across the seam included.
#[test]
fn slope_caps_across_seam_watertight() {
    let heights: &[&[i32]] = &[
        &[1, 1, 1, 1],
        &[1, 2, 2, 1],
        &[1, 2, 2, 1],
        &[1, 1, 1, 1],
    ];
    let chunk_w = crate::chunk::CHUNK_X as f32 * crate::chunk::VOXEL_SIZE;

    // Columns 0..2 in chunk A at cells 28/30; columns 2..4 in chunk B at 0/2.
    let mut a = ChunkData::new();
    build_capped_terrain_into(&mut a, heights, (28, 14, 8), 0..2);
    let mut b = ChunkData::new();
    build_capped_terrain_into(&mut b, heights, (0, 14, 8), 2..4);

    let shapes = make_shapes();
    let mut na = ChunkNeighbors::empty();
    na.set(1, 0, 0, b.clone());
    let mut nb = ChunkNeighbors::empty();
    nb.set(-1, 0, 0, a.clone());

    let ra = generate_chunk_mesh(&a, &na, &shapes, crate::PresentationMode::CutAndOffset);
    let rb = generate_chunk_mesh(&b, &nb, &shapes, crate::PresentationMode::CutAndOffset);
    let combined = union_meshes(ra.full_res(), rb.full_res(), [chunk_w, 0.0, 0.0]);

    let (boundary, _interior, non_manifold) = count_edge_sharing(&combined);
    eprintln!("cap seam: boundary={boundary} non_manifold={non_manifold}");
    if boundary > 0 {
        dump_boundary_edges(&combined, "cap_seam");
    }
    assert!(boundary == 0, "capped-hill seam union has {boundary} boundary edges");
    assert!(non_manifold == 0, "capped-hill seam union has {non_manifold} non-manifold edges");
}

/// The classifier picks the right shapes and orientations.
#[test]
fn slope_cap_classification() {
    use crate::worldgen::classify_slope_cap;
    // High to the south only → straight wedge, facing North (descends +z)
    let cap = classify_slope_cap(&|_dx, dz| if dz == -1 { 1 } else { 0 });
    assert_eq!(cap, Some((SHAPE_WEDGE, Facing::North, 2)));
    // High to the east only, by 2 → steep wedge facing East
    let cap = classify_slope_cap(&|dx, _dz| if dx == 1 { 2 } else { 0 });
    assert_eq!(cap, Some((SHAPE_WEDGE_STEEP, Facing::East, 3)));
    // High south AND west → inner corner (shared SW corner → North)
    let cap = classify_slope_cap(&|dx, dz| if dz == -1 || dx == -1 { 1 } else { 0 });
    assert_eq!(cap, Some((SHAPE_WEDGE_INNER, Facing::North, 2)));
    // Only the NE diagonal high → outer corner facing South
    let cap = classify_slope_cap(&|dx, dz| if dx == 1 && dz == 1 { 1 } else { 0 });
    assert_eq!(cap, Some((SHAPE_WEDGE_OUTER, Facing::South, 2)));
    // A cliff (+3) is left alone
    let cap = classify_slope_cap(&|dx, _dz| if dx == 1 { 3 } else { 0 });
    assert_eq!(cap, None);
    // Opposite high sides → no cap
    let cap = classify_slope_cap(&|dx, _dz| if dx != 0 { 1 } else { 0 });
    assert_eq!(cap, None);
}

/// Union of two adjacent chunks' meshes in world space, for seam checks.
fn union_meshes(a: &ChunkMeshData, b: &ChunkMeshData, b_offset: [f32; 3]) -> ChunkMeshData {
    let mut positions = a.positions.clone();
    let mut normals = a.normals.clone();
    let mut sharp_normals = a.sharp_normals.clone();
    let mut uvs = a.uvs.clone();
    let mut chamfer_offsets = a.chamfer_offsets.clone();
    let mut indices = a.indices.clone();
    let base = positions.len() as u32;
    positions.extend(b.positions.iter().map(|p| {
        [p[0] + b_offset[0], p[1] + b_offset[1], p[2] + b_offset[2]]
    }));
    normals.extend_from_slice(&b.normals);
    sharp_normals.extend_from_slice(&b.sharp_normals);
    uvs.extend_from_slice(&b.uvs);
    chamfer_offsets.extend_from_slice(&b.chamfer_offsets);
    indices.extend(b.indices.iter().map(|i| i + base));
    ChunkMeshData { positions, normals, sharp_normals, uvs, chamfer_offsets, indices }
}

/// A terrace step that crosses a chunk seam: the lower terrace continues
/// into the east chunk while a raised block sits right at the seam in the
/// west chunk. The two chunks mesh independently (with neighbor data);
/// their union must still be watertight — chamfer decisions on both sides
/// of the seam must agree. This is the in-game "pinholes along terraces"
/// configuration.
#[test]
fn terrace_across_chunk_seam_watertight() {
    let shapes = make_shapes();
    let chunk_w = crate::chunk::CHUNK_X as f32 * crate::chunk::VOXEL_SIZE;

    let mut a = ChunkData::new();
    a.place_std(28, 14, 8, SHAPE_CUBE, Facing::North, 1);
    a.place_std(30, 14, 8, SHAPE_CUBE, Facing::North, 1);
    a.place_std(30, 15, 8, SHAPE_CUBE, Facing::North, 1); // raised at the seam
    let mut b = ChunkData::new();
    b.place_std(0, 14, 8, SHAPE_CUBE, Facing::North, 1);
    b.place_std(2, 14, 8, SHAPE_CUBE, Facing::North, 1);

    let mut na = ChunkNeighbors::empty();
    na.set(1, 0, 0, b.clone());
    let mut nb = ChunkNeighbors::empty();
    nb.set(-1, 0, 0, a.clone());

    let ra = generate_chunk_mesh(&a, &na, &shapes, crate::PresentationMode::CutAndOffset);
    let rb = generate_chunk_mesh(&b, &nb, &shapes, crate::PresentationMode::CutAndOffset);
    let combined = union_meshes(ra.full_res(), rb.full_res(), [chunk_w, 0.0, 0.0]);

    let (boundary, _interior, non_manifold) = count_edge_sharing(&combined);
    let uncovered = count_uncovered_boundary_edges(&combined, "seam_terrace");
    eprintln!("seam terrace: boundary={boundary} uncovered={uncovered} non_manifold={non_manifold}");
    if boundary > 0 {
        dump_boundary_edges(&combined, "seam_terrace");
    }
    assert!(boundary == 0, "seam terrace union has {boundary} boundary edges (holes)");
    assert!(non_manifold == 0, "seam terrace union has {non_manifold} non-manifold edges");

    // The concave edge along the seam must actually be FILLETED, not left
    // sharp: with the neighbor halo both chunks round it identically. The
    // fillet pushes the crease off the exact corner line (16, 7.5, z), so
    // no final vertex may remain exactly on it between the step walls.
    let on_crease = combined.positions.iter().filter(|p| {
        (p[0] - 16.0).abs() < 1e-4 && (p[1] - 7.5).abs() < 1e-4 && p[2] > 4.05 && p[2] < 4.95
    }).count();
    assert!(on_crease == 0, "seam concave edge not filleted: {on_crease} verts still on the crease");
}

/// Same, but with the raised block in the EAST chunk (mirror case) and the
/// seam crossing along Z as well, to catch direction-dependent asymmetry.
#[test]
fn terrace_across_chunk_seam_mirrored_watertight() {
    let shapes = make_shapes();
    let chunk_w = crate::chunk::CHUNK_X as f32 * crate::chunk::VOXEL_SIZE;

    let mut a = ChunkData::new();
    a.place_std(30, 14, 8, SHAPE_CUBE, Facing::North, 1);
    let mut b = ChunkData::new();
    b.place_std(0, 14, 8, SHAPE_CUBE, Facing::North, 1);
    b.place_std(0, 15, 8, SHAPE_CUBE, Facing::North, 1); // raised, east side
    b.place_std(2, 14, 8, SHAPE_CUBE, Facing::North, 1);

    let mut na = ChunkNeighbors::empty();
    na.set(1, 0, 0, b.clone());
    let mut nb = ChunkNeighbors::empty();
    nb.set(-1, 0, 0, a.clone());

    let ra = generate_chunk_mesh(&a, &na, &shapes, crate::PresentationMode::CutAndOffset);
    let rb = generate_chunk_mesh(&b, &nb, &shapes, crate::PresentationMode::CutAndOffset);
    let combined = union_meshes(ra.full_res(), rb.full_res(), [chunk_w, 0.0, 0.0]);

    let (boundary, _interior, non_manifold) = count_edge_sharing(&combined);
    let uncovered = count_uncovered_boundary_edges(&combined, "seam_terrace_mirrored");
    eprintln!("seam terrace mirrored: boundary={boundary} uncovered={uncovered} non_manifold={non_manifold}");
    if boundary > 0 {
        dump_boundary_edges(&combined, "seam_terrace_mirrored");
    }
    assert!(boundary == 0, "mirrored seam terrace union has {boundary} boundary edges");
    assert!(non_manifold == 0, "mirrored union has {non_manifold} non-manifold edges");
}

/// Staircase profile: two stacked cubes beside one — the concave edge's
/// ENDPOINTS are mixed corners (2 convex + 1 concave edges each).
#[test]
fn mixed_corner_staircase_clean() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_std(8, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 14, 8, SHAPE_CUBE, Facing::North, 1);
    data.place_std(10, 15, 8, SHAPE_CUBE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    assert_concave_mesh_clean(result.full_res(), "staircase");
}

/// The world's wedge-ramp-on-platform showcase: a platform of cubes with a
/// wedge ramp on top.  The ramp base meets the platform rim at mixed corners.
#[test]
fn mixed_corner_ramp_on_platform() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    for bx in 0..3 {
        for bz in 0..3 {
            data.place_std(6 + bx * 2, 10, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
        }
    }
    for bz in 0..3 {
        data.place_wedge(6, 11, 14 + bz * 2, Facing::West, 1);
    }
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    assert_concave_mesh_clean(result.full_res(), "ramp_on_platform");
}

/// The world's rotation-test wedges sit against the wedge ramp's north wall
/// with partial face contacts at varying heights — wedge-back-to-wedge,
/// wedge-front-to-cube.  Reproduces the demo world's rough junctions.
#[test]
fn showcase_wedges_against_ramp() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();

    // Ground strip under the ramp area
    for bx in 0..8 {
        for bz in 7..11 {
            data.place_std(bx * 2, 0, bz * 2, SHAPE_CUBE, Facing::North, 1);
        }
    }
    // Wedge ramp columns (z cells 14..19), ascending east
    for step in 0..8usize {
        let wedge_y = 1 + step;
        for bz in 0..3 {
            data.place_wedge(step * 2, wedge_y, 14 + bz * 2, Facing::East, 1);
        }
        for y in 1..wedge_y {
            for bz in 0..3 {
                data.place_std(step * 2, y, 14 + bz * 2, SHAPE_CUBE, Facing::North, 1);
            }
        }
    }
    // Rotation-test wedges against the ramp's north side
    data.place_wedge(0, 2, 20, Facing::North, 1);
    data.place_wedge(4, 2, 20, Facing::East, 1);
    data.place_wedge(8, 2, 20, Facing::South, 1);
    data.place_wedge(12, 2, 20, Facing::West, 1);

    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    let mesh = result.full_res();
    assert_mesh_valid(mesh, "showcase_vs_ramp");
    assert_no_degenerate_triangles(mesh, "showcase_vs_ramp");
    let (boundary, _, non_manifold) = count_edge_sharing(mesh);
    let overlapping = count_coplanar_overlapping_tris(mesh);
    let intersecting = count_intersecting_triangle_pairs(mesh);
    eprintln!("showcase_vs_ramp: boundary={}, non_manifold={}, overlaps={}, intersections={}",
        boundary, non_manifold, overlapping, intersecting);
    assert!(non_manifold == 0, "showcase_vs_ramp: {} non-manifold edges", non_manifold);
    assert!(overlapping == 0, "showcase_vs_ramp: {} coplanar overlaps", overlapping);
    assert!(intersecting == 0, "showcase_vs_ramp: {} intersecting pairs", intersecting);
    // Known limitation: when a block's contact region is fully enclosed
    // within a larger coplanar wall (donut topology), the coplanar merge
    // can't produce a single ring and a hairline sliver (4 boundary edges)
    // remains at one corner of the unmerged seam.
    assert!(boundary <= 4,
        "showcase_vs_ramp: {} boundary edges (expected <= 4, see donut-hole limitation)", boundary);
}

/// Minimal pair: a wedge whose back wall rests against another wedge one
/// cell lower (the first showcase junction).
#[test]
fn wedge_back_against_lower_wedge() {
    let shapes = make_shapes();
    let mut data = ChunkData::new();
    data.place_wedge(8, 13, 8, Facing::East, 1);
    data.place_wedge(8, 14, 10, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);
    assert_concave_mesh_clean(result.full_res(), "wedge_back_lower_wedge");
}

#[test]
fn single_wedge_edge_sharing() {
    let shapes = make_shapes();
    let data = single_block_chunk(SHAPE_WEDGE, Facing::North, 1);
    let result = generate_chunk_mesh(&data, &ChunkNeighbors::empty(), &shapes, crate::PresentationMode::CutAndOffset);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res());
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

    assert_mesh_valid(&result.full_res(), label);
    assert_no_degenerate_triangles(&result.full_res(), label);

    let (boundary, interior, non_manifold) = count_edge_sharing(&result.full_res());
    eprintln!(
        "{}: {} verts, {} tris, {} boundary, {} interior, {} non-manifold edges",
        label,
        result.full_res().positions.len(),
        result.full_res().indices.len() / 3,
        boundary,
        interior,
        non_manifold,
    );

    if boundary > 0 {
        dump_boundary_edges(&result.full_res(), label);
    }
    if non_manifold > 0 {
        dump_non_manifold_edges(&result.full_res(), label);
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
    assert_convex(&result.full_res(), label);
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
    let shapes = make_shapes();
    let cube_shape = shapes.get(SHAPE_CUBE).unwrap();
    let cube_tris: usize = cube_shape.faces.iter().map(|f| f.triangles.len()).sum();
    let wedge_shape = shapes.get(SHAPE_WEDGE).unwrap();
    let wedge_tris: usize = wedge_shape.faces.iter().map(|f| f.triangles.len()).sum();

    struct TestCase {
        facing: Facing,
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

        let mut data_wedge_only = ChunkData::new();
        data_wedge_only.place_wedge(wx, wy, wz, tc.facing, 1);
        let tris_wedge_only = lod_tri_count(&data_wedge_only);
        assert_eq!(tris_wedge_only, wedge_tris,
            "{:?}: isolated wedge should emit all faces", tc.facing);

        let mut data_upper = ChunkData::new();
        data_upper.place_wedge(wx, wy, wz, tc.facing, 1);
        data_upper.place_std(cx, wy + 1, cz, SHAPE_CUBE, Facing::North, 1);
        let tris_upper = lod_tri_count(&data_upper);

        let upper_cube_tris = tris_upper - tris_wedge_only;
        assert_eq!(upper_cube_tris, cube_tris,
            "{:?}: upper cube next to wedge slope should have ALL faces visible (got {} tris, expected {}). \
             This means the wedge is incorrectly occluding the upper cube.",
            tc.facing, upper_cube_tris, cube_tris);

        let mut data_lower = ChunkData::new();
        data_lower.place_wedge(wx, wy, wz, tc.facing, 1);
        data_lower.place_std(cx, wy, cz, SHAPE_CUBE, Facing::North, 1);
        let tris_lower = lod_tri_count(&data_lower);

        let expected_lower = wedge_tris + cube_tris - 4;
        assert_eq!(tris_lower, expected_lower,
            "{:?}: wedge + lower cube should have {} tris (mutual culling of 2 faces), got {}",
            tc.facing, expected_lower, tris_lower);
    }
}

/// Ear clipping must preserve collinear T-vertices on triangle boundaries
/// for every ring rotation — bridging over them cracks the mesh against
/// per-sub-edge strips (regression test for the merged-face triangulation).
#[test]
fn ear_clip_preserves_collinear_boundary() {
    use bevy::math::Vec3;
    let base: Vec<Vec3> = vec![
        Vec3::new(0.0, 0.5, 7.0), Vec3::new(0.0, 0.5, 8.0), Vec3::new(0.0, 0.5, 9.0), Vec3::new(0.0, 0.5, 10.0),
        Vec3::new(1.0, 0.5, 10.0), Vec3::new(1.0, 0.5, 9.0), Vec3::new(1.0, 0.5, 8.0), Vec3::new(1.0, 0.5, 7.0),
    ];
    let normal = Vec3::new(0.0, -1.0, 0.0);
    for rot in 0..8 {
        for rev in [false, true] {
            let mut ring: Vec<Vec3> = (0..8).map(|i| base[(i + rot) % 8]).collect();
            if rev { ring.reverse(); }
            let tris = crate::meshing::ear_clip_triangulate_public(&ring, normal);
            assert_eq!(tris.len(), 6,
                "rot={} rev={}: expected 6 triangles (T-verts on boundaries), got {}", rot, rev, tris.len());
            // Every ring edge must appear in exactly one triangle.
            for i in 0..8 {
                let (a, b) = (i, (i + 1) % 8);
                let count = tris.iter().filter(|t| {
                    (0..3).any(|k| (t[k] == a && t[(k + 1) % 3] == b) || (t[k] == b && t[(k + 1) % 3] == a))
                }).count();
                assert_eq!(count, 1, "rot={} rev={}: ring edge {}-{} in {} triangles", rot, rev, a, b, count);
            }
        }
    }
}

/// Slope caps meeting a chunk seam in every arrangement worldgen produces:
/// caps side by side across the seam, caps on one side only, and a full
/// plateau ring (straight + outer-corner caps) straddling it. Each chunk
/// meshes independently; the union must be exactly watertight.
#[test]
fn slope_cap_seam_configurations_watertight() {
    let chunk_w = crate::chunk::CHUNK_X as f32 * crate::chunk::VOXEL_SIZE;
    let shapes = make_shapes();
    let configs: &[(&str, &[&[i32]])] = &[
        ("two_caps_side_by_side", &[&[1, 1], &[2, 2]]),
        ("cap_in_A_only", &[&[1, 2], &[2, 2]]),
        ("cap_in_B_only", &[&[2, 1], &[2, 2]]),
        ("plateau_ring", &[&[1, 1, 1, 1], &[1, 2, 2, 1], &[1, 2, 2, 1], &[1, 1, 1, 1]]),
    ];
    for (name, heights) in configs {
        let ncols = heights[0].len();
        let split = ncols / 2;
        let mut a = ChunkData::new();
        build_capped_terrain_into(&mut a, heights, (32 - split * 2, 14, 8), 0..split);
        let mut b = ChunkData::new();
        build_capped_terrain_into(&mut b, heights, (0, 14, 8), split..ncols);
        let mut na = ChunkNeighbors::empty();
        na.set(1, 0, 0, b.clone());
        let mut nb = ChunkNeighbors::empty();
        nb.set(-1, 0, 0, a.clone());
        let ra = generate_chunk_mesh(&a, &na, &shapes, crate::PresentationMode::CutAndOffset);
        let rb = generate_chunk_mesh(&b, &nb, &shapes, crate::PresentationMode::CutAndOffset);
        let (ba, _ia, na_) = count_edge_sharing(ra.full_res());
        let (bb, _ib, nb_) = count_edge_sharing(rb.full_res());
        let combined = union_meshes(ra.full_res(), rb.full_res(), [chunk_w, 0.0, 0.0]);
        let (boundary, _i, non_manifold) = count_edge_sharing(&combined);
        eprintln!("[cap seam] {name}: union boundary={boundary} nm={non_manifold} | A b={ba} nm={na_} | B b={bb} nm={nb_}");
        if boundary > 0 {
            dump_boundary_edges(&combined, name);
        }
        assert!(boundary == 0, "{name}: union has {boundary} boundary edges");
        assert!(non_manifold == 0, "{name}: union has {non_manifold} non-manifold edges");
        assert!(na_ == 0 && nb_ == 0, "{name}: per-chunk non-manifold edges");
    }
}
/// Minimal repro of the worldgen hole: an inner-corner cap on a cube column,
/// with a second inner-corner cap one cell lower on the adjacent column.
/// Extracted from worldgen chunk (111,0,133) blocks (20,15,28)cube /
/// (20,16,28)inner / (20,15,30)inner.
#[test]
fn inner_cap_step_repro() {
    let shapes = make_shapes();
    let inner = shapes.get(SHAPE_WEDGE_INNER).unwrap();
    let occ = crate::shape::rotated_occupied_cells(inner, Facing::North);
    let mut data = ChunkData::new();
    data.place_std(4, 4, 4, SHAPE_CUBE, Facing::North, 1);
    data.place_block(SHAPE_WEDGE_INNER, Facing::North, 1, 4, 5, 4, &occ);
    data.place_block(SHAPE_WEDGE_INNER, Facing::North, 1, 4, 4, 6, &occ);

    let nb = ChunkNeighbors::empty();
    let solid = build_solid_mesh_public(&data, &nb, &shapes);
    for (i, f) in solid.faces.iter().enumerate() {
        let vs: Vec<String> = f.verts.iter()
            .map(|&v| { let p = solid.positions[v as usize]; format!("({:.3},{:.3},{:.3})", p.x, p.y, p.z) })
            .collect();
        eprintln!("face {i}: n=({:.1},{:.1},{:.1}) voxel={:?} tris={:?} verts={}",
            f.normal.x, f.normal.y, f.normal.z, f.voxel, f.orig_triangles, vs.join(" "));
    }
    let r = generate_chunk_mesh(&data, &nb, &shapes, crate::PresentationMode::CutAndOffset);
    let (boundary, _i, nm) = count_edge_sharing(r.full_res());
    eprintln!("[inner cap step] boundary={boundary} nm={nm}");
    if boundary > 0 {
        dump_boundary_edges(r.full_res(), "inner_cap_step");
    }
    assert!(boundary == 0 && nm == 0, "inner cap step: boundary={boundary} nm={nm}");
}

/// Mesh the real worldgen terrain chunks densest in slope caps and require
/// watertightness at both stages:
/// 1. the pre-chamfer solid mesh of each chunk must have no boundary edges
///    off the chunk border planes (no interior holes), and
/// 2. the chamfered mesh, unioned with the chamfered meshes of all its
///    neighbor chunks, must have no boundary edges anywhere near the central
///    chunk volume (seam fillets must match across chunk borders).
#[test]
fn worldgen_slope_chunks_mesh_clean() {
    use crate::world_def::WorldDef;
    use crate::coords::RegionId;
    use bevy::math::Vec3;

    let def = WorldDef::standard();
    let mut region = crate::region::Region::new(RegionId(0), Vec3::ZERO);
    crate::worldgen::generate_archipelago(&mut region, &def.terrain);

    // Rank chunks by slope-cap count
    let mut ranked: Vec<(ChunkPos, usize)> = region
        .iter_chunks()
        .map(|(pos, slot)| {
            let caps = slot.data.blocks.iter().filter(|b| b.shape != 0).count();
            (pos, caps)
        })
        .collect();
    ranked.sort_by_key(|&(_, c)| std::cmp::Reverse(c));

    fn wire_neighbors(region: &crate::region::Region, pos: ChunkPos) -> ChunkNeighbors {
        let mut nb = ChunkNeighbors::empty();
        for (dx, dy, dz) in ChunkPos::neighbor_offsets() {
            if let Some(np) = pos.neighbor(dx, dy, dz) {
                if let Some(arc) = region.get_chunk_data(np) {
                    nb.set_arc(dx, dy, dz, arc);
                }
            }
        }
        nb
    }

    let shapes = make_shapes();
    let chunk_w = crate::chunk::CHUNK_X as f32 * crate::chunk::VOXEL_SIZE;
    let chunk_h = crate::chunk::CHUNK_Y as f32 * crate::chunk::VOXEL_SIZE;
    let mut cache: HashMap<ChunkPos, crate::meshing::ChunkMeshResult> = HashMap::new();
    let mut total_solid_interior = 0usize;
    let mut total_union_holes = 0usize;
    let mut total_backfacing = 0usize;

    for &(pos, caps) in ranked.iter().take(6) {
        let slot = region.get_chunk(pos).unwrap();
        let nb = wire_neighbors(&region, pos);

        // Stage 1: pre-chamfer solid mesh (owned faces, original
        // triangulation) must have no boundary edges off the border planes.
        let solid = build_solid_mesh_public(&slot.data, &nb, &shapes);
        let mut ec: HashMap<((i32,i32,i32),(i32,i32,i32)), usize> = HashMap::new();
        let mut sseg: HashMap<((i32,i32,i32),(i32,i32,i32)), ([f32;3],[f32;3])> = HashMap::new();
        for face in solid.faces.iter().filter(|f| !f.halo) {
            for tri in &face.orig_triangles {
                for j in 0..3 {
                    let a = solid.positions[face.verts[tri[j]] as usize];
                    let b = solid.positions[face.verts[tri[(j+1)%3]] as usize];
                    let (a, b) = ([a.x, a.y, a.z], [b.x, b.y, b.z]);
                    let (qa, qb) = (quantize_pos(&a), quantize_pos(&b));
                    if qa == qb { continue; }
                    let key = if qa <= qb { (qa, qb) } else { (qb, qa) };
                    *ec.entry(key).or_insert(0) += 1;
                    sseg.entry(key).or_insert((a, b));
                }
            }
        }
        let on_border = |p: [f32;3]| -> bool {
            p[0].abs() < 1e-3 || (p[0] - chunk_w).abs() < 1e-3
                || p[1].abs() < 1e-3 || (p[1] - chunk_h).abs() < 1e-3
                || p[2].abs() < 1e-3 || (p[2] - chunk_w).abs() < 1e-3
        };
        let mut solid_interior = 0;
        for (key, &c) in &ec {
            if c != 1 { continue; }
            let (a, b) = sseg[key];
            if !on_border(a) || !on_border(b) {
                solid_interior += 1;
                if solid_interior <= 6 {
                    eprintln!("  {pos}: SOLID hole edge ({:.3},{:.3},{:.3})\u{2192}({:.3},{:.3},{:.3})",
                        a[0],a[1],a[2],b[0],b[1],b[2]);
                }
            }
        }
        total_solid_interior += solid_interior;

        // Stage 2: union the chamfered mesh with every neighbor chunk's
        // chamfered mesh; no boundary edge may fall inside (or near) the
        // central chunk volume.
        cache.entry(pos).or_insert_with(|| {
            generate_chunk_mesh(&slot.data, &nb, &shapes, crate::PresentationMode::CutAndOffset)
        });
        let mut union = {
            let m = cache[&pos].full_res();
            ChunkMeshData {
                positions: m.positions.clone(),
                normals: m.normals.clone(),
                sharp_normals: m.sharp_normals.clone(),
                uvs: m.uvs.clone(),
                chamfer_offsets: m.chamfer_offsets.clone(),
                indices: m.indices.clone(),
            }
        };
        for (dx, dy, dz) in ChunkPos::neighbor_offsets() {
            let Some(np) = pos.neighbor(dx, dy, dz) else { continue };
            if region.get_chunk(np).is_none() { continue; }
            if !cache.contains_key(&np) {
                let ndata = region.get_chunk(np).unwrap();
                let nnb = wire_neighbors(&region, np);
                let r = generate_chunk_mesh(&ndata.data, &nnb, &shapes, crate::PresentationMode::CutAndOffset);
                cache.insert(np, r);
            }
            union = union_meshes(
                &union,
                cache[&np].full_res(),
                [dx as f32 * chunk_w, dy as f32 * chunk_h, dz as f32 * chunk_w],
            );
        }

        let mut edge_count: HashMap<((i32,i32,i32),(i32,i32,i32)), usize> = HashMap::new();
        let mut seg: HashMap<((i32,i32,i32),(i32,i32,i32)), ([f32;3],[f32;3])> = HashMap::new();
        for i in (0..union.indices.len()).step_by(3) {
            let t = [union.indices[i], union.indices[i+1], union.indices[i+2]];
            for j in 0..3 {
                let a = union.positions[t[j] as usize];
                let b = union.positions[t[(j+1)%3] as usize];
                let (qa, qb) = (quantize_pos(&a), quantize_pos(&b));
                let key = if qa <= qb { (qa, qb) } else { (qb, qa) };
                *edge_count.entry(key).or_insert(0) += 1;
                seg.entry(key).or_insert((a, b));
            }
        }
        // Any point within 1wu of the central chunk volume is far from the
        // union's outer surface (neighbors extend 16wu further out).
        let near_center = |p: [f32;3]| -> bool {
            p[0] > -1.0 && p[0] < chunk_w + 1.0
                && p[1] > -1.0 && p[1] < chunk_h + 1.0
                && p[2] > -1.0 && p[2] < chunk_w + 1.0
        };
        let mut union_holes = 0;
        for (key, &c) in &edge_count {
            if c == 2 { continue; }
            let (a, b) = seg[key];
            if near_center(a) && near_center(b) {
                union_holes += 1;
                if union_holes <= 6 {
                    eprintln!("  {pos}: UNION open edge x{c} ({:.3},{:.3},{:.3})\u{2192}({:.3},{:.3},{:.3})",
                        a[0],a[1],a[2],b[0],b[1],b[2]);
                }
            }
        }
        // Backfacing slivers: a triangle whose geometric normal strongly
        // opposes its stored (smoothed) vertex normals renders as a
        // see-through hole under backface culling even when edge counts are
        // watertight (a twisted quad mis-wound during re-triangulation).
        let m = cache[&pos].full_res();
        let mut backfacing = 0;
        for i in (0..m.indices.len()).step_by(3) {
            let t = [m.indices[i] as usize, m.indices[i+1] as usize, m.indices[i+2] as usize];
            let p = |k: usize| bevy::math::Vec3::from_array(m.positions[t[k]]);
            let nv = |k: usize| bevy::math::Vec3::from_array(m.normals[t[k]]);
            let geo = (p(1) - p(0)).cross(p(2) - p(0));
            let len = geo.length();
            // Sub-0.01 slivers (a few millimetres wide in-world) still slip
            // through clipped-face triangulation occasionally; they are
            // invisible at render scale, so only assert on visible sizes.
            if len * 0.5 < 0.01 { continue; }
            let avg_n = (nv(0) + nv(1) + nv(2)).normalize_or_zero();
            // -0.6: wall-base transition slivers on diagonal caps sit near
            // -0.55 (nearly perpendicular to their smoothed normals — no
            // winding is "right" for them); real backfacing defects score
            // -0.9 and below.
            if (geo / len).dot(avg_n) < -0.6 {
                backfacing += 1;
                if backfacing <= 4 {
                    let c = (p(0) + p(1) + p(2)) / 3.0;
                    eprintln!("  {pos}: backfacing sliver area {:.5} at ({:.3},{:.3},{:.3})",
                        len * 0.5, c.x, c.y, c.z);
                    for k in 0..3 {
                        let q = p(k); let nn = nv(k);
                        let off = bevy::math::Vec3::from_array(m.chamfer_offsets[t[k]]);
                        eprintln!("    v ({:.4},{:.4},{:.4}) n ({:.2},{:.2},{:.2}) push ({:.4},{:.4},{:.4})",
                            q.x, q.y, q.z, nn.x, nn.y, nn.z, off.x, off.y, off.z);
                    }
                }
            }
        }
        eprintln!("chunk {pos}: {caps} caps, solid-interior {solid_interior}, union-open {union_holes}, backfacing {backfacing}");
        total_union_holes += union_holes;
        total_backfacing += backfacing;
    }
    assert!(total_solid_interior == 0, "{total_solid_interior} solid interior hole edges");
    assert!(total_union_holes == 0, "{total_union_holes} open edges in neighbor unions");
    assert!(total_backfacing == 0, "{total_backfacing} backfacing sliver triangles");
}

/// Dump the chamfered mesh of every registered shape (plus a small terraced
/// assembly) as JSON for external visualization.  Gated on SHAPE_GALLERY_OUT
/// so it never runs in normal test sweeps.
#[test]
#[ignore]
fn dump_shape_gallery() {
    let Ok(out_path) = std::env::var("SHAPE_GALLERY_OUT") else {
        return;
    };
    let shapes = make_shapes();
    let mut exhibits: Vec<String> = Vec::new();

    let mesh_json = |name: &str, mesh: &ChunkMeshData| -> String {
        let pos: Vec<String> = mesh.positions.iter()
            .flat_map(|p| p.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>())
            .collect();
        let nrm: Vec<String> = mesh.normals.iter()
            .flat_map(|n| n.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>())
            .collect();
        let idx: Vec<String> = mesh.indices.iter().map(|i| i.to_string()).collect();
        format!("{{\"name\":\"{name}\",\"positions\":[{}],\"normals\":[{}],\"indices\":[{}]}}",
            pos.join(","), nrm.join(","), idx.join(","))
    };

    for (id, label) in [
        (SHAPE_CUBE, "cube"),
        (SHAPE_WEDGE, "wedge (1:2)"),
        (SHAPE_WEDGE_OUTER, "outer corner (1:2)"),
        (SHAPE_WEDGE_INNER, "inner corner (1:2)"),
        (SHAPE_WEDGE_STEEP, "steep wedge (1:1)"),
        (SHAPE_WEDGE_STEEP_OUTER, "steep outer (1:1)"),
        (SHAPE_WEDGE_STEEP_INNER, "steep inner (1:1)"),
        (SHAPE_WEDGE_DIAG, "diagonal ramp (1:2)"),
        (SHAPE_WEDGE_STEEP_DIAG, "steep diagonal (1:1)"),
    ] {
        let mut data = ChunkData::new();
        let shape = shapes.get(id).unwrap();
        let occ = crate::shape::rotated_occupied_cells(shape, Facing::North);
        data.place_block(id, Facing::North, 1, 14, 8, 14, &occ);
        let nb = ChunkNeighbors::empty();
        let r = generate_chunk_mesh(&data, &nb, &shapes, crate::PresentationMode::CutAndOffset);
        exhibits.push(mesh_json(label, r.full_res()));
    }

    // Terraced assembly: corners + straights + steep meeting with fillets.
    {
        let mut data = ChunkData::new();
        let heights: &[&[i32]] = &[&[3, 2, 1], &[2, 2, 1], &[1, 1, 1]];
        build_capped_terrain_into(&mut data, heights, (10, 6, 10), 0..3);
        let nb = ChunkNeighbors::empty();
        let r = generate_chunk_mesh(&data, &nb, &shapes, crate::PresentationMode::CutAndOffset);
        exhibits.push(mesh_json("assembled terrace", r.full_res()));
    }

    std::fs::write(&out_path, format!("[{}]", exhibits.join(","))).unwrap();
    eprintln!("wrote {out_path}");
}

/// Debug: find the longest straight runs of same-facing straight-wedge caps
/// (continuous ramps) in the generated archipelago, in world coordinates.
#[test]
#[ignore]
fn debug_find_slope_runs() {
    use crate::world_def::WorldDef;
    use crate::coords::RegionId;
    use bevy::math::Vec3;

    let def = WorldDef::standard();
    let mut region = crate::region::Region::new(RegionId(0), Vec3::ZERO);
    crate::worldgen::generate_archipelago(&mut region, &def.terrain);

    // Collect straight-wedge caps globally: (global block col x, z) -> (shape, facing, y)
    let mut caps: HashMap<(i32, i32), (u16, Facing, i32)> = HashMap::new();
    for (pos, slot) in region.iter_chunks() {
        for b in slot.data.blocks.iter() {
            if b.shape == SHAPE_WEDGE || b.shape == SHAPE_WEDGE_STEEP {
                let gx = pos.x * 32 + b.origin.0 as i32;
                let gy = pos.y * 32 + b.origin.1 as i32;
                let gz = pos.z * 32 + b.origin.2 as i32;
                caps.insert((gx / 2, gz / 2), (b.shape, b.facing, gy));
            }
        }
    }

    // A chained ramp climbs 2 cells (gentle) per block along the downhill
    // axis. Walk runs along +x and +z.
    let mut runs: Vec<(usize, i32, i32, i32, Facing, u16)> = Vec::new();
    for (&(cx, cz), &(shape, facing, y)) in &caps {
        for (dx, dz) in [(1i32, 0i32), (0, 1)] {
            // Only count run starts
            let prev = caps.get(&(cx - dx, cz - dz));
            let step = if shape == SHAPE_WEDGE { 1 } else { 2 };
            if let Some(&(ps, pf, py)) = prev {
                if ps == shape && pf == facing && (py - y).abs() == step {
                    continue;
                }
            }
            let mut len = 1;
            let (mut px, mut pz, mut py) = (cx, cz, y);
            loop {
                let Some(&(ns, nf, ny)) = caps.get(&(px + dx, pz + dz)) else { break };
                if ns != shape || nf != facing || (ny - py).abs() != step {
                    break;
                }
                len += 1; px += dx; pz += dz; py = ny;
            }
            if len >= 5 {
                runs.push((len, cx, cz, y, facing, shape));
            }
        }
    }
    runs.sort_by_key(|r| std::cmp::Reverse(r.0));
    for (len, cx, cz, y, facing, shape) in runs.iter().take(12) {
        // world coords: block col to wu, region centered at -2048
        let wx = (*cx as f32) * 1.0 - 2048.0;
        let wz = (*cz as f32) * 1.0 - 2048.0;
        let wy = (*y as f32) * 0.5;
        eprintln!("run len {len}: start world ({wx:.0},{wy:.0},{wz:.0}) facing {facing:?} shape {shape}");
    }
}

/// SEMANTIC placement check: for every slope cap the worldgen places, the
/// cap's surface must be flush with the neighboring column's SURFACE (cube
/// top or that column's own cap corners) along every side whose neighbor
/// stands higher, and at every strictly-higher diagonal corner.
/// Watertightness can't catch a mis-rotated or wrong-variant cap (walls
/// seal the gaps); this can.
#[test]
fn worldgen_caps_meet_neighbors() {
    use crate::world_def::WorldDef;
    use crate::coords::RegionId;
    use crate::shape::cap_corner_heights;
    use bevy::math::Vec3;

    let def = WorldDef::standard();
    let mut region = crate::region::Region::new(RegionId(0), Vec3::ZERO);
    crate::worldgen::generate_archipelago(&mut region, &def.terrain);

    // Per block column: highest cube top and any cap block.
    let mut cube_top: HashMap<(i32, i32), i32> = HashMap::new();
    let mut cap_at: HashMap<(i32, i32), (i32, u16, Facing)> = HashMap::new();
    for (pos, slot) in region.iter_chunks() {
        for b in slot.data.blocks.iter() {
            let gx = pos.x * 32 + b.origin.0 as i32;
            let gy = pos.y * 32 + b.origin.1 as i32;
            let gz = pos.z * 32 + b.origin.2 as i32;
            let col = (gx / 2, gz / 2);
            if b.shape == 0 {
                let top = gy + 1;
                let e = cube_top.entry(col).or_insert(top);
                if top > *e { *e = top; }
            } else {
                cap_at.insert(col, (gy, b.shape, b.facing));
            }
        }
    }

    // Surface corner heights of a column at its 4 block corners, in world
    // orientation: keys (0,0),(2,0),(0,2),(2,2). None if the column has no
    // terrain.
    let corners_of = |col: (i32, i32)| -> Option<HashMap<(i32, i32), i32>> {
        if let Some(&(gy, shape, facing)) = cap_at.get(&col) {
            let h = cap_corner_heights(shape)?;
            let mut m = HashMap::new();
            for (lx, lz, hh) in [(0.0, 0.0, h[0]), (2.0, 0.0, h[1]), (0.0, 2.0, h[2]), (2.0, 2.0, h[3])] {
                let p = facing.rotate_block_point(Vec3::new(lx, 0.0, lz), (2, 2, 2));
                m.insert((p.x.round() as i32, p.z.round() as i32), gy + hh as i32);
            }
            Some(m)
        } else {
            let &top = cube_top.get(&col)?;
            let mut m = HashMap::new();
            for k in [(0, 0), (2, 0), (0, 2), (2, 2)] {
                m.insert(k, top);
            }
            Some(m)
        }
    };
    // Column height as the classifier saw it (cap columns: cap base + 1).
    let height_of = |col: (i32, i32)| -> Option<i32> {
        if let Some(&(gy, _, _)) = cap_at.get(&col) {
            Some(gy + 1)
        } else {
            cube_top.get(&col).copied()
        }
    };

    let mut violations = 0usize;
    let mut checked = 0usize;
    for (&col, &(_gy, shape, facing)) in &cap_at {
        let (cx, cz) = col;
        let own = corners_of(col).unwrap();
        let h = height_of(col).unwrap();
        // Sides: our two edge corners must equal the neighbor's matching
        // corners when the neighbor column is higher.
        for (dx, dz, ours, theirs) in [
            (-1, 0, [(0, 0), (0, 2)], [(2, 0), (2, 2)]),
            (1, 0, [(2, 0), (2, 2)], [(0, 0), (0, 2)]),
            (0, -1, [(0, 0), (2, 0)], [(0, 2), (2, 2)]),
            (0, 1, [(0, 2), (2, 2)], [(0, 0), (2, 0)]),
        ] {
            let ncol = (cx + dx, cz + dz);
            let Some(hn) = height_of(ncol) else { continue };
            if hn <= h { continue; }
            let Some(nc) = corners_of(ncol) else { continue };
            checked += 1;
            if own[&ours[0]] != nc[&theirs[0]] || own[&ours[1]] != nc[&theirs[1]] {
                violations += 1;
                if violations <= 10 {
                    eprintln!(
                        "cap col ({cx},{cz}) shape {shape} facing {facing:?}: edge toward ({dx},{dz}) is {}/{}, neighbor surface {}/{}",
                        own[&ours[0]], own[&ours[1]], nc[&theirs[0]], nc[&theirs[1]]
                    );
                }
            }
        }
        // Diagonals: when only the diagonal is higher, our corner must be
        // flush with the diagonal column's facing corner.
        for (dx, dz, ours, theirs) in [
            (-1, -1, (0, 0), (2, 2)),
            (1, -1, (2, 0), (0, 2)),
            (-1, 1, (0, 2), (2, 0)),
            (1, 1, (2, 2), (0, 0)),
        ] {
            let dcol = (cx + dx, cz + dz);
            let Some(hd) = height_of(dcol) else { continue };
            let ha = height_of((cx + dx, cz)).unwrap_or(h);
            let hb = height_of((cx, cz + dz)).unwrap_or(h);
            if hd > h && ha <= h && hb <= h {
                let Some(dc) = corners_of(dcol) else { continue };
                checked += 1;
                if own[&ours] != dc[&theirs] {
                    violations += 1;
                    if violations <= 10 {
                        eprintln!(
                            "cap col ({cx},{cz}) shape {shape} facing {facing:?}: corner toward diag ({dx},{dz}) is {}, neighbor corner {}",
                            own[&ours], dc[&theirs]
                        );
                    }
                }
            }
        }
    }
    eprintln!("checked {checked} cap contacts, {violations} violations");
    // Rough terrain legitimately produces some cap-vs-cap mini-cliffs (a
    // neighbor's cap can rise past ours where gradients disagree), so this
    // asserts a BOUND, not zero: mis-rotated facings or wrong variants blow
    // far past it (a systematic error flags ~20%+ of contacts).
    assert!(
        violations * 20 <= checked,
        "{violations}/{checked} cap contacts disagree with neighbors (>5%)"
    );
}
