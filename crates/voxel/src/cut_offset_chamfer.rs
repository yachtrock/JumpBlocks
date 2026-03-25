use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::chunk::*;
use crate::meshing::*;
use crate::shape::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// 2D cross product of two vectors projected onto a face plane via its normal.
#[inline]
fn cross2d(a: Vec3, b: Vec3, face_normal: Vec3) -> f32 {
    a.cross(b).dot(face_normal)
}

/// Intersect two lines in a face plane.  Each line is (point, direction).
/// Returns None if the lines are parallel (collinear sharp edges).
fn intersect_lines_in_plane(
    p1: Vec3,
    d1: Vec3,
    p2: Vec3,
    d2: Vec3,
    face_normal: Vec3,
) -> Option<Vec3> {
    let denom = cross2d(d1, d2, face_normal);
    if denom.abs() < 1e-8 {
        return None;
    }
    let t = cross2d(p2 - p1, d2, face_normal) / denom;
    Some(p1 + t * d1)
}

/// Emit a quad as two triangles.  The `expected_normal` is used to determine
/// correct winding — the triangle normal is computed from the vertices, and
/// if it disagrees with `expected_normal` the winding is flipped.
fn emit_quad(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    expected_normal: Vec3,
) {
    let base = positions.len() as u32;
    positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
    let n: [f32; 3] = expected_normal.into();
    normals.extend_from_slice(&[n, n, n, n]);
    uvs.extend_from_slice(&[[0.0; 2]; 4]);

    // Compute actual triangle normal from first three verts to detect winding.
    let computed = (v1 - v0).cross(v2 - v0);
    if computed.dot(expected_normal) >= 0.0 {
        // CCW already matches expected outward normal
        indices.extend_from_slice(&[base, base + 1, base + 2]);
        indices.extend_from_slice(&[base, base + 2, base + 3]);
    } else {
        // Flip to CCW
        indices.extend_from_slice(&[base + 2, base + 1, base]);
        indices.extend_from_slice(&[base, base + 3, base + 2]);
    }
}

/// Emit a single triangle with automatic winding correction.
fn emit_tri(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    expected_normal: Vec3,
) {
    // Skip degenerate triangles.
    let area = (v1 - v0).cross(v2 - v0).length();
    if area < 1e-8 {
        return;
    }
    let base = positions.len() as u32;
    positions.extend_from_slice(&[v0.into(), v1.into(), v2.into()]);
    let n: [f32; 3] = expected_normal.into();
    normals.extend_from_slice(&[n, n, n]);
    uvs.extend_from_slice(&[[0.0; 2]; 3]);

    let computed = (v1 - v0).cross(v2 - v0);
    if computed.dot(expected_normal) >= 0.0 {
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    } else {
        indices.extend_from_slice(&[base + 2, base + 1, base]);
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Generate a chamfered mesh using the cut-and-offset approach.
///
/// For each face the algorithm performs *vertex splits* along existing edges:
///
/// 1. For every sharp edge on a face, each endpoint is split by sliding a copy
///    along the *adjacent* edge by `CHAMFER_WIDTH`.  The split vertex is
///    guaranteed to lie on an original edge of the face.
/// 2. The split vertices on opposite ends of a sharp edge define a *cut line*
///    across the face.
/// 3. Where two cut lines meet (corner with two sharp edges) the intersection
///    point is computed via a 2D line–line intersection in the face plane.
/// 4. These cuts subdivide the original face into:
///      - **Inner polygon** (the shrunk interior)
///      - **Edge strips** (between each sharp edge and its cut line)
///      - **Corner patches** (where two strips meet at a vertex)
/// 5. Cross-face chamfer strips bridge adjacent faces across each sharp edge,
///    with a center-line pushed outward for a fillet profile.
pub fn generate_cut_offset_chamfer(
    data: &ChunkData,
    neighbors: &ChunkNeighbors,
    shapes: &ShapeTable,
) -> ChunkMeshData {
    let solid = build_solid_mesh_public(data, neighbors, shapes);
    let edge_graph = build_edge_graph(&solid);

    // -- Identify sharp edges (excluding boundary-at-neighbor-seam) ----------
    let sharp_edges: HashSet<(u32, u32)> = edge_graph
        .iter()
        .filter(|&(&(a, b), ref info)| {
            if !is_sharp(info, &solid) {
                return false;
            }
            if info.faces.len() < 2 {
                let fi = info.faces[0];
                if is_boundary_edge_at_neighbor_seam(
                    a, b, &solid.faces[fi], &solid, data, neighbors,
                ) {
                    return false;
                }
            }
            true
        })
        .map(|(&key, _)| key)
        .collect();

    // -- Chamfered vertices: any vertex on a sharp edge ------------------------
    let chamfered_verts: HashSet<u32> = sharp_edges
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect();

    // -- Output buffers ------------------------------------------------------
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Per (edge_key, face_index) → (inner_v_at_a, inner_v_at_b) in edge-key
    // order.  Will be used for cross-face strip emission in future phases.
    let mut _edge_face_inners: HashMap<((u32, u32), usize), (Vec3, Vec3)> = HashMap::new();

    // -----------------------------------------------------------------------
    // Per-face: subdivide via vertex splits
    // -----------------------------------------------------------------------
    for (fi, face) in solid.faces.iter().enumerate() {
        let n = face.verts.len();
        let fn_ = face.normal;

        // -- Per-vertex: classify edges and compute split positions ----------

        // Is the edge *entering* / *leaving* this vertex sharp?
        let mut prev_sharp = vec![false; n];
        let mut next_sharp = vec![false; n];

        // Positions of the original vertices.
        let orig: Vec<Vec3> = (0..n)
            .map(|i| solid.positions[face.verts[i] as usize])
            .collect();

        for i in 0..n {
            let vi = face.verts[i];
            let vp = face.verts[(i + n - 1) % n];
            let vn = face.verts[(i + 1) % n];
            prev_sharp[i] = sharp_edges.contains(&edge_key(vp, vi));
            next_sharp[i] = sharp_edges.contains(&edge_key(vi, vn));
        }

        // Split vertex: slide vi along an adjacent edge by CHAMFER_WIDTH.
        // Split vertex vi by sliding toward target along the adjacent edge.
        // Returns the split position.
        let split_toward = |i: usize, target_idx: usize| -> Vec3 {
            let dir = (orig[target_idx] - orig[i]).normalize_or_zero();
            orig[i] + dir * CHAMFER_WIDTH
        };

        // Perpendicular offset of vertex i from the sharp edge i→j (or ip→i),
        // used when the adjacent edge is collinear with the sharp edge (midpoint
        // vertex) so there is no non-parallel edge to split along.
        let perp_offset = |i: usize, edge_start: usize, edge_end: usize| -> Vec3 {
            let edge_dir = (orig[edge_end] - orig[edge_start]).normalize_or_zero();
            let inward = edge_dir.cross(fn_).normalize_or_zero();
            orig[i] + inward * CHAMFER_WIDTH
        };

        // Check if the adjacent edge at vertex i (toward target_idx) is
        // collinear with the sharp edge (from edge_a to edge_b).
        let is_collinear = |i: usize, target_idx: usize, edge_a: usize, edge_b: usize| -> bool {
            let adj_dir = (orig[target_idx] - orig[i]).normalize_or_zero();
            let edge_dir = (orig[edge_b] - orig[edge_a]).normalize_or_zero();
            adj_dir.dot(edge_dir).abs() > 0.99
        };

        // -- Per sharp-edge: compute the two split vertices (one per end) ----

        #[derive(Clone)]
        struct CutLine {
            split_i: Vec3, // split of vi (start of sharp edge)
            split_j: Vec3, // split of vj (end of sharp edge)
            dir: Vec3,     // split_j - split_i
        }

        let mut cuts: Vec<Option<CutLine>> = vec![None; n];

        for i in 0..n {
            let j = (i + 1) % n;
            if !sharp_edges.contains(&edge_key(face.verts[i], face.verts[j])) {
                continue;
            }
            let ip = (i + n - 1) % n;
            let jn = (j + 1) % n;

            // At vi: split toward prev vertex, unless prev edge is collinear
            // with this sharp edge (midpoint vertex) — then use perpendicular.
            let si = if is_collinear(i, ip, i, j) {
                perp_offset(i, i, j)
            } else {
                split_toward(i, ip)
            };

            // At vj: split toward next vertex, unless next edge is collinear.
            let sj = if is_collinear(j, jn, i, j) {
                perp_offset(j, i, j)
            } else {
                split_toward(j, jn)
            };

            cuts[i] = Some(CutLine {
                split_i: si,
                split_j: sj,
                dir: sj - si,
            });
        }

        // -- Compute inner polygon vertices ----------------------------------
        // Walk each vertex:
        //   - neither adjacent edge sharp → original vertex
        //   - only prev sharp  → split of vi along *next* edge (== cut line
        //     of prev edge intersects next edge at this split point)
        //   - only next sharp  → split of vi along *prev* edge
        //   - both sharp       → intersection of the two cut lines

        let mut inner: Vec<Vec3> = Vec::with_capacity(n);

        for i in 0..n {
            let ip = (i + n - 1) % n; // face-edge index of prev edge
            let j = (i + 1) % n;

            let vi = face.verts[i];
            let is_chamfered = chamfered_verts.contains(&vi);

            match (prev_sharp[i], next_sharp[i]) {
                (false, false) if is_chamfered => {
                    // Vertex is on a sharp edge from another face, but has
                    // no sharp edges on THIS face.  We still need to cut
                    // because the vertex will be offset later.
                    // Compute inner via virtual cut lines along both adjacent
                    // edges, intersected in the face plane.
                    let d_prev = (orig[i] - orig[ip]).normalize_or_zero();
                    let d_next = (orig[j] - orig[i]).normalize_or_zero();
                    let sp = split_toward(i, ip); // on prev edge
                    let sn = split_toward(i, j); // on next edge
                    // Virtual cut for prev edge: through sn, direction d_prev
                    // Virtual cut for next edge: through sp, direction d_next
                    if let Some(pt) =
                        intersect_lines_in_plane(sn, d_prev, sp, d_next, fn_)
                    {
                        inner.push(pt);
                    } else {
                        // Parallel — perpendicular fallback
                        let inward = d_prev.cross(fn_).normalize_or_zero();
                        inner.push(orig[i] + inward * CHAMFER_WIDTH);
                    }
                }
                (false, false) => {
                    inner.push(orig[i]);
                }
                (true, false) => {
                    // Prev edge (ip) is sharp.  Split vi along the *next*
                    // edge (toward vj), unless next edge is collinear with
                    // the prev sharp edge — then use perpendicular offset.
                    if is_collinear(i, j, ip, i) {
                        inner.push(perp_offset(i, ip, i));
                    } else {
                        inner.push(split_toward(i, j));
                    }
                }
                (false, true) => {
                    // Next edge (i) is sharp.  Split vi along the *prev*
                    // edge (toward v_{i-1}), unless prev edge is collinear.
                    if is_collinear(i, ip, i, j) {
                        inner.push(perp_offset(i, i, j));
                    } else {
                        inner.push(split_toward(i, ip));
                    }
                }
                (true, true) => {
                    // Both adjacent edges are sharp.
                    let prev_dir = (orig[i] - orig[ip]).normalize_or_zero();
                    let next_dir = (orig[j] - orig[i]).normalize_or_zero();
                    let collinear = prev_dir.dot(next_dir).abs() > 0.99;

                    if collinear {
                        // Midpoint on a single long edge — both sharp edges
                        // are the same line.  Offset perpendicular to the
                        // edge along the face.
                        let edge_dir = prev_dir;
                        let inward = edge_dir.cross(fn_).normalize_or_zero();
                        inner.push(orig[i] + inward * CHAMFER_WIDTH);
                    } else {
                        // Normal case: intersect the two cut lines.
                        let cut_prev = &cuts[ip];
                        let cut_next = &cuts[i];
                        if let (Some(cp), Some(cn)) = (cut_prev, cut_next) {
                            if let Some(pt) = intersect_lines_in_plane(
                                cp.split_i, cp.dir, cn.split_i, cn.dir, fn_,
                            ) {
                                inner.push(pt);
                            } else {
                                // Shouldn't happen for non-collinear, but
                                // fall back to perpendicular offset.
                                let inward = prev_dir.cross(fn_).normalize_or_zero();
                                inner.push(orig[i] + inward * CHAMFER_WIDTH);
                            }
                        } else {
                            inner.push(orig[i]);
                        }
                    }
                }
            }
        }

        // -- Record inner positions for cross-face strips --------------------
        for i in 0..n {
            let j = (i + 1) % n;
            let vi = face.verts[i];
            let vj = face.verts[j];
            let ek = edge_key(vi, vj);
            if !sharp_edges.contains(&ek) {
                continue;
            }
            let (a, b) = if vi <= vj {
                (inner[i], inner[j])
            } else {
                (inner[j], inner[i])
            };
            _edge_face_inners.insert((ek, fi), (a, b));
        }

        // -- Emit inner polygon ----------------------------------------------
        let any_sharp = prev_sharp.iter().any(|&s| s) || next_sharp.iter().any(|&s| s);
        let inner_base = positions.len() as u32;
        for i in 0..n {
            positions.push(inner[i].into());
            normals.push(fn_.into());
            uvs.push([0.0, 0.0]);
        }
        if !face.orig_triangles.is_empty() {
            for tri in &face.orig_triangles {
                indices.push(inner_base + tri[2] as u32);
                indices.push(inner_base + tri[1] as u32);
                indices.push(inner_base + tri[0] as u32);
            }
        } else {
            triangulate_convex_polygon(&positions, inner_base, n, &mut indices);
        }

        if !any_sharp {
            continue; // nothing else to emit for this face
        }

        // -- Emit edge strips ------------------------------------------------
        // For each sharp edge (vi→vj), the strip is bounded by:
        //   outer side: the original edge (possibly truncated at corners)
        //   inner side: the cut line (= segment of the inner polygon)
        //
        // At each endpoint, if the OTHER edge at that vertex is also sharp,
        // the strip is truncated: outer uses the split vertex (which lies on
        // the original edge) and inner uses the inner-polygon vertex (the cut
        // line intersection).  Otherwise, outer is the original vertex and
        // inner is the split vertex (which equals the inner-polygon vertex).

        // Helper: detect collinear midpoint vertex (both adjacent edges are
        // sharp and effectively the same line).
        let is_collinear_midpoint = |i: usize| -> bool {
            if !prev_sharp[i] || !next_sharp[i] {
                return false;
            }
            let ip = (i + n - 1) % n;
            let j = (i + 1) % n;
            let prev_dir = (orig[i] - orig[ip]).normalize_or_zero();
            let next_dir = (orig[j] - orig[i]).normalize_or_zero();
            prev_dir.dot(next_dir).abs() > 0.99
        };

        for i in 0..n {
            let j = (i + 1) % n;
            if cuts[i].is_none() {
                continue;
            }

            // --- vi end ---
            // If the other edge at vi is sharp AND non-collinear, truncate.
            // If collinear (midpoint), no truncation — the strip passes through.
            let (outer_i, inner_i) = if prev_sharp[i] && !is_collinear_midpoint(i) {
                let ip = (i + n - 1) % n;
                let trunc = cuts[ip].as_ref().map_or(orig[i], |c| c.split_j);
                (trunc, inner[i])
            } else {
                (orig[i], inner[i])
            };

            // --- vj end ---
            let (outer_j, inner_j) = if next_sharp[j] && !is_collinear_midpoint(j) {
                let trunc = cuts[j].as_ref().map_or(orig[j], |c| c.split_i);
                (trunc, inner[j])
            } else {
                (orig[j], inner[j])
            };

            // Quad CW from outside: outer_i → inner_i → inner_j → outer_j
            emit_quad(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                outer_i,
                inner_i,
                inner_j,
                outer_j,
                fn_,
            );
        }

        // -- Emit corner patches ---------------------------------------------
        // At each vertex where the inner position differs from the original,
        // a corner patch fills the gap.  This covers:
        //   a) Both adjacent edges sharp (non-collinear) — gap between two
        //      truncated strips.
        //   b) Chamfered vertex with no sharp edges on this face — gap between
        //      inner polygon and original face boundary.
        // Collinear midpoint vertices are skipped (strip passes through).

        for i in 0..n {
            let vi = face.verts[i];
            let ip = (i + n - 1) % n;
            let j = (i + 1) % n;
            let is_chamfered = chamfered_verts.contains(&vi);

            let both_sharp = prev_sharp[i] && next_sharp[i];
            let no_sharp_but_chamfered =
                !prev_sharp[i] && !next_sharp[i] && is_chamfered;

            if !both_sharp && !no_sharp_but_chamfered {
                continue;
            }
            if both_sharp && is_collinear_midpoint(i) {
                continue;
            }

            // Compute the two split points (on the adjacent edges, at
            // CHAMFER_WIDTH from orig[i]).
            let sp = split_toward(i, ip); // on prev edge
            let sn = split_toward(i, j); // on next edge

            // For the both-sharp case, use the cut line split vertices
            // (which account for collinearity etc.).
            let trunc_prev = if both_sharp {
                cuts[ip].as_ref().map_or(sp, |c| c.split_j)
            } else {
                sp
            };
            let trunc_next = if both_sharp {
                cuts[i].as_ref().map_or(sn, |c| c.split_i)
            } else {
                sn
            };

            // Corner patch quad: orig → trunc_next → inner → trunc_prev
            emit_quad(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                orig[i],
                trunc_next,
                inner[i],
                trunc_prev,
                fn_,
            );

            // For chamfered-but-no-sharp-edges vertices, emit gap triangles
            // along each adjacent edge to fill between the corner patch and
            // the inner polygon.
            if no_sharp_but_chamfered {
                // Gap on prev side: inner[prev] → split_prev → inner[i]
                emit_tri(
                    &mut positions,
                    &mut normals,
                    &mut uvs,
                    &mut indices,
                    inner[ip],
                    trunc_prev,
                    inner[i],
                    fn_,
                );
                // Gap on next side: inner[i] → split_next → inner[next]
                emit_tri(
                    &mut positions,
                    &mut normals,
                    &mut uvs,
                    &mut indices,
                    inner[i],
                    trunc_next,
                    inner[j],
                    fn_,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // TODO: Cross-face chamfer strips & vertex offset (fillet shaping)
    // -----------------------------------------------------------------------
    // Future phases:
    // 1. Offset the outer vertices (on original sharp edges) inward along
    //    the bisector of adjacent face normals.
    // 2. Bridge the gap between adjacent faces' on-face strips with fillet
    //    surface quads.
    // 3. Smooth normals across the chamfer bands.
    // 4. Corner caps where 3+ chamfer strips converge.

    let chamfer_offsets = vec![[0.0; 3]; positions.len()];
    ChunkMeshData {
        positions,
        normals,
        uvs,
        chamfer_offsets,
        indices,
    }
}
