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

/// Emit a quad as two triangles.  Vertices must be in CW order when viewed
/// from outside (i.e. looking against the outward normal).  The function
/// reverses them to CCW for the engine.
fn emit_quad(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    normal: Vec3,
) {
    let base = positions.len() as u32;
    positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
    let n: [f32; 3] = normal.into();
    normals.extend_from_slice(&[n, n, n, n]);
    uvs.extend_from_slice(&[[0.0; 2]; 4]);
    // Reverse CW → CCW
    indices.extend_from_slice(&[base + 2, base + 1, base]);
    indices.extend_from_slice(&[base, base + 3, base + 2]);
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

    // -- Output buffers ------------------------------------------------------
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Per (edge_key, face_index) → (inner_v_at_a, inner_v_at_b) in edge-key
    // order, for cross-face strip emission later.
    let mut edge_face_inners: HashMap<((u32, u32), usize), (Vec3, Vec3)> = HashMap::new();

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
        // `split_toward(i, target_idx)` returns the split position.
        let split_toward = |i: usize, target_idx: usize| -> Vec3 {
            let dir = (orig[target_idx] - orig[i]).normalize_or_zero();
            orig[i] + dir * CHAMFER_WIDTH
        };

        // -- Per sharp-edge: compute the two split vertices (one per end) ----
        // Also compute the cut-line direction for intersection later.

        // For each face-edge index `i` (edge from vert i to vert i+1):
        //   split_i  = vi  split toward the *other* edge at vi  (i.e. toward v_{i-1})
        //   split_j  = vj  split toward the *other* edge at vj  (i.e. toward v_{j+1})
        //   cut_dir  = split_j - split_i  (direction of the cut line)

        #[derive(Clone)]
        struct CutLine {
            split_i: Vec3, // split of vi toward v_{i-1}
            split_j: Vec3, // split of vj toward v_{j+1}
            dir: Vec3,     // split_j - split_i (unnormalised is fine for intersection)
        }

        let mut cuts: Vec<Option<CutLine>> = vec![None; n];

        for i in 0..n {
            let j = (i + 1) % n;
            if !sharp_edges.contains(&edge_key(face.verts[i], face.verts[j])) {
                continue;
            }
            let ip = (i + n - 1) % n; // index of vertex before vi
            let jn = (j + 1) % n; // index of vertex after vj
            let si = split_toward(i, ip);
            let sj = split_toward(j, jn);
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

            match (prev_sharp[i], next_sharp[i]) {
                (false, false) => {
                    inner.push(orig[i]);
                }
                (true, false) => {
                    // Prev edge (ip) is sharp.  Split vi along the *next*
                    // edge (toward vj) — this is where prev's cut line exits
                    // the next edge.
                    inner.push(split_toward(i, j));
                }
                (false, true) => {
                    // Next edge (i) is sharp.  Split vi along the *prev*
                    // edge (toward v_{i-1}).
                    inner.push(split_toward(i, ip));
                }
                (true, true) => {
                    // Both adjacent edges are sharp.  Intersect the two cut
                    // lines in the face plane.
                    let cut_prev = &cuts[ip]; // cut for edge ip→i
                    let cut_next = &cuts[i]; // cut for edge i→j
                    if let (Some(cp), Some(cn)) = (cut_prev, cut_next) {
                        if let Some(pt) =
                            intersect_lines_in_plane(cp.split_i, cp.dir, cn.split_i, cn.dir, fn_)
                        {
                            inner.push(pt);
                        } else {
                            // Parallel / collinear — fall back to single offset
                            inner.push(split_toward(i, ip));
                        }
                    } else {
                        inner.push(orig[i]);
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
            edge_face_inners.insert((ek, fi), (a, b));
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

        for i in 0..n {
            let j = (i + 1) % n;
            if cuts[i].is_none() {
                continue;
            }

            // --- vi end ---
            let (outer_i, inner_i) = if prev_sharp[i] {
                // The OTHER edge at vi (prev) is also sharp.
                // Outer: split vi toward vj along this edge (the split vertex
                // for THIS edge's cut at vi end — lies on the prev edge).
                // But actually the split for this edge at vi is
                // split_toward(i, ip) which is on the PREV edge.
                //
                // The truncation point on the ORIGINAL edge (vi→vj line) is
                // the split of vi along the next edge for the OTHER cut, which
                // is cuts[ip].split_j.  But more simply:
                //   outer_i = split of vi along the next edge direction =
                //             the point on edge vi→vj at CHAMFER_WIDTH from vi
                //             ... NO — that would be on the SHARP edge, not the
                //             adjacent edge.
                //
                // Correct: the truncation point is the split vertex from the
                // PREV edge's cut line at the vi end.  The prev edge's cut at
                // vi is split_toward(i, j) (slide vi along the next edge,
                // which is the edge we're currently processing).
                // That's: vi + CHAMFER_WIDTH * dir(vi→vj).
                // This point lies on the current sharp edge itself.
                let ip = (i + n - 1) % n;
                let trunc = cuts[ip].as_ref().map_or(orig[i], |c| c.split_j);
                (trunc, inner[i])
            } else {
                (orig[i], inner[i])
            };

            // --- vj end ---
            let (outer_j, inner_j) = if next_sharp[j] {
                let jn = (j + 1) % n;
                let trunc = cuts[j].as_ref().map_or(orig[j], |c| c.split_i);
                let _ = jn; // suppress unused warning
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
        // At each vertex where both adjacent edges are sharp, a small quad
        // (or triangle) fills the gap between the two truncated strips.
        //
        // The four vertices of the corner patch are:
        //   orig[i]           — the original corner vertex
        //   trunc_from_prev   — where the prev edge's cut exits onto this edge
        //                       (= split vi along the next edge = cuts[ip].split_j)
        //   inner[i]          — cut-line intersection
        //   trunc_from_next   — where the next edge's cut exits onto the prev edge
        //                       (= split vi along the prev edge = cuts[i].split_i)

        for i in 0..n {
            if !prev_sharp[i] || !next_sharp[i] {
                continue;
            }
            let ip = (i + n - 1) % n;

            let trunc_prev = cuts[ip].as_ref().map_or(orig[i], |c| c.split_j);
            let trunc_next = cuts[i].as_ref().map_or(orig[i], |c| c.split_i);

            // CW from outside: orig → trunc_next → inner → trunc_prev
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
        }
    }

    // -----------------------------------------------------------------------
    // Cross-face chamfer strips
    // -----------------------------------------------------------------------
    // For each sharp edge shared by two faces, bridge the gap between the two
    // faces' inner edges with a fillet-profile strip (two half-quads with a
    // pushed center-line).

    for &ek in &sharp_edges {
        let Some(info) = edge_graph.get(&ek) else {
            continue;
        };
        if info.faces.len() < 2 {
            continue;
        }

        for pair in info.faces.windows(2) {
            let fi_a = pair[0];
            let fi_b = pair[1];
            let na = solid.faces[fi_a].normal;
            let nb = solid.faces[fi_b].normal;

            let Some(&(inner_a0, inner_a1)) = edge_face_inners.get(&(ek, fi_a)) else {
                continue;
            };
            let Some(&(inner_b0, inner_b1)) = edge_face_inners.get(&(ek, fi_b)) else {
                continue;
            };

            let avg_n = (na + nb).normalize_or_zero();
            let push = fillet_push_amount(na, nb, CHAMFER_WIDTH);

            let mid0 = (inner_a0 + inner_b0) * 0.5 + avg_n * push;
            let mid1 = (inner_a1 + inner_b1) * 0.5 + avg_n * push;

            // Half-quad: face A inner → center-line
            // CW from outside: inner_a0, inner_a1, mid1, mid0
            emit_quad(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                inner_a0,
                inner_a1,
                mid1,
                mid0,
                (na + avg_n).normalize_or_zero(),
            );

            // Half-quad: center-line → face B inner
            // CW from outside: mid0, mid1, inner_b1, inner_b0
            emit_quad(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                mid0,
                mid1,
                inner_b1,
                inner_b0,
                (nb + avg_n).normalize_or_zero(),
            );
        }
    }

    let chamfer_offsets = vec![[0.0; 3]; positions.len()];
    ChunkMeshData {
        positions,
        normals,
        uvs,
        chamfer_offsets,
        indices,
    }
}
