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
/// if it disagrees with `expected_normal` the winding is flipped.  Also
/// detects "bowtie" quads (vertices not in cyclic order) and switches to
/// the v1–v3 diagonal when needed.
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

    // Check if the v0–v2 diagonal produces two triangles with consistent
    // winding.  If not, the vertices form a bowtie and we need the v1–v3
    // diagonal instead.
    let n02_a = (v1 - v0).cross(v2 - v0);
    let n02_b = (v2 - v0).cross(v3 - v0);

    if n02_a.dot(n02_b) >= 0.0 {
        // v0–v2 diagonal is valid.  Check winding against expected normal.
        if n02_a.dot(expected_normal) >= 0.0 {
            indices.extend_from_slice(&[base, base + 1, base + 2]);
            indices.extend_from_slice(&[base, base + 2, base + 3]);
        } else {
            indices.extend_from_slice(&[base + 2, base + 1, base]);
            indices.extend_from_slice(&[base, base + 3, base + 2]);
        }
    } else {
        // Bowtie — use v1–v3 diagonal instead.
        let n13 = (v0 - v1).cross(v3 - v1);
        if n13.dot(expected_normal) >= 0.0 {
            indices.extend_from_slice(&[base + 1, base + 2, base + 3]);
            indices.extend_from_slice(&[base + 1, base + 3, base]);
        } else {
            indices.extend_from_slice(&[base + 3, base + 2, base + 1]);
            indices.extend_from_slice(&[base, base + 3, base + 1]);
        }
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

    // -- Per-vertex sharp edge directions (for detecting external chamfering) -
    // For each vertex, store the directions of all sharp edges emanating from it.
    let mut vert_sharp_dirs: HashMap<u32, Vec<Vec3>> = HashMap::new();
    for &(a, b) in &sharp_edges {
        let dir = (solid.positions[b as usize] - solid.positions[a as usize]).normalize_or_zero();
        vert_sharp_dirs.entry(a).or_default().push(dir);
        vert_sharp_dirs.entry(b).or_default().push(-dir);
    }
    let chamfered_verts: HashSet<u32> = vert_sharp_dirs.keys().copied().collect();

    // -- Precompute fillet push vectors per sharp edge and vertex ------------
    // Each sharp edge gets a push along the bisector of its two face normals.
    // Each sharp vertex gets an averaged push from all its incident edges.
    let mut edge_push: HashMap<(u32, u32), Vec3> = HashMap::new();
    let mut edge_bisector: HashMap<(u32, u32), Vec3> = HashMap::new();
    for &ek in &sharp_edges {
        let Some(info) = edge_graph.get(&ek) else { continue };
        if info.faces.len() < 2 {
            continue;
        }
        let na = solid.faces[info.faces[0]].normal;
        let nb = solid.faces[info.faces[1]].normal;
        let avg_n = (na + nb).normalize_or_zero();
        let push = fillet_push_amount(na, nb, CHAMFER_WIDTH);
        edge_push.insert(ek, avg_n * push);
        edge_bisector.insert(ek, avg_n);
    }

    // Per sharp vertex: average push from all incident sharp edges.
    let mut vert_push: HashMap<u32, Vec3> = HashMap::new();
    let mut vert_count: HashMap<u32, usize> = HashMap::new();
    for (&ek, &push) in &edge_push {
        let (a, b) = ek;
        *vert_push.entry(a).or_insert(Vec3::ZERO) += push;
        *vert_push.entry(b).or_insert(Vec3::ZERO) += push;
        *vert_count.entry(a).or_default() += 1;
        *vert_count.entry(b).or_default() += 1;
    }
    for (&v, push) in vert_push.iter_mut() {
        let count = vert_count.get(&v).copied().unwrap_or(1);
        *push /= count as f32;
    }

    // -- Output buffers ------------------------------------------------------
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut chamfer_offsets: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    /// Pad chamfer_offsets with zeros to match positions length.
    fn pad_offsets(offsets: &mut Vec<[f32; 3]>, positions: &[[f32; 3]]) {
        while offsets.len() < positions.len() {
            offsets.push([0.0; 3]);
        }
    }

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
        // A vertex gets a fully-offset inner position (intersection of cut
        // lines from both adjacent edges) when:
        //   - Both adjacent edges are sharp (non-collinear), OR
        //   - The vertex has "external chamfering" — sharp edges from OTHER
        //     faces beyond those on this face.
        // A vertex with only one sharp edge on this face and no external
        // chamfering uses the simpler split_toward approach.

        // Per-vertex: does this vertex have sharp edges in directions NOT
        // already handled by this face's own sharp edges?  Only those
        // directions (with a significant in-plane component) require extra
        // corner-patch geometry.
        let has_external_chamfer = |i: usize| -> bool {
            let vi = face.verts[i];
            let Some(all_dirs) = vert_sharp_dirs.get(&vi) else {
                return false;
            };

            // Directions of sharp edges on THIS face at this vertex.
            let ip = (i + n - 1) % n;
            let j = (i + 1) % n;
            let mut face_dirs: Vec<Vec3> = Vec::new();
            if prev_sharp[i] {
                face_dirs.push((orig[i] - orig[ip]).normalize_or_zero());
            }
            if next_sharp[i] {
                face_dirs.push((orig[j] - orig[i]).normalize_or_zero());
            }

            for &d in all_dirs {
                // Skip edges mostly perpendicular to this face — they don't
                // create in-plane displacement that needs a cut.
                let in_plane = d - fn_ * d.dot(fn_);
                if in_plane.length_squared() < 0.01 {
                    continue;
                }

                // Skip edges parallel to a sharp edge already on this face.
                let parallel = face_dirs.iter().any(|fd| fd.dot(d).abs() > 0.95);
                if !parallel {
                    return true;
                }
            }
            false
        };

        let mut inner: Vec<Vec3> = Vec::with_capacity(n);

        for i in 0..n {
            let ip = (i + n - 1) % n;
            let j = (i + 1) % n;
            let vi = face.verts[i];

            if !chamfered_verts.contains(&vi) {
                inner.push(orig[i]);
                continue;
            }

            let both_sharp = prev_sharp[i] && next_sharp[i];
            let needs_full_offset = both_sharp || has_external_chamfer(i);

            // Collinear midpoint: both edges sharp and same line.
            if both_sharp {
                let prev_dir = (orig[i] - orig[ip]).normalize_or_zero();
                let next_dir = (orig[j] - orig[i]).normalize_or_zero();
                if prev_dir.dot(next_dir).abs() > 0.99 {
                    let inward = prev_dir.cross(fn_).normalize_or_zero();
                    inner.push(orig[i] + inward * CHAMFER_WIDTH);
                    continue;
                }
            }

            if needs_full_offset {
                // Full intersection of cut lines from both edges.
                let d_prev = (orig[i] - orig[ip]).normalize_or_zero();
                let d_next = (orig[j] - orig[i]).normalize_or_zero();

                let sp = if prev_sharp[i] && is_collinear(i, ip, i, j) {
                    perp_offset(i, i, j)
                } else {
                    split_toward(i, ip)
                };
                let sn = if next_sharp[i] && is_collinear(i, j, ip, i) {
                    perp_offset(i, ip, i)
                } else {
                    split_toward(i, j)
                };

                if let Some(pt) = intersect_lines_in_plane(sn, d_prev, sp, d_next, fn_) {
                    inner.push(pt);
                } else {
                    let inward = d_prev.cross(fn_).normalize_or_zero();
                    inner.push(orig[i] + inward * CHAMFER_WIDTH);
                }
            } else if prev_sharp[i] {
                // Only prev edge sharp, no external chamfer — simple split.
                if is_collinear(i, j, ip, i) {
                    inner.push(perp_offset(i, ip, i));
                } else {
                    inner.push(split_toward(i, j));
                }
            } else {
                // Only next edge sharp, no external chamfer — simple split.
                if is_collinear(i, ip, i, j) {
                    inner.push(perp_offset(i, i, j));
                } else {
                    inner.push(split_toward(i, ip));
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

        // Helper: detect collinear midpoint vertex.
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

        // Determine which vertices will get corner patches.
        let needs_patch: Vec<bool> = (0..n)
            .map(|i| {
                let vi = face.verts[i];
                chamfered_verts.contains(&vi)
                    && !is_collinear_midpoint(i)
                    && ((prev_sharp[i] && next_sharp[i]) || has_external_chamfer(i))
            })
            .collect();

        // Triangulate the inner polygon using emit_quad/emit_tri which
        // handle non-convex polygons and auto-correct winding.
        // The full intersection offset can make the polygon non-convex,
        // so we can't use triangulate_convex_polygon.
        let any_chamfered = (0..n).any(|i| chamfered_verts.contains(&face.verts[i]));

        if n == 3 {
            emit_tri(
                &mut positions, &mut normals, &mut uvs, &mut indices,
                inner[0], inner[1], inner[2], fn_,
            );
        } else if n == 4 {
            emit_quad(
                &mut positions, &mut normals, &mut uvs, &mut indices,
                inner[0], inner[1], inner[2], inner[3], fn_,
            );
        } else if !face.orig_triangles.is_empty() && !any_chamfered {
            // No vertices moved — safe to reuse original triangulation.
            let inner_base = positions.len() as u32;
            for i in 0..n {
                positions.push(inner[i].into());
                normals.push(fn_.into());
                uvs.push([0.0, 0.0]);
            }
            for tri in &face.orig_triangles {
                indices.push(inner_base + tri[2] as u32);
                indices.push(inner_base + tri[1] as u32);
                indices.push(inner_base + tri[0] as u32);
            }
        } else {
            // 5+ vertices with chamfered verts — fan from vertex 0 using
            // emit_tri for winding correction.
            for k in 1..n - 1 {
                emit_tri(
                    &mut positions, &mut normals, &mut uvs, &mut indices,
                    inner[0], inner[k], inner[k + 1], fn_,
                );
            }
        }

        let any_patch = needs_patch.iter().any(|&p| p);
        if !any_sharp && !any_patch {
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

            let vi = face.verts[i];
            let vj = face.verts[j];

            // --- vi end ---
            // Truncate if vi has a corner patch.  A corner patch exists when:
            //   - both edges sharp (non-collinear), OR
            //   - vertex has external chamfering (sharp edges from other faces)
            let (outer_i, inner_i) = if needs_patch[i] {
                let trunc = if prev_sharp[i] {
                    let ip = (i + n - 1) % n;
                    cuts[ip].as_ref().map_or(split_toward(i, j), |c| c.split_j)
                } else {
                    split_toward(i, j)
                };
                (trunc, inner[i])
            } else {
                (orig[i], inner[i])
            };

            // --- vj end ---
            let (outer_j, inner_j) = if needs_patch[j] {
                let trunc = if next_sharp[j] {
                    cuts[j].as_ref().map_or(split_toward(j, i), |c| c.split_i)
                } else {
                    split_toward(j, i)
                };
                (trunc, inner[j])
            } else {
                (orig[j], inner[j])
            };

            // Quad: outer_i → inner_i → inner_j → outer_j
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

        // -- Emit corner patches + gap triangles --------------------------------

        for i in 0..n {
            if !needs_patch[i] {
                continue;
            }

            let ip = (i + n - 1) % n;
            let j = (i + 1) % n;

            let split_on_next_edge = if prev_sharp[i] {
                cuts[ip].as_ref().map_or(split_toward(i, j), |c| c.split_j)
            } else {
                split_toward(i, j)
            };
            let split_on_prev_edge = if next_sharp[i] {
                cuts[i].as_ref().map_or(split_toward(i, ip), |c| c.split_i)
            } else {
                split_toward(i, ip)
            };

            // Corner patch quad:
            //   orig → split_on_prev_edge → inner → split_on_next_edge
            emit_quad(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                orig[i],
                split_on_prev_edge,
                inner[i],
                split_on_next_edge,
                fn_,
            );

            // Gap triangles on non-sharp edges.
            if !prev_sharp[i] {
                emit_tri(
                    &mut positions, &mut normals, &mut uvs, &mut indices,
                    inner[ip], split_on_prev_edge, inner[i], fn_,
                );
            }
            if !next_sharp[i] {
                emit_tri(
                    &mut positions, &mut normals, &mut uvs, &mut indices,
                    inner[i], split_on_next_edge, inner[j], fn_,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Compute fillet offsets + smooth normals
    // -----------------------------------------------------------------------
    // For every emitted vertex, check if it lies on a sharp edge or at a
    // sharp vertex.  If so, apply the fillet push and set a smooth normal.
    chamfer_offsets.resize(positions.len(), [0.0; 3]);

    for idx in 0..positions.len() {
        let p = Vec3::from_array(positions[idx]);

        // 1. Check if at an original sharp vertex (endpoint of sharp edges).
        //    Use averaged push from all incident sharp edges.
        let mut at_vert = false;
        for (&v, &push) in &vert_push {
            if (p - solid.positions[v as usize]).length_squared() < 1e-6 {
                chamfer_offsets[idx] = push.into();
                normals[idx] = push.normalize_or_zero().into();
                at_vert = true;
                break;
            }
        }
        if at_vert {
            continue;
        }

        // 2. Check if on a sharp edge (between two endpoints).
        for &(a, b) in &sharp_edges {
            let pa = solid.positions[a as usize];
            let pb = solid.positions[b as usize];
            let edge_vec = pb - pa;
            let edge_len_sq = edge_vec.length_squared();
            if edge_len_sq < 1e-8 {
                continue;
            }

            let t = (p - pa).dot(edge_vec) / edge_len_sq;
            if t < -0.01 || t > 1.01 {
                continue;
            }

            let closest = pa + edge_vec * t.clamp(0.0, 1.0);
            let dist_sq = (p - closest).length_squared();

            if dist_sq < 1e-6 {
                if let Some(&push) = edge_push.get(&edge_key(a, b)) {
                    chamfer_offsets[idx] = push.into();
                    if let Some(&bisect) = edge_bisector.get(&edge_key(a, b)) {
                        normals[idx] = bisect.into();
                    }
                }
                break;
            }
        }
    }

    // Apply offsets to positions so we see the filleted version.
    for i in 0..positions.len() {
        positions[i][0] += chamfer_offsets[i][0];
        positions[i][1] += chamfer_offsets[i][1];
        positions[i][2] += chamfer_offsets[i][2];
    }

    ChunkMeshData {
        positions,
        normals,
        uvs,
        chamfer_offsets,
        indices,
    }
}
