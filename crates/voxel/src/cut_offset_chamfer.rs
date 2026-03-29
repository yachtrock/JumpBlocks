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

/// Compute the two triangles for a quad.  Picks the diagonal (v0–v2 or
/// v1–v3) and winding so that:
///   - bowtie quads (vertices not in cyclic order) use the valid diagonal;
///   - non-planar quads fold OUTWARD relative to `expected_normal` (the
///     off-diagonal vertex behind the first triangle's plane) — quads on a
///     convex fillet surface must fold away from the surface, not into it.
fn quad_triangulation(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    expected_normal: Vec3,
    base: u32,
) -> [u32; 6] {
    let n02_a = (v1 - v0).cross(v2 - v0);
    let n02_b = (v2 - v0).cross(v3 - v0);
    let n13_a = (v2 - v1).cross(v3 - v1);
    let n13_b = (v3 - v1).cross(v0 - v1);

    let ok02 = n02_a.dot(n02_b) >= 0.0;
    let ok13 = n13_a.dot(n13_b) >= 0.0;

    // How far the opposite vertex sits IN FRONT of the outward-oriented
    // plane of the first triangle.  Positive = convexity violation.
    let front_dist = |n: Vec3, anchor: Vec3, other: Vec3| -> f32 {
        let nh = n.normalize_or_zero();
        let nh = if nh.dot(expected_normal) >= 0.0 { nh } else { -nh };
        (other - anchor).dot(nh)
    };
    let front02 = front_dist(n02_a, v0, v3);
    let front13 = front_dist(n13_a, v1, v0);

    let use02 = if ok02 && ok13 {
        front02 <= front13 + 1e-6
    } else {
        ok02
    };

    if use02 {
        if n02_a.dot(expected_normal) >= 0.0 {
            [base, base + 1, base + 2, base, base + 2, base + 3]
        } else {
            [base + 2, base + 1, base, base, base + 3, base + 2]
        }
    } else {
        // n13_a is the normal of the first emitted triangle [v1, v2, v3].
        if n13_a.dot(expected_normal) >= 0.0 {
            [base + 1, base + 2, base + 3, base + 1, base + 3, base]
        } else {
            [base + 3, base + 2, base + 1, base, base + 3, base + 1]
        }
    }
}

/// A quad emitted into the mesh: where its 4 vertices and 6 indices live,
/// plus the outward reference normal.  Quad diagonals are re-evaluated after
/// fillet offsets are known (see `generate_cut_offset_chamfer`), since the
/// fold direction only appears once vertices are pushed.
struct QuadRecord {
    base: u32,
    index_start: usize,
    expected_normal: Vec3,
}

/// Emit a quad as two triangles and record it for post-push re-triangulation.
/// Quads with coincident corners (cut points can coincide at T-vertices on
/// contact seams) collapse to a single triangle.
fn emit_quad(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    quads: &mut Vec<QuadRecord>,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    expected_normal: Vec3,
) {
    const EPS_SQ: f32 = 1e-10;
    let mut poly: Vec<Vec3> = Vec::with_capacity(4);
    for v in [v0, v1, v2, v3] {
        if poly.last().map_or(true, |l: &Vec3| l.distance_squared(v) > EPS_SQ) {
            poly.push(v);
        }
    }
    if poly.len() >= 2 && poly[0].distance_squared(*poly.last().unwrap()) <= EPS_SQ {
        poly.pop();
    }
    match poly.len() {
        0..=2 => return,
        3 => {
            emit_tri(positions, normals, uvs, indices, poly[0], poly[1], poly[2], expected_normal);
            return;
        }
        _ => {
            // Diagonal coincidence: a zero-width butterfly, nothing to emit.
            if poly[0].distance_squared(poly[2]) <= EPS_SQ
                || poly[1].distance_squared(poly[3]) <= EPS_SQ
            {
                return;
            }
        }
    }
    let (v0, v1, v2, v3) = (poly[0], poly[1], poly[2], poly[3]);

    let base = positions.len() as u32;
    positions.extend_from_slice(&[v0.into(), v1.into(), v2.into(), v3.into()]);
    let n: [f32; 3] = expected_normal.into();
    normals.extend_from_slice(&[n, n, n, n]);
    uvs.extend_from_slice(&[[0.0; 2]; 4]);

    quads.push(QuadRecord {
        base,
        index_start: indices.len(),
        expected_normal,
    });
    indices.extend_from_slice(&quad_triangulation(v0, v1, v2, v3, expected_normal, base));
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
/// The fillet has a UNIFORM radius of `CHAMFER_WIDTH` on every edge and
/// corner.  Each sharp edge is rounded by a cylinder of that radius tangent
/// to both faces, and each sharp corner by a sphere tangent to all incident
/// faces — so the setback (perpendicular distance from an edge to its cut
/// line) varies with the dihedral angle: `setback = radius / tan(θ/2)`.
///
/// For each face the algorithm performs *vertex splits* along existing edges:
///
/// 1. Every sharp edge gets a *cut line* parallel to it at its per-edge
///    setback.  Cut-line endpoints, strip truncations, and corner-patch
///    splits lie on original edges at a per-(edge, endpoint) arc-length
///    shared by all faces of that edge (so cross-face boundaries match).
/// 2. Where two cut lines meet (corner with two sharp edges) the intersection
///    point is computed via a 2D line–line intersection in the face plane.
/// 3. These cuts subdivide the original face into:
///      - **Inner polygon** (the shrunk interior)
///      - **Edge strips** (between each sharp edge and its cut line)
///      - **Corner patches** (where two strips meet at a vertex)
/// 4. Fillet offsets push original-edge vertices onto their edge's cylinder
///    and original corners onto their sphere, so every emitted vertex lies
///    exactly on the rounded surface.  Quad diagonals are then re-picked with
///    the offsets applied so non-planar quads fold outward.
pub fn generate_cut_offset_chamfer(
    data: &ChunkData,
    neighbors: &ChunkNeighbors,
    shapes: &ShapeTable,
) -> ChunkMeshData {
    let mut solid = build_solid_mesh_public(data, neighbors, shapes);

    // -- Build connected components of face-adjacent blocks -------------------
    // Blocks that only share a vertex or edge (not a full face) should not
    // chamfer together. We union-find blocks by face-adjacency so that edges
    // between different components are suppressed.
    let block_component = {
        // Collect unique blocks from the chunk data — NOT from surviving
        // faces.  Fully-occluded blocks (or blocks whose faces merged into a
        // neighbor's) still bridge their neighbors into one component.
        let mut blocks: Vec<((usize, usize, usize), (u8, u8, u8))> = Vec::new();
        for block in &data.blocks {
            let Some(shape) = shapes.get(block.shape) else { continue };
            let (ox, oy, oz) = block.origin;
            let voxel = (ox as usize, oy as usize, oz as usize);
            if !blocks.iter().any(|(v, _)| *v == voxel) {
                blocks.push((voxel, shape.size));
            }
        }
        // Faces may reference neighbor-chunk voxels not present in data.
        for face in &solid.faces {
            if !blocks.iter().any(|(v, _)| *v == face.voxel) {
                blocks.push((face.voxel, face.block_size));
            }
        }
        // Simple union-find via labels
        let n = blocks.len();
        let mut label: Vec<usize> = (0..n).collect();

        fn find(label: &mut Vec<usize>, mut i: usize) -> usize {
            while label[i] != i {
                label[i] = label[label[i]];
                i = label[i];
            }
            i
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if blocks_are_face_adjacent(blocks[i].0, blocks[i].1, blocks[j].0, blocks[j].1) {
                    let ri = find(&mut label, i);
                    let rj = find(&mut label, j);
                    if ri != rj {
                        label[ri] = rj;
                    }
                }
            }
        }
        // Build map: voxel_origin → component root
        let mut map: HashMap<(usize, usize, usize), usize> = HashMap::new();
        for i in 0..n {
            let root = find(&mut label, i);
            map.insert(blocks[i].0, root);
        }
        map
    };

    // -- Split vertices shared between different components -------------------
    // When blocks only touch at a vertex or edge, the solid mesh merges their
    // vertices. We duplicate those shared vertices so each component gets its
    // own copies, making the chamfer pipelines fully independent.
    // Collect split vertex indices so fillet application can avoid cross-component matching.
    let mut split_vert_indices: HashSet<u32> = HashSet::new();
    {
        // Build: vertex → set of components that use it
        let mut vert_comps: HashMap<u32, HashSet<usize>> = HashMap::new();
        for face in &solid.faces {
            let comp = block_component.get(&face.voxel).copied().unwrap_or(0);
            for &vi in &face.verts {
                vert_comps.entry(vi).or_default().insert(comp);
            }
        }

        // For vertices used by 2+ components, create duplicates for all but the first
        let mut remap: HashMap<(u32, usize), u32> = HashMap::new(); // (orig_vert, component) → new_vert
        for (&vi, comps) in &vert_comps {
            if comps.len() < 2 {
                continue;
            }
            // First component keeps the original vertex, others get duplicates
            for (idx, &comp) in comps.iter().enumerate() {
                if idx == 0 {
                    remap.insert((vi, comp), vi);
                } else {
                    let new_idx = solid.positions.len() as u32;
                    solid.positions.push(solid.positions[vi as usize]);
                    remap.insert((vi, comp), new_idx);
                }
            }
        }

        // Record all split vertex indices (originals + duplicates)
        for (&(_vi, _comp), &new_vi) in &remap {
            split_vert_indices.insert(new_vi);
        }

        // Rewrite face vertex indices using the remap
        let split_count = remap.len();
        let shared_verts = vert_comps.iter().filter(|(_, c)| c.len() >= 2).count();
        if shared_verts > 0 {
            warn!("[vertex-split] {} shared vertices across components, {} remap entries", shared_verts, split_count);
        }
        if !remap.is_empty() {
            for face in &mut solid.faces {
                let comp = block_component.get(&face.voxel).copied().unwrap_or(0);
                for vi in &mut face.verts {
                    if let Some(&new_vi) = remap.get(&(*vi, comp)) {
                        *vi = new_vi;
                    }
                }
            }
        }
    }

    // Rebuild edge graph after vertex splitting
    let edge_graph = build_edge_graph(&solid);

    // -- Identify sharp edges (excluding boundary-at-neighbor-seam) ----------
    let sharp_edges: HashSet<(u32, u32)> = edge_graph
        .iter()
        .filter(|&(&(a, b), ref info)| {
            if !is_sharp(info, &solid) {
                return false;
            }
            // Suppress chamfer at edges where any face belongs to a
            // different connected component (vertex/edge-only contact).
            if info.faces.len() >= 2 {
                let comp0 = block_component.get(&solid.faces[info.faces[0]].voxel);
                let has_cross = info.faces.iter().skip(1).any(|&fi| {
                    block_component.get(&solid.faces[fi].voxel) != comp0
                });
                if has_cross {
                    return false;
                }
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
    // For each vertex, store the directions of all sharp edges emanating from it,
    // tagged with the connected component so cross-component edges are ignored.
    let mut vert_sharp_dirs: HashMap<u32, Vec<(Vec3, usize)>> = HashMap::new();
    for &(a, b) in &sharp_edges {
        let dir = (solid.positions[b as usize] - solid.positions[a as usize]).normalize_or_zero();
        // Find component from any face that owns this edge
        let edge_info = edge_graph.get(&edge_key(a, b)).unwrap();
        let comp = block_component.get(&solid.faces[edge_info.faces[0]].voxel).copied().unwrap_or(usize::MAX);
        vert_sharp_dirs.entry(a).or_default().push((dir, comp));
        vert_sharp_dirs.entry(b).or_default().push((-dir, comp));
    }
    let chamfered_verts: HashSet<u32> = vert_sharp_dirs.keys().copied().collect();

    // -- Precompute fillet push vectors per sharp edge and vertex ------------
    // Each sharp edge gets a push along the bisector of its two face normals,
    // plus a per-edge SETBACK so that every edge fillet has a uniform radius
    // of CHAMFER_WIDTH regardless of dihedral angle.
    // Each sharp vertex gets a push onto a sphere of the same radius.
    let mut edge_push: HashMap<(u32, u32), Vec3> = HashMap::new();
    let mut edge_bisector: HashMap<(u32, u32), Vec3> = HashMap::new();
    let mut edge_setback: HashMap<(u32, u32), f32> = HashMap::new();
    for &ek in &sharp_edges {
        let Some(info) = edge_graph.get(&ek) else { continue };
        if info.faces.len() < 2 {
            continue;
        }
        let (va, vb) = ek;
        let fi_a = info.faces[0];
        let fi_b = info.faces[1];
        let na = solid.faces[fi_a].normal;
        let nb = solid.faces[fi_b].normal;
        let avg_n = (na + nb).normalize_or_zero();
        let push_amount = fillet_push_amount(na, nb, CHAMFER_WIDTH);
        edge_setback.insert(ek, fillet_setback_amount(na, nb, CHAMFER_WIDTH));

        // Detect convex vs concave using the signed dihedral angle.
        // (n1 × n2) · edge_dir gives the sine of the dihedral angle with
        // correct sign: positive → convex, negative → concave.
        let pa = solid.positions[va as usize];
        let pb = solid.positions[vb as usize];
        let edge_dir = (pb - pa).normalize_or_zero();
        // Signed dihedral: (na × nb) · edge_dir.
        // The sign depends on which face is "left" vs "right" of the edge.
        // Determine this from face A's vertex winding: if face A traverses
        // the edge va→vb in its polygon order, it's on the left.
        let face_a_verts = &solid.faces[fi_a].verts;
        let n = face_a_verts.len();
        let face_a_forward = (0..n).any(|i| {
            face_a_verts[i] == va && face_a_verts[(i + 1) % n] == vb
        });
        let signed_dihedral = na.cross(nb).dot(edge_dir);
        let is_convex = if face_a_forward {
            signed_dihedral < 0.0
        } else {
            signed_dihedral > 0.0
        };

        // Convex: push inward (-bisector). Concave: push outward (+bisector).
        let sign = if is_convex { -1.0 } else { 1.0 };
        edge_push.insert(ek, avg_n * sign * push_amount);
        // Bisector always unsigned — used for surface normals (always outward).
        edge_bisector.insert(ek, avg_n);
    }

    // -- Shared split arc-lengths per (edge, endpoint) ------------------------
    // Split points (cut-line endpoints, strip truncations, corner-patch
    // splits) lie ON original edges shared between faces.  Both faces of an
    // edge must place them at the SAME arc-length or the mesh cracks.  For
    // each (edge, endpoint) pair we take the max over every incident sharp
    // edge's cut-line crossing:  arc = setback / sin(angle between edges).
    // Edges with no request fall back to CHAMFER_WIDTH (the 90° value).
    let split_arc: HashMap<((u32, u32), u32), f32> = {
        let mut map: HashMap<((u32, u32), u32), f32> = HashMap::new();
        let mut register = |sharp_v: u32, adj_v: u32, arc: f32| {
            let e = map.entry((edge_key(sharp_v, adj_v), sharp_v)).or_insert(0.0);
            if arc > *e {
                *e = arc;
            }
        };
        for face in &solid.faces {
            let n = face.verts.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let vi = face.verts[i];
                let vj = face.verts[j];
                let ek = edge_key(vi, vj);
                if !sharp_edges.contains(&ek) {
                    continue;
                }
                let Some(&d) = edge_setback.get(&ek) else { continue };
                let pi = solid.positions[vi as usize];
                let pj = solid.positions[vj as usize];
                let edge_dir = (pj - pi).normalize_or_zero();

                // Endpoint vi: the cut line crosses the previous edge (vi→vp).
                let vp = face.verts[(i + n - 1) % n];
                let adj = (solid.positions[vp as usize] - pi).normalize_or_zero();
                let sin = adj.cross(edge_dir).length();
                if sin > 0.2 {
                    register(vi, vp, (d / sin).min(CHAMFER_WIDTH * 2.5));
                }

                // Endpoint vj: the cut line crosses the next edge (vj→vn).
                let vn = face.verts[(j + 1) % n];
                let adj = (solid.positions[vn as usize] - pj).normalize_or_zero();
                let sin = adj.cross(edge_dir).length();
                if sin > 0.2 {
                    register(vj, vn, (d / sin).min(CHAMFER_WIDTH * 2.5));
                }
            }
        }
        map
    };

    // Per sharp vertex: compute push from a sphere of radius R tangent to
    // all incident face planes.  Direction from incident edge pushes.
    let mut vert_push: HashMap<u32, Vec3> = HashMap::new();
    let mut vert_bisector: HashMap<u32, Vec3> = HashMap::new();
    {
        // Collect unique face normals per vertex.
        let mut vert_face_normals: HashMap<u32, Vec<Vec3>> = HashMap::new();
        // Accumulate signed push direction to determine convex/concave.
        let mut vert_push_dir: HashMap<u32, Vec3> = HashMap::new();
        let mut vert_edge_count: HashMap<u32, usize> = HashMap::new();

        for &(a, b) in &sharp_edges {
            let ek = edge_key(a, b);
            let Some(info) = edge_graph.get(&ek) else { continue };
            for &fi in &info.faces {
                let n = solid.faces[fi].normal;
                for &v in &[a, b] {
                    let norms = vert_face_normals.entry(v).or_default();
                    if !norms.iter().any(|existing| existing.dot(n) > 0.99) {
                        norms.push(n);
                    }
                }
            }
            if let Some(&push) = edge_push.get(&ek) {
                *vert_push_dir.entry(a).or_insert(Vec3::ZERO) += push;
                *vert_push_dir.entry(b).or_insert(Vec3::ZERO) += push;
                *vert_edge_count.entry(a).or_default() += 1;
                *vert_edge_count.entry(b).or_default() += 1;
            }
        }

        for (&v, face_normals) in &vert_face_normals {
            if face_normals.len() < 2 {
                continue;
            }

            let r = CHAMFER_WIDTH;

            let avg_n: Vec3 = face_normals.iter().copied().sum::<Vec3>().normalize_or_zero();

            // For a single convex block, all vertices are convex.
            // Use the same sign as the edge convex detection at this vertex.
            // Count how many incident edges are convex vs concave.
            let mut convex_edges = 0i32;
            let mut concave_edges = 0i32;
            for &(a, b) in &sharp_edges {
                if a != v && b != v { continue; }
                let ek = edge_key(a, b);
                if let Some(&push) = edge_push.get(&ek) {
                    if let Some(&bisect) = edge_bisector.get(&ek) {
                        // Convex edges have push opposite to bisector
                        if push.dot(bisect) < 0.0 {
                            convex_edges += 1;
                        } else {
                            concave_edges += 1;
                        }
                    }
                }
            }
            // Mixed corner (both convex and concave incident edges): no
            // single sphere is tangent to all faces — a signed solve pushes
            // the vertex against the concave crest and pinches.  Instead,
            // place the vertex where the incident edges' fillet CREST LINES
            // meet: each sharp edge's crest is the edge translated by its
            // own push vector (already correctly signed per edge).  Least
            // squares over crest lines:  min Σ_e |(I - d dᵀ)(δ - push_e)|².
            if convex_edges > 0 && concave_edges > 0 {
                let mut a = bevy::math::Mat3::ZERO;
                let mut rhs = Vec3::ZERO;
                let mut bis_sum = Vec3::ZERO;
                for &(ea, eb) in &sharp_edges {
                    if ea != v && eb != v {
                        continue;
                    }
                    let ek = edge_key(ea, eb);
                    let Some(&push) = edge_push.get(&ek) else { continue };
                    let d = (solid.positions[eb as usize] - solid.positions[ea as usize])
                        .normalize_or_zero();
                    // Projector onto the plane perpendicular to the edge.
                    let outer = bevy::math::Mat3::from_cols(d * d.x, d * d.y, d * d.z);
                    let proj = bevy::math::Mat3::IDENTITY - outer;
                    a += proj;
                    rhs += proj * push;
                    if let Some(&bis) = edge_bisector.get(&ek) {
                        bis_sum += bis;
                    }
                }
                if a.determinant().abs() > 1e-6 {
                    vert_push.insert(v, a.inverse() * rhs);
                    vert_bisector.insert(v, bis_sum.normalize_or_zero());
                }
                continue;
            }

            let sign = if convex_edges >= concave_edges { -1.0 } else { 1.0 };

            // Solve N δ = sign * r for the sphere center offset δ.  With a
            // UNIFORM fillet radius r on every edge, the corner surface is an
            // exact sphere of radius r tangent to all incident face planes —
            // consistent with the edge cylinders by construction.
            let delta = if face_normals.len() == 3 {
                let n = bevy::math::Mat3::from_cols(
                    face_normals[0],
                    face_normals[1],
                    face_normals[2],
                ).transpose();
                let det = n.determinant();
                if det.abs() < 1e-6 {
                    continue;
                }
                n.inverse() * Vec3::splat(sign * r)
            } else if face_normals.len() == 2 {
                // Mid-edge vertex on a straight crease: the surface is the
                // edge CYLINDER, center at r/sin(θ/2) along the bisector.
                // This makes the vertex push match the edge push exactly.
                let k = (face_normals[0] + face_normals[1]).length();
                let sin_half = (k / 2.0).clamp(0.05, 1.0);
                avg_n * sign * (r / sin_half)
            } else {
                // Fallback for 4+ faces: least-squares-ish along the average.
                let cos_a = face_normals.iter()
                    .map(|n| n.dot(avg_n))
                    .sum::<f32>() / face_normals.len() as f32;
                let sin_sq = (1.0 - cos_a * cos_a).max(0.0);
                if sin_sq < 1e-6 { continue; }
                let sin_a = sin_sq.sqrt();
                avg_n * sign * r / sin_a
            };

            let delta_len = delta.length();
            if delta_len < 1e-6 {
                continue;
            }

            // The vertex offset: from original corner to the sphere surface.
            let offset = delta * (1.0 - r / delta_len).max(0.0);

            vert_push.insert(v, offset);
            // Exact sphere normal at the pushed point: from center toward
            // the surface point (outward for convex, inward for concave).
            vert_bisector.insert(v, (delta * sign).normalize_or_zero());
        }
    }

    // -- Build vertex-to-component map for fillet application ----------------
    let solid_vert_comp: HashMap<u32, usize> = {
        let mut m: HashMap<u32, usize> = HashMap::new();
        for face in solid.faces.iter() {
            let comp = block_component.get(&face.voxel).copied().unwrap_or(0);
            for &vi in &face.verts {
                m.insert(vi, comp);
            }
        }
        m
    };

    // -- Output buffers ------------------------------------------------------
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut chamfer_offsets: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // Emitted quads, re-triangulated after fillet offsets are known.
    let mut quads: Vec<QuadRecord> = Vec::new();
    // Component tag per emitted vertex — used to prevent cross-component
    // fillet push contamination at split vertices.
    let mut emitted_comp: Vec<usize> = Vec::new();

    // Per (edge_key, face_index) → (inner_v_at_a, inner_v_at_b) in edge-key
    // order.  Will be used for cross-face strip emission in future phases.
    let mut _edge_face_inners: HashMap<((u32, u32), usize), (Vec3, Vec3)> = HashMap::new();

    // -----------------------------------------------------------------------
    // Per-face: subdivide via vertex splits
    // -----------------------------------------------------------------------
    for (fi, face) in solid.faces.iter().enumerate() {
        let n = face.verts.len();
        let fn_ = face.normal;
        let face_comp_val = block_component.get(&face.voxel).copied().unwrap_or(0);

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

        // Per-edge setback: distance from a sharp edge to its cut line.
        let setback_of = |a: usize, b: usize| -> f32 {
            edge_setback
                .get(&edge_key(face.verts[a], face.verts[b]))
                .copied()
                .unwrap_or(CHAMFER_WIDTH)
        };

        // Shared split arc-length from vertex i along the edge toward target.
        let arc_of = |i: usize, target_idx: usize| -> f32 {
            split_arc
                .get(&(edge_key(face.verts[i], face.verts[target_idx]), face.verts[i]))
                .copied()
                .unwrap_or(CHAMFER_WIDTH)
        };

        // Split vertex vi by sliding toward target along the adjacent edge.
        // Uses the shared per-(edge, endpoint) arc-length so both faces of
        // the edge place the split at the same position.
        let split_toward = |i: usize, target_idx: usize| -> Vec3 {
            let dir = (orig[target_idx] - orig[i]).normalize_or_zero();
            orig[i] + dir * arc_of(i, target_idx)
        };

        // Point at the sharp edge's exact cut line: perpendicular offset of
        // vertex i from the edge (edge_start→edge_end) by that edge's setback.
        // Used for exact cut lines and for collinear midpoint vertices.
        let perp_offset = |i: usize, edge_start: usize, edge_end: usize| -> Vec3 {
            let edge_dir = (orig[edge_end] - orig[edge_start]).normalize_or_zero();
            let inward = edge_dir.cross(fn_).normalize_or_zero();
            orig[i] + inward * setback_of(edge_start, edge_end)
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
        let face_comp = block_component.get(&face.voxel).copied().unwrap_or(usize::MAX);
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

            for &(d, comp) in all_dirs {
                // Only consider sharp edges from the same connected component.
                if comp != face_comp {
                    continue;
                }

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

            let both_sharp = prev_sharp[i] && next_sharp[i];
            let has_ext = has_external_chamfer(i);
            let needs_full_offset = both_sharp || has_ext;

            if !prev_sharp[i] && !next_sharp[i] && !has_ext {
                inner.push(orig[i]);
                continue;
            }

            // Collinear midpoint: both edges sharp and same line.
            if both_sharp {
                let prev_dir = (orig[i] - orig[ip]).normalize_or_zero();
                let next_dir = (orig[j] - orig[i]).normalize_or_zero();
                if prev_dir.dot(next_dir).abs() > 0.99 {
                    let inward = prev_dir.cross(fn_).normalize_or_zero();
                    inner.push(orig[i] + inward * setback_of(ip, i));
                    continue;
                }
            }

            if needs_full_offset {
                // Full intersection of the cut lines from both edges.  Sharp
                // edges use their EXACT cut line (parallel at the per-edge
                // setback); non-sharp edges (external chamfer) use a pseudo
                // cut through the shared split point on the opposite edge.
                let d_prev = (orig[i] - orig[ip]).normalize_or_zero();
                let d_next = (orig[j] - orig[i]).normalize_or_zero();

                // Point on the NEXT edge's cut line (parallel to d_next).
                let sp = if next_sharp[i] {
                    perp_offset(i, i, j)
                } else if is_collinear(i, ip, i, j) {
                    perp_offset(i, i, j)
                } else {
                    split_toward(i, ip)
                };
                // Point on the PREV edge's cut line (parallel to d_prev).
                let sn = if prev_sharp[i] {
                    perp_offset(i, ip, i)
                } else if is_collinear(i, j, ip, i) {
                    perp_offset(i, ip, i)
                } else {
                    split_toward(i, j)
                };

                if let Some(pt) = intersect_lines_in_plane(sn, d_prev, sp, d_next, fn_) {
                    inner.push(pt);
                } else {
                    let inward = d_prev.cross(fn_).normalize_or_zero();
                    inner.push(orig[i] + inward * setback_of(ip, i));
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
        // A collinear midpoint normally passes through untruncated, but if
        // OTHER faces have sharp edges ending at it (e.g. a block leaning
        // against a longer merged run), they truncate and patch there — this
        // face must match or the boundary chains crack.
        let needs_patch: Vec<bool> = (0..n)
            .map(|i| {
                let vi = face.verts[i];
                chamfered_verts.contains(&vi)
                    && (!is_collinear_midpoint(i) || has_external_chamfer(i))
                    && ((prev_sharp[i] && next_sharp[i]) || has_external_chamfer(i))
            })
            .collect();

        // Triangulate the inner polygon using emit_quad/emit_tri which
        // handle non-convex polygons and auto-correct winding.
        // The full intersection offset can make the polygon non-convex,
        // so we can't use triangulate_convex_polygon.
        if n == 3 {
            emit_tri(
                &mut positions, &mut normals, &mut uvs, &mut indices,
                inner[0], inner[1], inner[2], fn_,
            );
        } else if n == 4 {
            emit_quad(
                &mut positions, &mut normals, &mut uvs, &mut indices, &mut quads,
                inner[0], inner[1], inner[2], inner[3], fn_,
            );
        } else if !face.orig_triangles.is_empty() {
            // Use original triangulation — preserves all boundary vertices
            // (like collinear midpoints) so edges match adjacent faces.
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
            // Fallback for faces without orig_triangles.
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

            // --- vi end ---
            // Truncate if vi has a corner patch.  A corner patch exists when:
            //   - both edges sharp (non-collinear), OR
            //   - vertex has external chamfering (sharp edges from other faces)
            let (outer_i, inner_i) = if needs_patch[i] {
                let trunc = if prev_sharp[i] && !is_collinear_midpoint(i) {
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
                let trunc = if next_sharp[j] && !is_collinear_midpoint(j) {
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
                &mut quads,
                outer_i,
                inner_i,
                inner_j,
                outer_j,
                fn_,
            );
        }

        // -- Emit corner patches + gap geometry ---------------------------------

        // Precompute split points for each patched vertex so we can
        // reference a neighbor's split when building gap quads.
        let mut patch_split_next: Vec<Option<Vec3>> = vec![None; n];
        let mut patch_split_prev: Vec<Option<Vec3>> = vec![None; n];

        for i in 0..n {
            if !needs_patch[i] {
                continue;
            }
            let ip = (i + n - 1) % n;
            let j = (i + 1) % n;

            patch_split_next[i] = Some(if prev_sharp[i] && !is_collinear_midpoint(i) {
                cuts[ip].as_ref().map_or(split_toward(i, j), |c| c.split_j)
            } else {
                split_toward(i, j)
            });
            patch_split_prev[i] = Some(if next_sharp[i] && !is_collinear_midpoint(i) {
                cuts[i].as_ref().map_or(split_toward(i, ip), |c| c.split_i)
            } else {
                split_toward(i, ip)
            });
        }

        for i in 0..n {
            if !needs_patch[i] {
                continue;
            }

            let ip = (i + n - 1) % n;
            let j = (i + 1) % n;

            let split_on_next_edge = patch_split_next[i].unwrap();
            let split_on_prev_edge = patch_split_prev[i].unwrap();

            // Corner patch quad:
            //   orig → split_on_prev_edge → inner → split_on_next_edge
            emit_quad(
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                &mut quads,
                orig[i],
                split_on_prev_edge,
                inner[i],
                split_on_next_edge,
                fn_,
            );

            // Gap geometry on non-sharp edges.
            // When both endpoints have patches, emit a quad connecting all
            // four points instead of two separate triangles (which leave a
            // diamond-shaped gap between the split points).
            if !prev_sharp[i] {
                if needs_patch[ip] {
                    // Both endpoints patched — vertex ip emits this quad in
                    // its next-edge pass; never emit from the prev side.
                } else {
                    emit_tri(
                        &mut positions, &mut normals, &mut uvs, &mut indices,
                        inner[ip], split_on_prev_edge, inner[i], fn_,
                    );
                }
            }
            if !next_sharp[i] {
                if needs_patch[j] {
                    // Both endpoints patched — the next side always emits.
                    let split_j = patch_split_prev[j].unwrap();
                    emit_quad(
                        &mut positions, &mut normals, &mut uvs, &mut indices, &mut quads,
                        inner[i], split_on_next_edge, split_j, inner[j], fn_,
                    );
                } else {
                    emit_tri(
                        &mut positions, &mut normals, &mut uvs, &mut indices,
                        inner[i], split_on_next_edge, inner[j], fn_,
                    );
                }
            }
        }

        // Tag all vertices emitted for this face with its component.
        emitted_comp.resize(positions.len(), face_comp_val);
    }

    // -----------------------------------------------------------------------
    // Compute fillet offsets + smooth normals
    // -----------------------------------------------------------------------
    // For every emitted vertex, check if it lies on a sharp edge or at a
    // sharp vertex.  If so, apply the fillet push and set a smooth normal.
    chamfer_offsets.resize(positions.len(), [0.0; 3]);

    // Capture normals before chamfer smoothing — these are the original flat
    // face normals that we'll blend back to when chamfer_amount goes to 0.
    let sharp_normals = normals.clone();

    for idx in 0..positions.len() {
        let p = Vec3::from_array(positions[idx]);

        // 1. Check if at an original sharp vertex (endpoint of sharp edges).
        //    Only match against vertices from the same connected component
        //    to prevent cross-component fillet contamination at split positions.
        let my_comp = emitted_comp.get(idx).copied().unwrap_or(usize::MAX);
        let mut at_vert = false;
        for (&v, &push) in &vert_push {
            if solid_vert_comp.get(&v).copied().unwrap_or(usize::MAX) != my_comp {
                continue;
            }
            if (p - solid.positions[v as usize]).length_squared() < 1e-6 {
                chamfer_offsets[idx] = push.into();
                if let Some(&bisect) = vert_bisector.get(&v) {
                    normals[idx] = bisect.into();
                }
                at_vert = true;
                break;
            }
        }
        if at_vert {
            continue;
        }

        // 2. Check if on a sharp edge (between two endpoints).
        //    Only match against edges from the same connected component.
        for &(a, b) in &sharp_edges {
            if solid_vert_comp.get(&a).copied().unwrap_or(usize::MAX) != my_comp {
                continue;
            }
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
                let ek = edge_key(a, b);
                if let Some(&push) = edge_push.get(&ek) {
                    chamfer_offsets[idx] = push.into();
                    if let Some(&bisect) = edge_bisector.get(&ek) {
                        normals[idx] = bisect.into();
                    }
                }
                break;
            }
        }
    }

    // -- Re-triangulate quads with the fillet offsets applied -----------------
    // Quads are planar at emission time (they lie in their face plane); the
    // fillet pushes fold them.  Re-pick each quad's diagonal using the pushed
    // positions so the fold goes outward instead of cutting into the surface.
    for quad in &quads {
        let b = quad.base as usize;
        let pushed = |k: usize| -> Vec3 {
            Vec3::from_array(positions[b + k]) + Vec3::from_array(chamfer_offsets[b + k])
        };
        let tri = quad_triangulation(
            pushed(0),
            pushed(1),
            pushed(2),
            pushed(3),
            quad.expected_normal,
            quad.base,
        );
        indices[quad.index_start..quad.index_start + 6].copy_from_slice(&tri);
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
        sharp_normals,
        uvs,
        chamfer_offsets,
        indices,
    }
}
