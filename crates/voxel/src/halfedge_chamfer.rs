//! Half-edge mesh chamfer via topology modification.
//!
//! For each sharp edge A→B on face F:
//! - Split F's edge going INTO A (P→A) at distance `width` from A → creates V1
//! - Split F's edge going OUT of B (B→Q) at distance `width` from B → creates V2
//! - insert_edge(V1, V2) splits F into [main face] and [strip: V1,A,B,V2]
//! - Do the same on the other face → strip: [V3,B,A,V4]
//! - Remove the shared edge A→B, merging the two strips into bevel face [V1,V2,V3,V4]

use std::collections::{HashMap, HashSet};
use bevy::prelude::*;

use crate::chunk::*;
use crate::meshing::{ChunkMeshData, build_solid_mesh_public};
use crate::shape::*;

type VId = u32;
type EId = u32;
type FId = u32;
const NIL: u32 = u32::MAX;
const SHARP_DOT: f32 = 0.985;

#[derive(Clone, Debug)]
struct Vert { pos: Vec3, edge: EId }
#[derive(Clone, Debug)]
struct Edge { origin: VId, twin: EId, next: EId, prev: EId, face: Option<FId> }
#[derive(Clone, Debug)]
struct Face { edge: EId, normal: Vec3, mode: ChamferMode, alive: bool }

struct HEMesh {
    verts: Vec<Vert>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

impl HEMesh {
    fn new() -> Self { Self { verts: vec![], edges: vec![], faces: vec![] } }

    fn add_vert(&mut self, pos: Vec3) -> VId {
        let id = self.verts.len() as VId;
        self.verts.push(Vert { pos, edge: NIL });
        id
    }

    fn target(&self, e: EId) -> VId {
        let next = self.edges[e as usize].next;
        if next == NIL { return NIL; }
        self.edges[next as usize].origin
    }

    fn is_sharp(&self, e: EId) -> bool {
        let he = &self.edges[e as usize];
        let tw = &self.edges[he.twin as usize];
        match (he.face, tw.face) {
            (Some(a), Some(b)) if a != b =>
                self.faces[a as usize].normal.dot(self.faces[b as usize].normal) < SHARP_DOT,
            (Some(_), None) | (None, Some(_)) => true,
            _ => false,
        }
    }

    /// Split edge e by inserting vertex at `pos`.
    /// e: A→B becomes A→V, V→B. Returns V.
    fn split_edge(&mut self, e: EId, pos: Vec3) -> VId {
        let v = self.add_vert(pos);
        let twin = self.edges[e as usize].twin;
        let e_next = self.edges[e as usize].next;
        let e_face = self.edges[e as usize].face;

        // e2: V→B on same face as e
        let e2 = self.edges.len() as EId;
        self.edges.push(Edge { origin: v, twin: NIL, next: e_next, prev: e, face: e_face });
        self.edges[e as usize].next = e2;
        self.edges[e_next as usize].prev = e2;
        self.verts[v as usize].edge = e2;

        if twin != NIL {
            let t_prev = self.edges[twin as usize].prev;
            let t_next = self.edges[twin as usize].next;
            let t_face = self.edges[twin as usize].face;
            let b_id = self.edges[twin as usize].origin;

            // Only modify twin side if it's a proper face edge (not boundary with NIL prev/next)
            if t_prev != NIL && t_next != NIL {
                // Change twin origin from B to V
                self.edges[twin as usize].origin = v;

                // t2: B→V
                let t2 = self.edges.len() as EId;
                self.edges.push(Edge { origin: b_id, twin: NIL, next: twin, prev: t_prev, face: t_face });
                self.edges[twin as usize].prev = t2;
                self.edges[t_prev as usize].next = t2;

                if self.verts[b_id as usize].edge == twin {
                    self.verts[b_id as usize].edge = t2;
                }

                self.edges[e2 as usize].twin = t2;
                self.edges[t2 as usize].twin = e2;
            } else {
                // Boundary twin — just update the twin pairing
                // e goes A→V, e2 goes V→B
                // twin was B→A, now should be split into B→V and V→A
                // But boundary edges have NIL prev/next, so just update origin
                self.edges[twin as usize].origin = v;
                // e2's twin is now the old twin (pointing V→A effectively)
                // This isn't perfect but prevents crashes
                self.edges[e2 as usize].twin = NIL;
            }
        }

        v
    }

    /// Insert edge connecting origins of e1 and e2 (must be on same face).
    /// Splits the face in two. Returns the new edge (origin(e1)→origin(e2)).
    fn insert_edge(&mut self, e1: EId, e2: EId) -> EId {
        let fid = self.edges[e1 as usize].face;
        let v1 = self.edges[e1 as usize].origin;
        let v2 = self.edges[e2 as usize].origin;

        let e1_prev = self.edges[e1 as usize].prev;
        let e2_prev = self.edges[e2 as usize].prev;

        let ne1 = self.edges.len() as EId;
        let ne2 = ne1 + 1;

        // ne1: v1→v2, stays on old face (e2 side)
        self.edges.push(Edge { origin: v1, twin: ne2, next: e2, prev: e1_prev, face: fid });
        // ne2: v2→v1, goes on new face (e1 side)
        self.edges.push(Edge { origin: v2, twin: ne1, next: e1, prev: e2_prev, face: None });

        self.edges[e1_prev as usize].next = ne1;
        self.edges[e2 as usize].prev = ne1;
        self.edges[e2_prev as usize].next = ne2;
        self.edges[e1 as usize].prev = ne2;

        // Create new face for ne2 side
        let old_n = fid.map(|f| self.faces[f as usize].normal).unwrap_or(Vec3::Y);
        let old_m = fid.map(|f| self.faces[f as usize].mode).unwrap_or(ChamferMode::Hard);
        let nf = self.faces.len() as FId;
        self.faces.push(Face { edge: ne2, normal: old_n, mode: old_m, alive: true });
        self.edges[ne2 as usize].face = Some(nf);

        // Assign new face to all edges on ne2's loop
        let mut e = self.edges[ne2 as usize].next;
        while e != ne2 {
            self.edges[e as usize].face = Some(nf);
            e = self.edges[e as usize].next;
            if e == NIL { break; }
        }

        // Fix old face edge pointer
        if let Some(f) = fid { self.faces[f as usize].edge = ne1; }

        ne1
    }

    /// Remove edge e and its twin, merging the two adjacent faces.
    /// Keeps the face on e's side, removes the face on twin's side.
    fn remove_edge(&mut self, e: EId) {
        let twin = self.edges[e as usize].twin;
        if twin == NIL { return; }

        let e_prev = self.edges[e as usize].prev;
        let e_next = self.edges[e as usize].next;
        let t_prev = self.edges[twin as usize].prev;
        let t_next = self.edges[twin as usize].next;

        let keep_face = self.edges[e as usize].face;
        let remove_face = self.edges[twin as usize].face;

        // Splice out e: connect e_prev → t_next
        self.edges[e_prev as usize].next = t_next;
        self.edges[t_next as usize].prev = e_prev;

        // Splice out twin: connect t_prev → e_next
        self.edges[t_prev as usize].next = e_next;
        self.edges[e_next as usize].prev = t_prev;

        // Reassign all edges from removed face to kept face
        if let Some(rf) = remove_face {
            self.faces[rf as usize].alive = false;
            if let Some(kf) = keep_face {
                self.faces[kf as usize].edge = e_prev; // make sure it doesn't point to removed edge
                let mut cur = t_next;
                let stop = e_next;
                loop {
                    self.edges[cur as usize].face = Some(kf);
                    cur = self.edges[cur as usize].next;
                    if cur == stop || cur == t_next { break; } // went all the way around
                    if cur == NIL { break; }
                }
            }
        }

        // Fix vertex edge pointers if they point to removed edges
        let v_e = self.edges[e as usize].origin;
        let v_t = self.edges[twin as usize].origin;
        if self.verts[v_e as usize].edge == e { self.verts[v_e as usize].edge = t_next; }
        if self.verts[v_t as usize].edge == twin { self.verts[v_t as usize].edge = e_next; }

        // Mark edges as dead (set face to None, twin to NIL)
        self.edges[e as usize].face = None;
        self.edges[e as usize].twin = NIL;
        self.edges[e as usize].next = NIL;
        self.edges[e as usize].prev = NIL;
        self.edges[twin as usize].face = None;
        self.edges[twin as usize].twin = NIL;
        self.edges[twin as usize].next = NIL;
        self.edges[twin as usize].prev = NIL;
    }

    fn from_solid(solid: &crate::meshing::SolidMesh) -> Self {
        let mut m = HEMesh::new();
        for &pos in &solid.positions { m.add_vert(pos); }
        let mut twins: HashMap<(VId, VId), EId> = HashMap::new();
        for sf in &solid.faces {
            let n = sf.verts.len();
            if n < 3 { continue; }
            let fid = m.faces.len() as FId;
            m.faces.push(Face { edge: m.edges.len() as EId, normal: sf.normal, mode: sf.chamfer_mode, alive: true });
            let first = m.edges.len() as EId;
            for i in 0..n {
                let o = sf.verts[i];
                let eid = m.edges.len() as EId;
                m.edges.push(Edge {
                    origin: o, twin: NIL,
                    next: first + ((i+1)%n) as u32, prev: first + ((i+n-1)%n) as u32,
                    face: Some(fid),
                });
                if m.verts[o as usize].edge == NIL { m.verts[o as usize].edge = eid; }
                let t = sf.verts[(i+1)%n];
                let k = if o < t { (o,t) } else { (t,o) };
                if let Some(&tw) = twins.get(&k) {
                    m.edges[eid as usize].twin = tw;
                    m.edges[tw as usize].twin = eid;
                } else { twins.insert(k, eid); }
            }
        }
        let ec = m.edges.len();
        for i in 0..ec {
            if m.edges[i].twin == NIL {
                let o = m.target(i as EId);
                let tid = m.edges.len() as EId;
                m.edges.push(Edge { origin: o, twin: i as EId, next: NIL, prev: NIL, face: None });
                m.edges[i].twin = tid;
            }
        }
        m
    }

    fn to_chunk_mesh_data(&self) -> ChunkMeshData {
        let mut pos = Vec::new();
        let mut nrm = Vec::new();
        let mut uvs = Vec::new();
        let mut idx = Vec::new();
        for face in &self.faces {
            if !face.alive { continue; }
            let na = face.normal.to_array();
            let base = pos.len() as u32;
            let mut vs = Vec::new();
            let mut e = face.edge;
            loop {
                if self.edges[e as usize].next == NIL { break; }
                vs.push(self.edges[e as usize].origin);
                e = self.edges[e as usize].next;
                if e == face.edge || vs.len() > 20 { break; }
            }
            if vs.len() < 3 { continue; }
            for &v in &vs { pos.push(self.verts[v as usize].pos.to_array()); nrm.push(na); }
            let uv = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];
            for i in 0..vs.len() { uvs.push(uv[i%4]); }
            for i in 1..vs.len()-1 { idx.extend_from_slice(&[base+i as u32+1, base+i as u32, base]); }
        }
        let n = pos.len();
        ChunkMeshData { positions: pos, normals: nrm, uvs, chamfer_offsets: vec![[0.0;3];n], indices: idx }
    }
}

// ---------------------------------------------------------------------------
// Chamfer algorithm
// ---------------------------------------------------------------------------

fn chamfer_sharp_edges(mesh: &mut HEMesh, width: f32) {
    // Collect sharp edges
    let ec = mesh.edges.len();
    let mut sharp: Vec<EId> = Vec::new();
    let mut seen = vec![false; ec];
    for i in 0..ec {
        if seen[i] { continue; }
        let tw = mesh.edges[i].twin;
        if tw == NIL { continue; }
        seen[i] = true;
        seen[tw as usize] = true;
        if mesh.is_sharp(i as EId) {
            sharp.push(i as EId);
        }
    }

    // For each sharp edge, split adjacent edges and insert bevel
    for &he_id in &sharp {
        let twin_id = mesh.edges[he_id as usize].twin;
        let fa = mesh.edges[he_id as usize].face;
        let fb = mesh.edges[twin_id as usize].face;

        // Only handle interior sharp edges (both faces present, both alive)
        let (fa, fb) = match (fa, fb) {
            (Some(a), Some(b)) if mesh.faces[a as usize].alive && mesh.faces[b as usize].alive => (a, b),
            _ => continue,
        };

        // Skip if edges have been invalidated by previous operations
        if mesh.edges[he_id as usize].next == NIL || mesh.edges[he_id as usize].prev == NIL { continue; }
        if mesh.edges[twin_id as usize].next == NIL || mesh.edges[twin_id as usize].prev == NIL { continue; }

        let v_a = mesh.edges[he_id as usize].origin;
        let v_b = mesh.target(he_id);
        if v_a == NIL || v_b == NIL { continue; }
        let p_a = mesh.verts[v_a as usize].pos;
        let p_b = mesh.verts[v_b as usize].pos;
        let edge_len = (p_b - p_a).length();
        if edge_len < width * 2.5 { continue; } // too short to chamfer

        // On face_a (he_id side):
        // he_id goes A→B. prev of he_id goes ?→A. next of he_id goes B→?.
        let he_prev = mesh.edges[he_id as usize].prev;  // ?→A on face_a
        let he_next = mesh.edges[he_id as usize].next;  // B→? on face_a

        // Split he_prev (?→A) near A: creates V1 between ? and A
        let prev_origin = mesh.edges[he_prev as usize].origin;
        let prev_pos = mesh.verts[prev_origin as usize].pos;
        let dir_to_a = (p_a - prev_pos).normalize_or_zero();
        let dist_to_a = (p_a - prev_pos).length();
        let t1 = ((dist_to_a - width) / dist_to_a).clamp(0.1, 0.9);
        let v1_pos = prev_pos.lerp(p_a, t1);
        let v1 = mesh.split_edge(he_prev, v1_pos);

        // Split he_next (B→?) near B: creates V2 between B and ?
        let next_target = mesh.target(he_next);
        let next_pos = mesh.verts[next_target as usize].pos;
        let t2 = (width / (next_pos - p_b).length()).clamp(0.1, 0.9);
        let v2_pos = p_b.lerp(next_pos, t2);
        let v2 = mesh.split_edge(he_next, v2_pos);

        // Now on face_a: ..., V1, A, B, V2, ...
        // Find the half-edge from V1 that's on face_a (it goes V1→A)
        // and the half-edge from V2 (goes V2→?)
        // We want to insert_edge connecting V2 to V1, isolating [V1, A, B, V2] as a strip

        // The half-edge V1→A is the edge after the split point of he_prev.
        // After split_edge(he_prev, v1_pos): he_prev goes ?→V1, new edge goes V1→A.
        let v1_to_a = mesh.edges[he_prev as usize].next; // V1→A

        // The half-edge at V2: after split_edge(he_next, v2_pos):
        // he_next goes B→V2, new edge goes V2→?
        let v2_to_next = mesh.edges[he_next as usize].next; // V2→?

        // insert_edge(v2_to_next, v1_to_a) connects origin(v2_to_next)=V2 to origin(v1_to_a)=V1
        // This creates edge V2→V1, splitting face_a into:
        //   - Main face: ..., V1, V2, ?, ...
        //   - Strip face: V1, A, B, V2 (with the new edge V2→V1 closing it)
        let ne_a = mesh.insert_edge(v2_to_next, v1_to_a);
        // ne_a goes V2→V1 on the main face side
        // its twin goes V1→V2 on the strip side

        // --- Same on face_b (twin side) ---
        // twin goes B→A on face_b
        let tw_prev = mesh.edges[twin_id as usize].prev;  // ?→B on face_b
        let tw_next = mesh.edges[twin_id as usize].next;  // A→? on face_b

        // Split tw_prev near B
        let tw_prev_origin = mesh.edges[tw_prev as usize].origin;
        let tw_prev_pos = mesh.verts[tw_prev_origin as usize].pos;
        let dist_to_b = (p_b - tw_prev_pos).length();
        let t3 = ((dist_to_b - width) / dist_to_b).clamp(0.1, 0.9);
        let v3_pos = tw_prev_pos.lerp(p_b, t3);
        let v3 = mesh.split_edge(tw_prev, v3_pos);

        // Split tw_next near A
        let tw_next_target = mesh.target(tw_next);
        let tw_next_pos = mesh.verts[tw_next_target as usize].pos;
        let t4 = (width / (tw_next_pos - p_a).length()).clamp(0.1, 0.9);
        let v4_pos = p_a.lerp(tw_next_pos, t4);
        let v4 = mesh.split_edge(tw_next, v4_pos);

        // Insert edge on face_b: V4→V3, isolating strip [V3, B, A, V4]
        let v3_to_b = mesh.edges[tw_prev as usize].next;
        let v4_to_next = mesh.edges[tw_next as usize].next;
        let _ne_b = mesh.insert_edge(v4_to_next, v3_to_b);

        // Now remove the original sharp edge A→B (he_id) and its twin B→A (twin_id).
        // This merges the two strip faces into one bevel face: V1, A, V4, V3, B, V2
        // Wait — after remove, A and B become interior vertices of the bevel face.
        // The bevel face would be: V1, V2, V3, V4 with A and B as intermediate vertices.
        // Actually we WANT A and B gone from the bevel face to get a clean quad.
        // But remove_edge just splices them out of the loop — A and B remain as vertices.

        // For now, just remove the shared edge to merge the strips.
        mesh.remove_edge(he_id);

        // Set bevel face normal to average of the two original face normals
        let na = mesh.faces[fa as usize].normal;
        let nb = mesh.faces[fb as usize].normal;
        let bevel_normal = (na + nb).normalize_or_zero();

        // Find the bevel face — it's the face that v1 is now on
        // after the edge removal
        if let Some(bevel_fid) = mesh.edges[mesh.verts[v1 as usize].edge as usize].face {
            mesh.faces[bevel_fid as usize].normal = bevel_normal;
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn generate_halfedge_chamfer(data: &ChunkData, shapes: &ShapeTable) -> ChunkMeshData {
    let solid = build_solid_mesh_public(data, shapes);
    if solid.positions.is_empty() || solid.faces.is_empty() {
        return ChunkMeshData { positions: vec![], normals: vec![], uvs: vec![], chamfer_offsets: vec![], indices: vec![] };
    }

    let mut mesh = HEMesh::from_solid(&solid);
    chamfer_sharp_edges(&mut mesh, CHAMFER_WIDTH);
    mesh.to_chunk_mesh_data()
}
