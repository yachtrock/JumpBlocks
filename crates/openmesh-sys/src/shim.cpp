#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/Traits.hh>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include "shim.h"

// Use float-based traits (game engine typical)
struct GameTraits : public OpenMesh::DefaultTraits {
    typedef OpenMesh::Vec3f Point;
    typedef OpenMesh::Vec3f Normal;
};

typedef OpenMesh::PolyMesh_ArrayKernelT<GameTraits> PolyMesh;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static OpenMesh::Vec3f face_normal(PolyMesh& mesh, PolyMesh::FaceHandle fh) {
    // Newell's method for polygon normal
    OpenMesh::Vec3f n(0, 0, 0);
    std::vector<OpenMesh::Vec3f> pts;
    for (auto fv = mesh.fv_begin(fh); fv != mesh.fv_end(fh); ++fv) {
        pts.push_back(mesh.point(*fv));
    }
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& cur = pts[i];
        const auto& nxt = pts[(i + 1) % pts.size()];
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2]);
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0]);
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1]);
    }
    float len = n.length();
    if (len > 1e-8f) n /= len;
    return n;
}

static float dihedral_angle_deg(PolyMesh& mesh, PolyMesh::EdgeHandle eh) {
    auto hh0 = mesh.halfedge_handle(eh, 0);
    auto hh1 = mesh.halfedge_handle(eh, 1);
    auto f0 = mesh.face_handle(hh0);
    auto f1 = mesh.face_handle(hh1);
    if (!f0.is_valid() || !f1.is_valid()) return 0.0f; // boundary
    auto n0 = face_normal(mesh, f0);
    auto n1 = face_normal(mesh, f1);
    float dot = n0 | n1; // OpenMesh dot product operator
    dot = std::max(-1.0f, std::min(1.0f, dot));
    return acosf(dot) * (180.0f / M_PI);
}

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

OmeshHandle omesh_new(void) {
    auto* m = new PolyMesh();
    return static_cast<OmeshHandle>(m);
}

void omesh_free(OmeshHandle handle) {
    auto* m = static_cast<PolyMesh*>(handle);
    delete m;
}

uint32_t omesh_add_vertex(OmeshHandle handle, float x, float y, float z) {
    auto* m = static_cast<PolyMesh*>(handle);
    auto vh = m->add_vertex(PolyMesh::Point(x, y, z));
    return static_cast<uint32_t>(vh.idx());
}

int32_t omesh_add_face(OmeshHandle handle, const uint32_t* vert_indices, uint32_t n_verts) {
    auto* m = static_cast<PolyMesh*>(handle);
    std::vector<PolyMesh::VertexHandle> vhs;
    vhs.reserve(n_verts);
    for (uint32_t i = 0; i < n_verts; ++i) {
        vhs.push_back(PolyMesh::VertexHandle(static_cast<int>(vert_indices[i])));
    }
    auto fh = m->add_face(vhs);
    return fh.is_valid() ? static_cast<int32_t>(fh.idx()) : -1;
}

uint32_t omesh_chamfer(OmeshHandle handle, float angle_threshold_deg, float width) {
    auto* m = static_cast<PolyMesh*>(handle);

    // Pass 1: identify sharp edges
    struct SharpEdge {
        PolyMesh::EdgeHandle eh;
        PolyMesh::HalfedgeHandle hh0, hh1;
        PolyMesh::FaceHandle f0, f1;
    };
    std::vector<SharpEdge> sharp_edges;

    for (auto eit = m->edges_begin(); eit != m->edges_end(); ++eit) {
        auto eh = *eit;
        if (m->is_boundary(eh)) continue;
        float angle = dihedral_angle_deg(*m, eh);
        if (angle >= angle_threshold_deg) {
            SharpEdge se;
            se.eh = eh;
            se.hh0 = m->halfedge_handle(eh, 0);
            se.hh1 = m->halfedge_handle(eh, 1);
            se.f0 = m->face_handle(se.hh0);
            se.f1 = m->face_handle(se.hh1);
            sharp_edges.push_back(se);
        }
    }

    if (sharp_edges.empty()) return 0;

    // Pass 2: for each sharp edge, compute inset vertices and insert chamfer quad
    //
    // For each sharp edge (v0->v1), we:
    //   - Compute inset points on each adjacent face
    //   - Create 4 new vertices (2 per face side)
    //   - Add a chamfer quad between them
    //
    // We collect all new geometry first, then add it.

    struct ChamferQuad {
        OpenMesh::Vec3f p0, p1, p2, p3;
    };
    std::vector<ChamferQuad> quads;

    // We also need to inset the original face vertices.
    // Strategy: for each face adjacent to a sharp edge, move its edge-vertices
    // inward along the face plane by `width`.

    // Collect per-face per-edge inset data
    // face -> list of (edge_vertex_idx_in_face, inset_direction)
    struct FaceInset {
        PolyMesh::VertexHandle vh;  // vertex to inset
        OpenMesh::Vec3f dir;        // inset direction (unit, in face plane)
    };
    std::map<int, std::vector<FaceInset>> face_insets;

    for (auto& se : sharp_edges) {
        auto v0 = m->to_vertex_handle(se.hh0);
        auto v1 = m->from_vertex_handle(se.hh0);
        auto p0 = m->point(v0);
        auto p1 = m->point(v1);

        auto edge_dir = (p1 - p0);
        float edge_len = edge_dir.length();
        if (edge_len < 1e-8f) continue;
        edge_dir /= edge_len;

        // Normals of the two adjacent faces
        auto n0 = face_normal(*m, se.f0);
        auto n1 = face_normal(*m, se.f1);

        // Inset direction for face 0: perpendicular to edge, in the face plane, pointing inward
        auto inset0 = (n0 % edge_dir); // cross product
        float len0 = inset0.length();
        if (len0 > 1e-8f) inset0 /= len0;

        auto inset1 = (n1 % edge_dir);
        float len1 = inset1.length();
        if (len1 > 1e-8f) inset1 /= len1;

        // Ensure inset directions point INTO the face (away from edge)
        // Check by dotting with vector from edge midpoint to face centroid
        auto mid = (p0 + p1) * 0.5f;

        // Face 0 centroid
        OpenMesh::Vec3f c0(0, 0, 0);
        int cnt0 = 0;
        for (auto fv = m->fv_begin(se.f0); fv != m->fv_end(se.f0); ++fv) {
            c0 += m->point(*fv); cnt0++;
        }
        c0 /= (float)cnt0;
        if ((inset0 | (c0 - mid)) < 0) inset0 = -inset0;

        // Face 1 centroid
        OpenMesh::Vec3f c1(0, 0, 0);
        int cnt1 = 0;
        for (auto fv = m->fv_begin(se.f1); fv != m->fv_end(se.f1); ++fv) {
            c1 += m->point(*fv); cnt1++;
        }
        c1 /= (float)cnt1;
        if ((inset1 | (c1 - mid)) < 0) inset1 = -inset1;

        // The 4 chamfer vertices:
        //   face0 side: p0 + inset0*width, p1 + inset0*width
        //   face1 side: p0 + inset1*width, p1 + inset1*width
        ChamferQuad q;
        q.p0 = p0 + inset0 * width;
        q.p1 = p1 + inset0 * width;
        q.p2 = p1 + inset1 * width;
        q.p3 = p0 + inset1 * width;
        quads.push_back(q);

        // Record insets for face vertex adjustment
        face_insets[se.f0.idx()].push_back({v0, inset0});
        face_insets[se.f0.idx()].push_back({v1, inset0});
        face_insets[se.f1.idx()].push_back({v0, inset1});
        face_insets[se.f1.idx()].push_back({v1, inset1});
    }

    // Pass 3: adjust original face vertices by insetting them
    // For vertices shared by multiple sharp edges on the same face, average the insets
    std::map<int, OpenMesh::Vec3f> vert_offsets; // vertex idx -> accumulated offset
    std::map<int, int> vert_counts;

    for (auto& [face_idx, insets] : face_insets) {
        for (auto& fi : insets) {
            int vi = fi.vh.idx();
            if (vert_offsets.find(vi) == vert_offsets.end()) {
                vert_offsets[vi] = fi.dir * width;
                vert_counts[vi] = 1;
            } else {
                vert_offsets[vi] += fi.dir * width;
                vert_counts[vi]++;
            }
        }
    }

    // Apply averaged offsets to original vertices
    for (auto& [vi, offset] : vert_offsets) {
        auto vh = PolyMesh::VertexHandle(vi);
        auto p = m->point(vh);
        auto avg_offset = offset / (float)vert_counts[vi];
        m->set_point(vh, p + avg_offset);
    }

    // Pass 4: add chamfer quads
    uint32_t n_chamfer = 0;
    for (auto& q : quads) {
        auto v0 = m->add_vertex(q.p0);
        auto v1 = m->add_vertex(q.p1);
        auto v2 = m->add_vertex(q.p2);
        auto v3 = m->add_vertex(q.p3);
        std::vector<PolyMesh::VertexHandle> fverts = {v0, v1, v2, v3};
        auto fh = m->add_face(fverts);
        if (fh.is_valid()) n_chamfer++;
    }

    return n_chamfer;
}

uint32_t omesh_n_vertices(OmeshHandle handle) {
    auto* m = static_cast<PolyMesh*>(handle);
    return static_cast<uint32_t>(m->n_vertices());
}

uint32_t omesh_n_faces(OmeshHandle handle) {
    auto* m = static_cast<PolyMesh*>(handle);
    return static_cast<uint32_t>(m->n_faces());
}

void omesh_export_sizes(OmeshHandle handle,
                        uint32_t* out_n_verts,
                        uint32_t* out_n_indices) {
    auto* m = static_cast<PolyMesh*>(handle);

    // Count: each face is fan-triangulated, duplicating vertices per face
    uint32_t total_verts = 0;
    uint32_t total_indices = 0;
    for (auto fit = m->faces_begin(); fit != m->faces_end(); ++fit) {
        uint32_t n = 0;
        for (auto fv = m->fv_begin(*fit); fv != m->fv_end(*fit); ++fv) {
            n++;
        }
        total_verts += n;
        if (n >= 3) total_indices += (n - 2) * 3;
    }
    *out_n_verts = total_verts;
    *out_n_indices = total_indices;
}

void omesh_export(OmeshHandle handle,
                  float* out_positions,
                  float* out_normals,
                  uint32_t* out_indices) {
    auto* m = static_cast<PolyMesh*>(handle);

    uint32_t vert_offset = 0;
    uint32_t idx_offset = 0;

    for (auto fit = m->faces_begin(); fit != m->faces_end(); ++fit) {
        auto fh = *fit;
        auto n = face_normal(*m, fh);

        std::vector<OpenMesh::Vec3f> pts;
        for (auto fv = m->fv_begin(fh); fv != m->fv_end(fh); ++fv) {
            pts.push_back(m->point(*fv));
        }

        uint32_t base = vert_offset;
        for (size_t i = 0; i < pts.size(); ++i) {
            out_positions[(vert_offset + i) * 3 + 0] = pts[i][0];
            out_positions[(vert_offset + i) * 3 + 1] = pts[i][1];
            out_positions[(vert_offset + i) * 3 + 2] = pts[i][2];
            out_normals[(vert_offset + i) * 3 + 0] = n[0];
            out_normals[(vert_offset + i) * 3 + 1] = n[1];
            out_normals[(vert_offset + i) * 3 + 2] = n[2];
        }
        vert_offset += static_cast<uint32_t>(pts.size());

        // Fan triangulation
        for (size_t i = 1; i + 1 < pts.size(); ++i) {
            out_indices[idx_offset++] = base;
            out_indices[idx_offset++] = base + static_cast<uint32_t>(i);
            out_indices[idx_offset++] = base + static_cast<uint32_t>(i + 1);
        }
    }
}

} // extern "C"
