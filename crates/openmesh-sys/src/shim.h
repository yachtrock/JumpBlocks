#ifndef OPENMESH_SHIM_H
#define OPENMESH_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* OmeshHandle;

OmeshHandle omesh_new(void);
void omesh_free(OmeshHandle mesh);

uint32_t omesh_add_vertex(OmeshHandle mesh, float x, float y, float z);
int32_t omesh_add_face(OmeshHandle mesh, const uint32_t* vert_indices, uint32_t n_verts);

// Chamfer sharp edges. Returns the number of chamfer faces inserted.
// angle_threshold_deg: edges with dihedral angle above this are "sharp" (e.g. 10.0)
// width: chamfer inset distance
uint32_t omesh_chamfer(OmeshHandle mesh, float angle_threshold_deg, float width);

// Query mesh counts after chamfering
uint32_t omesh_n_vertices(OmeshHandle mesh);
uint32_t omesh_n_faces(OmeshHandle mesh);

// Export mesh data.
// positions: float[n_vertices * 3]
// normals: float[n_vertices * 3] (per-face normals, duplicated per face-vertex)
// indices: uint32_t[] (triangulated)
// Returns total number of indices written.
// Call with NULL pointers first to query sizes via omesh_export_sizes.
void omesh_export_sizes(OmeshHandle mesh,
                        uint32_t* out_n_verts,
                        uint32_t* out_n_indices);
void omesh_export(OmeshHandle mesh,
                  float* out_positions,
                  float* out_normals,
                  uint32_t* out_indices);

#ifdef __cplusplus
}
#endif

#endif // OPENMESH_SHIM_H
