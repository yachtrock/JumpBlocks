// Vertex shader for chunk meshes with chamfer amount control.
// Reads the chamfer offset from vertex color (RGB) and scales it
// by chamfer_amount (0 = flat/no chamfer, 1 = full chamfer).
// This allows smooth fillet fade-out with distance.

#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

struct DitherFadeUniform {
    fade: f32,
    invert: f32,
    chamfer_amount: f32,
    _pad: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(200)
var<uniform> dither_fade: DitherFadeUniform;

@vertex
fn vertex(vertex_no_morph: Vertex) -> VertexOutput {
    var out: VertexOutput;
    var vertex = vertex_no_morph;

    let mesh_world_from_local = mesh_functions::get_world_from_local(vertex_no_morph.instance_index);
    var world_from_local = mesh_world_from_local;

    // Chamfer offset is stored in vertex color RGB.
    // Scale it back: at chamfer_amount=1 use authored positions (full chamfer),
    // at chamfer_amount=0 subtract the offset to flatten.
#ifdef VERTEX_COLORS
    let chamfer_offset = vertex.color.xyz;
    let undo = (1.0 - dither_fade.chamfer_amount) * chamfer_offset;
    vertex.position = vertex.position - undo;
#endif

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex_no_morph.instance_index
    );
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex_no_morph.instance_index
    );
#endif

    // Don't pass vertex color through — it's used for chamfer data, not actual color.

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex_no_morph.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex_no_morph.instance_index, mesh_world_from_local[3]);
#endif

    return out;
}
