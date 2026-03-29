// Prepass fragment shader that applies dither discard before the standard
// prepass runs. This ensures fully-faded meshes don't write to the shadow
// or depth buffer.

#import bevy_pbr::{
    pbr_prepass_functions,
    pbr_bindings,
    pbr_types,
    pbr_functions,
    prepass_io,
    mesh_bindings::mesh,
}

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output
#endif

#ifdef BINDLESS
#import bevy_pbr::pbr_bindings::material_indices
#endif

struct DitherFadeUniform {
    fade: f32,
    invert: f32,
    chamfer_amount: f32,
    _pad: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(200)
var<uniform> dither_fade: DitherFadeUniform;

const BAYER_4X4 = array<f32, 16>(
     0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
    12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
     3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
    15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0,
);

#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(
#ifdef MESHLET_MESH_MATERIAL_PASS
    @builtin(position) frag_coord: vec4<f32>,
#else
    in: prepass_io::VertexOutput,
    @builtin(front_facing) is_front: bool,
#endif
) -> prepass_io::FragmentOutput {
#ifdef MESHLET_MESH_MATERIAL_PASS
    let in = resolve_vertex_output(frag_coord);
    let is_front = true;
#endif

    // --- Dither fade discard (same logic as main pass) ---
    let fade = dither_fade.fade;
    if fade >= 1.0 {
        discard;
    }
    if fade > 0.0 {
        let x = u32(in.position.x) % 4u;
        let y = u32(in.position.y) % 4u;
        var threshold = BAYER_4X4[y * 4u + x];
        if dither_fade.invert > 0.5 {
            threshold = 1.0 - threshold;
        }
        if fade > threshold {
            discard;
        }
    }

#ifndef MESHLET_MESH_MATERIAL_PASS
#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif
    pbr_prepass_functions::prepass_alpha_discard(in);
#endif

    var out: prepass_io::FragmentOutput;

#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.unclipped_depth;
#endif

#ifdef NORMAL_PREPASS
    out.normal = vec4(in.world_normal * 0.5 + vec3(0.5), 1.0);
#endif

#ifdef MOTION_VECTOR_PREPASS
    out.motion_vector = in.motion_vector;
#endif

    return out;
}
#endif // PREPASS_FRAGMENT
