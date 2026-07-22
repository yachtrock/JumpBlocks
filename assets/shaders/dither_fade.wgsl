// Dither fade fragment shader — extends the standard PBR pipeline with
// screen-space ordered dithering for smooth LOD transitions.

#import bevy_pbr::{
    pbr_types,
    pbr_functions::alpha_discard,
    pbr_fragment::pbr_input_from_standard_material,
    decal::clustered::apply_decals,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}
#endif

#ifdef VISIBILITY_RANGE_DITHER
#import bevy_pbr::pbr_functions::visibility_range_dither;
#endif

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output
#endif

#ifdef OIT_ENABLED
#import bevy_core_pipeline::oit::oit_draw
#endif

// --- Dither fade uniforms ---

struct DitherFadeUniform {
    /// 0.0 = fully visible, 1.0 = fully invisible
    fade: f32,
    /// 0.0 = normal dither, 1.0 = inverted pattern (for complementary crossfade)
    invert: f32,
    /// 0.0 = no chamfer (flat), 1.0 = full chamfer. Controlled by vertex shader.
    chamfer_amount: f32,
    _pad: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(200)
var<uniform> dither_fade: DitherFadeUniform;

// 4x4 Bayer ordered dithering matrix (values 0..15, normalized to 0..1)
const BAYER_4X4 = array<f32, 16>(
     0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
    12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
     3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
    15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0,
);

// --- Fragment entry ---

@fragment
fn fragment(
#ifdef MESHLET_MESH_MATERIAL_PASS
    @builtin(position) frag_coord: vec4<f32>,
#else
    vertex_output: VertexOutput,
    @builtin(front_facing) is_front: bool,
#endif
) -> FragmentOutput {
#ifdef MESHLET_MESH_MATERIAL_PASS
    let vertex_output = resolve_vertex_output(frag_coord);
    let is_front = true;
#endif

    var in = vertex_output;

    // --- Dither fade discard ---
    let fade = dither_fade.fade;
    if fade >= 1.0 {
        discard;
    }
    if fade > 0.0 {
        let x = u32(in.position.x) % 4u;
        let y = u32(in.position.y) % 4u;
        var threshold = BAYER_4X4[y * 4u + x];
        // Invert the pattern so the incoming mesh covers exactly the pixels
        // the outgoing mesh discards, and vice versa.
        if dither_fade.invert > 0.5 {
            threshold = 1.0 - threshold;
        }
        if fade > threshold {
            discard;
        }
    }

#ifdef VISIBILITY_RANGE_DITHER
    visibility_range_dither(in.position, in.visibility_range_dither);
#endif

    // Standard PBR pipeline from here on
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    apply_decals(&pbr_input);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

#ifdef OIT_ENABLED
    let alpha_mode = pbr_input.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;
    if alpha_mode != pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE {
        oit_draw(in.position, out.color);
        discard;
    }
#endif

    return out;
}
