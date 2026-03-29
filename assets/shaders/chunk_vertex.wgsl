// Vertex shader for chunk meshes with chamfer amount control.
// Reads the ChamferOffset custom vertex attribute (@location(10))
// and scales it by chamfer_amount (0 = flat, 1 = full chamfer).

#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    forward_io::VertexOutput,
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

// Custom vertex input that includes the chamfer offset attribute.
struct ChunkVertex {
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>,
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(4) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef HAS_CHAMFER_OFFSET
    @location(10) chamfer_offset: vec3<f32>,
#endif
#ifdef HAS_SHARP_NORMAL
    @location(11) sharp_normal: vec3<f32>,
#endif
};

@vertex
fn vertex(in: ChunkVertex) -> VertexOutput {
    var out: VertexOutput;

    var position = in.position;

    // Scale back the chamfer offset: at chamfer_amount=1 the mesh is fully
    // chamfered (as authored), at chamfer_amount=0 the chamfer is removed.
#ifdef HAS_CHAMFER_OFFSET
    let undo = (1.0 - dither_fade.chamfer_amount) * in.chamfer_offset;
    position = position - undo;
#endif

    // Interpolate normal from smooth (chamfered) → sharp (flat face) as chamfer fades.
    var normal = in.normal;
#ifdef HAS_SHARP_NORMAL
    normal = mix(in.sharp_normal, in.normal, dither_fade.chamfer_amount);
#endif

    let mesh_world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    var world_from_local = mesh_world_from_local;

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        normal,
        in.instance_index
    );
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS_A
    out.uv = in.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = in.uv_b;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        in.tangent,
        in.instance_index
    );
#endif

#ifdef VERTEX_COLORS
    out.color = in.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        in.instance_index, mesh_world_from_local[3]);
#endif

    return out;
}
