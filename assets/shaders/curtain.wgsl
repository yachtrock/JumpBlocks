#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct CurtainUniforms {
    /// 0.0 = fully black, 1.0 = fully revealed
    progress: f32,
    /// Number of star points (unused for now, hardcoded to 5)
    points: f32,
    /// Aspect ratio (width / height)
    aspect: f32,
    _pad: f32,
};

@group(2) @binding(0)
var<uniform> uniforms: CurtainUniforms;

// 5-pointed star SDF (based on Inigo Quilez sdStar5)
// p: point, r: outer radius, rf: inner radius factor (0..1, lower = pointier)
fn sd_star5(p_in: vec2<f32>, r: f32, rf: f32) -> f32 {
    let k1 = vec2<f32>(0.809016994375, -0.587785252292); // cos/sin of pi/5
    let k2 = vec2<f32>(-k1.x, k1.y);

    var p = vec2<f32>(abs(p_in.x), -p_in.y);
    p = p - 2.0 * max(dot(k1, p), 0.0) * k1;
    p = p - 2.0 * max(dot(k2, p), 0.0) * k2;

    p = vec2<f32>(abs(p.x), p.y - r);
    let ba = rf * vec2<f32>(-k1.y, k1.x) - vec2<f32>(0.0, 1.0);
    let h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);

    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map UV to centered coordinates with aspect correction
    let uv = in.uv - vec2<f32>(0.5, 0.5);
    let p = vec2<f32>(uv.x * uniforms.aspect, uv.y);

    // Solid black until the reveal begins
    if uniforms.progress <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // The star radius grows from 0 to cover the full screen
    // Need to reach corners: sqrt(aspect^2 + 1) * 0.5
    let max_radius = sqrt(uniforms.aspect * uniforms.aspect + 1.0) * 0.5;
    // Ease-in: starts slow, accelerates over time — t^3
    let t = uniforms.progress;
    let eased = t * t * t;
    let radius = eased * max_radius * 2.0; // overshoot so star fully clears the screen
    let inner_factor = 0.38; // how concave the inner vertices are

    let d = sd_star5(p, radius, inner_factor);

    // Inside star = transparent (show game), outside = black
    if d < -0.002 {
        discard;
    }

    // Soft anti-aliased edge
    let alpha = smoothstep(-0.002, 0.002, d);
    return vec4<f32>(0.0, 0.0, 0.0, alpha);
}
