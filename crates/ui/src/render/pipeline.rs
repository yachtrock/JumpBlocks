use bevy::asset::load_embedded_asset;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_resource::*;

use crate::draw_cmd::UiVertex;

/// Bind group layout descriptors — shared between pipeline creation and bind group creation.
pub fn globals_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ui_globals_layout",
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(UiGlobals::min_size()),
            },
            count: None,
        }],
    )
}

pub fn atlas_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ui_atlas_layout",
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    )
}

/// Cached shader and vertex layout for the UI overlay — pipeline is specialized per surface format.
#[derive(Resource)]
pub struct UiOverlayPipeline {
    shader: Handle<Shader>,
    vertex_layout: VertexBufferLayout,
}

impl FromWorld for UiOverlayPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader: Handle<Shader> = load_embedded_asset!(world, "shader.wgsl");

        let vertex_layout = VertexBufferLayout {
            array_stride: size_of::<UiVertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 2,
                },
            ],
        };

        Self {
            shader,
            vertex_layout,
        }
    }
}

impl SpecializedRenderPipeline for UiOverlayPipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("ui_overlay_pipeline".into()),
            layout: vec![globals_layout_descriptor(), atlas_layout_descriptor()],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: None,
                buffers: vec![self.vertex_layout.clone()],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: None,
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// Uniform data for the vertex shader.
#[derive(Clone, Copy, ShaderType)]
pub struct UiGlobals {
    pub screen_size: Vec2,
}
