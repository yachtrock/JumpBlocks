use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::ViewTarget;

use crate::draw_cmd::{UiBatch, UiVertex, build_batches};

use super::extract::ExtractedUiFrame;
use super::pipeline::{UiGlobals, UiOverlayPipeline, atlas_layout_descriptor, globals_layout_descriptor};

/// Render-world resource holding GPU buffers and bind groups for the UI.
#[derive(Resource)]
pub struct UiRenderData {
    pub pipeline_id: Option<CachedRenderPipelineId>,
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub batches: Vec<UiBatch>,
    pub globals_bind_group: Option<BindGroup>,
    pub atlas_bind_group: Option<BindGroup>,
    pub atlas_texture: Option<Texture>,
    pub atlas_texture_view: Option<TextureView>,
    pub atlas_sampler: Option<Sampler>,
    pub globals_buffer: Option<Buffer>,
    pub current_atlas_size: (u32, u32),
}

impl Default for UiRenderData {
    fn default() -> Self {
        Self {
            pipeline_id: None,
            vertex_buffer: None,
            index_buffer: None,
            batches: Vec::new(),
            globals_bind_group: None,
            atlas_bind_group: None,
            atlas_texture: None,
            atlas_texture_view: None,
            atlas_sampler: None,
            globals_buffer: None,
            current_atlas_size: (0, 0),
        }
    }
}

/// Prepare system: builds GPU buffers from extracted draw commands.
pub fn prepare_ui_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<UiOverlayPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut specialized_pipelines: ResMut<SpecializedRenderPipelines<UiOverlayPipeline>>,
    extracted: Res<ExtractedUiFrame>,
    mut render_data: ResMut<UiRenderData>,
    view_targets: Query<&ViewTarget>,
) {
    // Specialize pipeline for the actual surface format from any active view
    if let Some(view_target) = view_targets.iter().next() {
        let format = view_target.out_texture_view_format();
        let id = specialized_pipelines.specialize(&pipeline_cache, &pipeline, format);
        render_data.pipeline_id = Some(id);
    }
    if !extracted.has_data || extracted.commands.is_empty() {
        render_data.batches.clear();
        return;
    }

    let dpi_scale = extracted.dpi_scale.max(1.0);

    // Build vertex/index data and batches
    let (vertices, indices, batches) = build_batches(&extracted.commands, dpi_scale);

    if vertices.is_empty() {
        render_data.batches.clear();
        return;
    }

    // Upload vertex buffer
    let vertex_bytes = bytemuck::cast_slice::<UiVertex, u8>(&vertices);
    render_data.vertex_buffer = Some(render_device.create_buffer_with_data(
        &BufferInitDescriptor {
            label: Some("ui_vertex_buffer"),
            contents: vertex_bytes,
            usage: BufferUsages::VERTEX,
        },
    ));

    // Upload index buffer
    let index_bytes = bytemuck::cast_slice::<u32, u8>(&indices);
    render_data.index_buffer = Some(render_device.create_buffer_with_data(
        &BufferInitDescriptor {
            label: Some("ui_index_buffer"),
            contents: index_bytes,
            usage: BufferUsages::INDEX,
        },
    ));

    render_data.batches = batches;

    // Handle atlas texture creation/update
    let (atlas_w, atlas_h) = extracted.atlas_size;
    if atlas_w == 0 || atlas_h == 0 {
        return;
    }

    // Recreate atlas texture if size changed
    if render_data.current_atlas_size != (atlas_w, atlas_h) {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("ui_atlas"),
            size: Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor::default());
        render_data.atlas_texture = Some(texture);
        render_data.atlas_texture_view = Some(view);
        render_data.current_atlas_size = (atlas_w, atlas_h);

        // Create sampler on first texture creation
        if render_data.atlas_sampler.is_none() {
            render_data.atlas_sampler = Some(render_device.create_sampler(&SamplerDescriptor {
                label: Some("ui_atlas_sampler"),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                ..default()
            }));
        }
    }

    // Upload dirty atlas regions
    if let Some(texture) = &render_data.atlas_texture {
        for upload in &extracted.atlas_uploads {
            render_queue.write_texture(
                TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: Origin3d {
                        x: upload.x,
                        y: upload.y,
                        z: 0,
                    },
                    aspect: TextureAspect::All,
                },
                &upload.pixels,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.width * 4),
                    rows_per_image: None,
                },
                Extent3d {
                    width: upload.width,
                    height: upload.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    // Update globals uniform with window physical size
    let globals = UiGlobals {
        screen_size: extracted.window_physical_size,
    };
    let mut globals_bytes = Vec::new();
    globals_bytes.extend_from_slice(bytemuck::bytes_of(&globals.screen_size.x));
    globals_bytes.extend_from_slice(bytemuck::bytes_of(&globals.screen_size.y));
    // Pad to 16 bytes (uniform alignment)
    globals_bytes.extend_from_slice(&[0u8; 8]);

    render_data.globals_buffer = Some(render_device.create_buffer_with_data(
        &BufferInitDescriptor {
            label: Some("ui_globals_buffer"),
            contents: &globals_bytes,
            usage: BufferUsages::UNIFORM,
        },
    ));

    // Get bind group layouts from the pipeline cache (matches the pipeline's layouts exactly)
    let globals_layout = pipeline_cache.get_bind_group_layout(&globals_layout_descriptor());
    let atlas_layout = pipeline_cache.get_bind_group_layout(&atlas_layout_descriptor());

    // Create bind groups — clone references to avoid borrow conflicts
    let globals_buf = render_data.globals_buffer.clone();
    let atlas_view = render_data.atlas_texture_view.clone();
    let atlas_sampler = render_data.atlas_sampler.clone();

    if let (Some(globals_buf), Some(atlas_view), Some(atlas_sampler)) =
        (globals_buf, atlas_view, atlas_sampler)
    {
        render_data.globals_bind_group = Some(render_device.create_bind_group(
            "ui_globals_bind_group",
            &globals_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        ));

        render_data.atlas_bind_group = Some(render_device.create_bind_group(
            "ui_atlas_bind_group",
            &atlas_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&atlas_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&atlas_sampler),
                },
            ],
        ));
    }
}
