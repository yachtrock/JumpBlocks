use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, ViewNode};
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderContext;
use bevy::render::view::ViewTarget;

use super::prepare::UiRenderData;

/// Render graph node that draws the UI overlay on top of the 3D scene.
#[derive(Default)]
pub struct UiOverlayNode;

impl ViewNode for UiOverlayNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view_target: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let render_data = world.get_resource::<UiRenderData>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(render_data) = render_data else {
            return Ok(());
        };

        // Skip if no batches to draw
        if render_data.batches.is_empty() {
            return Ok(());
        }

        // Get the specialized pipeline, skip if not ready yet
        let Some(pipeline_id) = render_data.pipeline_id else {
            return Ok(());
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
            return Ok(());
        };

        let (Some(vertex_buffer), Some(index_buffer)) =
            (&render_data.vertex_buffer, &render_data.index_buffer)
        else {
            return Ok(());
        };

        let (Some(globals_bg), Some(atlas_bg)) =
            (&render_data.globals_bind_group, &render_data.atlas_bind_group)
        else {
            return Ok(());
        };

        // Render to the output texture (after upscaling), preserving existing content
        let color_attachment = view_target.out_texture_color_attachment(None);

        let mut render_pass =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ui_overlay_pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, globals_bg, &[]);
        render_pass.set_bind_group(1, atlas_bg, &[]);
        render_pass.set_vertex_buffer(0, *vertex_buffer.slice(..));
        render_pass.set_index_buffer(*index_buffer.slice(..), IndexFormat::Uint32);

        for batch in &render_data.batches {
            // Set scissor rect if clipping is active
            if let Some([x, y, w, h]) = batch.clip {
                let x = x.max(0.0) as u32;
                let y = y.max(0.0) as u32;
                let w = w.max(1.0) as u32;
                let h = h.max(1.0) as u32;
                render_pass.set_scissor_rect(x, y, w, h);
            }

            render_pass.draw_indexed(
                batch.index_start..(batch.index_start + batch.index_count),
                0,
                0..1,
            );

            // Reset scissor after clipped batch
            if batch.clip.is_some() {
                // We'd need the full viewport size to reset — for now,
                // set a large scissor. The next batch with no clip will
                // work fine since we set it before each batch anyway.
                render_pass.set_scissor_rect(0, 0, 16384, 16384);
            }
        }

        Ok(())
    }
}
