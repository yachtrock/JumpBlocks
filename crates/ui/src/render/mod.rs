pub mod extract;
pub mod node;
pub mod pipeline;
pub mod prepare;

use bevy::asset::embedded_asset;
use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::prelude::*;
use bevy::render::render_graph::{RenderGraphExt, ViewNodeRunner};
use bevy::render::render_resource::SpecializedRenderPipelines;
use bevy::render::{Render, RenderApp, RenderSystems};
use crossbeam_channel::Receiver;

use crate::bridge::UiFrame;

use self::extract::{ExtractedUiFrame, UiFrameReceiver, extract_ui_frame};
use self::node::UiOverlayNode;
use self::pipeline::UiOverlayPipeline;
use self::prepare::{UiRenderData, prepare_ui_buffers};

/// Label for the UI overlay render graph node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, bevy::render::render_graph::RenderLabel)]
pub struct UiOverlayNodeLabel;

/// Registers the shader as an embedded asset (must happen during build phase).
pub struct UiRenderAssetPlugin;

impl Plugin for UiRenderAssetPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shader.wgsl");
    }
}

/// Set up the render-world resources, systems, and graph node.
/// Must be called from a plugin's `finish()` method (needs RenderDevice).
pub fn setup_render_world(app: &mut App, frame_rx: Receiver<UiFrame>) {
    let render_app = app.sub_app_mut(RenderApp);

    render_app
        .init_resource::<UiOverlayPipeline>()
        .init_resource::<SpecializedRenderPipelines<UiOverlayPipeline>>()
        .insert_resource(UiFrameReceiver(frame_rx))
        .insert_resource(ExtractedUiFrame::default())
        .insert_resource(UiRenderData::default())
        .add_systems(ExtractSchedule, extract_ui_frame)
        .add_systems(
            Render,
            prepare_ui_buffers.in_set(RenderSystems::Prepare),
        )
        .add_render_graph_node::<ViewNodeRunner<UiOverlayNode>>(Core3d, UiOverlayNodeLabel)
        .add_render_graph_edges(
            Core3d,
            (
                bevy::core_pipeline::core_3d::graph::Node3d::Upscaling,
                UiOverlayNodeLabel,
            ),
        );
}
