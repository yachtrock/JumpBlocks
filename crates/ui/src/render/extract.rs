use bevy::prelude::*;
use bevy::render::Extract;
use crossbeam_channel::Receiver;

use crate::bridge::{AtlasUpload, UiFrame};
use crate::draw_cmd::DrawCmd;

/// Render-world resource holding the receiver for UI frames.
#[derive(Resource)]
pub struct UiFrameReceiver(pub Receiver<UiFrame>);

/// Render-world resource holding the extracted frame data for the current frame.
#[derive(Resource, Default)]
pub struct ExtractedUiFrame {
    pub commands: Vec<DrawCmd>,
    pub atlas_uploads: Vec<AtlasUpload>,
    pub atlas_size: (u32, u32),
    pub dpi_scale: f32,
    pub window_physical_size: Vec2,
    pub has_data: bool,
}

/// Extract system: runs in ExtractSchedule, pulls the latest UiFrame from the channel.
pub fn extract_ui_frame(
    receiver: Res<UiFrameReceiver>,
    mut extracted: ResMut<ExtractedUiFrame>,
    windows: Extract<Query<&Window>>,
) {
    // Drain channel, keep only the latest frame
    let mut latest: Option<UiFrame> = None;
    while let Ok(frame) = receiver.0.try_recv() {
        latest = Some(frame);
    }

    if let Some(frame) = latest {
        extracted.commands = frame.commands;
        extracted.atlas_uploads = frame.atlas_uploads;
        extracted.atlas_size = frame.atlas_size;
        extracted.dpi_scale = frame.dpi_scale;
        extracted.has_data = true;
    } else {
        // No new frame — clear atlas uploads but keep previous commands
        extracted.atlas_uploads.clear();
    }

    // Extract window physical size for the shader
    if let Ok(window) = windows.single() {
        let scale = window.scale_factor();
        extracted.window_physical_size =
            Vec2::new(window.width() * scale, window.height() * scale);
        if extracted.dpi_scale == 0.0 {
            extracted.dpi_scale = scale;
        }
    }
}
