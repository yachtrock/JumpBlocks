/// A single draw command — one textured, tinted quad.
#[derive(Clone, Debug)]
pub struct DrawCmd {
    /// Screen-space rect in logical pixels: [x, y, width, height]
    pub rect: [f32; 4],
    /// UV coordinates in the atlas: [u_min, v_min, u_max, v_max]
    pub uvs: [f32; 4],
    /// RGBA tint color (premultiplied alpha)
    pub color: [f32; 4],
    /// Which atlas page/texture to sample from
    pub atlas_page: u32,
    /// Optional scissor clip rect in logical pixels: [x, y, width, height]
    pub clip: Option<[f32; 4]>,
}

/// GPU vertex for a UI quad corner.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// A batch of draw commands sharing the same atlas page and clip rect.
#[derive(Debug)]
pub struct UiBatch {
    pub atlas_page: u32,
    pub clip: Option<[f32; 4]>,
    pub index_start: u32,
    pub index_count: u32,
}

/// Expand draw commands into vertex/index buffers and batch groups.
pub fn build_batches(
    commands: &[DrawCmd],
    dpi_scale: f32,
) -> (Vec<UiVertex>, Vec<u32>, Vec<UiBatch>) {
    let mut vertices = Vec::with_capacity(commands.len() * 4);
    let mut indices = Vec::with_capacity(commands.len() * 6);
    let mut batches: Vec<UiBatch> = Vec::new();

    for cmd in commands {
        let [x, y, w, h] = cmd.rect;
        let (x, y, w, h) = (x * dpi_scale, y * dpi_scale, w * dpi_scale, h * dpi_scale);
        let [u0, v0, u1, v1] = cmd.uvs;
        let color = cmd.color;

        let base = vertices.len() as u32;

        vertices.push(UiVertex { position: [x, y], uv: [u0, v0], color });
        vertices.push(UiVertex { position: [x + w, y], uv: [u1, v0], color });
        vertices.push(UiVertex { position: [x + w, y + h], uv: [u1, v1], color });
        vertices.push(UiVertex { position: [x, y + h], uv: [u0, v1], color });

        let idx_start = indices.len() as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Try to merge with previous batch
        let scaled_clip = cmd.clip.map(|[cx, cy, cw, ch]| {
            [cx * dpi_scale, cy * dpi_scale, cw * dpi_scale, ch * dpi_scale]
        });

        let can_merge = batches.last().is_some_and(|b| {
            b.atlas_page == cmd.atlas_page && b.clip == scaled_clip
        });

        if can_merge {
            batches.last_mut().unwrap().index_count += 6;
        } else {
            batches.push(UiBatch {
                atlas_page: cmd.atlas_page,
                clip: scaled_clip,
                index_start: idx_start,
                index_count: 6,
            });
        }
    }

    (vertices, indices, batches)
}
