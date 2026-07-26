//! Block placement and preview system.
//!
//! Exposes a Rhai scripting API so action state scripts (e.g. `building.rhai`)
//! can control block preview and placement. The Rust side handles:
//! - Computing the valid build position each frame (pre-system)
//! - Rendering the translucent preview mesh (post-system)
//! - Writing voxels into chunks and triggering re-meshing (post-system)
//!
//! All positions are in chunk-local cell coordinates — the script never sees
//! world-space values. The Rust side converts to world space using each chunk's
//! transform, so chunks can have arbitrary transforms in the future.
//!
//! Script API:
//! - `build_position()` → `#{ x, y, z }` (chunk-local cell coords) or `()`
//! - `show_preview(facing)` — show preview at the current build position
//! - `show_preview_in_hand(facing)` — show preview floating in front of player
//! - `hide_preview()` — hide the preview
//! - `place_block(shape, facing, texture)` — place at the current build position
//! - `rotate_facing_right(facing)` / `rotate_facing_left(facing)` — helpers

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use jumpblocks_voxel::chunk::{
    Chunk, ChunkData, BlockModification, BLOCK_CELLS, CHUNK_X, CHUNK_Y, CHUNK_Z, VOXEL_SIZE,
};
use jumpblocks_voxel::shape::{
    cap_corner_heights, rotated_occupied_cells, Facing, ShapeTable, SHAPE_CUBE, SHAPE_WEDGE,
};
use jumpblocks_voxel::worldgen::classify_slope_cap;
use rhai::{Dynamic, Map, INT};

use crate::action_state::{ActionState, ActionStateEngine};
use crate::camera::OrbitCamera;
use crate::player::Player;

pub struct BuildingPlugin;

impl Plugin for BuildingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(BuildingApi::new())
            .init_resource::<PreviewResources>()
            .add_systems(Startup, register_building_api)
            .add_systems(
                Update,
                (
                    compute_build_context
                        .before(crate::action_state::action_state_update),
                    apply_building_commands
                        .after(crate::action_state::action_state_update),
                    process_chunk_modifications,
                ),
            );
    }
}

// ---------------------------------------------------------------------------
// Shared API data (accessed by both Rhai closures and ECS systems)
// ---------------------------------------------------------------------------

/// A valid build position within a chunk (chunk-local coordinates).
#[derive(Clone)]
struct BuildPos {
    /// Cell coordinates within the chunk.
    x: usize,
    y: usize,
    z: usize,
    /// Which chunk entity this position belongs to.
    chunk_entity: Entity,
}

/// How the preview should be displayed this frame.
#[derive(Clone, Default)]
enum PreviewState {
    #[default]
    Hidden,
    /// Show at the current build position (Rust resolves to world space via chunk transform).
    AtBuildPosition { shape: u16, facing: Facing },
    /// Show floating in front of the player (no chunk context needed).
    InHand { shape: u16, facing: Facing },
}

/// A request to place a block, queued by the script.
#[derive(Clone)]
struct PlaceRequest {
    shape: u16,
    facing: Facing,
    texture: u16,
    /// Auto-shape tool: reshape the surrounding terrain contour afterwards.
    auto: bool,
}

/// Inner shared state between Rhai script and ECS systems.
struct BuildingApiInner {
    /// Valid build position (set by pre-system, read by script via `build_position()`).
    build_pos: Option<BuildPos>,
    /// Player world position (for in-hand preview calculation).
    player_pos: Vec3,
    /// Camera forward direction (for in-hand preview calculation).
    camera_forward: Vec3,
    /// Preview display command (set by script, read by post-system).
    preview: PreviewState,
    /// Block placement requests (set by script, drained by post-system).
    place_requests: Vec<PlaceRequest>,
    /// Current selection from the build UI (persists across frames).
    selected_shape: u16,
    selected_texture: u16,
    auto_shape: bool,
    in_build_area: bool,
}

impl BuildingApiInner {
    /// The shape a placement/preview will actually use: the auto-shape tool
    /// always drops cubes (the contour pass turns them into slopes).
    fn effective_shape(&self) -> u16 {
        if self.auto_shape {
            SHAPE_CUBE
        } else {
            self.selected_shape
        }
    }
}

impl BuildingApiInner {
    fn new() -> Self {
        Self {
            build_pos: None,
            player_pos: Vec3::ZERO,
            camera_forward: Vec3::NEG_Z,
            preview: PreviewState::Hidden,
            place_requests: Vec::new(),
            selected_shape: SHAPE_CUBE,
            selected_texture: 20, // TEX_WHITE
            auto_shape: false,
            in_build_area: false,
        }
    }

    fn clear_frame(&mut self) {
        self.build_pos = None;
        self.preview = PreviewState::Hidden;
        self.place_requests.clear();
        // Selection fields persist — they mirror the build UI state.
    }
}

/// Resource providing the building API shared state.
#[derive(Resource, Clone)]
pub struct BuildingApi {
    inner: Arc<Mutex<BuildingApiInner>>,
}

impl BuildingApi {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(BuildingApiInner::new())),
        }
    }

    /// Mirror the build UI's current selection into the scripting API.
    /// Called each frame by the build UI (see `build_ui::update_build_area`).
    pub fn set_selection(&self, shape: u16, texture: u16, auto: bool, in_area: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.selected_shape = shape;
        inner.selected_texture = texture;
        inner.auto_shape = auto;
        inner.in_build_area = in_area;
    }
}

// ---------------------------------------------------------------------------
// Preview entity resources
// ---------------------------------------------------------------------------

#[derive(Component)]
struct BlockPreview;

#[derive(Resource, Default)]
struct PreviewResources {
    entity: Option<Entity>,
    material: Option<Handle<StandardMaterial>>,
    /// One preview mesh per shape id, built lazily from the ShapeTable.
    meshes: HashMap<u16, Handle<Mesh>>,
    /// Which shape the live preview entity currently shows.
    current_shape: Option<u16>,
}

// ---------------------------------------------------------------------------
// Rhai API registration (startup system)
// ---------------------------------------------------------------------------

fn register_building_api(mut engine: ResMut<ActionStateEngine>, api: Res<BuildingApi>) {
    let inner = api.inner.clone();

    // build_position() -> #{ x, y, z } | ()
    {
        let inner = inner.clone();
        engine.rhai_engine_mut().register_fn(
            "build_position",
            move || -> Dynamic {
                let data = inner.lock().unwrap();
                match &data.build_pos {
                    Some(pos) => {
                        let mut map = Map::new();
                        map.insert("x".into(), Dynamic::from(pos.x as INT));
                        map.insert("y".into(), Dynamic::from(pos.y as INT));
                        map.insert("z".into(), Dynamic::from(pos.z as INT));
                        Dynamic::from(map)
                    }
                    None => Dynamic::UNIT,
                }
            },
        );
    }

    // show_preview(facing)
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("show_preview", move |facing: INT| {
                let mut data = inner.lock().unwrap();
                data.preview = PreviewState::AtBuildPosition {
                    shape: data.effective_shape(),
                    facing: facing_from_int(facing),
                };
            });
    }

    // show_preview_in_hand(facing)
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("show_preview_in_hand", move |facing: INT| {
                let mut data = inner.lock().unwrap();
                data.preview = PreviewState::InHand {
                    shape: data.effective_shape(),
                    facing: facing_from_int(facing),
                };
            });
    }

    // hide_preview()
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("hide_preview", move || {
                let mut data = inner.lock().unwrap();
                data.preview = PreviewState::Hidden;
            });
    }

    // place_block(shape, facing, texture)
    {
        let inner = inner.clone();
        engine.rhai_engine_mut().register_fn(
            "place_block",
            move |shape: INT, facing: INT, texture: INT| {
                let mut data = inner.lock().unwrap();
                if data.build_pos.is_some() {
                    let auto = data.auto_shape;
                    let shape = if auto { SHAPE_CUBE } else { shape as u16 };
                    data.place_requests.push(PlaceRequest {
                        shape,
                        facing: facing_from_int(facing),
                        texture: texture as u16,
                        auto,
                    });
                }
            },
        );
    }

    // selected_shape() -> int — shape chosen in the build UI's shape menu
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("selected_shape", move || -> INT {
                inner.lock().unwrap().effective_shape() as INT
            });
    }

    // selected_texture() -> int — material chosen in the hotbar
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("selected_texture", move || -> INT {
                inner.lock().unwrap().selected_texture as INT
            });
    }

    // auto_shape_mode() -> bool — whether the auto-shape tool is active
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("auto_shape_mode", move || -> bool {
                inner.lock().unwrap().auto_shape
            });
    }

    // in_build_area() -> bool — whether building is currently available
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("in_build_area", move || -> bool {
                inner.lock().unwrap().in_build_area
            });
    }

    // rotate_facing_right(facing) -> int
    engine
        .rhai_engine_mut()
        .register_fn("rotate_facing_right", |facing: INT| -> INT {
            (facing + 1) % 4
        });

    // rotate_facing_left(facing) -> int
    engine
        .rhai_engine_mut()
        .register_fn("rotate_facing_left", |facing: INT| -> INT {
            (facing + 3) % 4
        });

    info!("[building] Registered Rhai API functions");
}

// ---------------------------------------------------------------------------
// Pre-system: compute build context before script runs
// ---------------------------------------------------------------------------

fn compute_build_context(
    api: Res<BuildingApi>,
    action_query: Query<(&ActionState, &Transform), With<Player>>,
    camera_query: Query<&Transform, (With<OrbitCamera>, Without<Player>)>,
    chunks: Query<(Entity, &Chunk, &Transform)>,
) {
    let mut inner = api.inner.lock().unwrap();

    inner.clear_frame();

    let Ok((action_state, player_transform)) = action_query.single() else {
        return;
    };
    if !action_state
        .0
        .as_deref()
        .is_some_and(|s| s == "building")
    {
        return;
    }

    let Ok(camera_transform) = camera_query.single() else {
        return;
    };

    let player_pos = player_transform.translation;
    let cam_forward = camera_transform.forward().as_vec3();
    inner.player_pos = player_pos;
    inner.camera_forward = cam_forward;

    inner.build_pos = find_placement_position(player_pos, cam_forward, &chunks);
}

// ---------------------------------------------------------------------------
// Post-system: apply script-driven commands after script runs
// ---------------------------------------------------------------------------

fn apply_building_commands(
    mut commands: Commands,
    api: Res<BuildingApi>,
    mut chunks: Query<(&mut Chunk, &Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut preview_res: ResMut<PreviewResources>,
    preview_query: Query<Entity, With<BlockPreview>>,
    build_locks: Option<Res<crate::challenge::BuildLocks>>,
    messages: Option<ResMut<crate::challenge::HudMessages>>,
    shape_table: Res<ShapeTable>,
    mut build_state: Option<ResMut<crate::build_ui::BuildState>>,
    catalog: Option<Res<crate::build_ui::ShapeCatalog>>,
    build_areas: Option<Res<crate::build_ui::BuildAreas>>,
) {
    let inner = api.inner.lock().unwrap();

    // --- Handle place requests ---
    let mut messages = messages;
    for req in &inner.place_requests {
        if let Some(ref pos) = inner.build_pos {
            if let Ok((mut chunk, chunk_transform)) = chunks.get_mut(pos.chunk_entity) {
                // Locked zones / active challenge runs prohibit modification —
                // but designated build areas (e.g. the spawn sandbox) override
                // zone locks. Nothing overrides the mid-run freeze.
                if let Some(ref locks) = build_locks {
                    let world_pos = chunk_transform.transform_point(Vec3::new(
                        (pos.x as f32 + 1.0) * VOXEL_SIZE,
                        (pos.y as f32 + 0.5) * VOXEL_SIZE,
                        (pos.z as f32 + 1.0) * VOXEL_SIZE,
                    ));
                    let in_designated_area = build_areas
                        .as_ref()
                        .is_some_and(|a| a.contains(world_pos));
                    let allowed = !locks.building_disabled
                        && (in_designated_area || locks.can_build_at(world_pos));
                    if !allowed {
                        if let Some(ref mut msgs) = messages {
                            if locks.building_disabled {
                                msgs.push("Can't build during a challenge run!");
                            } else {
                                msgs.push("This zone is locked — turn in trophies at its pedestal first.");
                            }
                        }
                        continue;
                    }
                }

                // Material cost: bulkier shapes cost more of the material.
                let cost = catalog.as_ref().map(|c| c.cost_of(req.shape)).unwrap_or(0);
                if let Some(ref mut state) = build_state {
                    let slot = state
                        .materials
                        .iter_mut()
                        .find(|m| m.texture == req.texture);
                    match slot {
                        Some(slot) if slot.count >= cost => {
                            slot.count -= cost;
                        }
                        Some(slot) => {
                            if let Some(ref mut msgs) = messages {
                                msgs.push(format!(
                                    "Not enough {} — need {}, have {}.",
                                    slot.name, cost, slot.count
                                ));
                            }
                            continue;
                        }
                        None => {}
                    }
                }

                if req.shape == SHAPE_WEDGE {
                    chunk.data.place_wedge(pos.x, pos.y, pos.z, req.facing, req.texture);
                } else if let Some(shape) = shape_table.get(req.shape) {
                    let occ = rotated_occupied_cells(shape, req.facing);
                    chunk
                        .data
                        .place_block(req.shape, req.facing, req.texture, pos.x, pos.y, pos.z, &occ);
                } else {
                    chunk.data.place_std(pos.x, pos.y, pos.z, req.shape, req.facing, req.texture);
                }
                chunk.pending_modifications.push(BlockModification {
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    shape: req.shape,
                    facing: req.facing,
                    texture: req.texture,
                });
                info!(
                    "Placed block at ({}, {}, {}) shape={} facing={:?}",
                    pos.x, pos.y, pos.z, req.shape, req.facing
                );

                // Auto-shape tool: re-cap the surrounding columns so the
                // terrain contour flows smoothly through the new block.
                if req.auto {
                    let mods = auto_reshape_contour(
                        &mut chunk.data,
                        pos.x,
                        pos.z,
                        req.texture,
                        &shape_table,
                    );
                    if !mods.is_empty() {
                        info!("Auto-shape re-capped {} columns", mods.len());
                        chunk.pending_modifications.extend(mods);
                    }
                }
            }
        }
    }

    // --- Handle preview ---
    match &inner.preview {
        PreviewState::Hidden => {
            if preview_res.entity.is_some() {
                for entity in preview_query.iter() {
                    commands.entity(entity).despawn();
                }
                preview_res.entity = None;
            }
        }
        PreviewState::AtBuildPosition { shape, facing } => {
            if let Some(ref pos) = inner.build_pos {
                if let Ok((_, chunk_transform)) = chunks.get(pos.chunk_entity) {
                    // The preview mesh is centered on the 2×2-cell footprint
                    // at its base, so rotation about Y stays in place.
                    let local_pos = Vec3::new(
                        (pos.x as f32 + 1.0) * VOXEL_SIZE,
                        pos.y as f32 * VOXEL_SIZE,
                        (pos.z as f32 + 1.0) * VOXEL_SIZE,
                    );
                    let world_pos = chunk_transform.transform_point(local_pos);
                    let world_rotation = chunk_transform.rotation
                        * Quat::from_rotation_y(facing.rotation_radians());
                    let transform =
                        Transform::from_translation(world_pos).with_rotation(world_rotation);
                    spawn_or_update_preview(
                        &mut commands,
                        &mut preview_res,
                        &mut meshes,
                        &mut materials,
                        &shape_table,
                        *shape,
                        transform,
                    );
                }
            }
        }
        PreviewState::InHand { shape, facing } => {
            let forward_flat =
                Vec3::new(inner.camera_forward.x, 0.0, inner.camera_forward.z)
                    .normalize_or_zero();
            let hand_pos = inner.player_pos + forward_flat * 1.5 + Vec3::Y * 0.5;
            let transform = Transform::from_translation(hand_pos)
                .with_rotation(Quat::from_rotation_y(facing.rotation_radians()));
            spawn_or_update_preview(
                &mut commands,
                &mut preview_res,
                &mut meshes,
                &mut materials,
                &shape_table,
                *shape,
                transform,
            );
        }
    }
}

fn spawn_or_update_preview(
    commands: &mut Commands,
    res: &mut PreviewResources,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    shape_table: &ShapeTable,
    shape: u16,
    transform: Transform,
) {
    if res.material.is_none() {
        res.material = Some(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 0.6, 1.0, 0.4),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            cull_mode: None,
            double_sided: true,
            ..default()
        }));
    }
    let mesh = res
        .meshes
        .entry(shape)
        .or_insert_with(|| {
            let m = shape_table
                .get(shape)
                .map(build_shape_preview_mesh)
                .unwrap_or_else(build_wedge_preview_mesh);
            meshes.add(m)
        })
        .clone();

    if let Some(entity) = res.entity {
        if commands.get_entity(entity).is_ok() {
            let mut e = commands.entity(entity);
            e.insert(transform);
            if res.current_shape != Some(shape) {
                e.insert(Mesh3d(mesh));
                res.current_shape = Some(shape);
            }
            return;
        }
        res.entity = None;
    }

    let entity = commands
        .spawn((
            BlockPreview,
            Mesh3d(mesh),
            MeshMaterial3d(res.material.clone().unwrap()),
            transform,
        ))
        .id();
    res.entity = Some(entity);
    res.current_shape = Some(shape);
}

/// Build a translucent preview mesh straight from a shape definition.
/// Vertices are in cell units with the origin at the block corner; the mesh
/// is re-centered on the 2×2 footprint so facing rotation spins in place.
fn build_shape_preview_mesh(shape: &jumpblocks_voxel::shape::BlockShape) -> Mesh {
    use bevy::mesh::{Indices, PrimitiveTopology};

    let center = Vec3::new(
        shape.size.0 as f32 * 0.5,
        0.0,
        shape.size.2 as f32 * 0.5,
    );
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for face in &shape.faces {
        let base = positions.len() as u32;
        for v in &face.vertices {
            positions.push((((*v) - center) * VOXEL_SIZE).to_array());
        }
        for t in &face.triangles {
            indices.extend([base + t[0] as u32, base + t[1] as u32, base + t[2] as u32]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_normals();
    mesh
}

// ---------------------------------------------------------------------------
// Auto-shape tool: contour reshaping
// ---------------------------------------------------------------------------

/// How far (in block columns) around a placed block the contour is re-capped.
const AUTO_RADIUS: i32 = 3;

/// What tops a terrain block column.
struct ColumnTop {
    /// Cube-surface height in cells: with a slope cap, the height its base
    /// cube layer would restore to (worldgen convention: cap base at h-1).
    height: i32,
    /// Texture of the topmost block (used for the replacement cap).
    texture: u16,
    /// The cap block currently on top, if any: (id, shape, facing).
    cap: Option<(jumpblocks_voxel::chunk::BlockId, u16, Facing)>,
}

/// Survey a block column (block coords, i.e. cells/2). Returns `None` for
/// empty columns and for columns topped by something that isn't a plain
/// cube or a slope cap (don't reshape a player's wedge sculpture).
fn column_top(data: &ChunkData, bx: usize, bz: usize) -> Option<ColumnTop> {
    use jumpblocks_voxel::chunk::Cell;
    let cx = bx * 2;
    let cz = bz * 2;
    for y in (0..CHUNK_Y).rev() {
        let cell = data.get_cell(cx, y, cz);
        let Cell::Local(id) = cell else {
            if cell.is_occupied() {
                // External cell (neighbor-owned) — leave the column alone.
                return None;
            }
            continue;
        };
        let block = data.get_block(id)?;
        if cap_corner_heights(block.shape).is_some() {
            return Some(ColumnTop {
                height: block.origin.1 as i32 + 1,
                texture: block.texture,
                cap: Some((id, block.shape, block.facing)),
            });
        }
        if block.shape == SHAPE_CUBE {
            return Some(ColumnTop {
                height: y as i32 + 1,
                texture: block.texture,
                cap: None,
            });
        }
        return None;
    }
    None
}

/// After an auto-shape placement at cell `(cell_x, *, cell_z)`, re-cap the
/// surrounding columns so the terrain flows smoothly through the new block:
/// strip existing slope caps back to flat cubes, then re-run the worldgen
/// cap classifier against the updated height field.
fn auto_reshape_contour(
    data: &mut ChunkData,
    cell_x: usize,
    cell_z: usize,
    center_texture: u16,
    shapes: &ShapeTable,
) -> Vec<BlockModification> {
    let bx = (cell_x / 2) as i32;
    let bz = (cell_z / 2) as i32;
    let max_bx = (CHUNK_X / 2) as i32 - 1;
    let max_bz = (CHUNK_Z / 2) as i32 - 1;
    let in_bounds = |ix: i32, iz: i32| ix >= 0 && iz >= 0 && ix <= max_bx && iz <= max_bz;

    // 1. Height survey, one ring beyond the reshape radius for the deltas.
    let mut heights: HashMap<(i32, i32), (i32, u16)> = HashMap::new();
    for ix in bx - AUTO_RADIUS - 1..=bx + AUTO_RADIUS + 1 {
        for iz in bz - AUTO_RADIUS - 1..=bz + AUTO_RADIUS + 1 {
            if !in_bounds(ix, iz) {
                continue;
            }
            if let Some(top) = column_top(data, ix as usize, iz as usize) {
                heights.insert((ix, iz), (top.height, top.texture));
            }
        }
    }

    let mut mods = Vec::new();

    // 2. Normalize: strip existing caps inside the radius back to flat cubes.
    for ix in bx - AUTO_RADIUS..=bx + AUTO_RADIUS {
        for iz in bz - AUTO_RADIUS..=bz + AUTO_RADIUS {
            if !in_bounds(ix, iz) {
                continue;
            }
            let Some(top) = column_top(data, ix as usize, iz as usize) else {
                continue;
            };
            if let Some((id, cap_shape, cap_facing)) = top.cap {
                let Some(shape) = shapes.get(cap_shape) else { continue };
                let occ = rotated_occupied_cells(shape, cap_facing);
                data.remove_block(id, &occ);
                let base = top.height - 1;
                if base >= 0 && (base as usize) < CHUNK_Y {
                    data.place_std(
                        ix as usize * 2,
                        base as usize,
                        iz as usize * 2,
                        SHAPE_CUBE,
                        Facing::North,
                        top.texture,
                    );
                }
            }
        }
    }

    // 3. Re-classify each column against the surveyed height field.
    for ix in bx - AUTO_RADIUS..=bx + AUTO_RADIUS {
        for iz in bz - AUTO_RADIUS..=bz + AUTO_RADIUS {
            if !in_bounds(ix, iz) {
                continue;
            }
            let Some(&(h, tex)) = heights.get(&(ix, iz)) else {
                continue;
            };
            if h < 1 {
                continue;
            }
            // Unsurveyed neighbors (chunk edge / empty) count as level.
            let delta =
                |dx: i32, dz: i32| heights.get(&(ix + dx, iz + dz)).map(|&(nh, _)| nh - h).unwrap_or(0);
            let Some((shape_id, facing, ch)) = classify_slope_cap(&delta) else {
                continue;
            };
            let base = h - 1;
            if base < 0 || base + ch > CHUNK_Y as i32 {
                continue;
            }

            // Swap the flat top cube for the cap.
            let cx = ix as usize * 2;
            let cz = iz as usize * 2;
            let Some(top) = column_top(data, ix as usize, iz as usize) else {
                continue;
            };
            if top.cap.is_some() || top.height != h {
                continue;
            }
            if let jumpblocks_voxel::chunk::Cell::Local(id) =
                data.get_cell(cx, base as usize, cz)
            {
                data.remove_block(id, &BLOCK_CELLS);
            }
            let Some(shape) = shapes.get(shape_id) else { continue };
            let occ = rotated_occupied_cells(shape, facing);
            let texture = if ix == bx && iz == bz { center_texture } else { tex };
            data.place_block(shape_id, facing, texture, cx, base as usize, cz, &occ);
            mods.push(BlockModification {
                x: cx,
                y: base as usize,
                z: cz,
                shape: shape_id,
                facing,
                texture,
            });
        }
    }

    mods
}

// ---------------------------------------------------------------------------
// Chunk modification processing
// ---------------------------------------------------------------------------

fn process_chunk_modifications(
    mut chunks: Query<(&jumpblocks_voxel::coords::ChunkCoord, &mut Chunk)>,
) {
    // First pass: apply modifications and snapshot the fresh data.
    let mut modified: Vec<(jumpblocks_voxel::coords::ChunkPos, Arc<jumpblocks_voxel::chunk::ChunkData>)> =
        Vec::new();
    for (coord, mut chunk) in chunks.iter_mut() {
        if chunk.pending_modifications.is_empty() {
            continue;
        }
        let count = chunk.pending_modifications.len();
        chunk.pending_modifications.clear();
        chunk.mark_dirty();
        modified.push((coord.pos, Arc::new(chunk.data.clone())));
        info!(
            "Applied {} block modification(s), chunk marked dirty for re-mesh",
            count
        );
    }
    if modified.is_empty() {
        return;
    }

    // Second pass: neighbors mesh with a halo of our blocks (seam fillets),
    // so they need the fresh data and a re-mesh too.
    for (coord, mut chunk) in chunks.iter_mut() {
        for (pos, arc) in &modified {
            let dx = pos.x - coord.pos.x;
            let dy = pos.y - coord.pos.y;
            let dz = pos.z - coord.pos.z;
            if (dx == 0 && dy == 0 && dz == 0)
                || dx.abs() > 1 || dy.abs() > 1 || dz.abs() > 1
            {
                continue;
            }
            chunk.neighbors.set_arc(dx, dy, dz, arc.clone());
            chunk.mark_dirty();
        }
    }
}

// ---------------------------------------------------------------------------
// Placement position search
// ---------------------------------------------------------------------------

const BUILD_RAY_MAX_DIST: f32 = 6.0;
const BUILD_RAY_STEP: f32 = 0.25;

fn find_placement_position(
    player_pos: Vec3,
    camera_forward: Vec3,
    chunks: &Query<(Entity, &Chunk, &Transform)>,
) -> Option<BuildPos> {
    let ray_dir = camera_forward.normalize_or_zero();
    if ray_dir == Vec3::ZERO {
        return None;
    }

    let ray_origin = player_pos + Vec3::Y * 0.5;

    let steps = ((BUILD_RAY_MAX_DIST / BUILD_RAY_STEP) as usize).max(1);

    let mut prev_world = ray_origin;

    for i in 1..=steps {
        let t = i as f32 * BUILD_RAY_STEP;
        let sample_world = ray_origin + ray_dir * t;

        for (chunk_entity, chunk, chunk_transform) in chunks.iter() {
            let local = chunk_transform
                .compute_affine()
                .inverse()
                .transform_point3(sample_world);

            let vx = (local.x / VOXEL_SIZE).floor() as i32;
            let vy = (local.y / VOXEL_SIZE).floor() as i32;
            let vz = (local.z / VOXEL_SIZE).floor() as i32;

            if vx < 0 || vy < 0 || vz < 0 {
                continue;
            }
            let (ux, uy, uz) = (vx as usize, vy as usize, vz as usize);
            if ux >= CHUNK_X || uy >= CHUNK_Y || uz >= CHUNK_Z {
                continue;
            }

            if chunk.data.is_occupied(ux, uy, uz) {
                if let Some(pos) =
                    try_place_at_world(prev_world, chunks)
                {
                    return Some(pos);
                }
                if let Some(pos) = search_down_for_build(ux, uy, uz, chunk, chunk_entity) {
                    return Some(pos);
                }
            }
        }

        prev_world = sample_world;
    }

    try_place_at_world(prev_world, chunks)
}

fn try_place_at_world(
    world_pos: Vec3,
    chunks: &Query<(Entity, &Chunk, &Transform)>,
) -> Option<BuildPos> {
    for (chunk_entity, chunk, chunk_transform) in chunks.iter() {
        let local = chunk_transform
            .compute_affine()
            .inverse()
            .transform_point3(world_pos);

        let vx = (local.x / VOXEL_SIZE).floor() as i32;
        let vy = (local.y / VOXEL_SIZE).floor() as i32;
        let vz = (local.z / VOXEL_SIZE).floor() as i32;

        if vx < 0 || vy < 0 || vz < 0 {
            continue;
        }
        let (ux, uy, uz) = (vx as usize, vy as usize, vz as usize);
        if ux >= CHUNK_X || uy >= CHUNK_Y || uz >= CHUNK_Z {
            continue;
        }

        let (ux, uy, uz) = (ux & !1, uy, uz & !1);

        if chunk.data.can_place_std(ux, uy, uz) {
            return Some(BuildPos {
                x: ux,
                y: uy,
                z: uz,
                chunk_entity,
            });
        }

        if let Some(pos) = search_down_for_build(ux, uy, uz, chunk, chunk_entity) {
            return Some(pos);
        }
    }
    None
}

fn search_down_for_build(
    x: usize,
    start_y: usize,
    z: usize,
    chunk: &Chunk,
    chunk_entity: Entity,
) -> Option<BuildPos> {
    let x = x & !1;
    let z = z & !1;
    if start_y + 1 < CHUNK_Y && chunk.data.can_place_std(x, start_y + 1, z) {
        return Some(BuildPos {
            x,
            y: start_y + 1,
            z,
            chunk_entity,
        });
    }

    for y in (0..=start_y).rev() {
        if chunk.data.can_place_std(x, y, z) {
            return Some(BuildPos {
                x,
                y,
                z,
                chunk_entity,
            });
        }
        if chunk.data.is_occupied(x, y, z) && y + 1 <= start_y {
            if chunk.data.can_place_std(x, y + 1, z) {
                return Some(BuildPos {
                    x,
                    y: y + 1,
                    z,
                    chunk_entity,
                });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Preview mesh
// ---------------------------------------------------------------------------

fn build_wedge_preview_mesh() -> Mesh {
    use bevy::mesh::{Indices, PrimitiveTopology};

    let h = VOXEL_SIZE;

    // Wedge: back wall (south/-Z) is tall, slope descends toward north/+Z
    let positions = vec![
        [-h, -h, -h], // 0: back-left bottom
        [h, -h, -h],  // 1: back-right bottom
        [h, -h, h],   // 2: front-right bottom
        [-h, -h, h],  // 3: front-left bottom
        [-h, h, -h],  // 4: back-left top
        [h, h, -h],   // 5: back-right top
    ];

    let indices = vec![
        // Bottom face
        0u32, 2, 1, 0, 3, 2, // Back wall
        4, 1, 5, 4, 0, 1, // Slope face
        4, 5, 2, 4, 2, 3, // Left side
        0, 4, 3, // Right side
        1, 2, 5,
    ];

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_normals();
    mesh
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn facing_from_int(v: INT) -> Facing {
    match v & 3 {
        0 => Facing::North,
        1 => Facing::East,
        2 => Facing::South,
        3 => Facing::West,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jumpblocks_voxel::shape::{SHAPE_WEDGE_OUTER};

    /// Flat plateau of cube columns at `height` cells over block columns
    /// [b0, b1] × [b0, b1].
    fn flat_terrain(height: usize, b0: usize, b1: usize) -> ChunkData {
        let mut data = ChunkData::new();
        for bx in b0..=b1 {
            for bz in b0..=b1 {
                for y in 0..height {
                    data.place_std(bx * 2, y, bz * 2, SHAPE_CUBE, Facing::North, 3);
                }
            }
        }
        data
    }

    #[test]
    fn auto_shape_caps_neighbors_of_a_bump() {
        let shapes = ShapeTable::default();
        let mut data = flat_terrain(6, 3, 13);

        // The auto tool just placed one cube on top of column (8, 8).
        data.place_std(16, 6, 16, SHAPE_CUBE, Facing::North, 12);

        let mods = auto_reshape_contour(&mut data, 16, 16, 12, &shapes);
        assert!(!mods.is_empty(), "reshape should have re-capped columns");

        // The bump column itself stays flat (all neighbors lower).
        let center = column_top(&data, 8, 8).expect("center column");
        assert_eq!(center.height, 7);
        assert!(center.cap.is_none(), "bump top should stay a cube");

        // Side neighbors get straight wedges rising toward the bump.
        for (bx, bz, facing) in [
            (7, 8, Facing::East),
            (9, 8, Facing::West),
            (8, 7, Facing::South),
            (8, 9, Facing::North),
        ] {
            let top = column_top(&data, bx, bz).expect("side neighbor");
            let (_, shape, f) = top.cap.expect("side neighbor should wear a cap");
            assert_eq!(shape, SHAPE_WEDGE, "column ({bx},{bz})");
            assert_eq!(f, facing, "column ({bx},{bz})");
        }

        // Diagonal neighbors get outer (hill) corners.
        for (bx, bz) in [(7, 7), (9, 7), (7, 9), (9, 9)] {
            let top = column_top(&data, bx, bz).expect("diagonal neighbor");
            let (_, shape, _) = top.cap.expect("diagonal neighbor should wear a cap");
            assert_eq!(shape, SHAPE_WEDGE_OUTER, "column ({bx},{bz})");
        }

        // Columns beyond the radius stay untouched flat cubes.
        let far = column_top(&data, 12, 12).expect("far column");
        assert!(far.cap.is_none());
        assert_eq!(far.height, 6);
    }

    #[test]
    fn auto_shape_is_idempotent_when_recapping() {
        let shapes = ShapeTable::default();
        let mut data = flat_terrain(6, 3, 13);
        data.place_std(16, 6, 16, SHAPE_CUBE, Facing::North, 12);

        auto_reshape_contour(&mut data, 16, 16, 12, &shapes);
        // A second pass over the same spot (e.g. stacking another cube after
        // caps exist) must strip and re-derive caps without corrupting cells.
        data.place_std(16, 7, 16, SHAPE_CUBE, Facing::North, 12);
        auto_reshape_contour(&mut data, 16, 16, 12, &shapes);

        let center = column_top(&data, 8, 8).expect("center column");
        assert_eq!(center.height, 8);
        assert!(center.cap.is_none());

        // Side neighbors now see a +2 step → steep wedge family.
        let top = column_top(&data, 7, 8).expect("side neighbor");
        let (_, shape, _) = top.cap.expect("cap expected");
        assert_eq!(shape, jumpblocks_voxel::shape::SHAPE_WEDGE_STEEP);
    }
}
