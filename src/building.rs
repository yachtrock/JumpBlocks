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
//! - `set_selected_shape(shape)` — tell the placement system which shape to validate
//! - `place_block(shape, facing, texture)` — place at the current build position
//! - `rotate_facing_right(facing)` / `rotate_facing_left(facing)` — helpers

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use jumpblocks_voxel::chunk::{
    Chunk, ChunkData, BlockModification, CHUNK_X, CHUNK_Y, CHUNK_Z, VOXEL_SIZE,
};
use jumpblocks_voxel::shape::{Facing, ShapeTable, SHAPE_CUBE};
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
    AtBuildPosition { facing: Facing },
    /// Show floating in front of the player (no chunk context needed).
    InHand { facing: Facing },
}

/// A request to place a block, queued by the script.
#[derive(Clone)]
struct PlaceRequest {
    shape: u16,
    facing: Facing,
    texture: u16,
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
    /// The shape the script wants to place (persists across frames).
    selected_shape: u16,
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
        }
    }

    fn clear_frame(&mut self) {
        self.build_pos = None;
        self.preview = PreviewState::Hidden;
        self.place_requests.clear();
        // Note: selected_shape is NOT cleared — it persists across frames.
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
    mesh: Option<Handle<Mesh>>,
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
                    data.place_requests.push(PlaceRequest {
                        shape: shape as u16,
                        facing: facing_from_int(facing),
                        texture: texture as u16,
                    });
                }
            },
        );
    }

    // set_selected_shape(shape) — tell the placement system what shape to validate for
    {
        let inner = inner.clone();
        engine
            .rhai_engine_mut()
            .register_fn("set_selected_shape", move |shape: INT| {
                let mut data = inner.lock().unwrap();
                data.selected_shape = shape as u16;
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
    shapes: Res<ShapeTable>,
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

    let occupied = shapes
        .get(inner.selected_shape)
        .map(|s| s.occupied_cells.as_slice())
        .unwrap_or(&jumpblocks_voxel::chunk::BLOCK_CELLS);

    inner.build_pos = find_placement_position(player_pos, cam_forward, &chunks, occupied);
}

// ---------------------------------------------------------------------------
// Post-system: apply script-driven commands after script runs
// ---------------------------------------------------------------------------

fn apply_building_commands(
    mut commands: Commands,
    api: Res<BuildingApi>,
    shapes: Res<ShapeTable>,
    mut chunks: Query<(&mut Chunk, &Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut preview_res: ResMut<PreviewResources>,
    preview_query: Query<Entity, With<BlockPreview>>,
) {
    let inner = api.inner.lock().unwrap();

    // --- Handle place requests ---
    for req in &inner.place_requests {
        if let Some(ref pos) = inner.build_pos {
            if let Ok((mut chunk, _)) = chunks.get_mut(pos.chunk_entity) {
                let occupied = shapes
                    .get(req.shape)
                    .map(|s| s.occupied_cells.as_slice())
                    .unwrap_or(&jumpblocks_voxel::chunk::BLOCK_CELLS);

                chunk.data.place_block(
                    req.shape, req.facing, req.texture,
                    pos.x, pos.y, pos.z, occupied,
                );
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
        PreviewState::AtBuildPosition { facing } => {
            if let Some(ref pos) = inner.build_pos {
                if let Ok((_, chunk_transform)) = chunks.get(pos.chunk_entity) {
                    let local_pos = Vec3::new(
                        pos.x as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5,
                        pos.y as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5,
                        pos.z as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5,
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
                        transform,
                    );
                }
            }
        }
        PreviewState::InHand { facing } => {
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
    transform: Transform,
) {
    if res.material.is_none() {
        res.material = Some(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 0.6, 1.0, 0.4),
            alpha_mode: AlphaMode::Blend,
            ..default()
        }));
    }
    if res.mesh.is_none() {
        res.mesh = Some(meshes.add(build_wedge_preview_mesh()));
    }

    if let Some(entity) = res.entity {
        if commands.get_entity(entity).is_ok() {
            commands.entity(entity).insert(transform);
            return;
        }
        res.entity = None;
    }

    let entity = commands
        .spawn((
            BlockPreview,
            Mesh3d(res.mesh.clone().unwrap()),
            MeshMaterial3d(res.material.clone().unwrap()),
            transform,
        ))
        .id();
    res.entity = Some(entity);
}

// ---------------------------------------------------------------------------
// Chunk modification processing
// ---------------------------------------------------------------------------

fn process_chunk_modifications(mut chunks: Query<&mut Chunk>) {
    for mut chunk in chunks.iter_mut() {
        if chunk.pending_modifications.is_empty() {
            continue;
        }
        let count = chunk.pending_modifications.len();
        chunk.pending_modifications.clear();
        chunk.mark_dirty();
        info!(
            "Applied {} block modification(s), chunk marked dirty for re-mesh",
            count
        );
    }
}

// ---------------------------------------------------------------------------
// Placement position search
// ---------------------------------------------------------------------------

const BUILD_RAY_MAX_DIST: f32 = 6.0;
const BUILD_RAY_STEP: f32 = 0.25;

/// Find the best placement position for a block along the camera ray.
///
/// Algorithm:
/// 1. Step along the ray until we hit an occupied cell.
/// 2. Determine which face of the cell was hit (approach direction).
/// 3. Try placing the block adjacent to that face, testing all valid origins
///    so that one of the block's occupied cells lands in the adjacent empty cell.
/// 4. Among valid origins, pick the one whose block center is closest to the
///    aim point for the most intuitive placement.
fn find_placement_position(
    player_pos: Vec3,
    camera_forward: Vec3,
    chunks: &Query<(Entity, &Chunk, &Transform)>,
    occupied_cells: &[(u8, u8, u8)],
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
            let inv = chunk_transform.compute_affine().inverse();
            let local = inv.transform_point3(sample_world);

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
                // Determine approach direction from the previous ray sample.
                let prev_local = inv.transform_point3(prev_world);
                let aim_local = local;

                if let Some(pos) = find_adjacent_placement(
                    &chunk.data,
                    chunk_entity,
                    ux, uy, uz,
                    prev_local,
                    aim_local,
                    occupied_cells,
                ) {
                    return Some(pos);
                }
            }
        }

        prev_world = sample_world;
    }

    // End of ray — try placing in the air if adjacent to something.
    try_place_near_world(prev_world, chunks, occupied_cells)
}

/// Given a hit cell, find the best adjacent position to place a block.
///
/// Determines the approach face from the previous ray sample, then tries all
/// possible block origins that would put one of the block's cells in the
/// adjacent empty space. Picks the origin whose block center is closest to
/// the aim point.
fn find_adjacent_placement(
    data: &ChunkData,
    chunk_entity: Entity,
    hit_x: usize,
    hit_y: usize,
    hit_z: usize,
    prev_local: Vec3,
    aim_local: Vec3,
    occupied_cells: &[(u8, u8, u8)],
) -> Option<BuildPos> {
    // Build a prioritized list of adjacent directions.
    // Primary: the face the ray approached from.
    // Secondary: all other faces, ordered by alignment with the approach direction.
    let approach = Vec3::new(
        prev_local.x - aim_local.x,
        prev_local.y - aim_local.y,
        prev_local.z - aim_local.z,
    );

    let mut face_dirs: [(i32, i32, i32); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    // Sort faces so the one most aligned with the approach direction comes first.
    face_dirs.sort_by(|a, b| {
        let dot_a = a.0 as f32 * approach.x + a.1 as f32 * approach.y + a.2 as f32 * approach.z;
        let dot_b = b.0 as f32 * approach.x + b.1 as f32 * approach.y + b.2 as f32 * approach.z;
        dot_b.partial_cmp(&dot_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    for &(fdx, fdy, fdz) in &face_dirs {
        let adj_x = hit_x as i32 + fdx;
        let adj_y = hit_y as i32 + fdy;
        let adj_z = hit_z as i32 + fdz;

        if adj_x < 0 || adj_y < 0 || adj_z < 0 {
            continue;
        }

        // Try all origins that would place one of the block's cells at (adj_x, adj_y, adj_z).
        let mut best: Option<BuildPos> = None;
        let mut best_dist_sq = f32::MAX;
        let aim_cell = Vec3::new(adj_x as f32, adj_y as f32, adj_z as f32);

        for &(ox, oy, oz) in occupied_cells {
            let origin_x = adj_x - ox as i32;
            let origin_y = adj_y - oy as i32;
            let origin_z = adj_z - oz as i32;

            if origin_x < 0 || origin_y < 0 || origin_z < 0 {
                continue;
            }
            let (ux, uy, uz) = (origin_x as usize, origin_y as usize, origin_z as usize);

            if data.can_place(ux, uy, uz, occupied_cells) {
                // Compute block center for tie-breaking.
                let center = block_center(ux, uy, uz, occupied_cells);
                let dist_sq = (center - aim_cell).length_squared();
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best = Some(BuildPos {
                        x: ux,
                        y: uy,
                        z: uz,
                        chunk_entity,
                    });
                }
            }
        }

        if best.is_some() {
            return best;
        }
    }

    // Fallback: search up/down near the hit column.
    search_vertical_for_build(hit_x, hit_y, hit_z, data, chunk_entity, occupied_cells)
}

/// Try placing a block near a world position (end-of-ray fallback).
fn try_place_near_world(
    world_pos: Vec3,
    chunks: &Query<(Entity, &Chunk, &Transform)>,
    occupied_cells: &[(u8, u8, u8)],
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

        // Try origins near this cell that would cover it.
        if let Some(pos) = try_origins_near(ux, uy, uz, &chunk.data, chunk_entity, occupied_cells) {
            return Some(pos);
        }

        if let Some(pos) =
            search_vertical_for_build(ux, uy, uz, &chunk.data, chunk_entity, occupied_cells)
        {
            return Some(pos);
        }
    }
    None
}

/// Try all block origins that would place one of the block's cells at `(cx, cy, cz)`.
fn try_origins_near(
    cx: usize,
    cy: usize,
    cz: usize,
    data: &ChunkData,
    chunk_entity: Entity,
    occupied_cells: &[(u8, u8, u8)],
) -> Option<BuildPos> {
    let mut best: Option<BuildPos> = None;
    let mut best_dist_sq = f32::MAX;
    let target = Vec3::new(cx as f32, cy as f32, cz as f32);

    for &(ox, oy, oz) in occupied_cells {
        let origin_x = cx as i32 - ox as i32;
        let origin_y = cy as i32 - oy as i32;
        let origin_z = cz as i32 - oz as i32;

        if origin_x < 0 || origin_y < 0 || origin_z < 0 {
            continue;
        }
        let (ux, uy, uz) = (origin_x as usize, origin_y as usize, origin_z as usize);

        if data.can_place(ux, uy, uz, occupied_cells) {
            let center = block_center(ux, uy, uz, occupied_cells);
            let dist_sq = (center - target).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best = Some(BuildPos {
                    x: ux,
                    y: uy,
                    z: uz,
                    chunk_entity,
                });
            }
        }
    }
    best
}

/// Search vertically near a column for a valid placement.
fn search_vertical_for_build(
    x: usize,
    start_y: usize,
    z: usize,
    data: &ChunkData,
    chunk_entity: Entity,
    occupied_cells: &[(u8, u8, u8)],
) -> Option<BuildPos> {
    // Try one above first.
    if start_y + 1 < CHUNK_Y {
        if let Some(pos) = try_origins_near(x, start_y + 1, z, data, chunk_entity, occupied_cells) {
            return Some(pos);
        }
    }

    // Search downward.
    for y in (0..=start_y).rev() {
        if let Some(pos) = try_origins_near(x, y, z, data, chunk_entity, occupied_cells) {
            return Some(pos);
        }
        // If we hit an occupied cell, try just above it.
        if data.is_occupied(x, y, z) && y + 1 <= start_y {
            if let Some(pos) = try_origins_near(x, y + 1, z, data, chunk_entity, occupied_cells) {
                return Some(pos);
            }
            break;
        }
    }
    None
}

/// Compute the center of a block's occupied cells given its origin.
fn block_center(ox: usize, oy: usize, oz: usize, occupied_cells: &[(u8, u8, u8)]) -> Vec3 {
    let n = occupied_cells.len() as f32;
    let mut sum = Vec3::ZERO;
    for &(dx, dy, dz) in occupied_cells {
        sum += Vec3::new(
            (ox + dx as usize) as f32,
            (oy + dy as usize) as f32,
            (oz + dz as usize) as f32,
        );
    }
    sum / n
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
