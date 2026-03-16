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

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use jumpblocks_voxel::chunk::{
    Chunk, BlockModification, CHUNK_X, CHUNK_Y, CHUNK_Z, VOXEL_SIZE,
};
use jumpblocks_voxel::shape::{Facing, SHAPE_WEDGE};
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
}

impl BuildingApiInner {
    fn new() -> Self {
        Self {
            build_pos: None,
            player_pos: Vec3::ZERO,
            camera_forward: Vec3::NEG_Z,
            preview: PreviewState::Hidden,
            place_requests: Vec::new(),
        }
    }

    fn clear_frame(&mut self) {
        self.build_pos = None;
        self.preview = PreviewState::Hidden;
        self.place_requests.clear();
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
) {
    let inner = api.inner.lock().unwrap();

    // --- Handle place requests ---
    for req in &inner.place_requests {
        if let Some(ref pos) = inner.build_pos {
            if let Ok((mut chunk, _)) = chunks.get_mut(pos.chunk_entity) {
                if req.shape == SHAPE_WEDGE {
                    chunk.data.place_wedge(pos.x, pos.y, pos.z, req.facing, req.texture);
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
