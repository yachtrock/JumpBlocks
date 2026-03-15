//! Block placement and preview system.
//!
//! When the player is in the "building" action state, this module:
//! 1. Finds the grid position in front of the player to preview block placement
//! 2. Draws a translucent preview block at that position (or held in hand if no valid spot)
//! 3. Handles `place_block` / `rotate_left` / `rotate_right` events from the action state script
//! 4. On placement, writes the voxel into the chunk and triggers re-meshing

use bevy::prelude::*;
use jumpblocks_voxel::chunk::{
    Chunk, Voxel, VoxelModification, CHUNK_X, CHUNK_Y, CHUNK_Z, VOXEL_HEIGHT, VOXEL_WIDTH,
};
use jumpblocks_voxel::shape::{Facing, SHAPE_WEDGE};

use crate::action_state::{ActionState, ActionStateEmits};
use crate::camera::OrbitCamera;
use crate::player::Player;

pub struct BuildingPlugin;

impl Plugin for BuildingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BuildingState>().add_systems(
            Update,
            (
                building_system.after(crate::action_state::action_state_update),
                process_chunk_modifications,
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

/// Marker for the translucent preview entity.
#[derive(Component)]
struct BlockPreview;

/// Persistent building state across frames.
#[derive(Resource)]
pub struct BuildingState {
    /// Current facing direction for the block being placed.
    pub facing: Facing,
    /// The preview entity (spawned once, reused).
    preview_entity: Option<Entity>,
    /// Handle to the translucent material for preview blocks.
    preview_material: Option<Handle<StandardMaterial>>,
    /// Cached preview mesh handle.
    preview_mesh: Option<Handle<Mesh>>,
    /// Was the player in building state last frame?
    was_building: bool,
}

impl Default for BuildingState {
    fn default() -> Self {
        Self {
            facing: Facing::North,
            was_building: false,
            preview_entity: None,
            preview_material: None,
            preview_mesh: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Finding placement position
// ---------------------------------------------------------------------------

/// Result of searching for a valid placement spot.
struct PlacementResult {
    chunk_entity: Entity,
    x: usize,
    y: usize,
    z: usize,
    world_pos: Vec3,
}

/// Find the voxel grid position in front of the player that sits on a chunk.
fn find_placement_position(
    player_pos: Vec3,
    camera_forward: Vec3,
    chunks: &Query<(Entity, &Chunk, &Transform)>,
) -> Option<PlacementResult> {
    let forward_flat = Vec3::new(camera_forward.x, 0.0, camera_forward.z).normalize_or_zero();
    let target_world = player_pos + forward_flat * 2.0;

    for (chunk_entity, chunk, chunk_transform) in chunks.iter() {
        let chunk_origin = chunk_transform.translation;
        let local = target_world - chunk_origin;

        let vx = (local.x / VOXEL_WIDTH).floor() as i32;
        let vy = (local.y / VOXEL_HEIGHT).floor() as i32;
        let vz = (local.z / VOXEL_WIDTH).floor() as i32;

        if vx < 0 || vy < 0 || vz < 0 {
            continue;
        }
        let (ux, uy, uz) = (vx as usize, vy as usize, vz as usize);
        if ux >= CHUNK_X || uy >= CHUNK_Y || uz >= CHUNK_Z {
            continue;
        }

        // Must be a valid build spot (empty + has orthogonal neighbor).
        // If the target is occupied, try one voxel up.
        let (final_x, final_y, final_z) = if chunk.data.can_build_at(ux, uy, uz) {
            (ux, uy, uz)
        } else if chunk.data.get(ux, uy, uz).is_filled() {
            let uy_up = uy + 1;
            if chunk.data.can_build_at(ux, uy_up, uz) {
                (ux, uy_up, uz)
            } else {
                continue;
            }
        } else {
            continue;
        };

        let world_pos = chunk_origin
            + Vec3::new(
                final_x as f32 * VOXEL_WIDTH + VOXEL_WIDTH * 0.5,
                final_y as f32 * VOXEL_HEIGHT + VOXEL_HEIGHT * 0.5,
                final_z as f32 * VOXEL_WIDTH + VOXEL_WIDTH * 0.5,
            );

        return Some(PlacementResult {
            chunk_entity,
            x: final_x,
            y: final_y,
            z: final_z,
            world_pos,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Main building system
// ---------------------------------------------------------------------------

fn building_system(
    mut commands: Commands,
    mut building_state: ResMut<BuildingState>,
    action_emits: Res<ActionStateEmits>,
    action_query: Query<(&ActionState, &Transform), With<Player>>,
    camera_query: Query<&Transform, (With<OrbitCamera>, Without<Player>)>,
    chunks: Query<(Entity, &Chunk, &Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    preview_query: Query<Entity, With<BlockPreview>>,
) {
    let Ok((action_state, player_transform)) = action_query.single() else {
        return;
    };

    let is_building = action_state
        .0
        .as_deref()
        .is_some_and(|s| s == "building");

    // Clean up preview when exiting building state
    if !is_building {
        if building_state.was_building {
            for entity in preview_query.iter() {
                commands.entity(entity).despawn();
            }
            building_state.preview_entity = None;
            building_state.was_building = false;
            building_state.facing = Facing::North;
        }
        return;
    }

    // Handle rotate events
    for emit in &action_emits.0 {
        match emit.as_str() {
            "rotate_left" => {
                building_state.facing = rotate_facing_left(building_state.facing);
                info!("Block facing: {:?}", building_state.facing);
            }
            "rotate_right" => {
                building_state.facing = rotate_facing_right(building_state.facing);
                info!("Block facing: {:?}", building_state.facing);
            }
            _ => {}
        }
    }

    building_state.was_building = true;

    let Ok(camera_transform) = camera_query.single() else {
        return;
    };

    let player_pos = player_transform.translation;
    let cam_forward = camera_transform.forward().as_vec3();
    let placement = find_placement_position(player_pos, cam_forward, &chunks);

    // Handle place_block: queue a deferred command to write voxel data
    let wants_place = action_emits.0.iter().any(|e| e == "place_block");
    if wants_place {
        if let Some(ref pl) = placement {
            let voxel = Voxel::new(SHAPE_WEDGE, building_state.facing, 1);
            let entity = pl.chunk_entity;
            let (x, y, z) = (pl.x, pl.y, pl.z);
            commands.queue(move |world: &mut World| {
                if let Some(mut chunk) = world.get_mut::<Chunk>(entity) {
                    chunk.data.set(x, y, z, voxel);
                    chunk.pending_modifications.push(VoxelModification {
                        x,
                        y,
                        z,
                        voxel,
                    });
                }
            });
            info!(
                "Placed wedge at ({}, {}, {}) facing {:?}",
                pl.x, pl.y, pl.z, building_state.facing
            );
        }
    }

    // Ensure preview material exists
    if building_state.preview_material.is_none() {
        building_state.preview_material = Some(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 0.6, 1.0, 0.4),
            alpha_mode: AlphaMode::Blend,
            ..default()
        }));
    }

    // Ensure preview mesh exists
    if building_state.preview_mesh.is_none() {
        building_state.preview_mesh = Some(meshes.add(build_wedge_preview_mesh()));
    }

    // Compute preview transform
    let preview_transform = if let Some(ref pl) = placement {
        Transform::from_translation(pl.world_pos)
            .with_rotation(Quat::from_rotation_y(building_state.facing.rotation_radians()))
    } else {
        // No valid position: hold block in front of player
        let forward_flat = Vec3::new(cam_forward.x, 0.0, cam_forward.z).normalize_or_zero();
        let hand_pos = player_pos + forward_flat * 1.5 + Vec3::Y * 0.5;
        Transform::from_translation(hand_pos)
            .with_rotation(Quat::from_rotation_y(building_state.facing.rotation_radians()))
    };

    // Spawn or update preview entity
    if let Some(entity) = building_state.preview_entity {
        if commands.get_entity(entity).is_ok() {
            commands.entity(entity).insert(preview_transform);
        } else {
            building_state.preview_entity = None;
        }
    }

    if building_state.preview_entity.is_none() {
        let mesh_handle = building_state.preview_mesh.clone().unwrap();
        let mat = building_state.preview_material.clone().unwrap();
        let entity = commands
            .spawn((
                BlockPreview,
                Mesh3d(mesh_handle),
                MeshMaterial3d(mat),
                preview_transform,
            ))
            .id();
        building_state.preview_entity = Some(entity);
    }
}

/// Build a simple wedge preview mesh (centered on origin, voxel-sized).
fn build_wedge_preview_mesh() -> Mesh {
    use bevy::mesh::{Indices, PrimitiveTopology};

    // Wedge: back wall (south/-Z) is tall, slope descends toward north/+Z
    let hw = VOXEL_WIDTH * 0.5;
    let hh = VOXEL_HEIGHT * 0.5;

    let positions = vec![
        // Bottom quad
        [-hw, -hh, -hw], // 0: back-left bottom
        [hw, -hh, -hw],  // 1: back-right bottom
        [hw, -hh, hw],   // 2: front-right bottom
        [-hw, -hh, hw],  // 3: front-left bottom
        // Top edge (back wall only)
        [-hw, hh, -hw], // 4: back-left top
        [hw, hh, -hw],  // 5: back-right top
    ];

    let indices = vec![
        // Bottom face
        0u32, 2, 1, 0, 3, 2, // Back wall (south face, -Z)
        4, 1, 5, 4, 0, 1, // Slope face (from top-back to bottom-front)
        4, 5, 2, 4, 2, 3, // Left triangle side
        0, 4, 3, // Right triangle side
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
// Chunk modification processing
// ---------------------------------------------------------------------------

/// System: drains pending voxel modifications and triggers chunk re-meshing.
fn process_chunk_modifications(mut chunks: Query<&mut Chunk>) {
    for mut chunk in chunks.iter_mut() {
        if chunk.pending_modifications.is_empty() {
            continue;
        }

        let count = chunk.pending_modifications.len();
        chunk.pending_modifications.clear();
        chunk.mark_dirty();

        info!(
            "Applied {} voxel modification(s), chunk marked dirty for re-mesh",
            count
        );
    }
}

// ---------------------------------------------------------------------------
// Facing rotation helpers
// ---------------------------------------------------------------------------

fn rotate_facing_right(f: Facing) -> Facing {
    match f {
        Facing::North => Facing::East,
        Facing::East => Facing::South,
        Facing::South => Facing::West,
        Facing::West => Facing::North,
    }
}

fn rotate_facing_left(f: Facing) -> Facing {
    match f {
        Facing::North => Facing::West,
        Facing::West => Facing::South,
        Facing::South => Facing::East,
        Facing::East => Facing::North,
    }
}
