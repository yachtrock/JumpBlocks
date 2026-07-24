use std::path::Path;

use avian3d::prelude::*;
use bevy::prelude::*;
use jumpblocks_voxel::chunk_lod::{ChunkDitherMaterial, DitherFadeExtension};
use jumpblocks_voxel::coords::{ChunkPos, RegionId};
use jumpblocks_voxel::persistence;
use jumpblocks_voxel::streaming::{ChunkMaterial, WorldSavePath};
use jumpblocks_voxel::world_def::{WorldDef, cell_base_world};
use jumpblocks_voxel::world_grid::WorldGrid;

use crate::layers::GameLayer;

/// World-space origin of the main region. The archipelago is authored so
/// that Haven Isle's center (chunk 128,128) sits at the world origin.
pub const REGION_ORIGIN: Vec3 = Vec3::new(-2048.0, 0.0, -2048.0);

/// Bump this whenever the terrain generator's output changes shape (new
/// block shapes, different heightmaps, ...).  Saved worlds stamped with an
/// older version — or with no region.meta at all — are regenerated from
/// scratch on load instead of serving stale terrain.
pub const WORLDGEN_VERSION: u32 = 2;

/// Resource communicating the spawn point to the player system.
#[derive(Resource)]
pub struct SpawnPoint(pub Vec3);

/// Spawn position derived from the world definition (with clearance).
fn def_spawn_point(def: &WorldDef) -> Vec3 {
    cell_base_world(REGION_ORIGIN, def.spawn_cell) + Vec3::Y * 2.0
}

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        // Insert default spawn point immediately so it's available even if
        // setup_world hasn't run yet. setup_world overwrites with the real value.
        app.insert_resource(SpawnPoint(def_spawn_point(&WorldDef::standard())));

        app.init_resource::<WorldGrid>()
            .add_systems(Startup, setup_world)
            .add_systems(Update, attach_chunk_collision_layers);
    }
}

/// Attaches CollisionLayers to newly spawned chunk entities that don't have them yet.
fn attach_chunk_collision_layers(
    mut commands: Commands,
    chunks: Query<Entity, (With<jumpblocks_voxel::chunk::Chunk>, Without<CollisionLayers>)>,
) {
    for entity in chunks.iter() {
        commands.entity(entity).insert(
            CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
        );
    }
}

pub fn setup_world(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    dither_materials: Option<ResMut<Assets<ChunkDitherMaterial>>>,
    mut world_grid: ResMut<WorldGrid>,
    save_path: Option<Res<WorldSavePath>>,
    debug_start: Option<Res<crate::DebugStart>>,
) {
    // "Sea" safety net: an infinite plane just below the walkable sea floor.
    let ground = commands.spawn((
        RigidBody::Static,
        Collider::half_space(Vec3::Y),
        CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
    ));
    let ground_entity = ground.id();

    if let (Some(mut meshes), Some(mut materials)) = (meshes, materials) {
        commands.entity(ground_entity).insert((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1500.0)))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.15, 0.35, 0.55),
                ..default()
            })),
        ));

        // Directional light — pointing straight down for clean platformer shadows
        commands.spawn((
            DirectionalLight {
                illuminance: 8_000.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        ));

        // Low ambient — IBL on the camera provides the real ambient coloring
        commands.insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.9, 0.9, 1.0),
            brightness: 50.0,
            ..default()
        });

    }

    // Default dither material for streamed chunks (must be outside the
    // rendering block — streaming needs this even before meshes are ready)
    if let Some(mut dither_mats) = dither_materials {
        let chunk_mat = dither_mats.add(ChunkDitherMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.6, 0.5, 0.4),
                ..default()
            },
            extension: DitherFadeExtension { fade: 0.0, invert: false, chamfer_amount: 1.0 },
        });
        commands.insert_resource(ChunkMaterial(chunk_mat));
    }

    // --- Create a region and load chunk data ---
    let region_id = world_grid.create_region(REGION_ORIGIN);
    world_grid.active_region = region_id;

    let loaded_from_disk = if let Some(ref save) = save_path {
        // A save is only usable if it was produced by the current terrain
        // generator; otherwise it would serve stale terrain (e.g. worlds
        // saved before slope blocks existed have none). Regenerate stale
        // saves from scratch — this discards the old chunks.
        let meta = persistence::load_region_meta_from_disk(&save.0, region_id)
            .ok()
            .flatten();
        let fresh = meta.map(|m| m.worldgen_version == WORLDGEN_VERSION).unwrap_or(false);
        if fresh {
            load_region_from_disk(&save.0, region_id, &mut world_grid)
        } else {
            let regions_dir = save.0.join("regions");
            if regions_dir.exists() {
                warn!(
                    "[world] Save at {} was generated by an older terrain \
                     generator — regenerating (old chunks discarded)",
                    save.0.display()
                );
                if let Err(e) = std::fs::remove_dir_all(&regions_dir) {
                    error!("[world] Failed to clear stale region data: {}", e);
                }
            }
            false
        }
    } else {
        false
    };

    if !loaded_from_disk {
        // No saved world — populate with demo data in memory
        populate_region(region_id, &mut world_grid);

        // Save the demo chunks to disk if we have a save path
        if let Some(ref save) = save_path {
            save_region_to_disk(&save.0, region_id, &mut world_grid);
        }
    }

    let chunk_count = world_grid
        .get_region(region_id)
        .map(|r| r.chunk_count())
        .unwrap_or(0);
    info!(
        "[world] Region {} with {} chunks ({})",
        region_id,
        chunk_count,
        if loaded_from_disk { "loaded from disk" } else { "generated" }
    );

    // Spawn point from the authored world definition (--warp overrides,
    // so debug warps survive the load-time spawn hold).
    let spawn = debug_start
        .as_ref()
        .and_then(|d| d.warp)
        .unwrap_or_else(|| def_spawn_point(&WorldDef::standard()));
    commands.insert_resource(SpawnPoint(spawn));
}

// ---------------------------------------------------------------------------
// Disk I/O
// ---------------------------------------------------------------------------

/// Load all chunks for a region from disk into the WorldGrid.
fn load_region_from_disk(
    world_dir: &Path,
    region_id: RegionId,
    world_grid: &mut WorldGrid,
) -> bool {
    let chunks_dir = persistence::region_chunks_dir(world_dir, region_id);
    let Ok(entries) = std::fs::read_dir(&chunks_dir) else {
        return false;
    };

    let mut count = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if path.extension().and_then(|e| e.to_str()) != Some("chunk") {
            continue;
        }

        // Parse "x_y_z" from filename
        let parts: Vec<&str> = stem.split('_').collect();
        if parts.len() != 3 {
            continue;
        }
        let Ok(x) = parts[0].parse::<i32>() else { continue };
        let Ok(y) = parts[1].parse::<i32>() else { continue };
        let Ok(z) = parts[2].parse::<i32>() else { continue };
        let pos = ChunkPos::new(x, y, z);

        match persistence::load_chunk_from_disk(world_dir, region_id, pos) {
            Ok(Some(data)) => {
                if let Some(region) = world_grid.get_region_mut(region_id) {
                    region.set_chunk(pos, data);
                    count += 1;
                }
            }
            Ok(None) => {}
            Err(e) => {
                error!("[world] Failed to load chunk {}: {}", pos, e);
            }
        }
    }

    count > 0
}

/// Save all chunks in a region to disk.
fn save_region_to_disk(
    world_dir: &Path,
    region_id: RegionId,
    world_grid: &mut WorldGrid,
) {
    let Some(region) = world_grid.get_region_mut(region_id) else {
        return;
    };
    match persistence::save_dirty_chunks(world_dir, region) {
        Ok(n) => info!("[world] Saved {} chunks to disk", n),
        Err(e) => error!("[world] Failed to save chunks: {}", e),
    }
    // Stamp the save with the terrain generator version so future loads can
    // tell whether the chunks are current.
    let meta = persistence::RegionMeta {
        region_id: region_id.0,
        world_origin: [REGION_ORIGIN.x, REGION_ORIGIN.y, REGION_ORIGIN.z],
        chunk_count: region.chunk_count() as u32,
        data_generation: region.dirty.data_gen.0,
        worldgen_version: WORLDGEN_VERSION,
    };
    if let Err(e) = persistence::save_region_meta_to_disk(world_dir, &meta) {
        error!("[world] Failed to save region metadata: {}", e);
    }
}

/// Generate the authored archipelago (terrain + challenge courses) for a region.
fn populate_region(region_id: RegionId, world_grid: &mut WorldGrid) {
    let region = world_grid.get_region_mut(region_id).unwrap();
    let def = WorldDef::standard();
    let count = def.build_into_region(region);
    info!(
        "[world] Generated archipelago: {} islands, {} challenges, {} chunks",
        def.terrain.islands.len(),
        def.challenges.len(),
        count
    );
}

/// Build a world to disk (CLI --new-world command).
/// Creates the demo chunks and saves them without starting the game.
pub fn build_world_to_disk(world_dir: &Path) {
    let mut world_grid = WorldGrid::new();
    let region_id = world_grid.create_region(REGION_ORIGIN);
    populate_region(region_id, &mut world_grid);

    let region = world_grid.get_region_mut(region_id).unwrap();
    match persistence::save_dirty_chunks(world_dir, region) {
        Ok(n) => println!("Saved {} chunks", n),
        Err(e) => eprintln!("Failed to save: {}", e),
    }

    // Save region metadata
    let meta = persistence::RegionMeta {
        region_id: region_id.0,
        world_origin: [REGION_ORIGIN.x, REGION_ORIGIN.y, REGION_ORIGIN.z],
        chunk_count: region.chunk_count() as u32,
        data_generation: region.dirty.data_gen.0,
        worldgen_version: WORLDGEN_VERSION,
    };
    if let Err(e) = persistence::save_region_meta_to_disk(world_dir, &meta) {
        eprintln!("Failed to save region metadata: {}", e);
    }
}
