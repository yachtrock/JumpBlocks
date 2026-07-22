use std::path::Path;

use avian3d::prelude::*;
use bevy::prelude::*;
use jumpblocks_voxel::chunk_lod::{ChunkDitherMaterial, DitherFadeExtension};
use jumpblocks_voxel::coords::{ChunkPos, RegionId};
use jumpblocks_voxel::persistence;
use jumpblocks_voxel::streaming::{ChunkMaterial, WorldSavePath};
use jumpblocks_voxel::world_grid::WorldGrid;
use jumpblocks_voxel::worldgen::{self, IslandGenConfig};

use crate::layers::GameLayer;

/// Resource communicating the spawn point to the player system.
#[derive(Resource)]
pub struct SpawnPoint(pub Vec3);

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        // Insert default spawn point immediately so it's available even if
        // setup_world hasn't run yet. setup_world overwrites with the real value.
        let gen_config = IslandGenConfig::default();
        let spawn_y = worldgen::island_spawn_height(&gen_config) + 3.0;
        app.insert_resource(SpawnPoint(Vec3::new(0.0, spawn_y, 0.0)));

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

fn setup_world(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    dither_materials: Option<ResMut<Assets<ChunkDitherMaterial>>>,
    mut world_grid: ResMut<WorldGrid>,
    save_path: Option<Res<WorldSavePath>>,
) {
    let has_rendering = meshes.is_some() && materials.is_some();

    // Ground plane
    let ground = commands.spawn((
        RigidBody::Static,
        Collider::half_space(Vec3::Y),
        CollisionLayers::new([GameLayer::Default, GameLayer::CameraBlocking], LayerMask::ALL),
    ));
    let ground_entity = ground.id();

    let platforms = [
        Vec3::new(5.0, 1.0, 0.0),
        Vec3::new(10.0, 2.5, 3.0),
        Vec3::new(15.0, 4.0, -1.0),
        Vec3::new(12.0, 5.5, -6.0),
        Vec3::new(7.0, 7.0, -8.0),
    ];

    let platform_entities: Vec<(Entity, Vec3)> = platforms
        .iter()
        .map(|&pos| {
            let e = commands
                .spawn((
                    Transform::from_translation(pos),
                    RigidBody::Static,
                    Collider::cuboid(4.0, 0.5, 4.0),
                    CollisionLayers::new(
                        [GameLayer::Default, GameLayer::CameraBlocking],
                        LayerMask::ALL,
                    ),
                ))
                .id();
            (e, pos)
        })
        .collect();

    if let (Some(mut meshes), Some(mut materials)) = (meshes, materials) {
        commands.entity(ground_entity).insert((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(50.0)))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.6, 0.3),
                ..default()
            })),
        ));

        let platform_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.6),
            ..default()
        });
        let platform_mesh = meshes.add(Cuboid::new(4.0, 0.5, 4.0));

        for (entity, _pos) in &platform_entities {
            commands.entity(*entity).insert((
                Mesh3d(platform_mesh.clone()),
                MeshMaterial3d(platform_material.clone()),
            ));
        }

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
    let region_id = world_grid.create_region(Vec3::new(-2048.0, 0.0, -2048.0));
    world_grid.active_region = region_id;

    let loaded_from_disk = if let Some(ref save) = save_path {
        load_region_from_disk(&save.0, region_id, &mut world_grid)
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

    // Compute spawn point above island surface
    let gen_config = IslandGenConfig::default();
    let spawn_y = worldgen::island_spawn_height(&gen_config) + 3.0; // +3 for clearance
    commands.insert_resource(SpawnPoint(Vec3::new(0.0, spawn_y, 0.0)));
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
}

/// Generate island terrain for a region.
fn populate_region(region_id: RegionId, world_grid: &mut WorldGrid) {
    let region = world_grid.get_region_mut(region_id).unwrap();
    let config = IslandGenConfig::default();
    let count = worldgen::generate_island(region, &config);
    info!("[world] Generated island with {} chunks", count);
}

/// Build a world to disk (CLI --new-world command).
/// Creates the demo chunks and saves them without starting the game.
pub fn build_world_to_disk(world_dir: &Path) {
    let mut world_grid = WorldGrid::new();
    let region_id = world_grid.create_region(Vec3::new(-2048.0, 0.0, -2048.0));
    populate_region(region_id, &mut world_grid);

    let region = world_grid.get_region_mut(region_id).unwrap();
    match persistence::save_dirty_chunks(world_dir, region) {
        Ok(n) => println!("Saved {} chunks", n),
        Err(e) => eprintln!("Failed to save: {}", e),
    }

    // Save region metadata
    let meta = persistence::RegionMeta {
        region_id: region_id.0,
        world_origin: [-2048.0, 0.0, -2048.0],
        chunk_count: region.chunk_count() as u32,
        data_generation: region.dirty.data_gen.0,
    };
    if let Err(e) = persistence::save_region_meta_to_disk(world_dir, &meta) {
        eprintln!("Failed to save region metadata: {}", e);
    }
}
