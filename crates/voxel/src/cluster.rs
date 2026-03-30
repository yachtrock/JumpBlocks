//! Chunk cluster system: merges distant chunks into 4×4 XZ groups
//! for reduced draw calls at distance.
//!
//! Rules:
//! - If all chunks in a 4×4 XZ cluster are beyond `cluster_radius` and
//!   have LOD meshes, merge them into a single cluster entity
//! - Clustered chunks are marked with `Clustered` and set Hidden
//! - LOD1 only renders for chunks NOT in an active cluster
//! - When approaching, cluster is destroyed and members are un-clustered

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};

use crate::chunk_lod::{ChunkDitherMaterial, ChunkLodMaterials, ChunkLodMesh, DitherFadeExtension, LodDebugMode, LodTier, LodConfig};
use crate::coords::{ChunkCoord, ChunkPos, CHUNK_WORLD_SIZE};
use crate::streaming::StreamingAnchor;

/// Size of a cluster in chunks along each XZ axis.
pub const CLUSTER_SIZE: i32 = 4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// XZ cluster key — identifies a 4×4 group of chunk columns.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ClusterKey {
    pub cx: i32,
    pub cz: i32,
}

impl ClusterKey {
    pub fn from_chunk(pos: ChunkPos) -> Self {
        Self {
            cx: pos.x.div_euclid(CLUSTER_SIZE),
            cz: pos.z.div_euclid(CLUSTER_SIZE),
        }
    }
}

/// Marker on chunks that are part of an active cluster.
/// When present, the LOD system should NOT render this chunk.
#[derive(Component)]
pub struct Clustered;

/// Component on a cluster entity.
#[derive(Component)]
pub struct ChunkCluster {
    pub key: ClusterKey,
    pub members: Vec<Entity>,
}

/// Marker for cluster entities.
#[derive(Component)]
pub struct ClusterMarker;

/// Resource tracking render stats.
#[derive(Resource, Default)]
pub struct RenderStats {
    pub lod0_count: usize,
    pub lod1_count: usize,
    pub cluster_count: usize,
    pub impostor_count: usize,
}

#[derive(Resource, Debug, Clone)]
pub struct ClusterConfig {
    pub cluster_radius: i32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self { cluster_radius: 5 }
    }
}

struct ChunkInfo {
    entity: Entity,
    #[allow(dead_code)]
    pos: ChunkPos,
    world_pos: Vec3,
    lod_mesh: Option<Handle<Mesh>>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Collects render stats every frame.
pub fn render_stats_system(
    chunks: Query<(&LodTier, &ChunkLodMaterials, Option<&Clustered>), With<ChunkCoord>>,
    clusters: Query<&ChunkCluster>,
    mut stats: ResMut<RenderStats>,
    dither_materials: Res<Assets<ChunkDitherMaterial>>,
) {
    let mut lod0 = 0usize;
    let mut lod1 = 0usize;

    for (tier, mats, clustered) in chunks.iter() {
        // Clustered chunks don't render individually
        if clustered.is_some() {
            continue;
        }

        let main_visible = dither_materials
            .get(&mats.main_handle)
            .map(|m| m.extension.fade < 0.99)
            .unwrap_or(false);
        let child_visible = dither_materials
            .get(&mats.child_handle)
            .map(|m| m.extension.fade < 0.99)
            .unwrap_or(false);

        if main_visible { lod0 += 1; }
        if child_visible { lod1 += 1; }
    }

    stats.lod0_count = lod0;
    stats.lod1_count = lod1;
    stats.cluster_count = clusters.iter().count();
    stats.impostor_count = 0;
}

/// Main cluster management system.
pub fn cluster_management_system(
    mut commands: Commands,
    config: Res<ClusterConfig>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
    chunks: Query<(Entity, &ChunkCoord, &GlobalTransform, &ChunkLodMesh, &LodTier, Option<&Clustered>)>,
    existing_clusters: Query<(Entity, &ChunkCluster)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
    debug_mode: Res<LodDebugMode>,
) {
    let Ok(anchor_tf) = anchor_query.single() else { return };
    let anchor_pos = anchor_tf.translation();

    // Group chunks by cluster key

    let mut cluster_map: HashMap<ClusterKey, Vec<ChunkInfo>> = HashMap::new();

    for (entity, coord, tf, lod_mesh, _tier, _clustered) in chunks.iter() {
        let key = ClusterKey::from_chunk(coord.pos);
        cluster_map.entry(key).or_default().push(ChunkInfo {
            entity,
            pos: coord.pos,
            world_pos: tf.translation(),
            lod_mesh: lod_mesh.lod.clone(),
        });
    }

    // Determine which clusters should be active
    let mut desired_clusters: HashMap<ClusterKey, Vec<Entity>> = HashMap::new();

    for (key, members) in &cluster_map {
        // All members need LOD meshes
        if !members.iter().all(|m| m.lod_mesh.is_some()) {
            continue;
        }

        // Check distance — use average position
        let avg_pos = members.iter().map(|m| m.world_pos).sum::<Vec3>() / members.len() as f32;
        let dist = ((anchor_pos - avg_pos) / CHUNK_WORLD_SIZE).abs();
        let max_dist = dist.x.max(dist.y).max(dist.z) as i32;

        if max_dist < config.cluster_radius {
            continue;
        }

        desired_clusters.insert(*key, members.iter().map(|m| m.entity).collect());
    }

    // Build set of currently active cluster keys
    let current_clusters: HashMap<ClusterKey, Entity> = existing_clusters
        .iter()
        .map(|(e, c)| (c.key, e))
        .collect();

    // Remove clusters that should no longer exist
    for (key, cluster_entity) in &current_clusters {
        if !desired_clusters.contains_key(key) {
            // Un-cluster members
            if let Ok((_, cluster)) = existing_clusters.get(*cluster_entity) {
                for &member in &cluster.members {
                    if let Ok(mut ec) = commands.get_entity(member) {
                        ec.remove::<Clustered>();
                        ec.insert(Visibility::Inherited);
                    }
                }
            }
            commands.entity(*cluster_entity).despawn();
        }
    }

    // Create clusters that don't exist yet
    for (key, member_entities) in &desired_clusters {
        if current_clusters.contains_key(key) {
            continue;
        }

        let members = &cluster_map[key];

        // Merge meshes
        let origin = members.first().map(|m| m.world_pos).unwrap_or(Vec3::ZERO);
        let merged = merge_lod_meshes(members, origin, &meshes);

        if merged.positions.is_empty() {
            continue;
        }

        let merged_handle = meshes.add(build_cluster_mesh(&merged));

        let cluster_color = match *debug_mode {
            LodDebugMode::Normal => Color::srgb(0.6, 0.5, 0.4),
            LodDebugMode::Tinted => Color::srgb(0.9, 0.3, 0.3),
        };
        let mat = dither_materials.add(ChunkDitherMaterial {
            base: StandardMaterial {
                base_color: cluster_color,
                ..default()
            },
            extension: DitherFadeExtension { fade: 0.0, invert: false, chamfer_amount: 0.0 },
        });

        commands.spawn((
            ClusterMarker,
            ChunkCluster {
                key: *key,
                members: member_entities.clone(),
            },
            Mesh3d(merged_handle),
            MeshMaterial3d(mat),
            Transform::from_translation(origin),
            Visibility::default(),
        ));

        // Mark members as clustered and hide them
        for &entity in member_entities {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.insert((Clustered, Visibility::Hidden));
            }
        }
    }
}

/// Updates cluster material colors when debug mode changes.
pub fn cluster_debug_color_system(
    debug_mode: Res<LodDebugMode>,
    clusters: Query<&MeshMaterial3d<ChunkDitherMaterial>, With<ClusterMarker>>,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
) {
    if !debug_mode.is_changed() {
        return;
    }
    let color = match *debug_mode {
        LodDebugMode::Normal => Color::srgb(0.6, 0.5, 0.4),
        LodDebugMode::Tinted => Color::srgb(0.9, 0.3, 0.3),
    };
    for mat_handle in clusters.iter() {
        if let Some(mat) = dither_materials.get_mut(&mat_handle.0) {
            mat.base.base_color = color;
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh merging
// ---------------------------------------------------------------------------

struct MergedMeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

fn merge_lod_meshes(
    members: &[ChunkInfo],
    origin: Vec3,
    mesh_assets: &Assets<Mesh>,
) -> MergedMeshData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for member in members {
        let Some(ref handle) = member.lod_mesh else { continue };
        let Some(mesh) = mesh_assets.get(handle) else { continue };

        let Some(pos_attr) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { continue };
        let Some(norm_attr) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL) else { continue };
        let Some(uv_attr) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else { continue };
        let Some(mesh_indices) = mesh.indices() else { continue };

        let bevy::mesh::VertexAttributeValues::Float32x3(pos_data) = pos_attr else { continue };
        let bevy::mesh::VertexAttributeValues::Float32x3(norm_data) = norm_attr else { continue };
        let bevy::mesh::VertexAttributeValues::Float32x2(uv_data) = uv_attr else { continue };

        let base_idx = positions.len() as u32;
        let offset = member.world_pos - origin;

        for p in pos_data {
            positions.push([p[0] + offset.x, p[1] + offset.y, p[2] + offset.z]);
        }
        normals.extend_from_slice(norm_data);
        uvs.extend_from_slice(uv_data);

        match mesh_indices {
            Indices::U32(idx) => {
                for &i in idx { indices.push(base_idx + i); }
            }
            Indices::U16(idx) => {
                for &i in idx { indices.push(base_idx + i as u32); }
            }
        }
    }

    MergedMeshData { positions, normals, uvs, indices }
}

fn build_cluster_mesh(data: &MergedMeshData) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    let n = data.positions.len();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs.clone());
    mesh.insert_attribute(crate::meshing::ATTRIBUTE_CHAMFER_OFFSET, vec![[0.0f32; 3]; n]);
    mesh.insert_attribute(crate::meshing::ATTRIBUTE_SHARP_NORMAL, data.normals.clone());
    mesh.insert_indices(Indices::U32(data.indices.clone()));
    mesh
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct ClusterPlugin;

impl Plugin for ClusterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClusterConfig>()
            .init_resource::<RenderStats>()
            .add_systems(Update, (
                render_stats_system,
                cluster_management_system,
                cluster_debug_color_system,
            ));
    }
}
