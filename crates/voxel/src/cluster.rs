//! Chunk cluster system: merges distant chunks into 4×4 XZ groups
//! for reduced draw calls at distance.
//!
//! When all chunks in a 4×4 XZ cluster are beyond the `reduced_radius`,
//! their LOD meshes are merged into a single combined mesh entity.
//! Individual chunk meshes are hidden while the cluster is active.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};

use crate::chunk_lod::{ChunkDitherMaterial, ChunkLodMaterials, ChunkLodMesh, DitherFadeExtension, LodTier, LodConfig};
use crate::coords::{ChunkCoord, ChunkPos, CHUNK_WORLD_SIZE};
use crate::streaming::StreamingAnchor;

/// Size of a cluster in chunks along each XZ axis.
pub const CLUSTER_SIZE: i32 = 4;

/// World-space size of a cluster.
pub const CLUSTER_WORLD_SIZE: f32 = CLUSTER_SIZE as f32 * CHUNK_WORLD_SIZE;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// XZ cluster key — identifies a 4×4 group of chunk columns.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ClusterKey {
    /// Cluster X index (chunk_x / CLUSTER_SIZE).
    pub cx: i32,
    /// Cluster Z index (chunk_z / CLUSTER_SIZE).
    pub cz: i32,
}

impl ClusterKey {
    pub fn from_chunk(pos: ChunkPos) -> Self {
        Self {
            cx: pos.x.div_euclid(CLUSTER_SIZE),
            cz: pos.z.div_euclid(CLUSTER_SIZE),
        }
    }

    /// World-space origin of this cluster (min corner).
    pub fn world_origin(&self, region_origin: Vec3) -> Vec3 {
        region_origin + Vec3::new(
            self.cx as f32 * CLUSTER_WORLD_SIZE,
            0.0,
            self.cz as f32 * CLUSTER_WORLD_SIZE,
        )
    }
}

/// Component on a cluster entity.
#[derive(Component)]
pub struct ChunkCluster {
    pub key: ClusterKey,
    /// Chunk entities that are part of this cluster (and currently hidden).
    pub members: Vec<Entity>,
}

/// Marker so we can query cluster entities.
#[derive(Component)]
pub struct ClusterMarker;

/// Resource tracking render stats for the debug panel.
#[derive(Resource, Default)]
pub struct RenderStats {
    pub lod0_count: usize,
    pub lod1_count: usize,
    pub cluster_count: usize,
    pub impostor_count: usize,
}

/// Config for the cluster system.
#[derive(Resource, Debug, Clone)]
pub struct ClusterConfig {
    /// Minimum distance (chunk units) before clusters activate.
    /// Should be >= reduced_radius from LodConfig.
    pub cluster_radius: i32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            cluster_radius: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Collects render stats every frame.
pub fn render_stats_system(
    chunks: Query<(&LodTier, &ChunkLodMaterials), With<ChunkCoord>>,
    clusters: Query<&ChunkCluster>,
    mut stats: ResMut<RenderStats>,
    dither_materials: Res<Assets<ChunkDitherMaterial>>,
) {
    let mut lod0 = 0usize;
    let mut lod1 = 0usize;

    for (tier, mats) in chunks.iter() {
        // Check if the main mesh is actually visible (fade < 1)
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

/// Manages cluster lifecycle: creates, updates, and destroys cluster entities.
pub fn cluster_management_system(
    mut commands: Commands,
    config: Res<ClusterConfig>,
    lod_config: Res<LodConfig>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
    chunks: Query<(Entity, &ChunkCoord, &GlobalTransform, &ChunkLodMesh, &LodTier)>,
    mut existing_clusters: Query<(Entity, &mut ChunkCluster)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
    mesh_assets: Res<Assets<Mesh>>,
) {
    let Ok(anchor_tf) = anchor_query.single() else { return };
    let anchor_pos = anchor_tf.translation();

    // Group chunks by cluster key
    let mut cluster_map: HashMap<ClusterKey, Vec<(Entity, ChunkPos, Vec3, Option<Handle<Mesh>>, LodTier)>> = HashMap::new();

    for (entity, coord, tf, lod_mesh, tier) in chunks.iter() {
        let key = ClusterKey::from_chunk(coord.pos);
        cluster_map.entry(key).or_default().push((
            entity,
            coord.pos,
            tf.translation(),
            lod_mesh.lod.clone(),
            *tier,
        ));
    }

    // Track which cluster keys should exist
    let mut active_keys: HashMap<ClusterKey, Vec<Entity>> = HashMap::new();

    for (key, members) in &cluster_map {
        // Check if all members are far enough for clustering
        let cluster_center = Vec3::new(
            members.iter().map(|m| m.2.x).sum::<f32>() / members.len() as f32,
            members.iter().map(|m| m.2.y).sum::<f32>() / members.len() as f32,
            members.iter().map(|m| m.2.z).sum::<f32>() / members.len() as f32,
        );
        let dist = ((anchor_pos - cluster_center) / CHUNK_WORLD_SIZE).abs();
        let max_dist = dist.x.max(dist.y).max(dist.z) as i32;

        if max_dist < config.cluster_radius {
            continue; // Too close — don't cluster
        }

        // All members need LOD meshes to merge
        let all_have_lod = members.iter().all(|m| m.3.is_some());
        if !all_have_lod {
            continue;
        }

        let entities: Vec<Entity> = members.iter().map(|m| m.0).collect();
        active_keys.insert(*key, entities);
    }

    // Remove clusters that are no longer needed
    for (cluster_entity, cluster) in existing_clusters.iter() {
        if !active_keys.contains_key(&cluster.key) {
            // Un-hide member chunks
            for &member in &cluster.members {
                commands.entity(member).insert(Visibility::Inherited);
            }
            commands.entity(cluster_entity).despawn();
        }
    }

    // Create or update clusters
    let existing_keys: HashMap<ClusterKey, Entity> = existing_clusters
        .iter()
        .map(|(e, c)| (c.key, e))
        .collect();

    for (key, member_entities) in &active_keys {
        if existing_keys.contains_key(key) {
            continue; // Already exists
        }

        // Merge LOD meshes into a single mesh
        let members = &cluster_map[key];
        let merged = merge_lod_meshes(members, &mesh_assets);

        if merged.positions.is_empty() {
            continue;
        }

        let merged_handle = meshes.add(build_cluster_mesh(&merged));

        // Compute cluster center for transform
        let min_pos = members.iter().map(|m| m.2).reduce(|a, b| a.min(b)).unwrap();

        let mat = dither_materials.add(ChunkDitherMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.6, 0.5, 0.4),
                ..default()
            },
            extension: DitherFadeExtension { fade: 0.0, invert: false, chamfer_amount: 0.0 },
        });

        // Spawn cluster entity
        commands.spawn((
            ClusterMarker,
            ChunkCluster {
                key: *key,
                members: member_entities.clone(),
            },
            Mesh3d(merged_handle),
            MeshMaterial3d(mat),
            Transform::from_translation(min_pos),
            Visibility::default(),
        ));

        // Hide individual chunks
        for &entity in member_entities {
            commands.entity(entity).insert(Visibility::Hidden);
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
    members: &[(Entity, ChunkPos, Vec3, Option<Handle<Mesh>>, LodTier)],
    mesh_assets: &Assets<Mesh>,
) -> MergedMeshData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    // Use the first member's position as the origin offset
    let origin = members.first().map(|m| m.2).unwrap_or(Vec3::ZERO);

    for (_, _, world_pos, mesh_handle, _) in members {
        let Some(handle) = mesh_handle else { continue };
        let Some(mesh) = mesh_assets.get(handle) else { continue };

        let Some(pos_attr) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { continue };
        let Some(norm_attr) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL) else { continue };
        let Some(uv_attr) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else { continue };
        let Some(mesh_indices) = mesh.indices() else { continue };

        let bevy::mesh::VertexAttributeValues::Float32x3(pos_data) = pos_attr else { continue };
        let bevy::mesh::VertexAttributeValues::Float32x3(norm_data) = norm_attr else { continue };
        let bevy::mesh::VertexAttributeValues::Float32x2(uv_data) = uv_attr else { continue };

        let base_idx = positions.len() as u32;
        let offset = *world_pos - origin;

        for p in pos_data {
            positions.push([p[0] + offset.x, p[1] + offset.y, p[2] + offset.z]);
        }
        normals.extend_from_slice(norm_data);
        uvs.extend_from_slice(uv_data);

        match mesh_indices {
            Indices::U32(idx) => {
                for &i in idx {
                    indices.push(base_idx + i);
                }
            }
            Indices::U16(idx) => {
                for &i in idx {
                    indices.push(base_idx + i as u32);
                }
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
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs.clone());
    // Add zero-filled custom attributes to match chunk vertex layout
    let n = data.positions.len();
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
            ));
    }
}
