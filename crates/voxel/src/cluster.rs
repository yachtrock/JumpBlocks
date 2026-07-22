//! Chunk cluster system: merges distant chunk columns into 4×4 XZ groups.
//!
//! Top-down rendering decision:
//! 1. Group all chunks by 4×4 XZ cluster key
//! 2. For each cluster: is every chunk in the group at Reduced tier?
//!    - Yes → render the merged cluster mesh, hide all individual members
//!    - No → render individual chunks (LOD0/LOD1 per the LOD system)

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};

use crate::chunk_lod::{ChunkDitherMaterial, ChunkLodMaterials, ChunkLodMesh, DitherFadeExtension, LodDebugMode, LodTier, LodTransition};
use crate::coords::{ChunkCoord, ChunkPos, CHUNK_WORLD_SIZE};
use crate::streaming::StreamingAnchor;

/// Size of a cluster in chunks along each XZ axis.
pub const CLUSTER_SIZE: i32 = 4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// XZ cluster key — identifies a 4×4 group of chunk columns.
/// Always aligned: chunk x=0..3 → cluster 0, x=4..7 → cluster 1, etc.
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
#[derive(Component)]
pub struct Clustered;

/// Component on a cluster entity.
#[derive(Component)]
pub struct ChunkCluster {
    pub key: ClusterKey,
    pub members: Vec<Entity>,
}

#[derive(Component)]
pub struct ClusterMarker;

/// Resource tracking render stats.
#[derive(Resource, Default)]
pub struct RenderStats {
    pub lod0_count: usize,
    pub lod1_count: usize,
    pub cluster_count: usize,
    pub impostor_count: usize,
    pub mesh_full_count: usize,
    pub mesh_lod_only_count: usize,
    pub mesh_none_count: usize,
    pub meshing_count: usize,
    pub total_chunks: usize,
    pub clustered_count: usize,
}

#[derive(Resource, Debug, Clone)]
pub struct ClusterConfig {
    /// XZ distance (in chunk units) beyond which clusters activate.
    pub cluster_radius: i32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self { cluster_radius: 3 }
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

struct Member {
    entity: Entity,
    world_pos: Vec3,
    has_lod_mesh: bool,
    lod_mesh_handle: Option<Handle<Mesh>>,
    tier: LodTier,
    transitioning: bool,
    has_blocks: bool,
}

/// Collects render stats every frame.
pub fn render_stats_system(
    chunks: Query<(&LodTier, Option<&ChunkLodMaterials>, Option<&Clustered>, Option<&crate::ChunkMeshLevel>), With<ChunkCoord>>,
    all_chunks: Query<&crate::chunk::Chunk, With<ChunkCoord>>,
    clusters: Query<&ChunkCluster>,
    mut stats: ResMut<RenderStats>,
    dither_materials: Res<Assets<ChunkDitherMaterial>>,
) {
    let mut lod0 = 0usize;
    let mut lod1 = 0usize;
    let mut mesh_full = 0usize;
    let mut mesh_lod_only = 0usize;
    let mut mesh_none = 0usize;
    let mut clustered = 0usize;
    let mut meshing = 0usize;
    let mut total = 0usize;

    for (_tier, mats, is_clustered, mesh_level) in chunks.iter() {
        total += 1;
        match mesh_level.copied().unwrap_or(crate::ChunkMeshLevel::None) {
            crate::ChunkMeshLevel::Full => mesh_full += 1,
            crate::ChunkMeshLevel::LodOnly => mesh_lod_only += 1,
            crate::ChunkMeshLevel::None => mesh_none += 1,
        }

        if is_clustered.is_some() {
            clustered += 1;
            continue;
        }

        if let Some(mats) = mats {
            let main_vis = dither_materials.get(&mats.main_handle).map(|m| m.extension.fade < 0.99).unwrap_or(false);
            let child_vis = dither_materials.get(&mats.child_handle).map(|m| m.extension.fade < 0.99).unwrap_or(false);
            if main_vis { lod0 += 1; }
            if child_vis { lod1 += 1; }
        }
    }

    for chunk in all_chunks.iter() {
        if chunk.state == crate::chunk::ChunkState::Meshing { meshing += 1; }
    }

    stats.lod0_count = lod0;
    stats.lod1_count = lod1;
    stats.cluster_count = clusters.iter().count();
    stats.impostor_count = 0;
    stats.mesh_full_count = mesh_full;
    stats.mesh_lod_only_count = mesh_lod_only;
    stats.mesh_none_count = mesh_none;
    stats.meshing_count = meshing;
    stats.total_chunks = total;
    stats.clustered_count = clustered;
}

/// Unified cluster management. Runs every frame.
///
/// Decision tree per cluster group:
/// 1. Compute XZ distance from anchor to cluster center
/// 2. If close (< cluster_radius): dissolve cluster, show individual chunks
/// 3. If far (>= cluster_radius): check if all members have LOD meshes and
///    are at Reduced tier. If yes → activate cluster, hide members.
pub fn cluster_management_system(
    mut commands: Commands,
    config: Res<ClusterConfig>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
    chunks: Query<(Entity, &ChunkCoord, &GlobalTransform, Option<&ChunkLodMesh>, &LodTier, Option<&Clustered>, &crate::chunk::Chunk, Option<&LodTransition>)>,
    existing_clusters: Query<(Entity, &ChunkCluster)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
    debug_mode: Res<LodDebugMode>,
) {
    let Ok(anchor_tf) = anchor_query.single() else { return };
    let anchor_pos = anchor_tf.translation();

    // --- Step 1: Group all chunks by cluster key ---

    let mut groups: HashMap<ClusterKey, Vec<Member>> = HashMap::new();
    // We need to compute the region offset from the first chunk we see
    let mut region_offset: Option<Vec3> = None;

    for (entity, coord, tf, lod_mesh, tier, _clustered, chunk, transition) in chunks.iter() {
        if region_offset.is_none() {
            region_offset = Some(tf.translation() - coord.pos.to_world_offset());
        }

        let key = ClusterKey::from_chunk(coord.pos);
        groups.entry(key).or_default().push(Member {
            entity,
            world_pos: tf.translation(),
            has_lod_mesh: lod_mesh.map(|m| m.lod.is_some()).unwrap_or(false),
            lod_mesh_handle: lod_mesh.and_then(|m| m.lod.clone()),
            tier: *tier,
            transitioning: transition.is_some(),
            has_blocks: !chunk.data.blocks.is_empty(),
        });
    }

    let region_off = region_offset.unwrap_or(Vec3::ZERO);

    // --- Step 2: Decide per cluster group ---
    let current_clusters: HashMap<ClusterKey, Entity> = existing_clusters
        .iter()
        .map(|(e, c)| (c.key, e))
        .collect();

    let mut should_be_active: HashMap<ClusterKey, Vec<Entity>> = HashMap::new();

    for (key, members) in &groups {
        // Compute cluster center in world space (XZ center of the 4×4 group)
        let cluster_center = region_off + Vec3::new(
            (key.cx * CLUSTER_SIZE) as f32 * CHUNK_WORLD_SIZE + CLUSTER_SIZE as f32 * CHUNK_WORLD_SIZE * 0.5,
            0.0,
            (key.cz * CLUSTER_SIZE) as f32 * CHUNK_WORLD_SIZE + CLUSTER_SIZE as f32 * CHUNK_WORLD_SIZE * 0.5,
        );

        let dx = ((anchor_pos.x - cluster_center.x) / CHUNK_WORLD_SIZE).abs();
        let dz = ((anchor_pos.z - cluster_center.z) / CHUNK_WORLD_SIZE).abs();
        let xz_dist = dx.max(dz) as i32;

        if xz_dist < config.cluster_radius {
            continue; // Too close — render individually
        }

        // Cluster eligibility:
        // - Every chunk with blocks must have an LOD mesh, be at Reduced tier,
        //   and NOT be in an active LOD transition (to or from Full)
        // - If ANY chunk is Full or transitioning, the whole cluster breaks apart
        // - Empty chunks (no blocks) are always fine
        let all_ready = members.iter().all(|m| {
            if m.has_blocks {
                m.has_lod_mesh && m.tier == LodTier::Reduced && !m.transitioning
            } else {
                true
            }
        });

        let has_geometry = members.iter().any(|m| m.has_lod_mesh);

        if all_ready && has_geometry {
            should_be_active.insert(*key, members.iter().map(|m| m.entity).collect());
        } else if has_geometry && !all_ready {
            // Debug: log why this group can't cluster
            let not_ready: Vec<_> = members.iter()
                .filter(|m| m.has_blocks && !m.has_lod_mesh)
                .collect();
            if !not_ready.is_empty() {
                trace!(
                    "Cluster ({},{}) blocked: {} chunks with blocks but no LOD mesh",
                    key.cx, key.cz, not_ready.len()
                );
            }
        }
    }

    // --- Step 3: Dissolve clusters that shouldn't exist or have stale membership ---
    let mut dissolved_keys: std::collections::HashSet<ClusterKey> = std::collections::HashSet::new();
    for (key, cluster_entity) in &current_clusters {
        let should_dissolve = if let Some(desired_members) = should_be_active.get(key) {
            // Check if membership changed (new chunks loaded into this group)
            if let Ok((_, cluster)) = existing_clusters.get(*cluster_entity) {
                let mut current_set: Vec<Entity> = cluster.members.clone();
                let mut desired_set: Vec<Entity> = desired_members.clone();
                current_set.sort();
                desired_set.sort();
                current_set != desired_set
            } else {
                true
            }
        } else {
            true // Not in desired set at all
        };

        if should_dissolve {
            if let Ok((_, cluster)) = existing_clusters.get(*cluster_entity) {
                for &member in &cluster.members {
                    if let Ok(mut ec) = commands.get_entity(member) {
                        ec.remove::<Clustered>();
                        ec.insert(Visibility::Inherited);
                    }
                }
            }
            commands.entity(*cluster_entity).despawn();
            dissolved_keys.insert(*key);
        }
    }

    // --- Step 4: Create clusters that should exist and don't yet ---
    // (despawns above are deferred, so consult dissolved_keys rather than
    // querying entity liveness — a stale cluster dissolved this frame must
    // be rebuilt this frame to avoid a one-frame gap)
    for (key, member_entities) in &should_be_active {
        if current_clusters.contains_key(key) && !dissolved_keys.contains(key) {
            continue; // Valid cluster already exists
        }

        let members = &groups[key];

        // Compute origin for the merged mesh (min corner of the cluster in world space)
        let origin = region_off + Vec3::new(
            (key.cx * CLUSTER_SIZE) as f32 * CHUNK_WORLD_SIZE,
            0.0,
            (key.cz * CLUSTER_SIZE) as f32 * CHUNK_WORLD_SIZE,
        );

        // Find the actual min Y of member world positions for proper Y offset
        let min_y = members.iter()
            .filter(|m| m.has_lod_mesh)
            .map(|m| m.world_pos.y)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let origin = Vec3::new(origin.x, min_y, origin.z);

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
            base: StandardMaterial { base_color: cluster_color, ..default() },
            extension: DitherFadeExtension { fade: 0.0, invert: false, chamfer_amount: 0.0 },
        });

        commands.spawn((
            ClusterMarker,
            ChunkCluster { key: *key, members: member_entities.clone() },
            Mesh3d(merged_handle),
            MeshMaterial3d(mat),
            Transform::from_translation(origin),
            Visibility::default(),
        ));

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
    if !debug_mode.is_changed() { return; }
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
    members: &[Member],
    origin: Vec3,
    mesh_assets: &Assets<Mesh>,
) -> MergedMeshData {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for member in members {
        let Some(ref handle) = member.lod_mesh_handle else { continue };
        let Some(mesh) = mesh_assets.get(handle) else { continue };

        let Some(bevy::mesh::VertexAttributeValues::Float32x3(pos_data)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { continue };
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(norm_data)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL) else { continue };
        let Some(bevy::mesh::VertexAttributeValues::Float32x2(uv_data)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else { continue };
        let Some(mesh_indices) = mesh.indices() else { continue };

        let base_idx = positions.len() as u32;
        let offset = member.world_pos - origin;

        for p in pos_data {
            positions.push([p[0] + offset.x, p[1] + offset.y, p[2] + offset.z]);
        }
        normals.extend_from_slice(norm_data);
        uvs.extend_from_slice(uv_data);

        match mesh_indices {
            Indices::U32(idx) => { for &i in idx { indices.push(base_idx + i); } }
            Indices::U16(idx) => { for &i in idx { indices.push(base_idx + i as u32); } }
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
