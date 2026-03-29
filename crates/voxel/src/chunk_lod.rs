//! LOD tier management with dither-fade transitions.
//!
//! Each chunk entity always has a permanent child holding the alternate LOD mesh.
//! During transitions, both meshes are visible with complementary dither patterns.
//! No entities are spawned or despawned during transitions.
//!
//! Structure:
//!   Chunk entity:  Mesh3d(full_res) + dither material (fade_a)
//!     └─ LodChild: Mesh3d(lod)      + dither material (fade_b)
//!
//! At rest: one has fade=0 (visible), the other fade=1 (invisible).
//! During crossfade: they animate toward each other over `transition_duration`.

use bevy::prelude::*;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::coords::{ChunkCoord, CHUNK_WORLD_SIZE};
use crate::streaming::StreamingAnchor;

// ---------------------------------------------------------------------------
// Dither fade material
// ---------------------------------------------------------------------------

/// Type alias for the extended material used by all chunk meshes.
pub type ChunkDitherMaterial = ExtendedMaterial<StandardMaterial, DitherFadeExtension>;

/// Material extension that adds screen-space dither fading to StandardMaterial.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
#[uniform(200, DitherFadeUniform)]
pub struct DitherFadeExtension {
    /// 0.0 = fully visible, 1.0 = fully invisible.
    pub fade: f32,
    /// If true, uses the inverted dither pattern (for the incoming mesh in a crossfade).
    pub invert: bool,
}

#[derive(Clone, Default, ShaderType)]
pub struct DitherFadeUniform {
    pub fade: f32,
    pub invert: f32,
}

impl From<&DitherFadeExtension> for DitherFadeUniform {
    fn from(ext: &DitherFadeExtension) -> Self {
        Self {
            fade: ext.fade,
            invert: if ext.invert { 1.0 } else { 0.0 },
        }
    }
}

impl MaterialExtension for DitherFadeExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/dither_fade.wgsl".into()
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The detail tier a chunk is currently rendered at.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LodTier {
    #[default]
    Full,
    Reduced,
    Hidden,
}

/// The tier the LOD system has *decided* this chunk should be at.
/// Separate from LodTier so we can animate the transition.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LodTarget {
    #[default]
    Full,
    Reduced,
    Hidden,
}

impl LodTarget {
    pub fn as_tier(self) -> LodTier {
        match self {
            LodTarget::Full => LodTier::Full,
            LodTarget::Reduced => LodTier::Reduced,
            LodTarget::Hidden => LodTier::Hidden,
        }
    }
}

/// LOD configuration resource.
#[derive(Resource, Debug, Clone)]
pub struct LodConfig {
    /// Chunks within this distance (in chunk units) get full detail.
    pub full_radius: i32,
    /// Chunks within this distance get reduced LOD. Beyond this → hidden.
    pub reduced_radius: i32,
    /// Duration of the dither crossfade in seconds.
    pub transition_duration: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            full_radius: 2,
            reduced_radius: 6,
            transition_duration: 0.4,
        }
    }
}

/// Holds the LOD mesh handles generated during meshing.
#[derive(Component)]
pub struct ChunkLodMesh {
    pub full_res: Option<Handle<Mesh>>,
    pub lod: Option<Handle<Mesh>>,
}

/// Permanent child entity that holds the alternate LOD mesh.
#[derive(Component)]
pub struct LodChild(pub Entity);

/// Marker on the child entity so we can query it.
#[derive(Component)]
pub struct LodChildMarker;

/// Active transition state on a chunk entity.
#[derive(Component)]
pub struct LodTransition {
    /// What we're transitioning to.
    pub to: LodTier,
    /// Progress 0.0 → 1.0.
    pub progress: f32,
    /// Duration in seconds.
    pub duration: f32,
    /// Material handle for the main mesh (parent).
    pub main_material: Handle<ChunkDitherMaterial>,
    /// Material handle for the child LOD mesh.
    pub child_material: Handle<ChunkDitherMaterial>,
}

/// Optional debug materials for visualizing LOD tiers.
#[derive(Resource)]
pub struct LodDebugMaterials {
    pub full: Handle<ChunkDitherMaterial>,
    pub reduced: Handle<ChunkDitherMaterial>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Assigns LodTarget based on distance to the streaming anchor.
pub fn lod_target_assignment_system(
    config: Res<LodConfig>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
    mut chunks: Query<(&ChunkCoord, &mut LodTarget)>,
) {
    let Ok(anchor_transform) = anchor_query.single() else {
        return;
    };
    let anchor_pos = anchor_transform.translation();

    for (coord, mut target) in chunks.iter_mut() {
        let chunk_center = coord.pos.to_world_offset()
            + Vec3::splat(CHUNK_WORLD_SIZE * 0.5);

        let dist_chunks = ((anchor_pos - chunk_center) / CHUNK_WORLD_SIZE).abs();
        let max_dist = dist_chunks.x.max(dist_chunks.y).max(dist_chunks.z) as i32;

        let new_target = if max_dist <= config.full_radius {
            LodTarget::Full
        } else if max_dist <= config.reduced_radius {
            LodTarget::Reduced
        } else {
            LodTarget::Hidden
        };

        if *target != new_target {
            *target = new_target;
        }
    }
}

/// Creates the permanent LodChild when a chunk first gets its meshes.
/// The child starts invisible (fade=1).
pub fn lod_child_setup_system(
    mut commands: Commands,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
    chunks: Query<(Entity, &ChunkLodMesh), Added<ChunkLodMesh>>,
    debug_mats: Option<Res<LodDebugMaterials>>,
) {
    for (entity, lod_mesh) in chunks.iter() {
        let Some(ref lod_handle) = lod_mesh.lod else {
            continue;
        };

        // Child starts invisible
        let child_mat = dither_materials.add(ChunkDitherMaterial {
            base: base_material_for_tier(LodTier::Reduced, &debug_mats),
            extension: DitherFadeExtension { fade: 1.0, invert: false },
        });

        let child = commands.spawn((
            LodChildMarker,
            Mesh3d(lod_handle.clone()),
            MeshMaterial3d(child_mat),
            Transform::default(),
            Visibility::default(),
        )).id();

        commands.entity(entity).add_child(child);
        commands.entity(entity).insert(LodChild(child));
    }
}

/// Starts transitions when LodTarget differs from LodTier.
pub fn lod_transition_start_system(
    mut commands: Commands,
    config: Res<LodConfig>,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
    chunks: Query<
        (Entity, &LodTier, &LodTarget, &LodChild),
        (Changed<LodTarget>, Without<LodTransition>),
    >,
    debug_mats: Option<Res<LodDebugMaterials>>,
) {
    for (entity, current_tier, target, lod_child) in chunks.iter() {
        let target_tier = target.as_tier();
        if *current_tier == target_tier {
            continue;
        }

        // Determine which is "main" (parent mesh) and which is "child" (LOD mesh)
        // Parent always has full_res, child always has lod.
        //
        // Full → Reduced: main fades out (0→1), child fades in (1→0)
        // Reduced → Full: main fades in (1→0), child fades out (0→1)
        // Any → Hidden:   both fade out
        // Hidden → Any:   appropriate one fades in

        // (main_start_fade, main_invert, child_start_fade, child_invert)
        // The outgoing mesh uses normal dither (invert=false), fading 0→1
        // The incoming mesh uses inverted dither (invert=true), fading 1→0
        // This ensures complementary pixel coverage at all times.
        let (main_start, main_inv, child_start, child_inv) = match (*current_tier, target_tier) {
            (LodTier::Full, LodTier::Reduced) => (0.0, false, 1.0, true),   // main out, child in
            (LodTier::Reduced, LodTier::Full) => (1.0, true, 0.0, false),   // main in, child out
            (LodTier::Full, LodTier::Hidden) => (0.0, false, 1.0, false),   // main out, child stays hidden
            (LodTier::Reduced, LodTier::Hidden) => (1.0, false, 0.0, false),// child out, main stays hidden
            (LodTier::Hidden, LodTier::Full) => (1.0, true, 1.0, false),    // main in from hidden
            (LodTier::Hidden, LodTier::Reduced) => (1.0, false, 1.0, true), // child in from hidden
            _ => continue,
        };

        let main_mat = dither_materials.add(ChunkDitherMaterial {
            base: base_material_for_tier(LodTier::Full, &debug_mats),
            extension: DitherFadeExtension { fade: main_start, invert: main_inv },
        });

        let child_mat = dither_materials.add(ChunkDitherMaterial {
            base: base_material_for_tier(LodTier::Reduced, &debug_mats),
            extension: DitherFadeExtension { fade: child_start, invert: child_inv },
        });

        // Apply crossfade materials to both meshes.
        // Both always have Mesh3d — visibility is controlled purely by fade values.
        commands.entity(entity).insert(MeshMaterial3d(main_mat.clone()));
        commands.entity(lod_child.0).insert(MeshMaterial3d(child_mat.clone()));

        commands.entity(entity).insert(LodTransition {
            to: target_tier,
            progress: 0.0,
            duration: config.transition_duration,
            main_material: main_mat,
            child_material: child_mat,
        });
    }
}

/// Animates active LOD transitions: updates dither fade values on both meshes.
pub fn lod_transition_update_system(
    mut commands: Commands,
    time: Res<Time>,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
    mut chunks: Query<(Entity, &mut LodTransition, &mut LodTier, &LodChild)>,
    debug_mats: Option<Res<LodDebugMaterials>>,
) {
    let dt = time.delta_secs();

    for (entity, mut transition, mut tier, lod_child) in chunks.iter_mut() {
        transition.progress = (transition.progress + dt / transition.duration).min(1.0);
        let t = transition.progress;

        // Compute fade values based on transition direction
        let (main_fade, child_fade) = match (*tier, transition.to) {
            (LodTier::Full, LodTier::Reduced) => {
                // main: 0→1 (fading out), child: 1→0 (fading in)
                (t, 1.0 - t)
            }
            (LodTier::Reduced, LodTier::Full) => {
                // main: 1→0 (fading in), child: 0→1 (fading out)
                (1.0 - t, t)
            }
            (LodTier::Full, LodTier::Hidden) => {
                // main: 0→1 (fading out), child stays invisible
                (t, 1.0)
            }
            (LodTier::Reduced, LodTier::Hidden) => {
                // child: 0→1 (fading out), main stays invisible
                (1.0, t)
            }
            (LodTier::Hidden, LodTier::Full) => {
                // main: 1→0 (fading in), child stays invisible
                (1.0 - t, 1.0)
            }
            (LodTier::Hidden, LodTier::Reduced) => {
                // child: 1→0 (fading in), main stays invisible
                (1.0, 1.0 - t)
            }
            _ => (0.0, 1.0),
        };

        // Update fade values on both materials
        if let Some(mat) = dither_materials.get_mut(&transition.main_material) {
            mat.extension.fade = main_fade;
        }
        if let Some(mat) = dither_materials.get_mut(&transition.child_material) {
            mat.extension.fade = child_fade;
        }

        // Transition complete
        if t >= 1.0 {
            *tier = transition.to;

            // Set final materials and hide the invisible mesh
            let final_main_mat = final_material_for_tier(transition.to, true, &debug_mats, &mut dither_materials);
            let final_child_mat = final_material_for_tier(transition.to, false, &debug_mats, &mut dither_materials);

            // Set final materials with fade=0 or fade=1. No Mesh3d removal
            // or Visibility toggling — the shader discards all fragments at fade=1.
            commands.entity(entity)
                .insert(MeshMaterial3d(final_main_mat))
                .remove::<LodTransition>();
            commands.entity(lod_child.0)
                .insert(MeshMaterial3d(final_child_mat));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn base_material_for_tier(
    tier: LodTier,
    debug_mats: &Option<Res<LodDebugMaterials>>,
) -> StandardMaterial {
    if debug_mats.is_some() {
        match tier {
            LodTier::Full => StandardMaterial {
                base_color: Color::srgb(0.6, 0.5, 0.4),
                ..default()
            },
            LodTier::Reduced | LodTier::Hidden => StandardMaterial {
                base_color: Color::srgb(0.3, 0.7, 0.9),
                ..default()
            },
        }
    } else {
        StandardMaterial {
            base_color: Color::srgb(0.6, 0.5, 0.4),
            ..default()
        }
    }
}

/// Returns the final material for a tier after transition completes.
/// `is_main` = true for the parent (full_res mesh), false for the child (lod mesh).
fn final_material_for_tier(
    tier: LodTier,
    is_main: bool,
    debug_mats: &Option<Res<LodDebugMaterials>>,
    dither_materials: &mut Assets<ChunkDitherMaterial>,
) -> Handle<ChunkDitherMaterial> {
    // Main (parent) is visible when tier == Full, child is visible when tier == Reduced
    let visible = match tier {
        LodTier::Full => is_main,
        LodTier::Reduced => !is_main,
        LodTier::Hidden => false,
    };

    let fade = if visible { 0.0 } else { 1.0 };
    let base_tier = if is_main { LodTier::Full } else { LodTier::Reduced };

    // Always create a material with the correct fade value (no invert at rest)
    dither_materials.add(ChunkDitherMaterial {
        base: base_material_for_tier(base_tier, debug_mats),
        extension: DitherFadeExtension { fade, invert: false },
    })
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct LodPlugin;

impl Plugin for LodPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ChunkDitherMaterial>::default())
            .init_resource::<LodConfig>()
            .add_systems(Update, (
                lod_target_assignment_system,
                lod_child_setup_system,
                lod_transition_start_system
                    .after(lod_target_assignment_system)
                    .after(lod_child_setup_system),
                lod_transition_update_system.after(lod_transition_start_system),
            ));
    }
}
