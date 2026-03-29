//! LOD tier management with dither-fade transitions.
//!
//! Immediate-mode: every frame, compute the desired LOD state from distance
//! and write fade/chamfer values directly to persistent material handles.
//! No deferred material swaps, no pipeline re-specialization races.
//!
//! Structure:
//!   Chunk entity:  Mesh3d(full_res) + ChunkDitherMaterial (persistent handle)
//!     └─ LodChild: Mesh3d(lod)      + ChunkDitherMaterial (persistent handle)
//!
//! Both materials are created once and mutated in place every frame.

use bevy::prelude::*;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline, MaterialPlugin};
use bevy::render::render_resource::{AsBindGroup, ShaderType, SpecializedMeshPipelineError, RenderPipelineDescriptor};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::shader::ShaderRef;

use crate::coords::{ChunkCoord, CHUNK_WORLD_SIZE};
use crate::meshing::{ATTRIBUTE_CHAMFER_OFFSET, ATTRIBUTE_SHARP_NORMAL};
use crate::streaming::StreamingAnchor;

// ---------------------------------------------------------------------------
// Dither fade material
// ---------------------------------------------------------------------------

pub type ChunkDitherMaterial = ExtendedMaterial<StandardMaterial, DitherFadeExtension>;

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
#[uniform(200, DitherFadeUniform)]
pub struct DitherFadeExtension {
    pub fade: f32,
    pub invert: bool,
    pub chamfer_amount: f32,
}

#[derive(Clone, Default, ShaderType)]
pub struct DitherFadeUniform {
    pub fade: f32,
    pub invert: f32,
    pub chamfer_amount: f32,
    pub _pad: f32,
}

impl From<&DitherFadeExtension> for DitherFadeUniform {
    fn from(ext: &DitherFadeExtension) -> Self {
        Self {
            fade: ext.fade,
            invert: if ext.invert { 1.0 } else { 0.0 },
            chamfer_amount: ext.chamfer_amount,
            _pad: 0.0,
        }
    }
}

pub const CHAMFER_OFFSET_LOCATION: u32 = 10;
pub const SHARP_NORMAL_LOCATION: u32 = 11;

impl MaterialExtension for DitherFadeExtension {
    fn vertex_shader() -> ShaderRef {
        "shaders/chunk_vertex.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/dither_fade.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/dither_prepass.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let mut attrs = vec![
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        ];
        if layout.0.contains(Mesh::ATTRIBUTE_UV_1) {
            attrs.push(Mesh::ATTRIBUTE_UV_1.at_shader_location(3));
        }
        if layout.0.contains(Mesh::ATTRIBUTE_TANGENT) {
            attrs.push(Mesh::ATTRIBUTE_TANGENT.at_shader_location(4));
        }
        if layout.0.contains(Mesh::ATTRIBUTE_COLOR) {
            attrs.push(Mesh::ATTRIBUTE_COLOR.at_shader_location(5));
        }
        if layout.0.contains(ATTRIBUTE_CHAMFER_OFFSET) {
            attrs.push(ATTRIBUTE_CHAMFER_OFFSET.at_shader_location(CHAMFER_OFFSET_LOCATION));
            descriptor.vertex.shader_defs.push("HAS_CHAMFER_OFFSET".into());
        }
        if layout.0.contains(ATTRIBUTE_SHARP_NORMAL) {
            attrs.push(ATTRIBUTE_SHARP_NORMAL.at_shader_location(SHARP_NORMAL_LOCATION));
            descriptor.vertex.shader_defs.push("HAS_SHARP_NORMAL".into());
        }
        descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LodTier {
    #[default]
    Full,
    Reduced,
    Hidden,
}

#[derive(Resource, Debug, Clone)]
pub struct LodConfig {
    pub full_radius: i32,
    pub reduced_radius: i32,
    pub transition_duration: f32,
    pub chamfer_start: f32,
    pub chamfer_end: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            full_radius: 2,
            reduced_radius: 6,
            transition_duration: 0.4,
            chamfer_start: 1.0,
            chamfer_end: 2.0,
        }
    }
}

#[derive(Component)]
pub struct ChunkLodMesh {
    pub full_res: Option<Handle<Mesh>>,
    pub lod: Option<Handle<Mesh>>,
}

/// Persistent material handles, created once per chunk.
#[derive(Component)]
pub struct ChunkLodMaterials {
    pub main_handle: Handle<ChunkDitherMaterial>,
    pub child_handle: Handle<ChunkDitherMaterial>,
}

#[derive(Component)]
pub struct LodChild(pub Entity);

#[derive(Component)]
pub struct LodChildMarker;

/// Tracks an in-progress LOD crossfade.
#[derive(Component)]
pub struct LodTransition {
    pub from: LodTier,
    pub to: LodTier,
    pub blend: f32,
}

/// Optional debug coloring for LOD tiers.
#[derive(Resource)]
pub struct LodDebugMaterials {
    pub full_color: Color,
    pub reduced_color: Color,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// One-time setup: creates the LOD child entity and persistent material handles.
pub fn lod_setup_system(
    mut commands: Commands,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
    chunks: Query<
        (Entity, &ChunkLodMesh, &MeshMaterial3d<ChunkDitherMaterial>),
        (Added<ChunkLodMesh>, Without<LodChild>),
    >,
    debug_mats: Option<Res<LodDebugMaterials>>,
) {
    for (entity, lod_mesh, parent_mat) in chunks.iter() {
        let Some(ref lod_handle) = lod_mesh.lod else { continue };

        let base = dither_materials
            .get(&parent_mat.0)
            .map(|m| m.base.clone())
            .unwrap_or_else(|| StandardMaterial {
                base_color: Color::srgb(0.6, 0.5, 0.4),
                ..default()
            });

        let main_base = if let Some(ref dbg) = debug_mats {
            StandardMaterial { base_color: dbg.full_color, ..base.clone() }
        } else {
            base.clone()
        };
        let child_base = if let Some(ref dbg) = debug_mats {
            StandardMaterial { base_color: dbg.reduced_color, ..base }
        } else {
            base
        };

        let main_handle = dither_materials.add(ChunkDitherMaterial {
            base: main_base,
            extension: DitherFadeExtension { fade: 0.0, invert: false, chamfer_amount: 1.0 },
        });
        let child_handle = dither_materials.add(ChunkDitherMaterial {
            base: child_base,
            extension: DitherFadeExtension { fade: 1.0, invert: false, chamfer_amount: 0.0 },
        });

        let child = commands.spawn((
            LodChildMarker,
            Mesh3d(lod_handle.clone()),
            MeshMaterial3d(child_handle.clone()),
            Transform::default(),
            Visibility::default(),
        )).id();

        commands.entity(entity)
            .insert((
                MeshMaterial3d(main_handle.clone()),
                ChunkLodMaterials { main_handle, child_handle },
                LodTier::Full,
            ))
            .add_child(child);
        commands.entity(entity).insert(LodChild(child));
    }
}

/// Immediate-mode LOD update. Runs every frame for every chunk that has
/// materials set up. Computes distance, desired tier, chamfer, fade values
/// and writes them directly to the persistent material assets.
pub fn lod_update_system(
    config: Res<LodConfig>,
    time: Res<Time>,
    anchor_query: Query<&GlobalTransform, With<StreamingAnchor>>,
    mut chunks: Query<(
        Entity,
        &ChunkCoord,
        &ChunkLodMaterials,
        &LodChild,
        &mut LodTier,
        Option<&mut LodTransition>,
    )>,
    mut child_vis: Query<&mut Visibility, With<LodChildMarker>>,
    mut commands: Commands,
    mut dither_materials: ResMut<Assets<ChunkDitherMaterial>>,
) {
    let Ok(anchor_transform) = anchor_query.single() else { return };
    let anchor_pos = anchor_transform.translation();
    let dt = time.delta_secs();

    let ch_start = config.chamfer_start;
    let ch_range = (config.chamfer_end - ch_start).max(0.001);

    for (entity, coord, materials, lod_child, mut tier, transition_opt) in chunks.iter_mut() {
        let chunk_center = coord.pos.to_world_offset()
            + Vec3::splat(CHUNK_WORLD_SIZE * 0.5);
        let dist = ((anchor_pos - chunk_center) / CHUNK_WORLD_SIZE).abs();
        let max_dist = dist.x.max(dist.y).max(dist.z);

        // --- Desired tier ---
        let desired = if (max_dist as i32) <= config.full_radius {
            LodTier::Full
        } else if (max_dist as i32) <= config.reduced_radius {
            LodTier::Reduced
        } else {
            LodTier::Hidden
        };

        // --- Chamfer amount (only affects the main/full-res mesh) ---
        let chamfer = 1.0 - ((max_dist - ch_start) / ch_range).clamp(0.0, 1.0);

        // --- Compute fade values ---
        let (main_fade, main_inv, child_fade, child_inv) = if desired == *tier {
            // At rest — remove any leftover transition
            if transition_opt.is_some() {
                commands.entity(entity).remove::<LodTransition>();
            }
            rest_fades(*tier)
        } else if let Some(mut trans) = transition_opt {
            // Mid-transition — check if target changed
            if trans.to != desired {
                // Retarget: start fresh toward new desired
                trans.from = *tier;
                trans.to = desired;
                trans.blend = 0.0;
            }
            trans.blend = (trans.blend + dt / config.transition_duration).min(1.0);
            let result = transition_fades(trans.from, trans.to, trans.blend);

            if trans.blend >= 1.0 {
                *tier = trans.to;
                commands.entity(entity).remove::<LodTransition>();
            }
            result
        } else {
            // Start new transition
            commands.entity(entity).insert(LodTransition {
                from: *tier,
                to: desired,
                blend: 0.0,
            });
            // First frame: still show current state
            rest_fades(*tier)
        };

        // --- Write to materials ---
        if let Some(mat) = dither_materials.get_mut(&materials.main_handle) {
            mat.extension.fade = main_fade;
            mat.extension.invert = main_inv;
            mat.extension.chamfer_amount = chamfer;
        }
        if let Some(mat) = dither_materials.get_mut(&materials.child_handle) {
            mat.extension.fade = child_fade;
            mat.extension.invert = child_inv;
            mat.extension.chamfer_amount = 0.0; // LOD mesh has no chamfer data
        }

        // --- Set child visibility (prevents shadow/wireframe/depth artifacts) ---
        // Visibility::Hidden is the only way to fully exclude from all render passes.
        // Safe on the child since it has no descendants.
        if let Ok(mut vis) = child_vis.get_mut(lod_child.0) {
            *vis = if child_fade >= 1.0 {
                Visibility::Hidden
            } else {
                Visibility::Inherited
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fade values when at rest in a given tier (no transition).
fn rest_fades(tier: LodTier) -> (f32, bool, f32, bool) {
    match tier {
        LodTier::Full    => (0.0, false, 1.0, false), // main visible, child hidden
        LodTier::Reduced => (1.0, false, 0.0, false), // main hidden, child visible
        LodTier::Hidden  => (1.0, false, 1.0, false), // both hidden
    }
}

/// Fade values during a transition at blend `t` (0→1).
fn transition_fades(from: LodTier, to: LodTier, t: f32) -> (f32, bool, f32, bool) {
    match (from, to) {
        // Full → Reduced: main fades out (normal), child fades in (inverted)
        (LodTier::Full, LodTier::Reduced) => (t, false, 1.0 - t, true),
        // Reduced → Full: main fades in (inverted), child fades out (normal)
        (LodTier::Reduced, LodTier::Full) => (1.0 - t, true, t, false),
        // Full → Hidden: main fades out
        (LodTier::Full, LodTier::Hidden) => (t, false, 1.0, false),
        // Reduced → Hidden: child fades out
        (LodTier::Reduced, LodTier::Hidden) => (1.0, false, t, false),
        // Hidden → Full: main fades in
        (LodTier::Hidden, LodTier::Full) => (1.0 - t, true, 1.0, false),
        // Hidden → Reduced: child fades in
        (LodTier::Hidden, LodTier::Reduced) => (1.0, false, 1.0 - t, true),
        _ => rest_fades(to),
    }
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
                lod_setup_system,
                lod_update_system.after(lod_setup_system),
            ));
    }
}
