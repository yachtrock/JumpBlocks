use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_tnua::prelude::*;
use std::f32::consts::TAU;

use crate::player::{ControlScheme, Player};

pub struct EdgeDetectionPlugin;

impl Plugin for EdgeDetectionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (detect_edges, draw_edge_gizmos));
    }
}

/// Tuning for the edge detection raycasts.
#[derive(Component)]
pub struct EdgeDetectionSettings {
    /// How far down to cast rays from the player origin.
    pub ray_max_distance: f32,
}

impl Default for EdgeDetectionSettings {
    fn default() -> Self {
        Self {
            ray_max_distance: 3.0,
        }
    }
}

/// Tracks when the player is standing near or over an edge.
#[derive(Component, Default)]
pub struct PrecariousEdge {
    /// Whether the player is currently on an edge.
    pub on_edge: bool,
    /// World-space position of the closest point on the edge (at ground level).
    pub edge_point: Vec3,
    /// Direction along the edge (normalized, in XZ plane).
    pub edge_direction: Vec3,
    /// Direction pointing toward the void / off the edge.
    pub overhang_direction: Vec3,
    /// How far the player's center is from the edge (horizontal).
    pub distance_from_edge: f32,
}

/// Number of radial rays to cast around the player.
const RAY_COUNT: usize = 16;
/// Radius at which to sample ground (matches player capsule radius).
const SAMPLE_RADIUS: f32 = 0.35;
/// Number of binary search iterations for precise edge location.
const BISECT_STEPS: usize = 6;

fn detect_edges(
    spatial_query: SpatialQuery,
    mut player_query: Query<
        (
            Entity,
            &Transform,
            &TnuaController<ControlScheme>,
            &EdgeDetectionSettings,
            &mut PrecariousEdge,
        ),
        With<Player>,
    >,
) {
    let Ok((entity, transform, controller, settings, mut edge)) = player_query.single_mut()
    else {
        return;
    };

    // Only detect edges when grounded (walking/running), not while jumping/airborne
    let grounded = matches!(controller.is_airborne(), Ok(false));
    if !grounded {
        edge.on_edge = false;
        return;
    }

    let ray_max = settings.ray_max_distance;
    let filter = SpatialQueryFilter::from_excluded_entities([entity]);
    let origin = transform.translation;
    let down = Dir3::NEG_Y;

    // Cast center ray straight down
    let center_hit = spatial_query.cast_ray(origin, down, ray_max, true, &filter);

    if center_hit.is_some() {
        // Center is over solid ground — not precarious
        edge.on_edge = false;
        return;
    }

    // Center ray missed. Cast radial rays to see if we're near an edge.
    let mut hit_dirs: Vec<(Vec3, f32)> = Vec::with_capacity(RAY_COUNT);
    let mut miss_dirs: Vec<Vec3> = Vec::with_capacity(RAY_COUNT);
    let mut ray_results: Vec<bool> = Vec::with_capacity(RAY_COUNT);

    for i in 0..RAY_COUNT {
        let angle = (i as f32 / RAY_COUNT as f32) * TAU;
        let dir = Vec3::new(angle.cos(), 0.0, angle.sin());
        let ray_origin = origin + dir * SAMPLE_RADIUS;

        if let Some(hit) = spatial_query.cast_ray(ray_origin, down, ray_max, true, &filter) {
            hit_dirs.push((dir, hit.distance));
            ray_results.push(true);
        } else {
            miss_dirs.push(dir);
            ray_results.push(false);
        }
    }

    // If no radial rays hit, we're fully airborne — no edge
    if hit_dirs.is_empty() {
        edge.on_edge = false;
        return;
    }
    // If all hit, we're surrounded by ground (but center missed — odd geometry, ignore)
    if miss_dirs.is_empty() {
        edge.on_edge = false;
        return;
    }

    // We have a mix of hits and misses — we're on an edge!
    edge.on_edge = true;

    // Find an adjacent hit/miss boundary pair in the radial samples
    let mut boundary_hit_dir = hit_dirs[0].0;
    let mut boundary_miss_dir = miss_dirs[0];
    let mut boundary_hit_distance = hit_dirs[0].1;

    for i in 0..RAY_COUNT {
        let next = (i + 1) % RAY_COUNT;
        if ray_results[i] != ray_results[next] {
            let angle_a = (i as f32 / RAY_COUNT as f32) * TAU;
            let angle_b = (next as f32 / RAY_COUNT as f32) * TAU;
            let dir_a = Vec3::new(angle_a.cos(), 0.0, angle_a.sin());
            let dir_b = Vec3::new(angle_b.cos(), 0.0, angle_b.sin());

            if ray_results[i] {
                boundary_hit_dir = dir_a;
                boundary_miss_dir = dir_b;
                boundary_hit_distance = spatial_query
                    .cast_ray(origin + dir_a * SAMPLE_RADIUS, down, ray_max, true, &filter)
                    .map(|h| h.distance)
                    .unwrap_or(1.5);
            } else {
                boundary_hit_dir = dir_b;
                boundary_miss_dir = dir_a;
                boundary_hit_distance = spatial_query
                    .cast_ray(origin + dir_b * SAMPLE_RADIUS, down, ray_max, true, &filter)
                    .map(|h| h.distance)
                    .unwrap_or(1.5);
            }
            break;
        }
    }

    // Binary search angularly between the hit and miss directions to find precise edge
    let mut hit_dir = boundary_hit_dir;
    let mut miss_dir = boundary_miss_dir;

    for _ in 0..BISECT_STEPS {
        let mid = (hit_dir + miss_dir).normalize_or_zero();
        if mid == Vec3::ZERO {
            break;
        }
        let mid_origin = origin + mid * SAMPLE_RADIUS;
        if spatial_query
            .cast_ray(mid_origin, down, ray_max, true, &filter)
            .is_some()
        {
            hit_dir = mid;
        } else {
            miss_dir = mid;
        }
    }

    // Edge point is at the boundary between hit and miss, at ground level
    let edge_horizontal = (hit_dir + miss_dir) * 0.5 * SAMPLE_RADIUS;
    let ground_y = origin.y - boundary_hit_distance;
    edge.edge_point = Vec3::new(
        origin.x + edge_horizontal.x,
        ground_y,
        origin.z + edge_horizontal.z,
    );

    // Overhang direction: from ground side toward void side
    let ground_center: Vec3 = hit_dirs
        .iter()
        .map(|(d, _)| *d)
        .sum::<Vec3>()
        .normalize_or_zero();
    let void_center: Vec3 = miss_dirs.iter().sum::<Vec3>().normalize_or_zero();
    edge.overhang_direction = void_center;

    // Edge direction: perpendicular to overhang direction in XZ plane
    let to_void = (void_center - ground_center).normalize_or_zero();
    edge.edge_direction = Vec3::new(-to_void.z, 0.0, to_void.x).normalize_or_zero();

    // Distance from player center to the edge (horizontal only)
    let player_to_edge = Vec3::new(edge_horizontal.x, 0.0, edge_horizontal.z);
    edge.distance_from_edge = player_to_edge.length();
}

fn draw_edge_gizmos(mut gizmos: Gizmos, query: Query<&PrecariousEdge, With<Player>>) {
    let Ok(edge) = query.single() else {
        return;
    };

    if !edge.on_edge {
        return;
    }

    // Draw a capsule along the edge, width = player capsule diameter
    let capsule_half_length = SAMPLE_RADIUS; // half of player diameter
    let capsule_radius = 0.04;

    // Rotation: align capsule's local Y axis with the edge direction
    let rotation = if edge.edge_direction.length_squared() > 0.01 {
        Quat::from_rotation_arc(Vec3::Y, edge.edge_direction)
    } else {
        Quat::IDENTITY
    };

    let edge_color = Color::srgb(1.0, 0.4, 0.0);
    gizmos.primitive_3d(
        &Capsule3d::new(capsule_radius, capsule_half_length * 2.0),
        Isometry3d::new(edge.edge_point, rotation),
        edge_color,
    );

    // Arrow showing overhang direction
    let arrow_start = edge.edge_point;
    let arrow_end = edge.edge_point + edge.overhang_direction * 0.4;
    gizmos.line(arrow_start, arrow_end, Color::srgb(1.0, 0.0, 0.0));
}
