//! World definition — the authored content of the game world.
//!
//! JumpBlocks' world is an open archipelago (à la Bowser's Fury): islands
//! spread across a shallow walkable sea. Each island hosts platforming
//! **challenges**: a start pad in the world; touching it activates the
//! challenge and makes a **goal** appear. Reaching the goal earns a
//! **trophy**. Challenges can carry restrictions — a time limit and/or a
//! kill-height (fall volume). Each island also has a **locked zone** whose
//! terrain cannot be modified until enough of that island's trophies have
//! been turned in at the zone's pedestal.
//!
//! Everything here is plain data + deterministic stamping code so the same
//! definition can be built by the game, by tests, and by the web exporter.
//!
//! Coordinates are **region-local global cells** (see [`crate::stamp`]):
//! `world = region_origin + cell * VOXEL_SIZE`.

use bevy::prelude::*;

use crate::chunk::VOXEL_SIZE;
use crate::region::Region;
use crate::shape::Facing;
use crate::stamp::Stamper;
use crate::worldgen::{
    self, ArchipelagoConfig, IslandParams, TEX_BASALT, TEX_COURSE, TEX_DIRT, TEX_GOAL, TEX_GRASS,
    TEX_PAD, TEX_PEDESTAL, TEX_SAND, TEX_SKYSTONE, TEX_STONE,
};

// ---------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------

/// A build-locked zone on an island. While locked, terrain inside the zone
/// AABB cannot be modified. Turn in `trophies_required` trophies earned on
/// the zone's island at the pedestal to unlock it.
#[derive(Clone, Debug)]
pub struct ZoneDef {
    pub name: String,
    /// Index into [`WorldDef::terrain`]'s island list.
    pub island: usize,
    /// AABB in global cells (inclusive min, inclusive max).
    pub min_cell: IVec3,
    pub max_cell: IVec3,
    pub trophies_required: u32,
    /// Cell the player stands on when interacting with the pedestal
    /// (the pedestal column is stamped just next to it).
    pub pedestal_cell: IVec3,
}

/// A platforming challenge: start pad → (challenge activates, goal appears)
/// → reach the goal → trophy.
#[derive(Clone, Debug)]
pub struct ChallengeDef {
    pub name: String,
    /// Index into [`WorldDef::terrain`]'s island list (whose zone this trophy
    /// counts toward).
    pub island: usize,
    /// Cell the player stands on at the start pad.
    pub start_cell: IVec3,
    /// Cell where the goal entity floats when the challenge is active.
    pub goal_cell: IVec3,
    /// Time limit in seconds. `None` = untimed.
    pub time_limit: Option<f32>,
    /// Kill height in cells: while the challenge is active, dropping below
    /// this Y sends the player back to the start pad.
    pub kill_y_cell: Option<i32>,
}

/// A kinematic moving platform oscillating between two points.
#[derive(Clone, Debug)]
pub struct MovingPlatformDef {
    pub name: String,
    pub from_cell: IVec3,
    pub to_cell: IVec3,
    /// Half extents in world units.
    pub half_extents: Vec3,
    /// Seconds for a full out-and-back cycle.
    pub period: f32,
}

/// The full authored world: terrain generation config plus gameplay defs.
#[derive(Clone, Debug)]
pub struct WorldDef {
    pub terrain: ArchipelagoConfig,
    pub zones: Vec<ZoneDef>,
    pub challenges: Vec<ChallengeDef>,
    pub platforms: Vec<MovingPlatformDef>,
    /// Cell the player spawns standing on.
    pub spawn_cell: IVec3,
}

/// World-space center of a cell (the point half a voxel above its base is
/// `cell_stand_world`).
pub fn cell_center_world(region_origin: Vec3, cell: IVec3) -> Vec3 {
    region_origin
        + Vec3::new(
            (cell.x as f32 + 0.5) * VOXEL_SIZE,
            (cell.y as f32 + 0.5) * VOXEL_SIZE,
            (cell.z as f32 + 0.5) * VOXEL_SIZE,
        )
}

/// World-space position at the *base* of a cell, centered in XZ — where an
/// entity standing "on" the cell below should be placed.
pub fn cell_base_world(region_origin: Vec3, cell: IVec3) -> Vec3 {
    region_origin
        + Vec3::new(
            (cell.x as f32 + 0.5) * VOXEL_SIZE,
            cell.y as f32 * VOXEL_SIZE,
            (cell.z as f32 + 0.5) * VOXEL_SIZE,
        )
}

/// World-space AABB of a zone (min corner of min cell to max corner of max cell).
pub fn zone_world_aabb(region_origin: Vec3, zone: &ZoneDef) -> (Vec3, Vec3) {
    let min = region_origin + zone.min_cell.as_vec3() * VOXEL_SIZE;
    let max = region_origin + (zone.max_cell + IVec3::ONE).as_vec3() * VOXEL_SIZE;
    (min, max)
}

impl WorldDef {
    /// Terrain height in cells at a global-cell XZ position.
    pub fn ground(&self, gx: i32, gz: i32) -> i32 {
        worldgen::sample_terrain(
            &self.terrain,
            gx as f32 / crate::chunk::CHUNK_X as f32,
            gz as f32 / crate::chunk::CHUNK_Z as f32,
        )
        .height
    }

    /// The standard authored world: four islands, five challenges, three
    /// locked zones, two moving platforms.
    pub fn standard() -> Self {
        let terrain = ArchipelagoConfig {
            islands: vec![
                IslandParams {
                    name: "Haven Isle".to_string(),
                    center: (128.0, 128.0),
                    radius_chunks: 8.0,
                    peak_height: 80,
                    edge_height: 2,
                    perimeter_bumps: 5.0,
                    bump_amplitude: 0.25,
                    ridge_height: 8.0,
                    texture_surface: TEX_GRASS,
                    texture_body: TEX_DIRT,
                },
                IslandParams {
                    name: "Ember Isle".to_string(),
                    center: (108.0, 132.0),
                    radius_chunks: 5.5,
                    peak_height: 56,
                    edge_height: 2,
                    perimeter_bumps: 4.0,
                    bump_amplitude: 0.2,
                    ridge_height: 5.0,
                    texture_surface: TEX_BASALT,
                    texture_body: TEX_STONE,
                },
                IslandParams {
                    name: "Skyreach Spire".to_string(),
                    center: (141.0, 111.0),
                    radius_chunks: 4.2,
                    peak_height: 112,
                    edge_height: 4,
                    perimeter_bumps: 3.0,
                    bump_amplitude: 0.15,
                    ridge_height: 3.0,
                    texture_surface: TEX_SKYSTONE,
                    texture_body: TEX_STONE,
                },
                IslandParams {
                    name: "Step Cay".to_string(),
                    center: (137.0, 140.0),
                    radius_chunks: 2.5,
                    peak_height: 16,
                    edge_height: 2,
                    perimeter_bumps: 6.0,
                    bump_amplitude: 0.3,
                    ridge_height: 2.0,
                    texture_surface: TEX_GRASS,
                    texture_body: TEX_SAND,
                },
            ],
            flat_extent: 21.0,
            flat_height: 1,
            flat_texture: TEX_SAND,
            underground_depth: 16,
        };

        // Helper for ground heights during authoring
        let ground = |gx: i32, gz: i32| -> i32 {
            worldgen::sample_terrain(
                &terrain,
                gx as f32 / crate::chunk::CHUNK_X as f32,
                gz as f32 / crate::chunk::CHUNK_Z as f32,
            )
            .height
        };

        // --- Challenges ---
        // Start cells are where the player stands: pad is stamped one cell
        // below, so start_cell.y = ground + 1 (top of the pad layer).
        let c0_start = IVec3::new(4240, ground(4240, 4096) + 1, 4096);
        let c1_start = IVec3::new(3950, ground(3950, 4040) + 1, 4040);
        let c2_start = IVec3::new(3456, ground(3456, 4340) + 1, 4340);
        let c3_start = IVec3::new(3570, ground(3570, 4224) + 1, 4224);
        let c4_start = IVec3::new(4512, ground(4512, 3660) + 1, 3660);

        let challenges = vec![
            ChallengeDef {
                name: "First Steps".to_string(),
                island: 0,
                start_cell: c0_start,
                goal_cell: IVec3::new(4460, 34, 4180),
                time_limit: None,
                kill_y_cell: None,
            },
            ChallengeDef {
                name: "Ridge Runner".to_string(),
                island: 0,
                start_cell: c1_start,
                goal_cell: IVec3::new(4110, ground(4110, 4130) + 8, 4130),
                time_limit: Some(50.0),
                kill_y_cell: None,
            },
            ChallengeDef {
                name: "Magma Hop".to_string(),
                island: 1,
                start_cell: c2_start,
                goal_cell: IVec3::new(3456, 14, 4560),
                time_limit: None,
                kill_y_cell: Some(5),
            },
            ChallengeDef {
                name: "Ember Ascent".to_string(),
                island: 1,
                start_cell: c3_start,
                goal_cell: IVec3::new(3464, ground(3464, 4224) + 6, 4224),
                time_limit: Some(60.0),
                kill_y_cell: Some(5),
            },
            ChallengeDef {
                name: "Spire Spiral".to_string(),
                island: 2,
                start_cell: c4_start,
                goal_cell: IVec3::new(4542, 120, 3552),
                time_limit: Some(90.0),
                kill_y_cell: Some(30),
            },
        ];

        let zones = vec![
            ZoneDef {
                name: "Haven Summit".to_string(),
                island: 0,
                min_cell: IVec3::new(4048, 30, 4048),
                max_cell: IVec3::new(4144, 140, 4144),
                trophies_required: 2,
                pedestal_cell: IVec3::new(4160, ground(4160, 4096) + 1, 4096),
            },
            ZoneDef {
                name: "Ember Crater".to_string(),
                island: 1,
                min_cell: IVec3::new(3416, 20, 4184),
                max_cell: IVec3::new(3496, 100, 4264),
                trophies_required: 2,
                pedestal_cell: IVec3::new(3560, ground(3560, 4260) + 1, 4260),
            },
            ZoneDef {
                name: "Skyreach Crown".to_string(),
                island: 2,
                min_cell: IVec3::new(4480, 60, 3520),
                max_cell: IVec3::new(4544, 160, 3584),
                trophies_required: 4,
                pedestal_cell: IVec3::new(4512, ground(4512, 3640) + 1, 3640),
            },
        ];

        let spawn_cell = IVec3::new(4096, ground(4096, 4096) + 1, 4096);

        let mut def = Self { terrain, zones, challenges, platforms: Vec::new(), spawn_cell };

        // The Sea Ferry rides low enough to board from the sand (top at
        // ~2.25 wu; the measured max hop-up is 2.0 wu from a 0.5 wu floor).
        def.platforms.push(MovingPlatformDef {
            name: "Sea Ferry".to_string(),
            from_cell: IVec3::new(3872, 4, 4141),
            to_cell: IVec3::new(3616, 4, 4192),
            half_extents: Vec3::new(2.0, 0.25, 2.0),
            period: 30.0,
        });

        // The Spire Lift bridges the deliberate gap in the Spire Spiral
        // course — its endpoints are derived from the actual course stones
        // so it always lines up.
        let layout = def.course_layout(4);
        if let Some((a, b)) = layout.lift_gap {
            let s1 = layout.stones[a];
            let s2 = layout.stones[b];
            let dir = (s2 - s1).as_vec3().normalize_or_zero();
            let from = s1.as_vec3() + dir * 8.0;
            let to = s2.as_vec3() - dir * 8.0;
            def.platforms.push(MovingPlatformDef {
                name: "Spire Lift".to_string(),
                from_cell: IVec3::new(from.x as i32, s1.y, from.z as i32),
                to_cell: IVec3::new(to.x as i32, s2.y, to.z as i32),
                half_extents: Vec3::new(1.5, 0.25, 1.5),
                period: 10.0,
            });
        }

        def
    }

    /// Generate terrain and stamp all authored structures into the region.
    /// Returns the number of chunks in the region afterwards.
    pub fn build_into_region(&self, region: &mut Region) -> usize {
        worldgen::generate_archipelago(region, &self.terrain);

        let mut s = Stamper::new(region);

        // Start pads: 3×3 block pad under each start cell.
        for c in &self.challenges {
            let p = c.start_cell;
            s.platform(p.x - 2, p.y - 1, p.z - 2, 3, 1, 3, TEX_PAD);
        }

        // Pedestals: 1-block column, 3 cells tall, next to the stand cell.
        for z in &self.zones {
            let p = z.pedestal_cell;
            let base = p.y - 1;
            s.platform(p.x - 2, base, p.z - 2, 3, 1, 3, TEX_PEDESTAL);
            s.column(p.x + 2, base + 1, base + 4, p.z, TEX_PEDESTAL);
        }

        // Courses
        self.stamp_courses(&mut s);

        // Goal platforms: 3×3 pad two cells below each goal, with a wedge
        // "crown" on the outer edge (shows off chamfered slopes).
        for c in &self.challenges {
            let g = c.goal_cell;
            let base = g.y - 2;
            s.platform(g.x - 2, base, g.z - 2, 3, 1, 3, TEX_GOAL);
            s.place_wedge(g.x - 4, base, g.z, Facing::West, TEX_COURSE);
            s.place_wedge(g.x + 4, base, g.z, Facing::East, TEX_COURSE);
            s.place_wedge(g.x, base, g.z - 4, Facing::South, TEX_COURSE);
            s.place_wedge(g.x, base, g.z + 4, Facing::North, TEX_COURSE);
        }

        region.chunk_count()
    }

    fn stamp_courses(&self, s: &mut Stamper) {
        for idx in 0..self.challenges.len() {
            let layout = self.course_layout(idx);
            // Interior stones only: index 0 is the start pad and the last
            // entry is the goal platform, both stamped separately.
            for stone in &layout.stones[1..layout.stones.len().saturating_sub(1)] {
                s.platform(stone.x - 3, stone.y - 1, stone.z - 3, 3, 1, 3, TEX_COURSE);
            }
        }
    }

    /// The hop-by-hop layout of a challenge course. Every consecutive pair
    /// of standing positions (including the start pad and goal platform)
    /// respects the measured player envelope — see `sample_course`.
    pub fn course_layout(&self, idx: usize) -> CourseLayout {
        let c = &self.challenges[idx];
        let start = c.start_cell.as_vec3();
        // Standing surface on the goal platform is one cell below the goal
        // entity's float cell.
        let goal = (c.goal_cell - IVec3::Y).as_vec3();

        match idx {
            // First Steps: gentle bowed arc off Haven's east flank.
            0 => CourseLayout::plain(sample_course(&arc_path(start, goal, 30.0))),
            // Ridge Runner: terrain-hugging bowed run to the summit.
            1 => {
                let xz = arc_path(start, goal, 40.0);
                let path = move |t: f32| {
                    let p = xz(t);
                    let g = self.ground(p.x as i32, p.z as i32) as f32;
                    // Blend to the exact endpoint heights at both ends
                    let hover = g + 3.0;
                    let y = if t < 0.1 {
                        start.y + (hover - start.y) * (t / 0.1)
                    } else if t > 0.9 {
                        hover + (goal.y - hover) * ((t - 0.9) / 0.1)
                    } else {
                        hover
                    };
                    Vec3::new(p.x, y, p.z)
                };
                CourseLayout::plain(sample_course(&path))
            }
            // Magma Hop: low bowed arc across the sea.
            2 => CourseLayout::plain(sample_course(&arc_path(start, goal, 24.0))),
            // Ember Ascent: spiral hugging the cone.
            3 => {
                let center = Vec3::new(3456.0, 0.0, 4224.0);
                let spiral = spiral_path(center, start, goal, 160.0, 40.0);
                let path = move |t: f32| {
                    let p = spiral(t);
                    let g = self.ground(p.x as i32, p.z as i32) as f32 + 3.0;
                    // Follow terrain in the middle, honor endpoints
                    let y = if t < 0.1 {
                        start.y + (g - start.y) * (t / 0.1)
                    } else if t > 0.85 {
                        g + (goal.y - g) * ((t - 0.85) / 0.15)
                    } else {
                        g
                    };
                    Vec3::new(p.x, y, p.z)
                };
                CourseLayout::plain(sample_course(&path))
            }
            // Spire Spiral: free-floating spiral with a moving-platform gap.
            4 => {
                let center = Vec3::new(4512.0, 0.0, 3552.0);
                let spiral = spiral_path(center, start, goal, 110.0, 55.0);
                let lerp_y = move |t: f32| {
                    let p = spiral(t);
                    Vec3::new(p.x, start.y + (goal.y - start.y) * t, p.z)
                };
                let part1 = {
                    let f = |t: f32| lerp_y(t * 0.5);
                    sample_course(&f)
                };
                let part2 = {
                    let f = |t: f32| lerp_y(0.62 + t * 0.38);
                    sample_course(&f)
                };
                let gap_a = part1.len() - 1;
                let mut stones = part1;
                stones.extend(part2);
                CourseLayout { lift_gap: Some((gap_a, gap_a + 1)), stones }
            }
            _ => CourseLayout::plain(vec![c.start_cell, c.goal_cell - IVec3::Y]),
        }
    }
}

// ---------------------------------------------------------------------------
// Course layout
// ---------------------------------------------------------------------------

// The player's measured movement envelope (tests/jump_metrics.rs):
//   jump apex ~3.1 wu, max ledge hop-up 2.0 wu, max run gap >= 11 wu.
// Authoring uses comfortable fractions so courses are challenges, not
// pixel-perfect maxima.

/// Hard ceiling on the rise of a single hop (world units).
pub const PLAYER_MAX_RISE_WU: f32 = 2.0;
/// Hard ceiling on the edge-to-edge gap of a single hop (world units).
pub const PLAYER_MAX_GAP_WU: f32 = 11.0;

/// Authoring limit: rise per hop in cells (1.5 wu of the 2.0 measured).
const STEP_RISE_CELLS: f32 = 3.0;
/// Authoring limit: edge-to-edge gap per hop in cells (5 wu of the 11 measured).
const STEP_GAP_CELLS: f32 = 10.0;
/// Course stones are 3×3 blocks = 6×6 cells.
const STONE_CELLS: f32 = 6.0;

/// A challenge course as standing positions: `stones[0]` is the start pad,
/// the last entry is the goal platform surface.
pub struct CourseLayout {
    pub stones: Vec<IVec3>,
    /// Consecutive stone indices bridged by a moving platform, not a jump.
    pub lift_gap: Option<(usize, usize)>,
}

impl CourseLayout {
    fn plain(stones: Vec<IVec3>) -> Self {
        Self { stones, lift_gap: None }
    }
}

/// A bowed path from `start` to `end` (cells): lateral sine offset, linear Y.
fn arc_path(start: Vec3, end: Vec3, lateral: f32) -> impl Fn(f32) -> Vec3 {
    let dir = end - start;
    let perp = Vec3::new(-dir.z, 0.0, dir.x).normalize_or_zero();
    move |t: f32| start + dir * t + perp * lateral * (t * std::f32::consts::PI).sin()
}

/// A spiral (XZ only; Y = 0) around `center` starting at `start`'s bearing,
/// radius `r0`→`r1`, just over one full loop, blending to `end` at the tail.
fn spiral_path(center: Vec3, start: Vec3, end: Vec3, r0: f32, r1: f32) -> impl Fn(f32) -> Vec3 {
    let a0 = (start.z - center.z).atan2(start.x - center.x);
    let total = 2.0 * std::f32::consts::PI * 1.1;
    move |t: f32| {
        let angle = a0 + total * t;
        let radius = r0 + (r1 - r0) * t;
        let p = Vec3::new(
            center.x + angle.cos() * radius,
            0.0,
            center.z + angle.sin() * radius,
        );
        // Blend toward the exact endpoints so the course connects.
        if t < 0.08 {
            let k = t / 0.08;
            Vec3::new(start.x, 0.0, start.z).lerp(p, k)
        } else if t > 0.88 {
            let k = (t - 0.88) / 0.12;
            p.lerp(Vec3::new(end.x, 0.0, end.z), k * k)
        } else {
            p
        }
    }
}

/// Sample a path into stone positions such that every consecutive hop
/// (including from `path(0)` and to `path(1)`) stays inside the authoring
/// envelope: edge gap ≤ `STEP_GAP_CELLS`, |rise| ≤ `STEP_RISE_CELLS`.
///
/// Greedy: walk the path finely; when the next sample would violate a
/// constraint relative to the last emitted stone, emit the previous sample.
/// On terrain steeper than the envelope allows, the rise is clamped so the
/// stones ladder up in valid steps.
fn sample_course(path: &impl Fn(f32) -> Vec3) -> Vec<IVec3> {
    let to_cell = |p: Vec3| IVec3::new(
        p.x.round() as i32,
        p.y.round() as i32,
        p.z.round() as i32,
    );

    let mut out = vec![to_cell(path(0.0))];
    let mut last = path(0.0);
    let mut prev_sample = last;

    const FINE: i32 = 1200;
    for i in 1..=FINE {
        let t = i as f32 / FINE as f32;
        let p = path(t);
        let horiz = Vec2::new(p.x - last.x, p.z - last.z).length();
        let gap = horiz - STONE_CELLS;
        let rise = p.y - last.y;
        if gap > STEP_GAP_CELLS || rise.abs() > STEP_RISE_CELLS {
            // Emit the last sample that still satisfied the constraints,
            // clamping the rise for terrain steeper than one hop.
            let mut emit = prev_sample;
            let d = emit.y - last.y;
            if d.abs() > STEP_RISE_CELLS {
                emit.y = last.y + STEP_RISE_CELLS.copysign(d);
            }
            // Ensure forward progress even on near-vertical terrain.
            let advance = Vec2::new(emit.x - last.x, emit.z - last.z).length();
            if advance < 2.0 {
                emit.x = p.x;
                emit.z = p.z;
            }
            let cell = to_cell(emit);
            if cell != *out.last().unwrap() {
                out.push(cell);
            }
            last = emit;
        }
        prev_sample = p;
    }

    let end = to_cell(path(1.0));
    if end != *out.last().unwrap() {
        // The final leg's rise may still exceed one hop (e.g. a goal high
        // above the last stone); insert ladder steps if needed.
        let mut lastv = out.last().unwrap().as_vec3();
        let endv = end.as_vec3();
        loop {
            let d = endv.y - lastv.y;
            if d.abs() <= STEP_RISE_CELLS {
                break;
            }
            let t = STEP_RISE_CELLS / d.abs();
            lastv = lastv.lerp(endv, t.min(1.0));
            out.push(to_cell(lastv));
        }
        out.push(end);
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{RegionId, REGION_XZ};
    use crate::chunk::CHUNK_X;

    #[test]
    fn standard_def_is_consistent() {
        let def = WorldDef::standard();
        assert_eq!(def.terrain.islands.len(), 4);
        assert_eq!(def.challenges.len(), 5);
        assert_eq!(def.zones.len(), 3);
        let max_cell = REGION_XZ * CHUNK_X as i32;
        for c in &def.challenges {
            assert!(c.island < def.terrain.islands.len());
            for cell in [c.start_cell, c.goal_cell] {
                assert!(cell.x > 0 && cell.x < max_cell, "{} start/goal x in bounds", c.name);
                assert!(cell.z > 0 && cell.z < max_cell);
                assert!(cell.y > 0, "{} cells above sea", c.name);
            }
        }
        for z in &def.zones {
            assert!(z.island < def.terrain.islands.len());
            assert!(z.min_cell.x < z.max_cell.x && z.min_cell.y < z.max_cell.y);
            assert!(z.trophies_required > 0);
        }
        // Spawn on Haven, above ground
        assert!(def.spawn_cell.y > 2);
    }

    #[test]
    fn island_trophies_can_satisfy_zone_locks() {
        // Zones 0/1 must be unlockable with their own island's challenges;
        // the final zone may require trophies from everywhere.
        let def = WorldDef::standard();
        let total = def.challenges.len() as u32;
        for z in &def.zones {
            let island_trophies = def
                .challenges
                .iter()
                .filter(|c| c.island == z.island)
                .count() as u32;
            assert!(
                z.trophies_required <= island_trophies || z.trophies_required <= total,
                "zone {} requires {} trophies but only {} exist",
                z.name,
                z.trophies_required,
                total
            );
        }
    }

    #[test]
    fn build_produces_terrain_and_pads() {
        let def = WorldDef::standard();
        let mut region = Region::new(RegionId(0), Vec3::new(-2048.0, 0.0, -2048.0));
        let count = def.build_into_region(&mut region);
        assert!(count > 100, "archipelago should have many chunks, got {count}");

        // Each start pad's stand cell has a solid block underneath
        let s = Stamper::new(&mut region);
        for c in &def.challenges {
            let p = c.start_cell;
            assert!(
                s.is_occupied(p.x, p.y - 1, p.z),
                "start pad missing under {} at {:?}",
                c.name,
                p
            );
            // Goal platform exists two cells below the goal
            let g = c.goal_cell;
            assert!(
                s.is_occupied(g.x, g.y - 2, g.z),
                "goal platform missing for {} at {:?}",
                c.name,
                g
            );
        }
    }

    #[test]
    fn courses_respect_player_envelope() {
        // Every hop of every course must be clearable by the measured
        // player (tests/jump_metrics.rs): the authoring limits used here
        // are comfortable fractions of PLAYER_MAX_RISE_WU / _GAP_WU.
        let def = WorldDef::standard();
        let cells_to_wu = crate::chunk::VOXEL_SIZE;
        for idx in 0..def.challenges.len() {
            let layout = def.course_layout(idx);
            let name = &def.challenges[idx].name;
            assert!(layout.stones.len() >= 3, "{name} should have stones");
            for (i, pair) in layout.stones.windows(2).enumerate() {
                if layout.lift_gap == Some((i, i + 1)) {
                    continue; // bridged by the moving platform
                }
                let a = pair[0].as_vec3();
                let b = pair[1].as_vec3();
                let gap_wu =
                    (Vec2::new(b.x - a.x, b.z - a.z).length() - STONE_CELLS) * cells_to_wu;
                let rise_wu = (b.y - a.y) * cells_to_wu;
                assert!(
                    gap_wu <= STEP_GAP_CELLS * cells_to_wu + 0.6,
                    "{name} hop {i}: gap {gap_wu:.1} wu too wide ({:?} -> {:?})",
                    pair[0], pair[1]
                );
                assert!(
                    rise_wu.abs() <= STEP_RISE_CELLS * cells_to_wu + 0.26,
                    "{name} hop {i}: rise {rise_wu:.1} wu too tall ({:?} -> {:?})",
                    pair[0], pair[1]
                );
                // And well inside what the real player can actually do:
                assert!(gap_wu <= PLAYER_MAX_GAP_WU * 0.6);
                assert!(rise_wu.abs() <= PLAYER_MAX_RISE_WU * 0.8 + 0.26);
            }
            // Stones never sit below the sea surface
            for stone in &layout.stones {
                assert!(stone.y >= 2, "{name} stone under the sea: {stone:?}");
            }
        }
    }

    #[test]
    fn spire_lift_bridges_its_gap() {
        let def = WorldDef::standard();
        let layout = def.course_layout(4);
        let (a, b) = layout.lift_gap.expect("Spire Spiral should have a lift gap");
        let lift = def
            .platforms
            .iter()
            .find(|p| p.name == "Spire Lift")
            .expect("Spire Lift exists");
        // Lift endpoints hug the stones on both sides of the gap
        let s1 = layout.stones[a].as_vec3();
        let s2 = layout.stones[b].as_vec3();
        let from = lift.from_cell.as_vec3();
        let to = lift.to_cell.as_vec3();
        assert!(Vec2::new(from.x - s1.x, from.z - s1.z).length() <= 12.0);
        assert!(Vec2::new(to.x - s2.x, to.z - s2.z).length() <= 12.0);
        assert!((from.y - s1.y).abs() <= 3.0 && (to.y - s2.y).abs() <= 3.0);
    }

    #[test]
    fn spawn_and_starts_have_ground() {
        let def = WorldDef::standard();
        // Ground heights used during authoring must be solid terrain
        for c in &def.challenges {
            let g = def.ground(c.start_cell.x, c.start_cell.z);
            assert!(g > 0, "{} start is over the void", c.name);
        }
        assert!(def.ground(def.spawn_cell.x, def.spawn_cell.z) > 0);
    }
}
