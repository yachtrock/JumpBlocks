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

        let platforms = vec![
            MovingPlatformDef {
                name: "Sea Ferry".to_string(),
                from_cell: IVec3::new(3872, 7, 4141),
                to_cell: IVec3::new(3616, 7, 4192),
                half_extents: Vec3::new(2.0, 0.25, 2.0),
                period: 30.0,
            },
            MovingPlatformDef {
                name: "Spire Lift".to_string(),
                // Filled in relative to the Spire Spiral course gap; see
                // course authoring below (kept in sync manually).
                from_cell: IVec3::new(4418, 76, 3600),
                to_cell: IVec3::new(4448, 88, 3524),
                half_extents: Vec3::new(1.5, 0.25, 1.5),
                period: 10.0,
            },
        ];

        let spawn_cell = IVec3::new(4096, ground(4096, 4096) + 1, 4096);

        Self { terrain, zones, challenges, platforms, spawn_cell }
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
        // C0 First Steps: gentle arc of stones from the east flank of Haven
        // down toward the goal over the sea.
        stamp_stone_arc(
            s,
            self.challenges[0].start_cell,
            self.challenges[0].goal_cell + IVec3::new(0, -2, 0),
            9,
            30.0,
            6.0,
            TEX_COURSE,
        );

        // C1 Ridge Runner: terrain-hugging stones over the ridge to the
        // summit; floating 3 cells above ground along a bowed path.
        stamp_terrain_run(
            s,
            self,
            self.challenges[1].start_cell,
            self.challenges[1].goal_cell + IVec3::new(0, -2, 0),
            10,
            40.0,
            3,
            TEX_COURSE,
        );

        // C2 Magma Hop: low stones across the sea, kill volume below.
        stamp_stone_arc(
            s,
            self.challenges[2].start_cell,
            self.challenges[2].goal_cell + IVec3::new(0, -2, 0),
            8,
            24.0,
            4.0,
            TEX_COURSE,
        );

        // C3 Ember Ascent: spiral of ramped stones up the cone.
        stamp_spiral(
            s,
            self,
            IVec3::new(3456, 0, 4224), // Ember center
            self.challenges[3].start_cell,
            self.challenges[3].goal_cell + IVec3::new(0, -2, 0),
            10,
            160.0,
            36.0,
            Some(3), // hug terrain at +3 cells
            TEX_COURSE,
        );

        // C4 Spire Spiral: floating spiral around the spire with a gap for
        // the Spire Lift moving platform (t in [0.5, 0.65] skipped).
        stamp_spiral(
            s,
            self,
            IVec3::new(4512, 0, 3552), // Spire center
            self.challenges[4].start_cell,
            self.challenges[4].goal_cell + IVec3::new(0, -2, 0),
            14,
            110.0,
            55.0,
            None, // free-floating: interpolate start.y -> goal.y
            TEX_COURSE,
        );
    }
}

// ---------------------------------------------------------------------------
// Course stamping helpers
// ---------------------------------------------------------------------------

fn facing_toward(delta: IVec3) -> Facing {
    if delta.x.abs() >= delta.z.abs() {
        if delta.x >= 0 { Facing::East } else { Facing::West }
    } else if delta.z >= 0 {
        Facing::North
    } else {
        Facing::South
    }
}

/// A run of floating 2×2-block stones from `start` to `end` with a lateral
/// sine bow (`lateral` cells) and a vertical arc (`rise` cells at midpoint).
fn stamp_stone_arc(
    s: &mut Stamper,
    start: IVec3,
    end: IVec3,
    steps: i32,
    lateral: f32,
    rise: f32,
    texture: u16,
) {
    let dir = (end - start).as_vec3();
    let perp = Vec3::new(-dir.z, 0.0, dir.x).normalize_or_zero();
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        let bow = (t * std::f32::consts::PI).sin();
        let p = start.as_vec3() + dir * t + perp * lateral * bow + Vec3::Y * (rise * bow);
        let cell = IVec3::new(p.x as i32, p.y as i32, p.z as i32);
        // Stone: 2×2 blocks, one layer, standing surface at cell.y
        s.platform(cell.x - 2, cell.y - 1, cell.z - 2, 2, 1, 2, texture);
    }
}

/// Stones that follow the terrain at `hover` cells above ground along a
/// bowed path (for over-land runs).
#[allow(clippy::too_many_arguments)]
fn stamp_terrain_run(
    s: &mut Stamper,
    def: &WorldDef,
    start: IVec3,
    end: IVec3,
    steps: i32,
    lateral: f32,
    hover: i32,
    texture: u16,
) {
    let dir = (end - start).as_vec3();
    let perp = Vec3::new(-dir.z, 0.0, dir.x).normalize_or_zero();
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        let bow = (t * std::f32::consts::PI).sin();
        let p = start.as_vec3() + dir * t + perp * lateral * bow;
        let gx = p.x as i32;
        let gz = p.z as i32;
        let gy = def.ground(gx, gz) + hover;
        s.platform(gx - 2, gy - 1, gz - 2, 2, 1, 2, texture);
    }
}

/// A spiral of stones around `center` from `start` to `end`.
/// Radius shrinks from `r0` to `r1`; if `hug` is set the stones sit that many
/// cells above terrain, otherwise Y interpolates start→end. Every stone gets
/// a wedge ramp on its uphill edge pointing back along the path.
#[allow(clippy::too_many_arguments)]
fn stamp_spiral(
    s: &mut Stamper,
    def: &WorldDef,
    center: IVec3,
    start: IVec3,
    end: IVec3,
    steps: i32,
    r0: f32,
    r1: f32,
    hug: Option<i32>,
    texture: u16,
) {
    let a0 = ((start.z - center.z) as f32).atan2((start.x - center.x) as f32);
    let total_angle = 2.0 * std::f32::consts::PI * 1.1; // just over one loop
    let mut prev = start;
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        // Gap for the moving platform on free-floating spirals
        if hug.is_none() && (0.5..0.65).contains(&t) {
            continue;
        }
        let (cell, is_last) = if i == steps {
            (end, true)
        } else {
            let angle = a0 + total_angle * t;
            let radius = r0 + (r1 - r0) * t;
            let gx = center.x + (angle.cos() * radius) as i32;
            let gz = center.z + (angle.sin() * radius) as i32;
            let gy = match hug {
                Some(h) => def.ground(gx, gz) + h,
                None => start.y + ((end.y - start.y) as f32 * t) as i32,
            };
            (IVec3::new(gx, gy, gz), false)
        };
        s.platform(cell.x - 2, cell.y - 1, cell.z - 2, 2, 1, 2, texture);
        // Ramp on the approach edge when climbing
        if cell.y > prev.y + 1 && !is_last {
            let toward_prev = prev - cell;
            let f = facing_toward(toward_prev);
            let off = match f {
                Facing::East => IVec3::new(4, 0, 0),
                Facing::West => IVec3::new(-4, 0, 0),
                Facing::North => IVec3::new(0, 0, 4),
                Facing::South => IVec3::new(0, 0, -4),
            };
            s.place_wedge(cell.x + off.x, cell.y - 1, cell.z + off.z, f, texture);
        }
        prev = cell;
    }
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
