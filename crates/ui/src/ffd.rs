//! # Free-Form Deformation (FFD) via Verlet Physics
//!
//! A 4×4 grid of control points driven by a verlet physics simulation.
//! Used to add jiggle, squash, stretch, and lively motion to UI widgets.
//!
//! Each widget that wants deformation owns an [`FfdSim`]. The sim is stepped
//! each frame, and the resulting control-point positions are used as a bicubic
//! Bernstein FFD to warp draw commands through [`Canvas::begin_ffd`].
//!
//! ## Coordinate space
//!
//! The FFD maps normalized coordinates `(s, t)` in `[0, 1]²` to screen-space
//! pixel positions. At rest the control points are evenly spaced across the
//! widget's rect, so the FFD is an identity mapping. Forces and impulses
//! displace the points, creating deformation.

/// Number of control points per axis. Always 4 for bicubic Bernstein.
pub const GRID_SIZE: usize = 4;

/// Total number of control points.
pub const NUM_POINTS: usize = GRID_SIZE * GRID_SIZE;

/// A 4×4 verlet-driven free-form deformation simulation.
///
/// Control points are stored in row-major order: index `j * 4 + i` where
/// `i` is the column (horizontal) and `j` is the row (vertical).
/// The four corner indices in the 4×4 grid.
const CORNER_INDICES: &[usize] = &[0, GRID_SIZE - 1, NUM_POINTS - GRID_SIZE, NUM_POINTS - 1];

fn is_corner(idx: usize) -> bool {
    idx == 0 || idx == GRID_SIZE - 1 || idx == NUM_POINTS - GRID_SIZE || idx == NUM_POINTS - 1
}

pub struct FfdSim {
    /// Current positions in screen-space pixels.
    pub pos: [[f32; 2]; NUM_POINTS],
    /// Previous positions (for verlet integration).
    old_pos: [[f32; 2]; NUM_POINTS],
    /// Rest positions — the undeformed grid mapped to the widget rect.
    rest: [[f32; 2]; NUM_POINTS],
    /// Distance constraints between adjacent points.
    constraints: Vec<Constraint>,
    /// Global damping factor applied each step (0.0 = no damping, 1.0 = frozen).
    /// Typical: 0.03–0.08. Higher = motion dies faster.
    pub damping: f32,
    /// Spring rate pulling points back toward rest (0.0–1.0).
    /// This is the per-frame-at-60fps ratio. 0.05 = gentle, 0.2 = snappy.
    pub spring_rate: f32,
    /// Stiffness of distance constraints (0.0–1.0).
    /// Lower = softer jelly. 0.3 is a good default.
    pub constraint_stiffness: f32,
    /// Number of constraint-solving iterations per step.
    pub iterations: u32,
    /// Whether corner points are pinned (don't move from rest).
    pub pin_corners: bool,
}

struct Constraint {
    a: usize,
    b: usize,
    rest_len: f32,
}

impl FfdSim {
    /// Create a new simulation mapped to the given screen-space rectangle.
    ///
    /// The 4×4 grid is evenly distributed across `[x, y, w, h]`.
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        let mut pos = [[0.0f32; 2]; NUM_POINTS];
        let mut rest = [[0.0f32; 2]; NUM_POINTS];

        for j in 0..GRID_SIZE {
            for i in 0..GRID_SIZE {
                let idx = j * GRID_SIZE + i;
                let px = x + (i as f32 / (GRID_SIZE - 1) as f32) * w;
                let py = y + (j as f32 / (GRID_SIZE - 1) as f32) * h;
                pos[idx] = [px, py];
                rest[idx] = [px, py];
            }
        }

        let old_pos = pos;

        // Build distance constraints: horizontal and vertical neighbors only.
        // Omitting diagonals gives a softer, jelly-like deformation.
        let mut constraints = Vec::new();
        for j in 0..GRID_SIZE {
            for i in 0..GRID_SIZE {
                let idx = j * GRID_SIZE + i;
                // Right neighbor
                if i + 1 < GRID_SIZE {
                    let other = j * GRID_SIZE + (i + 1);
                    constraints.push(Constraint::between(&pos, idx, other));
                }
                // Down neighbor
                if j + 1 < GRID_SIZE {
                    let other = (j + 1) * GRID_SIZE + i;
                    constraints.push(Constraint::between(&pos, idx, other));
                }
            }
        }

        Self {
            pos,
            old_pos,
            rest,
            constraints,
            damping: 0.04,
            spring_rate: 0.08,
            constraint_stiffness: 0.3,
            iterations: 2,
            pin_corners: true,
        }
    }

    /// Resize the simulation to a new screen-space rectangle.
    ///
    /// Resets all points to their rest positions within the new rect.
    pub fn resize(&mut self, x: f32, y: f32, w: f32, h: f32) {
        for j in 0..GRID_SIZE {
            for i in 0..GRID_SIZE {
                let idx = j * GRID_SIZE + i;
                let px = x + (i as f32 / (GRID_SIZE - 1) as f32) * w;
                let py = y + (j as f32 / (GRID_SIZE - 1) as f32) * h;
                self.pos[idx] = [px, py];
                self.old_pos[idx] = [px, py];
                self.rest[idx] = [px, py];
            }
        }
        // Rebuild constraint rest lengths
        for c in &mut self.constraints {
            c.rest_len = dist(&self.pos[c.a], &self.pos[c.b]);
        }
    }

    /// Step the simulation forward by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let damp = 1.0 - self.damping;

        // Verlet integration
        for i in 0..NUM_POINTS {
            let [cx, cy] = self.pos[i];
            let [ox, oy] = self.old_pos[i];
            let vx = (cx - ox) * damp;
            let vy = (cy - oy) * damp;
            self.old_pos[i] = [cx, cy];
            self.pos[i] = [cx + vx, cy + vy];
        }

        // Spring back toward rest position.
        // Use a frame-rate-independent factor: 1 - (1 - spring_rate)^(dt * 60)
        // so the spring feels the same at any tick rate.
        let spring_factor = 1.0 - (1.0 - self.spring_rate).powf(dt * 60.0);
        for i in 0..NUM_POINTS {
            let [rx, ry] = self.rest[i];
            self.pos[i][0] += (rx - self.pos[i][0]) * spring_factor;
            self.pos[i][1] += (ry - self.pos[i][1]) * spring_factor;
        }

        // Solve distance constraints (only horizontal + vertical, skip diagonals
        // for a softer, jelly-like feel)
        for _ in 0..self.iterations {
            for c in &self.constraints {
                let [ax, ay] = self.pos[c.a];
                let [bx, by] = self.pos[c.b];
                let dx = bx - ax;
                let dy = by - ay;
                let d = (dx * dx + dy * dy).sqrt();
                if d < 1e-6 {
                    continue;
                }
                let diff = (c.rest_len - d) / d * 0.5 * self.constraint_stiffness;
                let ox = dx * diff;
                let oy = dy * diff;
                // Don't move pinned corners via constraints
                let a_pinned = self.pin_corners && is_corner(c.a);
                let b_pinned = self.pin_corners && is_corner(c.b);
                if !a_pinned {
                    self.pos[c.a] = [ax - ox, ay - oy];
                }
                if !b_pinned {
                    self.pos[c.b] = [bx + ox, by + oy];
                }
            }
        }

        // Pin corners
        if self.pin_corners {
            for &idx in CORNER_INDICES {
                self.pos[idx] = self.rest[idx];
                self.old_pos[idx] = self.rest[idx];
            }
        }
    }

    /// Apply a force (in pixels/s²) to all points.
    pub fn apply_force(&mut self, fx: f32, fy: f32, dt: f32) {
        let dt2 = dt * dt;
        for i in 0..NUM_POINTS {
            self.pos[i][0] += fx * dt2;
            self.pos[i][1] += fy * dt2;
        }
    }

    /// Apply a force to a single control point by grid coordinate.
    pub fn apply_force_at(&mut self, col: usize, row: usize, fx: f32, fy: f32, dt: f32) {
        let dt2 = dt * dt;
        let idx = row * GRID_SIZE + col;
        self.pos[idx][0] += fx * dt2;
        self.pos[idx][1] += fy * dt2;
    }

    /// Apply an impulse (instant velocity change in pixels/s) to all points.
    pub fn apply_impulse(&mut self, vx: f32, vy: f32) {
        for i in 0..NUM_POINTS {
            // Verlet velocity = pos - old_pos, so shift old_pos
            self.old_pos[i][0] -= vx;
            self.old_pos[i][1] -= vy;
        }
    }

    /// Apply a "jiggle" — random-ish impulse that creates lively bouncy motion.
    /// `strength` is the magnitude in pixels/frame of the impulse.
    /// Uses a simple deterministic pattern rather than true randomness.
    pub fn jiggle(&mut self, strength: f32, seed: u32) {
        for i in 0..NUM_POINTS {
            // Simple hash-based pseudo-random direction per point
            let h = hash(seed.wrapping_mul(31).wrapping_add(i as u32));
            let angle = (h as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            let vx = angle.cos() * strength;
            let vy = angle.sin() * strength;
            self.old_pos[i][0] -= vx;
            self.old_pos[i][1] -= vy;
        }
    }

    /// Apply a radial "pop" impulse from the center outward.
    /// Points further from center get proportionally larger impulses,
    /// creating a satisfying expansion/wobble. Great for menu open animations.
    /// `strength` is in pixels — the outermost points receive this full amount.
    pub fn pop(&mut self, strength: f32) {
        let cx = (self.rest[0][0] + self.rest[NUM_POINTS - 1][0]) * 0.5;
        let cy = (self.rest[0][1] + self.rest[NUM_POINTS - 1][1]) * 0.5;
        // Find the max distance from center for normalization
        let max_d = {
            let dx = self.rest[0][0] - cx;
            let dy = self.rest[0][1] - cy;
            (dx * dx + dy * dy).sqrt().max(1.0)
        };
        for i in 0..NUM_POINTS {
            let dx = self.pos[i][0] - cx;
            let dy = self.pos[i][1] - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d < 0.1 {
                continue;
            }
            // Scale by distance: outer points move more
            let scale = (d / max_d) * strength;
            self.old_pos[i][0] -= (dx / d) * scale;
            self.old_pos[i][1] -= (dy / d) * scale;
        }
    }

    /// Evaluate the bicubic Bernstein FFD at normalized coordinates `(s, t)`
    /// both in `[0, 1]`. Returns the deformed screen-space position.
    pub fn eval(&self, s: f32, t: f32) -> [f32; 2] {
        let mut result = [0.0f32; 2];
        for j in 0..GRID_SIZE {
            let bj = bernstein3(j, t);
            for i in 0..GRID_SIZE {
                let bi = bernstein3(i, s);
                let w = bi * bj;
                let idx = j * GRID_SIZE + i;
                result[0] += self.pos[idx][0] * w;
                result[1] += self.pos[idx][1] * w;
            }
        }
        result
    }

    /// Map a screen-space point within the widget rect to normalized `(s, t)`.
    ///
    /// Uses the rest (undeformed) positions to compute the mapping.
    pub fn screen_to_normalized(&self, px: f32, py: f32) -> (f32, f32) {
        let min_x = self.rest[0][0];
        let min_y = self.rest[0][1];
        let max_x = self.rest[NUM_POINTS - 1][0];
        let max_y = self.rest[NUM_POINTS - 1][1];
        let w = max_x - min_x;
        let h = max_y - min_y;
        let s = if w > 1e-6 { (px - min_x) / w } else { 0.0 };
        let t = if h > 1e-6 { (py - min_y) / h } else { 0.0 };
        (s, t)
    }

    /// Returns the rest-position bounding rect `[x, y, w, h]`.
    pub fn rest_rect(&self) -> [f32; 4] {
        let min_x = self.rest[0][0];
        let min_y = self.rest[0][1];
        let max_x = self.rest[NUM_POINTS - 1][0];
        let max_y = self.rest[NUM_POINTS - 1][1];
        [min_x, min_y, max_x - min_x, max_y - min_y]
    }

    /// Returns `true` if the sim is nearly at rest (all points close to rest positions).
    pub fn is_at_rest(&self, threshold: f32) -> bool {
        let thresh2 = threshold * threshold;
        for i in 0..NUM_POINTS {
            let dx = self.pos[i][0] - self.rest[i][0];
            let dy = self.pos[i][1] - self.rest[i][1];
            if dx * dx + dy * dy > thresh2 {
                return false;
            }
        }
        true
    }
}

impl Constraint {
    fn between(positions: &[[f32; 2]; NUM_POINTS], a: usize, b: usize) -> Self {
        Self {
            a,
            b,
            rest_len: dist(&positions[a], &positions[b]),
        }
    }
}

/// Cubic Bernstein basis function: C(3, i) * t^i * (1-t)^(3-i)
fn bernstein3(i: usize, t: f32) -> f32 {
    let u = 1.0 - t;
    match i {
        0 => u * u * u,
        1 => 3.0 * t * u * u,
        2 => 3.0 * t * t * u,
        3 => t * t * t,
        _ => 0.0,
    }
}

fn dist(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

/// Simple u32 hash for deterministic pseudo-random values.
fn hash(mut x: u32) -> u32 {
    x = x.wrapping_mul(0x9E3779B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x45D9F3B);
    x ^= x >> 16;
    x
}

/// Number of subdivisions per axis when tessellating a quad through the FFD.
/// Higher = smoother deformation, more vertices.
pub const FFD_SUBDIVISIONS: usize = 8;

/// Tessellate a rectangular region through an FFD, producing deformed quads.
///
/// Takes a source rect `[x, y, w, h]` and UV rect `[u0, v0, u1, v1]`,
/// subdivides into an `FFD_SUBDIVISIONS × FFD_SUBDIVISIONS` grid, and warps
/// each vertex position through the FFD while interpolating UVs linearly.
///
/// Returns a list of `(position, uv)` vertices and triangle indices.
pub fn tessellate_through_ffd(
    ffd: &FfdSim,
    rect: [f32; 4],
    uvs: [f32; 4],
) -> (Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<u32>) {
    let [rx, ry, rw, rh] = rect;
    let [u0, v0, u1, v1] = uvs;
    let n = FFD_SUBDIVISIONS;
    let verts = (n + 1) * (n + 1);

    let mut positions = Vec::with_capacity(verts);
    let mut tex_coords = Vec::with_capacity(verts);

    for j in 0..=n {
        let t_frac = j as f32 / n as f32;
        let py = ry + t_frac * rh;
        for i in 0..=n {
            let s_frac = i as f32 / n as f32;
            let px = rx + s_frac * rw;

            // Map screen position to FFD normalized coords
            let (s, t) = ffd.screen_to_normalized(px, py);
            // Evaluate FFD to get deformed position
            let deformed = ffd.eval(s, t);
            positions.push(deformed);

            // Linearly interpolate UVs
            let u = u0 + s_frac * (u1 - u0);
            let v = v0 + t_frac * (v1 - v0);
            tex_coords.push([u, v]);
        }
    }

    // Build triangle indices
    let mut indices = Vec::with_capacity(n * n * 6);
    for j in 0..n {
        for i in 0..n {
            let tl = (j * (n + 1) + i) as u32;
            let tr = tl + 1;
            let bl = ((j + 1) * (n + 1) + i) as u32;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, tr, br, tl, br, bl]);
        }
    }

    (positions, tex_coords, indices)
}
