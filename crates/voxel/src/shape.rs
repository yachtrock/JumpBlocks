use bevy::prelude::*;

/// Global chamfer fillet radius in world units.  Every rounded edge and
/// corner has this radius; the setback (cut distance along each face) is
/// derived from it per edge based on the dihedral angle.
pub const CHAMFER_WIDTH: f32 = 0.12;

/// Cardinal facing direction for blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Facing {
    #[default]
    North = 0, // +Z
    East = 1,  // +X
    South = 2, // -Z
    West = 3,  // -X
}

impl Facing {
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0 => Facing::North,
            1 => Facing::East,
            2 => Facing::South,
            3 => Facing::West,
            _ => unreachable!(),
        }
    }

    /// Returns the Y-axis rotation angle in radians for this facing.
    pub fn rotation_radians(&self) -> f32 {
        match self {
            Facing::North => 0.0,
            Facing::East => -std::f32::consts::FRAC_PI_2,
            Facing::South => std::f32::consts::PI,
            Facing::West => std::f32::consts::FRAC_PI_2,
        }
    }

    /// Rotate a point in block-local space by this facing.
    /// For a 1x1x1 block, coordinates are in 0..1 and rotation is around (0.5, y, 0.5).
    /// For larger blocks, use `rotate_block_point` with the block size.
    pub fn rotate_point(&self, p: Vec3) -> Vec3 {
        self.rotate_block_point(p, (1, 1, 1))
    }

    /// Rotate a point in block-local space for a block of given size.
    /// Rotation is around the center of the block's XZ footprint.
    pub fn rotate_block_point(&self, p: Vec3, size: (u8, u8, u8)) -> Vec3 {
        let cx_size = size.0 as f32;
        let cz_size = size.2 as f32;
        let half_x = cx_size * 0.5;
        let half_z = cz_size * 0.5;
        let cx = p.x - half_x;
        let cz = p.z - half_z;
        let (rx, rz, new_half_x, new_half_z) = match self {
            Facing::North => (cx, cz, half_x, half_z),
            Facing::East => (-cz, cx, half_z, half_x),
            Facing::South => (-cx, -cz, half_x, half_z),
            Facing::West => (cz, -cx, half_z, half_x),
        };
        Vec3::new(rx + new_half_x, p.y, rz + new_half_z)
    }

    /// Return the inverse facing (undoes this rotation).
    pub fn inverse(&self) -> Facing {
        match self {
            Facing::North => Facing::North,
            Facing::East => Facing::West,
            Facing::South => Facing::South,
            Facing::West => Facing::East,
        }
    }

    /// Rotate a normal vector by this facing.
    pub fn rotate_normal(&self, n: Vec3) -> Vec3 {
        match self {
            Facing::North => n,
            Facing::East => Vec3::new(-n.z, n.y, n.x),
            Facing::South => Vec3::new(-n.x, n.y, -n.z),
            Facing::West => Vec3::new(n.z, n.y, -n.x),
        }
    }
}

/// Which side of a cell a face is on, used for neighbor occlusion checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceSide {
    Top,    // +Y
    Bottom, // -Y
    North,  // +Z
    South,  // -Z
    East,   // +X
    West,   // -X
    /// Face is internal or diagonal — no simple neighbor occlusion.
    None,
}

impl FaceSide {
    pub fn neighbor_offset(&self) -> Option<(i32, i32, i32)> {
        match self {
            FaceSide::Top => Some((0, 1, 0)),
            FaceSide::Bottom => Some((0, -1, 0)),
            FaceSide::North => Some((0, 0, 1)),
            FaceSide::South => Some((0, 0, -1)),
            FaceSide::East => Some((1, 0, 0)),
            FaceSide::West => Some((-1, 0, 0)),
            FaceSide::None => None,
        }
    }

    /// Rotate a face side by a facing direction.
    pub fn rotated_by(&self, facing: Facing) -> FaceSide {
        match self {
            FaceSide::Top | FaceSide::Bottom | FaceSide::None => *self,
            _ => {
                let steps = facing as u8;
                let base = match self {
                    FaceSide::North => 0u8,
                    FaceSide::East => 1,
                    FaceSide::South => 2,
                    FaceSide::West => 3,
                    _ => unreachable!(),
                };
                match (base + 4 - steps) % 4 {
                    0 => FaceSide::North,
                    1 => FaceSide::East,
                    2 => FaceSide::South,
                    3 => FaceSide::West,
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn opposite(&self) -> FaceSide {
        match self {
            FaceSide::Top => FaceSide::Bottom,
            FaceSide::Bottom => FaceSide::Top,
            FaceSide::North => FaceSide::South,
            FaceSide::South => FaceSide::North,
            FaceSide::East => FaceSide::West,
            FaceSide::West => FaceSide::East,
            FaceSide::None => FaceSide::None,
        }
    }
}

/// An edge of a block face, defined by two vertex indices into the face's vertex list.
#[derive(Debug, Clone)]
pub struct VoxelEdge {
    /// Index of the start vertex in the parent face's vertices.
    pub v0: usize,
    /// Index of the end vertex in the parent face's vertices.
    pub v1: usize,
    /// Which neighbor sides this edge borders (for chamfer decisions).
    /// The edge is NOT chamfered if ANY of these neighbors occludes toward us.
    /// An empty vec means always chamfer (internal/diagonal edges).
    pub neighbor_sides: Vec<FaceSide>,
}

/// How much of a cell-side a face covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coverage {
    /// Face fully covers this cell-side (full quad). Occludes any neighbor coverage.
    Full,
    /// Face partially covers this cell-side. Only occludes a neighbor with the
    /// same partial ID (matching shapes that together fill the cell boundary).
    Partial(u8),
}

/// Well-known partial coverage IDs.
/// Wedge upper side triangle (the triangular portion of a pentagon side face).
pub const PARTIAL_WEDGE_UPPER_TRI: u8 = 1;
/// Wedge front face — fully covers its cells but the slope above exposes the
/// boundary from certain angles, so it must not be mutually culled with cubes.
pub const PARTIAL_WEDGE_FRONT: u8 = 2;
/// Sentinel partial ID that never matches another partial: the face is only
/// culled when a neighbor covers the whole cell-side. Used for the sloped
/// portions of slope-cap side faces, whose exact cross-section depends on the
/// corner heights — partially-overlapping contacts are resolved by contact
/// clipping instead of coverage matching.
pub const PARTIAL_NEVER_MATCH: u8 = 255;

/// Which cell within a block a face covers, and on which side.
#[derive(Debug, Clone)]
pub struct CellCover {
    /// Cell position within the block (0..size.x, 0..size.y, 0..size.z).
    pub cell: (u8, u8, u8),
    /// Which side of that cell this face covers.
    pub side: FaceSide,
    /// How much of this cell-side the face covers.
    pub coverage: Coverage,
}

/// A face of a block shape.
#[derive(Debug, Clone)]
pub struct BlockFace {
    /// Vertices in block-local space (x in 0..size.0, y in 0..size.1, z in 0..size.2).
    pub vertices: Vec<Vec3>,
    /// Triangle indices (into `vertices`).
    pub triangles: Vec<[usize; 3]>,
    /// The face normal in block-local space.
    pub normal: Vec3,
    /// Edges of this face, used for chamfering.
    pub edges: Vec<VoxelEdge>,
    /// Which cells this face covers and on which sides (for occlusion).
    /// Empty means diagonal/internal — never culled by neighbor check.
    pub cell_coverage: Vec<CellCover>,
}

/// A block shape definition.
#[derive(Debug, Clone)]
pub struct BlockShape {
    pub name: String,
    /// Dimensions of this block in cells.
    pub size: (u8, u8, u8),
    pub faces: Vec<BlockFace>,
    /// Which cells this shape occupies (relative to origin).
    pub occupied_cells: Vec<(u8, u8, u8)>,
}

/// Resource holding all registered block shapes. Index 0 is always the cube.
#[derive(Resource, Debug, Clone)]
pub struct ShapeTable {
    pub shapes: Vec<BlockShape>,
}

impl Default for ShapeTable {
    fn default() -> Self {
        let mut table = Self {
            shapes: Vec::new(),
        };
        table.shapes.push(cube_shape());    // shape 0: cube
        table.shapes.push(wedge_shape());   // shape 1: wedge/ramp (1:2)
        // Slope family (see `slope_cap_shape`): corner variants of the 1:2
        // wedge, and a steep 1:1 wedge with its corners. Canonical
        // orientation: high side/corner at -X/-Z, descending toward +X/+Z.
        table.shapes.push(slope_cap_shape("wedge_outer", [2, 1, 1, 1]));       // 2
        table.shapes.push(slope_cap_shape("wedge_inner", [2, 2, 2, 1]));       // 3
        table.shapes.push(slope_cap_shape("wedge_steep", [3, 3, 1, 1]));       // 4
        table.shapes.push(slope_cap_shape("wedge_steep_outer", [3, 1, 1, 1])); // 5
        table.shapes.push(slope_cap_shape("wedge_steep_inner", [3, 3, 3, 1])); // 6
        table
    }
}

/// Well-known shape indices.
pub const SHAPE_CUBE: u16 = 0;
pub const SHAPE_WEDGE: u16 = 1;
pub const SHAPE_WEDGE_OUTER: u16 = 2;
pub const SHAPE_WEDGE_INNER: u16 = 3;
pub const SHAPE_WEDGE_STEEP: u16 = 4;
pub const SHAPE_WEDGE_STEEP_OUTER: u16 = 5;
pub const SHAPE_WEDGE_STEEP_INNER: u16 = 6;

/// All slope-cap shape ids (everything that smooths a terrain step),
/// including the classic 1:2 wedge.
pub const SLOPE_SHAPES: [u16; 6] = [
    SHAPE_WEDGE,
    SHAPE_WEDGE_OUTER,
    SHAPE_WEDGE_INNER,
    SHAPE_WEDGE_STEEP,
    SHAPE_WEDGE_STEEP_OUTER,
    SHAPE_WEDGE_STEEP_INNER,
];

impl ShapeTable {
    pub fn get(&self, index: u16) -> Option<&BlockShape> {
        self.shapes.get(index as usize)
    }

    pub fn register(&mut self, shape: BlockShape) -> u16 {
        let idx = self.shapes.len() as u16;
        self.shapes.push(shape);
        idx
    }
}

// ---------------------------------------------------------------------------
// Slope-cap shape generator
// ---------------------------------------------------------------------------
//
// A slope cap is a terrain-smoothing block: a solid 1-cell base slab with a
// sloped top surface over a 2×2-cell footprint. The top surface is defined
// by its four corner heights (in cells, ≥1) at (x,z) ∈ {0,2}², folded along
// the (0,0)→(2,2) diagonal. Corner heights express the whole family:
//
//   [2,2,1,1]  straight 1:2 wedge   (equivalent to the hand-made SHAPE_WEDGE)
//   [2,1,1,1]  1:2 outer corner     (convex hill corner)
//   [2,2,2,1]  1:2 inner corner     (concave valley corner)
//   [3,3,1,1]  steep 1:1 wedge
//   [3,1,1,1]  steep outer corner
//   [3,3,3,1]  steep inner corner
//
// Canonical orientation: high side/corner at -X/-Z; `Facing` rotates it.

/// Top-surface height at (x, z) for corner heights
/// `[h00 (0,0), h10 (2,0), h01 (0,2), h11 (2,2)]`,
/// piecewise-linear with a fold along the (0,0)→(2,2) diagonal.
pub fn cap_height(h: [u8; 4], x: f32, z: f32) -> f32 {
    let [h00, h10, h01, h11] = h.map(|v| v as f32);
    if x >= z {
        h00 + (h10 - h00) * ((x - z) * 0.5) + (h11 - h00) * (z * 0.5)
    } else {
        h00 + (h01 - h00) * ((z - x) * 0.5) + (h11 - h00) * (x * 0.5)
    }
}

/// Build a face from a convex polygon: fan-triangulated, wound so that the
/// mesher's normal convention (`-(b-a)×(c-a)` points outward) holds.
fn polygon_face(mut verts: Vec<Vec3>, normal: Vec3, cell_coverage: Vec<CellCover>) -> BlockFace {
    // Newell normal of the polygon as wound
    let mut newell = Vec3::ZERO;
    for i in 0..verts.len() {
        let a = verts[i];
        let b = verts[(i + 1) % verts.len()];
        newell += Vec3::new(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    // The engine treats a triangle's outward normal as -(b-a)×(c-a), which
    // is the NEGATED Newell direction — so wind the polygon such that its
    // Newell normal points OPPOSITE the desired outward normal.
    if newell.dot(normal) > 0.0 {
        verts.reverse();
    }
    // Ear-clip triangulation covering EVERY vertex. Side polygons carry
    // collinear vertices (at every integer height, for exact seams against
    // stacked cubes); the chamfer pipeline re-tessellates faces using these
    // triangle indices at positions where collinearity no longer holds, so
    // a vertex left out of the triangulation becomes a T-junction crack.
    // Greedy rule: whenever a positive-area ear whose middle vertex is a
    // straight-run interior vertex exists, cut it first — this consumes
    // collinear vertices as soon as their ears gain area instead of
    // stranding them in a degenerate final triple.
    let n = verts.len();
    let area2 = |a: usize, b: usize, c: usize| -> f32 {
        (verts[b] - verts[a]).cross(verts[c] - verts[a]).length_squared()
    };
    let run_interior: Vec<bool> = (0..n)
        .map(|i| area2((i + n - 1) % n, i, (i + 1) % n) <= 1e-10)
        .collect();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    // Newell-based area of the remaining ring, for the stranding check.
    let ring_area2 = |ring: &[usize]| -> f32 {
        let mut nw = Vec3::ZERO;
        for i in 0..ring.len() {
            let a = verts[ring[i]];
            let b = verts[ring[(i + 1) % ring.len()]];
            nw += Vec3::new(
                (a.y - b.y) * (a.z + b.z),
                (a.z - b.z) * (a.x + b.x),
                (a.x - b.x) * (a.y + b.y),
            );
        }
        nw.length_squared()
    };
    while idx.len() >= 3 {
        let m = idx.len();
        let ear_at = |k: usize| -> ([usize; 3], f32) {
            let t = [idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]];
            (t, area2(t[0], t[1], t[2]))
        };
        // Cutting an ear must never leave a zero-area ring (that strands
        // any remaining run-interior vertices in degenerate triples).
        let strands = |k: usize| -> bool {
            if m == 3 {
                return false;
            }
            let mut rest: Vec<usize> = idx.clone();
            rest.remove(k);
            ring_area2(&rest) <= 1e-12
        };
        let mut cut = None;
        // Prefer positive ears that consume a straight-run interior vertex
        for k in 0..m {
            let (t, a) = ear_at(k);
            if a > 1e-10 && run_interior[idx[k]] && !strands(k) {
                cut = Some((k, t));
                break;
            }
        }
        if cut.is_none() {
            for k in 0..m {
                let (t, a) = ear_at(k);
                if a > 1e-10 && !strands(k) {
                    cut = Some((k, t));
                    break;
                }
            }
        }
        let Some((k, tri)) = cut else { break };
        triangles.push(tri);
        if m == 3 {
            break;
        }
        idx.remove(k);
    }
    debug_assert!(
        {
            let mut seen = vec![false; n];
            for t in &triangles {
                for &v in t {
                    seen[v] = true;
                }
            }
            seen.iter().all(|&s| s)
        },
        "polygon_face: triangulation dropped a vertex"
    );
    BlockFace {
        vertices: verts,
        triangles,
        normal,
        edges: Vec::new(),
        cell_coverage,
    }
}

/// Generate a slope cap from its four top-corner heights (see module notes).
pub fn slope_cap_shape(name: &str, h: [u8; 4]) -> BlockShape {
    let [h00, h10, h01, h11] = h;
    let max_h = h.iter().copied().max().unwrap();
    assert!(h.iter().all(|&v| v >= 1), "cap corners need >= 1 cell of base");

    let mut faces: Vec<BlockFace> = Vec::new();

    // -- Bottom -------------------------------------------------------------
    faces.push(polygon_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
        ],
        Vec3::NEG_Y,
        vec![
            CellCover { cell: (0, 0, 0), side: FaceSide::Bottom, coverage: Coverage::Full },
            CellCover { cell: (1, 0, 0), side: FaceSide::Bottom, coverage: Coverage::Full },
            CellCover { cell: (0, 0, 1), side: FaceSide::Bottom, coverage: Coverage::Full },
            CellCover { cell: (1, 0, 1), side: FaceSide::Bottom, coverage: Coverage::Full },
        ],
    ));

    // -- Sides ---------------------------------------------------------------
    // Each side runs between two top corners; its polygon is the bottom edge,
    // the two vertical corner edges (with vertices at every integer Y for
    // exact seams against stacked cubes), and the straight top edge.
    struct Side {
        /// Polygon corners a→b along the side (block-local XZ).
        a: (f32, f32),
        b: (f32, f32),
        ha: u8,
        hb: u8,
        normal: Vec3,
        face_side: FaceSide,
        /// Cell coordinates covered by this side per (step along a→b, layer).
        cells: [(u8, u8); 2],
    }
    let sides = [
        // South (-Z): from (0,0) to (2,0)
        Side { a: (0.0, 0.0), b: (2.0, 0.0), ha: h00, hb: h10, normal: Vec3::NEG_Z, face_side: FaceSide::South, cells: [(0, 0), (1, 0)] },
        // East (+X): from (2,0) to (2,2)
        Side { a: (2.0, 0.0), b: (2.0, 2.0), ha: h10, hb: h11, normal: Vec3::X, face_side: FaceSide::East, cells: [(1, 0), (1, 1)] },
        // North (+Z): from (2,2) to (0,2)
        Side { a: (2.0, 2.0), b: (0.0, 2.0), ha: h11, hb: h01, normal: Vec3::Z, face_side: FaceSide::North, cells: [(1, 1), (0, 1)] },
        // West (-X): from (0,2) to (0,0)
        Side { a: (0.0, 2.0), b: (0.0, 0.0), ha: h01, hb: h00, normal: Vec3::NEG_X, face_side: FaceSide::West, cells: [(0, 1), (0, 0)] },
    ];

    for s in sides {
        let mut verts: Vec<Vec3> = Vec::new();
        let at = |p: (f32, f32), y: f32| Vec3::new(p.0, y, p.1);
        // Bottom edge a → b
        verts.push(at(s.a, 0.0));
        verts.push(at(s.b, 0.0));
        // Up the b corner through integer heights
        for y in 1..=s.hb {
            verts.push(at(s.b, y as f32));
        }
        // Top edge b → a is straight; midpoint at integer height gets a
        // vertex so neighboring cube stacks meet real vertices.
        let mid_h = (s.ha as f32 + s.hb as f32) * 0.5;
        if mid_h.fract() == 0.0 && s.ha != s.hb {
            let mid = ((s.a.0 + s.b.0) * 0.5, (s.a.1 + s.b.1) * 0.5);
            verts.push(at(mid, mid_h));
        }
        // Down the a corner through integer heights
        for y in (1..=s.ha).rev() {
            verts.push(at(s.a, y as f32));
        }

        // Full coverage where the side spans the whole cell; the partially
        // covered rows under the sloped top edge get never-matching partial
        // coverage so the face stays alive when that region is exposed
        // (contact clipping trims whatever part rests against a neighbor).
        let mut coverage = Vec::new();
        for (step, &(cx, cz)) in s.cells.iter().enumerate() {
            // Height of the side profile at this cell's two x/z extents
            let t0 = step as f32 * 0.5;
            let t1 = t0 + 0.5;
            let h_at = |t: f32| s.ha as f32 + (s.hb as f32 - s.ha as f32) * t;
            let hmin = h_at(t0).min(h_at(t1));
            let hmax = h_at(t0).max(h_at(t1));
            for cy in 0..(hmin.floor() as u8) {
                coverage.push(CellCover {
                    cell: (cx, cy, cz),
                    side: s.face_side,
                    coverage: Coverage::Full,
                });
            }
            for cy in (hmin.floor() as u8)..(hmax.ceil() as u8) {
                coverage.push(CellCover {
                    cell: (cx, cy, cz),
                    side: s.face_side,
                    coverage: Coverage::Partial(PARTIAL_NEVER_MATCH),
                });
            }
        }

        faces.push(polygon_face(verts, s.normal, coverage));
    }

    // -- Top surface ---------------------------------------------------------
    let t00 = Vec3::new(0.0, h00 as f32, 0.0);
    let t10 = Vec3::new(2.0, h10 as f32, 0.0);
    let t01 = Vec3::new(0.0, h01 as f32, 2.0);
    let t11 = Vec3::new(2.0, h11 as f32, 2.0);
    // Walk an edge a→b, appending `a` plus a midpoint vertex when the side
    // faces inserted one on the same edge (integer mid height) — shared
    // boundary vertices must exist in BOTH adjacent faces.
    let push_edge = |verts: &mut Vec<Vec3>, a: Vec3, b: Vec3| {
        verts.push(a);
        let mid_h = (a.y + b.y) * 0.5;
        if mid_h.fract() == 0.0 && a.y != b.y {
            verts.push(Vec3::new((a.x + b.x) * 0.5, mid_h, (a.z + b.z) * 0.5));
        }
    };
    let planar = (h00 as i32 - h10 as i32 - h01 as i32 + h11 as i32) == 0;
    let tri_normal = |a: Vec3, b: Vec3, c: Vec3| -> Vec3 { (c - a).cross(b - a).normalize() };
    if planar {
        let n = tri_normal(t00, t10, t11);
        let mut verts = Vec::new();
        push_edge(&mut verts, t00, t10);
        push_edge(&mut verts, t10, t11);
        push_edge(&mut verts, t11, t01);
        push_edge(&mut verts, t01, t00);
        faces.push(polygon_face(verts, n, Vec::new()));
    } else {
        // Two triangles folded along the (0,0)→(2,2) diagonal; only their
        // outer edges get midpoints (the shared diagonal stays one segment).
        let n1 = tri_normal(t00, t10, t11);
        let mut v1 = Vec::new();
        push_edge(&mut v1, t00, t10);
        push_edge(&mut v1, t10, t11);
        v1.push(t11);
        faces.push(polygon_face(v1, n1, Vec::new()));
        let n2 = tri_normal(t00, t11, t01);
        let mut v2 = vec![t00];
        push_edge(&mut v2, t11, t01);
        push_edge(&mut v2, t01, t00);
        faces.push(polygon_face(v2, n2, Vec::new()));
    }

    // -- Occupied cells -------------------------------------------------------
    // A cell is occupied if the top surface rises above the cell's floor
    // anywhere over its footprint (checked at the subcell corners — the
    // piecewise-linear surface attains its extrema there).
    let mut occupied_cells = Vec::new();
    for cy in 0..max_h {
        for cx in 0..2u8 {
            for cz in 0..2u8 {
                let mut m: f32 = 0.0;
                for (dx, dz) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                    m = m.max(cap_height(h, cx as f32 + dx, cz as f32 + dz));
                }
                if m > cy as f32 + 0.001 {
                    occupied_cells.push((cx, cy, cz));
                }
            }
        }
    }

    BlockShape {
        name: name.to_string(),
        size: (2, max_h, 2),
        faces,
        occupied_cells,
    }
}

/// A shape's occupied cells rotated by a facing — needed when writing a
/// rotated block's cells into the chunk grid (matches the cell rotation the
/// mesher uses for occlusion checks).
pub fn rotated_occupied_cells(shape: &BlockShape, facing: Facing) -> Vec<(u8, u8, u8)> {
    if facing == Facing::North {
        return shape.occupied_cells.clone();
    }
    shape
        .occupied_cells
        .iter()
        .map(|&(x, y, z)| {
            let center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
            let r = facing.rotate_block_point(center, shape.size);
            (r.x.floor() as u8, r.y.floor() as u8, r.z.floor() as u8)
        })
        .collect()
}

/// Shape 0: the standard full cube (2x1x2).
fn cube_shape() -> BlockShape {
    let v = [
        Vec3::new(0.0, 0.0, 0.0), // 0: left  bottom back
        Vec3::new(2.0, 0.0, 0.0), // 1: right bottom back
        Vec3::new(0.0, 0.0, 2.0), // 2: left  bottom front
        Vec3::new(2.0, 0.0, 2.0), // 3: right bottom front
        Vec3::new(0.0, 1.0, 0.0), // 4: left  top    back
        Vec3::new(2.0, 1.0, 0.0), // 5: right top    back
        Vec3::new(0.0, 1.0, 2.0), // 6: left  top    front
        Vec3::new(2.0, 1.0, 2.0), // 7: right top    front
    ];

    let faces = vec![
        // Top (+Y)
        BlockFace {
            vertices: vec![v[4], v[5], v[7], v[6]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::Y,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::North] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 0), side: FaceSide::Top, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 0), side: FaceSide::Top, coverage: Coverage::Full },
                CellCover { cell: (0, 0, 1), side: FaceSide::Top, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 1), side: FaceSide::Top, coverage: Coverage::Full },
            ],
        },
        // Bottom (-Y)
        BlockFace {
            vertices: vec![v[2], v[3], v[1], v[0]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Y,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::North] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 0), side: FaceSide::Bottom, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 0), side: FaceSide::Bottom, coverage: Coverage::Full },
                CellCover { cell: (0, 0, 1), side: FaceSide::Bottom, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 1), side: FaceSide::Bottom, coverage: Coverage::Full },
            ],
        },
        // North (+Z)
        BlockFace {
            vertices: vec![v[3], v[2], v[6], v[7]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::Z,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::West] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::East] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 1), side: FaceSide::North, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 1), side: FaceSide::North, coverage: Coverage::Full },
            ],
        },
        // South (-Z)
        BlockFace {
            vertices: vec![v[0], v[1], v[5], v[4]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Z,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 0), side: FaceSide::South, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 0), side: FaceSide::South, coverage: Coverage::Full },
            ],
        },
        // East (+X)
        BlockFace {
            vertices: vec![v[1], v[3], v[7], v[5]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::X,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::North] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::South] },
            ],
            cell_coverage: vec![
                CellCover { cell: (1, 0, 0), side: FaceSide::East, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 1), side: FaceSide::East, coverage: Coverage::Full },
            ],
        },
        // West (-X)
        BlockFace {
            vertices: vec![v[2], v[0], v[4], v[6]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_X,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::North] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 0), side: FaceSide::West, coverage: Coverage::Full },
                CellCover { cell: (0, 0, 1), side: FaceSide::West, coverage: Coverage::Full },
            ],
        },
    ];

    BlockShape {
        name: "cube".to_string(),
        size: (2, 1, 2),
        faces,
        occupied_cells: vec![
            (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
        ],
    }
}

/// Shape 1: wedge/ramp (2x2x2) with solid base.
///
/// Bottom half is a solid cube base. Top half has a slope that descends
/// from back (south, z=0) at y=2 to front (north, z=2) at y=1.
/// Every edge has at least 1 cell of material.
fn wedge_shape() -> BlockShape {
    // All faces have vertices at every cell boundary to prevent T-junctions
    // with adjacent cube blocks (2x1x2). v[8]/v[9] are at y=1 midpoints.
    let v = [
        Vec3::new(0.0, 0.0, 0.0), //  0: left  bottom back
        Vec3::new(2.0, 0.0, 0.0), //  1: right bottom back
        Vec3::new(0.0, 0.0, 2.0), //  2: left  bottom front
        Vec3::new(2.0, 0.0, 2.0), //  3: right bottom front
        Vec3::new(0.0, 1.0, 2.0), //  4: left  mid    front
        Vec3::new(2.0, 1.0, 2.0), //  5: right mid    front
        Vec3::new(0.0, 2.0, 0.0), //  6: left  top    back
        Vec3::new(2.0, 2.0, 0.0), //  7: right top    back
        Vec3::new(0.0, 1.0, 0.0), //  8: left  mid    back  (y=1 boundary)
        Vec3::new(2.0, 1.0, 0.0), //  9: right mid    back  (y=1 boundary)
    ];

    let faces = vec![
        // Bottom (-Y): full 2x2 quad at y=0
        BlockFace {
            vertices: vec![v[2], v[3], v[1], v[0]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Y,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::North] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 0), side: FaceSide::Bottom, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 0), side: FaceSide::Bottom, coverage: Coverage::Full },
                CellCover { cell: (0, 0, 1), side: FaceSide::Bottom, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 1), side: FaceSide::Bottom, coverage: Coverage::Full },
            ],
        },
        // South/Back wall (-Z): 6-vertex polygon with y=1 midpoints
        // v[0]=(0,0,0), v[1]=(2,0,0), v[9]=(2,1,0), v[7]=(2,2,0), v[6]=(0,2,0), v[8]=(0,1,0)
        BlockFace {
            vertices: vec![v[0], v[1], v[9], v[7], v[6], v[8]],
            triangles: vec![[0, 1, 2], [0, 2, 5], [5, 2, 3], [5, 3, 4]],
            normal: Vec3::NEG_Z,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::East] },
                // Ridge (v[7]→v[6]) omitted — internal edge shared with slope
                VoxelEdge { v0: 4, v1: 5, neighbor_sides: vec![FaceSide::West] },
                VoxelEdge { v0: 5, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 0), side: FaceSide::South, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 0), side: FaceSide::South, coverage: Coverage::Full },
                CellCover { cell: (0, 1, 0), side: FaceSide::South, coverage: Coverage::Full },
                CellCover { cell: (1, 1, 0), side: FaceSide::South, coverage: Coverage::Full },
            ],
        },
        // North/Front wall (+Z): short front z=2, y=0..1
        BlockFace {
            vertices: vec![v[3], v[2], v[4], v[5]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::Z,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::West] },
                // Top edge shared with slope — omitted
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::East] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 1), side: FaceSide::North, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 1), side: FaceSide::North, coverage: Coverage::Full },
            ],
        },
        // Slope face: from back-top to front-mid — diagonal, never culled
        BlockFace {
            vertices: vec![v[6], v[7], v[5], v[4]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::new(0.0, 2.0, 1.0).normalize(),
            edges: vec![
                // Ridge (top/back) omitted — internal edge shared with south face
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                // Bottom/front edge shared with north face — omitted
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            cell_coverage: vec![], // diagonal — never culled
        },
        // West (-X): 5-vertex pentagon with y=1 midpoint
        // v[2]=(0,0,2), v[0]=(0,0,0), v[8]=(0,1,0), v[6]=(0,2,0), v[4]=(0,1,2)
        BlockFace {
            vertices: vec![v[2], v[0], v[8], v[6], v[4]],
            triangles: vec![[0, 1, 2], [0, 2, 4], [2, 3, 4]],
            normal: Vec3::NEG_X,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 3, v1: 4, neighbor_sides: vec![FaceSide::West] },
                VoxelEdge { v0: 4, v1: 0, neighbor_sides: vec![FaceSide::North] },
            ],
            cell_coverage: vec![
                CellCover { cell: (0, 0, 0), side: FaceSide::West, coverage: Coverage::Full },
                CellCover { cell: (0, 0, 1), side: FaceSide::West, coverage: Coverage::Full },
                CellCover { cell: (0, 1, 0), side: FaceSide::West, coverage: Coverage::Partial(PARTIAL_WEDGE_UPPER_TRI) },
                CellCover { cell: (0, 1, 1), side: FaceSide::West, coverage: Coverage::Partial(PARTIAL_WEDGE_UPPER_TRI) },
            ],
        },
        // East (+X): 5-vertex pentagon with y=1 midpoint
        // v[1]=(2,0,0), v[3]=(2,0,2), v[5]=(2,1,2), v[7]=(2,2,0), v[9]=(2,1,0)
        BlockFace {
            vertices: vec![v[1], v[3], v[5], v[7], v[9]],
            triangles: vec![[0, 1, 2], [0, 2, 4], [4, 2, 3]],
            normal: Vec3::X,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::North] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 3, v1: 4, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 4, v1: 0, neighbor_sides: vec![FaceSide::South] },
            ],
            cell_coverage: vec![
                CellCover { cell: (1, 0, 0), side: FaceSide::East, coverage: Coverage::Full },
                CellCover { cell: (1, 0, 1), side: FaceSide::East, coverage: Coverage::Full },
                CellCover { cell: (1, 1, 0), side: FaceSide::East, coverage: Coverage::Partial(PARTIAL_WEDGE_UPPER_TRI) },
                CellCover { cell: (1, 1, 1), side: FaceSide::East, coverage: Coverage::Partial(PARTIAL_WEDGE_UPPER_TRI) },
            ],
        },
    ];

    BlockShape {
        name: "wedge".to_string(),
        size: (2, 2, 2),
        faces,
        occupied_cells: vec![
            (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
            (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
        ],
    }
}
