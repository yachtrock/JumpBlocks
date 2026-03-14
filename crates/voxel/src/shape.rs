use bevy::prelude::*;

/// Global chamfer width in world units.
pub const CHAMFER_WIDTH: f32 = 0.06;

/// Cardinal facing direction for voxels.
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

    /// Rotate a point in normalized voxel space (0..1, 0..1, 0..1) by this facing
    /// around the center of the voxel (0.5, y, 0.5).
    pub fn rotate_point(&self, p: Vec3) -> Vec3 {
        let cx = p.x - 0.5;
        let cz = p.z - 0.5;
        let (rx, rz) = match self {
            Facing::North => (cx, cz),
            Facing::East => (-cz, cx),
            Facing::South => (-cx, -cz),
            Facing::West => (cz, -cx),
        };
        Vec3::new(rx + 0.5, p.y, rz + 0.5)
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

/// Which side of the voxel a face is on, used for neighbor occlusion checks.
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
}

/// How chamfer strip normals are computed for a face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChamferMode {
    /// Hard edge: chamfer strip gets a flat normal (average of face + outward).
    #[default]
    Hard,
    /// Smooth edge: chamfer strip normals interpolate from face normal (inner)
    /// to the edge outward direction (outer), creating a rounded appearance.
    Smooth,
}

/// An edge of a voxel face, defined by two vertex indices into the face's vertex list.
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

/// A face of a voxel shape.
#[derive(Debug, Clone)]
pub struct VoxelFace {
    /// Vertices in normalized voxel space (0..1, 0..1, 0..1 where Y=1 is VOXEL_HEIGHT).
    pub vertices: Vec<Vec3>,
    /// Triangle indices (into `vertices`).
    pub triangles: Vec<[usize; 3]>,
    /// The face normal in normalized voxel space.
    pub normal: Vec3,
    /// Which side of the voxel this face occupies (for neighbor occlusion).
    pub side: FaceSide,
    /// Edges of this face, used for chamfering.
    pub edges: Vec<VoxelEdge>,
    /// How chamfer normals blend on this face's edges.
    pub chamfer_mode: ChamferMode,
}

/// Which sides a shape fully occludes in its local frame.
/// Indexed as [Top, Bottom, North, South, East, West].
#[derive(Debug, Clone, Copy)]
pub struct OcclusionMask(pub [bool; 6]);

impl OcclusionMask {
    /// All sides occluded (full cube).
    pub const FULL: Self = Self([true; 6]);
    /// No sides occluded.
    pub const NONE: Self = Self([false; 6]);

    fn side_index(side: FaceSide) -> Option<usize> {
        match side {
            FaceSide::Top => Some(0),
            FaceSide::Bottom => Some(1),
            FaceSide::North => Some(2),
            FaceSide::South => Some(3),
            FaceSide::East => Some(4),
            FaceSide::West => Some(5),
            FaceSide::None => None,
        }
    }

    /// Check if a world-space side is occluded, accounting for the voxel's facing.
    /// `world_side` is the side to check in world space.
    /// `facing` is the voxel's facing direction.
    pub fn occludes_world_side(&self, world_side: FaceSide, facing: Facing) -> bool {
        // Un-rotate world side to local side
        let local_side = world_side.rotated_by(facing.inverse());
        if let Some(idx) = Self::side_index(local_side) {
            self.0[idx]
        } else {
            false
        }
    }
}

/// A voxel shape definition — a collection of faces.
#[derive(Debug, Clone)]
pub struct VoxelShape {
    pub name: String,
    pub faces: Vec<VoxelFace>,
    /// Which sides this shape fully occludes (in local/unrotated frame).
    pub occlusion: OcclusionMask,
}

/// Resource holding all registered voxel shapes. Index 0 is always the cube.
#[derive(Resource, Debug, Clone)]
pub struct ShapeTable {
    pub shapes: Vec<VoxelShape>,
}

impl Default for ShapeTable {
    fn default() -> Self {
        let mut table = Self {
            shapes: Vec::new(),
        };
        table.shapes.push(cube_shape());         // shape 0: hard-edge cube
        table.shapes.push(smooth_cube_shape());   // shape 1: smooth-edge cube
        table.shapes.push(wedge_shape());         // shape 2: wedge/ramp (hard chamfer)
        table.shapes.push(smooth_wedge_shape());  // shape 3: wedge/ramp (smooth fillet)
        table
    }
}

/// Well-known shape indices.
pub const SHAPE_CUBE: u16 = 0;
pub const SHAPE_SMOOTH_CUBE: u16 = 1;
pub const SHAPE_WEDGE: u16 = 2;
pub const SHAPE_SMOOTH_WEDGE: u16 = 3;

impl ShapeTable {
    pub fn get(&self, index: u16) -> Option<&VoxelShape> {
        self.shapes.get(index as usize)
    }

    pub fn register(&mut self, shape: VoxelShape) -> u16 {
        let idx = self.shapes.len() as u16;
        self.shapes.push(shape);
        idx
    }
}

/// Shape 0: the standard full cube.
fn cube_shape() -> VoxelShape {
    // Vertices of a unit cube in normalized space (0..1, 0..1, 0..1)
    //
    //     4-----5       Y+
    //    /|    /|        |
    //   6-----7 |        |
    //   | 0---|-1       +--- X+
    //   |/    |/        /
    //   2-----3        Z+
    //
    let v = [
        Vec3::new(0.0, 0.0, 0.0), // 0: left  bottom back
        Vec3::new(1.0, 0.0, 0.0), // 1: right bottom back
        Vec3::new(0.0, 0.0, 1.0), // 2: left  bottom front
        Vec3::new(1.0, 0.0, 1.0), // 3: right bottom front
        Vec3::new(0.0, 1.0, 0.0), // 4: left  top    back
        Vec3::new(1.0, 1.0, 0.0), // 5: right top    back
        Vec3::new(0.0, 1.0, 1.0), // 6: left  top    front
        Vec3::new(1.0, 1.0, 1.0), // 7: right top    front
    ];

    let faces = vec![
        // Top (+Y): vertices 4,5,7,6
        VoxelFace {
            vertices: vec![v[4], v[5], v[7], v[6]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::Y,
            side: FaceSide::Top,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::South] },  // back edge
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },   // right edge
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::North] },  // front edge
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },   // left edge
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // Bottom (-Y): vertices 2,3,1,0
        VoxelFace {
            vertices: vec![v[2], v[3], v[1], v[0]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Y,
            side: FaceSide::Bottom,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::North] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // North (+Z): vertices 3,2,6,7
        VoxelFace {
            vertices: vec![v[3], v[2], v[6], v[7]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::Z,
            side: FaceSide::North,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::West] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::East] },
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // South (-Z): vertices 0,1,5,4
        VoxelFace {
            vertices: vec![v[0], v[1], v[5], v[4]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Z,
            side: FaceSide::South,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // East (+X): vertices 1,3,7,5
        VoxelFace {
            vertices: vec![v[1], v[3], v[7], v[5]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::X,
            side: FaceSide::East,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::North] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::South] },
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // West (-X): vertices 2,0,4,6
        VoxelFace {
            vertices: vec![v[2], v[0], v[4], v[6]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_X,
            side: FaceSide::West,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] },
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::South] },
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::Top] },
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::North] },
            ],
            chamfer_mode: ChamferMode::Hard,
        },
    ];

    VoxelShape {
        name: "cube".to_string(),
        faces,
        occlusion: OcclusionMask::FULL,
    }
}

/// Shape 1: smooth-edge cube — same geometry as cube but with smooth chamfer normals.
fn smooth_cube_shape() -> VoxelShape {
    let mut shape = cube_shape();
    shape.name = "smooth_cube".to_string();
    for face in &mut shape.faces {
        face.chamfer_mode = ChamferMode::Smooth;
    }
    shape
}

/// Shape 2: wedge/ramp.
///
/// Default facing (North): slope descends from back (south, -Z) to front (north, +Z).
/// The tall wall is at the back (south side).
///
/// ```text
///     4-----5       Back wall (south)
///    /     /         Slope descends toward +Z
///   /     /
///  2-----3          Front is at ground level (y=0)
/// ```
///
/// Vertices:
///   0: (0,0,0)  left  bottom back
///   1: (1,0,0)  right bottom back
///   2: (0,0,1)  left  bottom front
///   3: (1,0,1)  right bottom front
///   4: (0,1,0)  left  top    back
///   5: (1,1,0)  right top    back
fn wedge_shape() -> VoxelShape {
    let v = [
        Vec3::new(0.0, 0.0, 0.0), // 0: left  bottom back
        Vec3::new(1.0, 0.0, 0.0), // 1: right bottom back
        Vec3::new(0.0, 0.0, 1.0), // 2: left  bottom front
        Vec3::new(1.0, 0.0, 1.0), // 3: right bottom front
        Vec3::new(0.0, 1.0, 0.0), // 4: left  top    back
        Vec3::new(1.0, 1.0, 0.0), // 5: right top    back
    ];

    let faces = vec![
        // Bottom (-Y): quad [2, 3, 1, 0]
        VoxelFace {
            vertices: vec![v[2], v[3], v[1], v[0]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Y,
            side: FaceSide::Bottom,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::North] },  // front
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },   // right
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::South] },  // back
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },   // left
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // Back/South wall (-Z): quad [0, 1, 5, 4]
        VoxelFace {
            vertices: vec![v[0], v[1], v[5], v[4]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Z,
            side: FaceSide::South,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] }, // bottom
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },   // right
                // Ridge (top) omitted — internal edge shared with slope face
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },   // left
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // Slope face: quad [4, 5, 3, 2] — from back-top to front-bottom
        // Normal is computed from world-space vertices at mesh time.
        VoxelFace {
            vertices: vec![v[4], v[5], v[3], v[2]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::new(0.0, 1.0, 1.0).normalize(),
            side: FaceSide::None, // diagonal — no single neighbor can occlude
            edges: vec![
                // Ridge (top) omitted — internal edge shared with back face
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },   // right diagonal (top→bottom)
                VoxelEdge { v0: 2, v1: 3, neighbor_sides: vec![FaceSide::North, FaceSide::Bottom] },  // bottom/front (right→left)
                VoxelEdge { v0: 3, v1: 0, neighbor_sides: vec![FaceSide::West] },   // left diagonal (bottom→top)
            ],
            chamfer_mode: ChamferMode::Smooth,
        },
        // Left/West triangle (-X): [0, 4, 2]
        VoxelFace {
            vertices: vec![v[0], v[4], v[2]],
            triangles: vec![[0, 1, 2]],
            normal: Vec3::NEG_X,
            side: FaceSide::West,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::South, FaceSide::Bottom] },  // back (bottom→top)
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::West] },   // hypotenuse — chamfer when side is exposed
                VoxelEdge { v0: 2, v1: 0, neighbor_sides: vec![FaceSide::Bottom] }, // bottom (front→back)
            ],
            chamfer_mode: ChamferMode::Hard,
        },
        // Right/East triangle (+X): [1, 3, 5]
        VoxelFace {
            vertices: vec![v[1], v[3], v[5]],
            triangles: vec![[0, 1, 2]],
            normal: Vec3::X,
            side: FaceSide::East,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_sides: vec![FaceSide::Bottom] }, // bottom (back→front)
                VoxelEdge { v0: 1, v1: 2, neighbor_sides: vec![FaceSide::East] },   // hypotenuse — chamfer when side is exposed
                VoxelEdge { v0: 2, v1: 0, neighbor_sides: vec![FaceSide::South, FaceSide::Bottom] },  // back (top→bottom)
            ],
            chamfer_mode: ChamferMode::Hard,
        },
    ];

    VoxelShape {
        name: "wedge".to_string(),
        faces,
        // Wedge only fully occludes Bottom and South (the full back wall).
        // Top=false, Bottom=true, North=false, South=true, East=false, West=false
        occlusion: OcclusionMask([false, true, false, true, false, false]),
    }
}

/// Shape 3: smooth wedge — same geometry as wedge but with smooth/fillet chamfer normals.
fn smooth_wedge_shape() -> VoxelShape {
    let mut shape = wedge_shape();
    shape.name = "smooth_wedge".to_string();
    for face in &mut shape.faces {
        face.chamfer_mode = ChamferMode::Smooth;
    }
    shape
}
