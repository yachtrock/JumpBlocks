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
                match (base + steps) % 4 {
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

/// An edge of a voxel face, defined by two vertex indices into the face's vertex list.
#[derive(Debug, Clone)]
pub struct VoxelEdge {
    /// Index of the start vertex in the parent face's vertices.
    pub v0: usize,
    /// Index of the end vertex in the parent face's vertices.
    pub v1: usize,
    /// Which neighboring face side this edge borders (for chamfer decisions).
    /// If the neighbor on this side is solid, the edge won't be chamfered.
    pub neighbor_side: FaceSide,
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
}

/// A voxel shape definition — a collection of faces.
#[derive(Debug, Clone)]
pub struct VoxelShape {
    pub name: String,
    pub faces: Vec<VoxelFace>,
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
        table.shapes.push(cube_shape());
        table
    }
}

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
                VoxelEdge { v0: 0, v1: 1, neighbor_side: FaceSide::South },  // back edge
                VoxelEdge { v0: 1, v1: 2, neighbor_side: FaceSide::East },   // right edge
                VoxelEdge { v0: 2, v1: 3, neighbor_side: FaceSide::North },  // front edge
                VoxelEdge { v0: 3, v1: 0, neighbor_side: FaceSide::West },   // left edge
            ],
        },
        // Bottom (-Y): vertices 2,3,1,0
        VoxelFace {
            vertices: vec![v[2], v[3], v[1], v[0]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Y,
            side: FaceSide::Bottom,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_side: FaceSide::North },
                VoxelEdge { v0: 1, v1: 2, neighbor_side: FaceSide::East },
                VoxelEdge { v0: 2, v1: 3, neighbor_side: FaceSide::South },
                VoxelEdge { v0: 3, v1: 0, neighbor_side: FaceSide::West },
            ],
        },
        // North (+Z): vertices 3,2,6,7
        VoxelFace {
            vertices: vec![v[3], v[2], v[6], v[7]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::Z,
            side: FaceSide::North,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_side: FaceSide::Bottom },
                VoxelEdge { v0: 1, v1: 2, neighbor_side: FaceSide::West },
                VoxelEdge { v0: 2, v1: 3, neighbor_side: FaceSide::Top },
                VoxelEdge { v0: 3, v1: 0, neighbor_side: FaceSide::East },
            ],
        },
        // South (-Z): vertices 0,1,5,4
        VoxelFace {
            vertices: vec![v[0], v[1], v[5], v[4]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_Z,
            side: FaceSide::South,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_side: FaceSide::Bottom },
                VoxelEdge { v0: 1, v1: 2, neighbor_side: FaceSide::East },
                VoxelEdge { v0: 2, v1: 3, neighbor_side: FaceSide::Top },
                VoxelEdge { v0: 3, v1: 0, neighbor_side: FaceSide::West },
            ],
        },
        // East (+X): vertices 1,3,7,5
        VoxelFace {
            vertices: vec![v[1], v[3], v[7], v[5]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::X,
            side: FaceSide::East,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_side: FaceSide::Bottom },
                VoxelEdge { v0: 1, v1: 2, neighbor_side: FaceSide::North },
                VoxelEdge { v0: 2, v1: 3, neighbor_side: FaceSide::Top },
                VoxelEdge { v0: 3, v1: 0, neighbor_side: FaceSide::South },
            ],
        },
        // West (-X): vertices 2,0,4,6
        VoxelFace {
            vertices: vec![v[2], v[0], v[4], v[6]],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normal: Vec3::NEG_X,
            side: FaceSide::West,
            edges: vec![
                VoxelEdge { v0: 0, v1: 1, neighbor_side: FaceSide::Bottom },
                VoxelEdge { v0: 1, v1: 2, neighbor_side: FaceSide::South },
                VoxelEdge { v0: 2, v1: 3, neighbor_side: FaceSide::Top },
                VoxelEdge { v0: 3, v1: 0, neighbor_side: FaceSide::North },
            ],
        },
    ];

    VoxelShape {
        name: "cube".to_string(),
        faces,
    }
}
