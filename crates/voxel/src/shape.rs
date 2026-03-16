use bevy::prelude::*;

/// Global chamfer width in world units.
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

/// Which cell within a block a face covers, and on which side.
#[derive(Debug, Clone)]
pub struct CellCover {
    /// Cell position within the block (0..size.x, 0..size.y, 0..size.z).
    pub cell: (u8, u8, u8),
    /// Which side of that cell this face covers.
    pub side: FaceSide,
    /// Whether this face fully covers that cell-side.
    pub full: bool,
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
        table.shapes.push(wedge_shape());   // shape 1: wedge/ramp
        table
    }
}

/// Well-known shape indices.
pub const SHAPE_CUBE: u16 = 0;
pub const SHAPE_WEDGE: u16 = 1;

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
                CellCover { cell: (0, 0, 0), side: FaceSide::Top, full: true },
                CellCover { cell: (1, 0, 0), side: FaceSide::Top, full: true },
                CellCover { cell: (0, 0, 1), side: FaceSide::Top, full: true },
                CellCover { cell: (1, 0, 1), side: FaceSide::Top, full: true },
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
                CellCover { cell: (0, 0, 0), side: FaceSide::Bottom, full: true },
                CellCover { cell: (1, 0, 0), side: FaceSide::Bottom, full: true },
                CellCover { cell: (0, 0, 1), side: FaceSide::Bottom, full: true },
                CellCover { cell: (1, 0, 1), side: FaceSide::Bottom, full: true },
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
                CellCover { cell: (0, 0, 1), side: FaceSide::North, full: true },
                CellCover { cell: (1, 0, 1), side: FaceSide::North, full: true },
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
                CellCover { cell: (0, 0, 0), side: FaceSide::South, full: true },
                CellCover { cell: (1, 0, 0), side: FaceSide::South, full: true },
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
                CellCover { cell: (1, 0, 0), side: FaceSide::East, full: true },
                CellCover { cell: (1, 0, 1), side: FaceSide::East, full: true },
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
                CellCover { cell: (0, 0, 0), side: FaceSide::West, full: true },
                CellCover { cell: (0, 0, 1), side: FaceSide::West, full: true },
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
                CellCover { cell: (0, 0, 0), side: FaceSide::Bottom, full: true },
                CellCover { cell: (1, 0, 0), side: FaceSide::Bottom, full: true },
                CellCover { cell: (0, 0, 1), side: FaceSide::Bottom, full: true },
                CellCover { cell: (1, 0, 1), side: FaceSide::Bottom, full: true },
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
                CellCover { cell: (0, 0, 0), side: FaceSide::South, full: true },
                CellCover { cell: (1, 0, 0), side: FaceSide::South, full: true },
                CellCover { cell: (0, 1, 0), side: FaceSide::South, full: true },
                CellCover { cell: (1, 1, 0), side: FaceSide::South, full: true },
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
                CellCover { cell: (0, 0, 1), side: FaceSide::North, full: true },
                CellCover { cell: (1, 0, 1), side: FaceSide::North, full: true },
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
                CellCover { cell: (0, 0, 0), side: FaceSide::West, full: true },
                CellCover { cell: (0, 0, 1), side: FaceSide::West, full: true },
                CellCover { cell: (0, 1, 0), side: FaceSide::West, full: false },
                CellCover { cell: (0, 1, 1), side: FaceSide::West, full: false },
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
                CellCover { cell: (1, 0, 0), side: FaceSide::East, full: true },
                CellCover { cell: (1, 0, 1), side: FaceSide::East, full: true },
                CellCover { cell: (1, 1, 0), side: FaceSide::East, full: false },
                CellCover { cell: (1, 1, 1), side: FaceSide::East, full: false },
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
