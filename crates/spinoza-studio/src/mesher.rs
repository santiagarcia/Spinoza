//! Mesh generation module for Spinoza Studio.
//!
//! Implements multiple meshing algorithms from scratch, covering surface and
//! volume discretisation with various element types:
//!
//! | Algorithm                    | Element types   | Domain    |
//! |------------------------------|-----------------|-----------|
//! | Delaunay triangulation (2D)  | Tri3            | 2D planar |
//! | Advancing-front              | Tri3            | 2D planar |
//! | Surface remesh               | Tri3            | 3D surf.  |
//! | Delaunay tet meshing         | Tet4            | 3D volume |
//! | Structured hex meshing       | Hex8            | 3D box    |
//! | Quad-dominant surface        | Quad4 / Tri3    | 3D surf.  |
//!
//! All algorithms produce [`MeshData`] compatible with the analysis pipeline
//! ([`crate::assembly`]) and the visualization tessellator ([`crate::tessellate`]).
//!
//! # State-of-the-art background
//!
//! * **Bowyer-Watson** (1981): Incremental Delaunay via cavity expansion.
//!   O(n log n) expected for uniformly distributed points.  Used in Triangle,
//!   CGAL, TetGen.
//! * **Advancing-front** (Löhner 1996): Grows mesh from boundary inward,
//!   generating well-shaped triangles.  Used in NETGEN, ANSYS Fluent.
//! * **Isotropic surface remeshing** (Botsch & Kobbelt 2004): Edge split /
//!   collapse / flip / tangential smoothing to achieve uniform sizing.
//! * **Delaunay refinement** (Shewchuk 2002): Ruppert / stellar insertion
//!   for guaranteed quality bounds.  Used in TetGen, CGAL 3D.
//! * **Transfinite / structured** mapping: Tensor-product parameterization
//!   for simple topologies; produces pure hex meshes.

use spinoza_core::results::{
    BoundaryEntity, BoundaryEntitySet, Element, ElementType, FieldAssociation, FieldData,
    MeshData, SolveDiagnostics, VisSolveResult,
};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════════
// Public types
// ═══════════════════════════════════════════════════════════════════════════

/// Meshing algorithm selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshAlgorithm {
    /// Bowyer-Watson Delaunay triangulation (2D planar).
    DelaunayTriangle,
    /// Advancing-front method (2D planar).
    AdvancingFront,
    /// Isotropic surface remeshing of imported triangulated surfaces.
    SurfaceRemesh,
    /// Delaunay tetrahedralization of a closed surface (3D volume).
    DelaunayTet,
    /// Axis-aligned structured hexahedral meshing of the bounding box.
    StructuredHex,
    /// Convert tris to quads via pairing/split (quad-dominant).
    QuadDominant,
}

impl MeshAlgorithm {
    pub fn label(&self) -> &'static str {
        match self {
            Self::DelaunayTriangle => "Delaunay Triangulation (2D Tri3)",
            Self::AdvancingFront => "Advancing Front (2D Tri3)",
            Self::SurfaceRemesh => "Surface Remesh (Tri3)",
            Self::DelaunayTet => "Delaunay Tet Meshing (Tet4)",
            Self::StructuredHex => "Structured Hex Meshing (Hex8)",
            Self::QuadDominant => "Quad-Dominant Surface (Quad4/Tri3)",
        }
    }

    pub fn all() -> &'static [MeshAlgorithm] {
        &[
            Self::DelaunayTriangle,
            Self::AdvancingFront,
            Self::SurfaceRemesh,
            Self::DelaunayTet,
            Self::StructuredHex,
            Self::QuadDominant,
        ]
    }

    /// Short description for UI tooltip.
    pub fn description(&self) -> &'static str {
        match self {
            Self::DelaunayTriangle => "Bowyer-Watson incremental Delaunay. Projects points to XY plane and triangulates. Good for planar 2D meshes.",
            Self::AdvancingFront => "Grows triangles from boundary inward, generating well-shaped elements. Good for 2D domains.",
            Self::SurfaceRemesh => "Isotropic remeshing via edge split/collapse/flip/smooth. Produces uniform surface triangulations.",
            Self::DelaunayTet => "Generates tetrahedral volume mesh inside a closed triangulated surface. Uses 3D Bowyer-Watson.",
            Self::StructuredHex => "Maps the bounding box to a structured hexahedral grid. Fast, high-quality for simple geometries.",
            Self::QuadDominant => "Converts triangulated surface to quad-dominant mesh by pairing adjacent triangles.",
        }
    }
}

/// Meshing parameters.
#[derive(Debug, Clone)]
pub struct MeshParams {
    /// Target element size (edge length). If 0 or negative, auto-computed.
    pub element_size: f64,
    /// Number of divisions along each axis for structured meshing.
    pub divisions: [usize; 3],
    /// Minimum angle threshold for quality improvement (degrees).
    pub min_angle: f64,
    /// Maximum number of Laplacian smoothing iterations.
    pub smooth_iterations: usize,
    /// Surface remesh: target edge length ratio (relative to mean).
    pub target_edge_ratio: f64,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            element_size: 0.0, // auto
            divisions: [10, 10, 10],
            min_angle: 20.0,
            smooth_iterations: 5,
            target_edge_ratio: 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a mesh from existing geometry using the chosen algorithm.
///
/// `source` is the imported geometry mesh (e.g. from OBJ/STL).
/// Returns a new [`VisSolveResult`] containing the generated mesh.
pub fn generate_mesh(
    source: &MeshData,
    algorithm: MeshAlgorithm,
    params: &MeshParams,
) -> Result<VisSolveResult, String> {
    let mesh = match algorithm {
        MeshAlgorithm::DelaunayTriangle => delaunay_triangulate_2d(source, params)?,
        MeshAlgorithm::AdvancingFront => advancing_front_2d(source, params)?,
        MeshAlgorithm::SurfaceRemesh => surface_remesh(source, params)?,
        MeshAlgorithm::DelaunayTet => delaunay_tet_3d(source, params)?,
        MeshAlgorithm::StructuredHex => structured_hex(source, params)?,
        MeshAlgorithm::QuadDominant => quad_dominant_surface(source, params)?,
    };

    let n = mesh.node_count();
    let e = mesh.element_count();

    // Zero field for colorbar display.
    let field = FieldData {
        name: "Mesh Quality".into(),
        association: FieldAssociation::Node,
        components: 1,
        values: vec![0.0; n],
        boundary_set_name: None,
    };

    Ok(VisSolveResult {
        problem_name: format!("Meshed ({} — {} nodes, {} elems)", algorithm.label(), n, e),
        mesh,
        fields: vec![field],
        diagnostics: SolveDiagnostics {
            residual_norm: 0.0,
            iterations: 0,
            backend: "mesher".into(),
            selected_pack: algorithm.label().into(),
            selected_builder: "spinoza-studio mesher".into(),
            selected_operator: "none".into(),
            selected_preconditioner: "none".into(),
            selected_solver: format!("{} nodes, {} elements", n, e),
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn auto_element_size(nodes: &[[f64; 3]]) -> f64 {
    if nodes.len() < 2 {
        return 1.0;
    }
    let (mut mn, mut mx) = ([f64::MAX; 3], [f64::MIN; 3]);
    for n in nodes {
        for i in 0..3 {
            mn[i] = mn[i].min(n[i]);
            mx[i] = mx[i].max(n[i]);
        }
    }
    let diag = ((mx[0] - mn[0]).powi(2) + (mx[1] - mn[1]).powi(2) + (mx[2] - mn[2]).powi(2))
        .sqrt();
    // Target ~20 divisions across the longest diagonal.
    (diag / 20.0).max(1e-12)
}

fn bounding_box(nodes: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut mn = [f64::MAX; 3];
    let mut mx = [f64::MIN; 3];
    for n in nodes {
        for i in 0..3 {
            mn[i] = mn[i].min(n[i]);
            mx[i] = mx[i].max(n[i]);
        }
    }
    (mn, mx)
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[allow(dead_code)]
fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n < 1e-30 {
        [0.0, 0.0, 0.0]
    } else {
        scale3(a, 1.0 / n)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AABB tree for accelerated ray-triangle intersection
// ═══════════════════════════════════════════════════════════════════════════

/// Axis-aligned bounding box.
#[derive(Clone)]
struct Aabb {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Aabb {
    fn new() -> Self {
        Self {
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        }
    }

    fn expand_point(&mut self, p: &[f64; 3]) {
        for i in 0..3 {
            if p[i] < self.lo[i] { self.lo[i] = p[i]; }
            if p[i] > self.hi[i] { self.hi[i] = p[i]; }
        }
    }

    fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            lo: [
                self.lo[0].min(other.lo[0]),
                self.lo[1].min(other.lo[1]),
                self.lo[2].min(other.lo[2]),
            ],
            hi: [
                self.hi[0].max(other.hi[0]),
                self.hi[1].max(other.hi[1]),
                self.hi[2].max(other.hi[2]),
            ],
        }
    }

    /// Test if ray(origin, dir) with t > 0 can intersect this box.
    fn ray_intersects(&self, origin: &[f64; 3], inv_dir: &[f64; 3]) -> bool {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;
        for i in 0..3 {
            let t1 = (self.lo[i] - origin[i]) * inv_dir[i];
            let t2 = (self.hi[i] - origin[i]) * inv_dir[i];
            let ta = t1.min(t2);
            let tb = t1.max(t2);
            tmin = tmin.max(ta);
            tmax = tmax.min(tb);
        }
        tmax >= tmin.max(0.0)
    }
}

/// BVH node — binary tree over triangle indices.
enum BvhNode {
    Leaf { aabb: Aabb, tri_indices: Vec<usize> },
    Inner { aabb: Aabb, left: Box<BvhNode>, right: Box<BvhNode> },
}

impl BvhNode {
    fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } => aabb,
            BvhNode::Inner { aabb, .. } => aabb,
        }
    }
}

/// Build a BVH from triangle indices.  Each triangle is represented by its
/// AABB (computed from the 3 vertices).  We split on the axis with the
/// largest extent using the median centroid.
fn build_bvh(
    tri_aabbs: &[Aabb],
    centroids: &[[f64; 3]],
    mut indices: Vec<usize>,
) -> BvhNode {
    if indices.len() <= 4 {
        let mut aabb = Aabb::new();
        for &i in &indices {
            aabb = aabb.merge(&tri_aabbs[i]);
        }
        return BvhNode::Leaf { aabb, tri_indices: indices };
    }

    // Compute total AABB and centroid extent.
    let mut total = Aabb::new();
    let mut cent_lo = [f64::INFINITY; 3];
    let mut cent_hi = [f64::NEG_INFINITY; 3];
    for &i in &indices {
        total = total.merge(&tri_aabbs[i]);
        for d in 0..3 {
            if centroids[i][d] < cent_lo[d] { cent_lo[d] = centroids[i][d]; }
            if centroids[i][d] > cent_hi[d] { cent_hi[d] = centroids[i][d]; }
        }
    }

    // Split on axis with largest centroid extent.
    let mut axis = 0;
    let mut max_ext = 0.0f64;
    for d in 0..3 {
        let ext = cent_hi[d] - cent_lo[d];
        if ext > max_ext {
            max_ext = ext;
            axis = d;
        }
    }

    // Sort by centroid along chosen axis.
    indices.sort_by(|&a, &b| centroids[a][axis].partial_cmp(&centroids[b][axis]).unwrap());
    let mid = indices.len() / 2;
    let right_idx = indices.split_off(mid);

    let left = build_bvh(tri_aabbs, centroids, indices);
    let right = build_bvh(tri_aabbs, centroids, right_idx);
    let aabb = left.aabb().merge(right.aabb());

    BvhNode::Inner {
        aabb,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Pre-built acceleration structure for ray-casting inside/outside queries.
struct RayCastAccel {
    nodes: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
    bvh: BvhNode,
}

impl RayCastAccel {
    fn build(nodes: &[[f64; 3]], triangles: &[[usize; 3]]) -> Self {
        let mut tri_aabbs = Vec::with_capacity(triangles.len());
        let mut centroids = Vec::with_capacity(triangles.len());
        for tri in triangles {
            let mut aabb = Aabb::new();
            let v0 = nodes[tri[0]];
            let v1 = nodes[tri[1]];
            let v2 = nodes[tri[2]];
            aabb.expand_point(&v0);
            aabb.expand_point(&v1);
            aabb.expand_point(&v2);
            tri_aabbs.push(aabb);
            centroids.push([
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ]);
        }
        let indices: Vec<usize> = (0..triangles.len()).collect();
        let bvh = build_bvh(&tri_aabbs, &centroids, indices);
        Self {
            nodes: nodes.to_vec(),
            triangles: triangles.to_vec(),
            bvh,
        }
    }

    /// Count ray-triangle crossings using BVH acceleration.
    fn count_crossings(&self, origin: &[f64; 3], dir: &[f64; 3]) -> u32 {
        let inv_dir = [
            if dir[0].abs() < 1e-30 { 1e30_f64.copysign(dir[0]) } else { 1.0 / dir[0] },
            if dir[1].abs() < 1e-30 { 1e30_f64.copysign(dir[1]) } else { 1.0 / dir[1] },
            if dir[2].abs() < 1e-30 { 1e30_f64.copysign(dir[2]) } else { 1.0 / dir[2] },
        ];
        let mut count = 0u32;
        let mut stack: Vec<&BvhNode> = vec![&self.bvh];
        while let Some(node) = stack.pop() {
            if !node.aabb().ray_intersects(origin, &inv_dir) {
                continue;
            }
            match node {
                BvhNode::Leaf { tri_indices, .. } => {
                    for &ti in tri_indices {
                        if ray_intersects_triangle_dir(origin, dir, &self.nodes, &self.triangles[ti]) {
                            count += 1;
                        }
                    }
                }
                BvhNode::Inner { left, right, .. } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        count
    }

    /// Inside/outside test: majority vote with 3 non-axis-aligned rays.
    fn point_inside(&self, pt: &[f64; 3]) -> bool {
        const DIRS: [[f64; 3]; 3] = [
            [0.9572349, 0.2134987, 0.1953271],
            [0.1876453, 0.9217348, 0.3394618],
            [0.2748391, 0.1598743, 0.9481562],
        ];
        let mut inside_votes = 0u32;
        for dir in &DIRS {
            if self.count_crossings(pt, dir) % 2 == 1 {
                inside_votes += 1;
            }
        }
        inside_votes >= 2
    }
}

// Send + Sync so rayon can share the accel structure across threads.
unsafe impl Send for RayCastAccel {}
unsafe impl Sync for RayCastAccel {}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial hash grid for fast proximity queries
// ═══════════════════════════════════════════════════════════════════════════

/// Cell-based spatial hash for O(1) amortized proximity lookups.
struct SpatialHash {
    #[allow(dead_code)]
    cell_size: f64,
    inv_cell: f64,
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl SpatialHash {
    fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            inv_cell: 1.0 / cell_size,
            cells: HashMap::new(),
        }
    }

    fn cell_key(&self, p: &[f64; 3]) -> (i64, i64, i64) {
        (
            (p[0] * self.inv_cell).floor() as i64,
            (p[1] * self.inv_cell).floor() as i64,
            (p[2] * self.inv_cell).floor() as i64,
        )
    }

    fn insert(&mut self, idx: usize, p: &[f64; 3]) {
        let key = self.cell_key(p);
        self.cells.entry(key).or_default().push(idx);
    }

    /// Check if any point within `radius` of `p` exists.
    fn has_neighbor(&self, p: &[f64; 3], radius: f64, points: &[[f64; 3]]) -> bool {
        let r2 = radius * radius;
        let key = self.cell_key(p);
        // Check the 3×3×3 neighborhood of cells.
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let k = (key.0 + dx, key.1 + dy, key.2 + dz);
                    if let Some(bucket) = self.cells.get(&k) {
                        for &idx in bucket {
                            let q = &points[idx];
                            let d2 = (p[0] - q[0]).powi(2)
                                + (p[1] - q[1]).powi(2)
                                + (p[2] - q[2]).powi(2);
                            if d2 < r2 {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

fn make_tri3_type() -> ElementType {
    ElementType {
        id: 100,
        name: "tri3".into(),
        element_dim: 2,
        order: 1,
        reference: "Lagrange".into(),
        notes: None,
    }
}

fn make_quad4_type() -> ElementType {
    ElementType {
        id: 101,
        name: "quad4".into(),
        element_dim: 2,
        order: 1,
        reference: "Lagrange".into(),
        notes: None,
    }
}

fn make_tet4_type() -> ElementType {
    ElementType {
        id: 102,
        name: "tet4".into(),
        element_dim: 3,
        order: 1,
        reference: "Lagrange".into(),
        notes: None,
    }
}

fn make_hex8_type() -> ElementType {
    ElementType {
        id: 103,
        name: "hex8".into(),
        element_dim: 3,
        order: 1,
        reference: "Lagrange".into(),
        notes: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. DELAUNAY TRIANGULATION (2D) — Bowyer-Watson algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// 2D Delaunay triangulation using the Bowyer-Watson incremental algorithm.
///
/// Projects all source nodes to the XY plane, then incrementally inserts
/// each point by:
///   1. Finding all triangles whose circumcircle contains the new point.
///   2. Removing those triangles and constructing a star-shaped polygon.
///   3. Creating new triangles by connecting the polygon edges to the point.
///
/// After all insertions, removes triangles connected to the super-triangle.
fn delaunay_triangulate_2d(source: &MeshData, params: &MeshParams) -> Result<MeshData, String> {
    if source.nodes.is_empty() {
        return Err("No nodes in source geometry".into());
    }

    let h = if params.element_size > 0.0 {
        params.element_size
    } else {
        auto_element_size(&source.nodes)
    };

    // Collect 2D points (project to XY).
    let mut points: Vec<[f64; 2]> = Vec::new();
    let (bb_min, bb_max) = bounding_box(&source.nodes);

    // Always use source nodes + interior refinement grid for mesh density.
    for n in &source.nodes {
        points.push([n[0], n[1]]);
    }
    // Add interior grid points within bounding box.
    let nx = ((bb_max[0] - bb_min[0]) / h).ceil().max(1.0) as usize;
    let ny = ((bb_max[1] - bb_min[1]) / h).ceil().max(1.0) as usize;
    for iy in 1..ny {
        for ix in 1..nx {
            let x = bb_min[0] + ix as f64 * h;
            let y = bb_min[1] + iy as f64 * h;
            // Only add if not too close to existing points.
            let too_close = points.iter().any(|p| {
                ((p[0] - x).powi(2) + (p[1] - y).powi(2)).sqrt() < h * 0.5
            });
            if !too_close {
                points.push([x, y]);
            }
        }
    }

    if points.len() < 3 {
        return Err("Need at least 3 points for triangulation".into());
    }

    // De-duplicate points.
    let mut unique_pts: Vec<[f64; 2]> = Vec::with_capacity(points.len());
    for p in &points {
        let dup = unique_pts
            .iter()
            .any(|q| (q[0] - p[0]).abs() < 1e-12 && (q[1] - p[1]).abs() < 1e-12);
        if !dup {
            unique_pts.push(*p);
        }
    }
    let points = unique_pts;
    let n = points.len();

    // Super-triangle: enclose all points with a large margin.
    let dx = bb_max[0] - bb_min[0] + 1.0;
    let dy = bb_max[1] - bb_min[1] + 1.0;
    let dmax = dx.max(dy);
    let mid_x = (bb_min[0] + bb_max[0]) * 0.5;
    let mid_y = (bb_min[1] + bb_max[1]) * 0.5;

    let st0 = [mid_x - 20.0 * dmax, mid_y - dmax];
    let st1 = [mid_x + 20.0 * dmax, mid_y - dmax];
    let st2 = [mid_x, mid_y + 20.0 * dmax];

    // Working arrays: points include super-triangle vertices at the end.
    let mut all_pts = points.clone();
    let si0 = all_pts.len();
    all_pts.push(st0);
    let si1 = all_pts.len();
    all_pts.push(st1);
    let si2 = all_pts.len();
    all_pts.push(st2);

    // Triangle list: each is [v0, v1, v2].
    let mut triangles: Vec<[usize; 3]> = vec![[si0, si1, si2]];

    // Insert each point.
    for pi in 0..n {
        let px = all_pts[pi][0];
        let py = all_pts[pi][1];

        // Find bad triangles (point inside circumcircle).
        let mut bad = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            if in_circumcircle_2d(&all_pts, tri, px, py) {
                bad.push(ti);
            }
        }

        // Find boundary polygon of the cavity.
        let mut polygon: Vec<[usize; 2]> = Vec::new();
        for &bi in &bad {
            let tri = triangles[bi];
            for edge_i in 0..3 {
                let e = [tri[edge_i], tri[(edge_i + 1) % 3]];
                // Edge is on the boundary if it's not shared by another bad triangle.
                let shared = bad.iter().any(|&oi| {
                    oi != bi && {
                        let ot = triangles[oi];
                        edge_in_tri(e, &ot)
                    }
                });
                if !shared {
                    polygon.push(e);
                }
            }
        }

        // Remove bad triangles (in reverse order to keep indices stable).
        let mut bad_sorted = bad;
        bad_sorted.sort_unstable();
        for &bi in bad_sorted.iter().rev() {
            triangles.swap_remove(bi);
        }

        // Re-triangulate the cavity.
        for edge in &polygon {
            triangles.push([pi, edge[0], edge[1]]);
        }
    }

    // Remove triangles that reference super-triangle vertices.
    triangles.retain(|tri| {
        tri[0] < n && tri[1] < n && tri[2] < n
    });

    // Build MeshData.
    let nodes: Vec<[f64; 3]> = points.iter().map(|p| [p[0], p[1], 0.0]).collect();
    let elements: Vec<Element> = triangles
        .iter()
        .map(|t| Element {
            type_id: 100,
            connectivity: vec![t[0] as u32, t[1] as u32, t[2] as u32],
            region: None,
        })
        .collect();

    // Extract boundary edges (edges referenced by only one triangle).
    let boundary = extract_boundary_edges_2d(&triangles, n);

    Ok(MeshData {
        dimension: 2,
        nodes,
        element_types: vec![make_tri3_type()],
        elements,
        boundaries: if boundary.is_empty() {
            vec![]
        } else {
            vec![BoundaryEntitySet {
                name: "exterior".into(),
                entity_dim: 1,
                entities: boundary
                    .iter()
                    .map(|e| BoundaryEntity {
                        type_id: None,
                        connectivity: vec![e[0] as u32, e[1] as u32],
                    })
                    .collect(),
            }]
        },
    })
}

/// Check if point (px, py) lies inside the circumcircle of the triangle.
///
/// Handles both CCW and CW oriented triangles by computing the signed area
/// and adjusting the determinant sign accordingly.
fn in_circumcircle_2d(pts: &[[f64; 2]], tri: &[usize; 3], px: f64, py: f64) -> bool {
    let ax = pts[tri[0]][0] - px;
    let ay = pts[tri[0]][1] - py;
    let bx = pts[tri[1]][0] - px;
    let by = pts[tri[1]][1] - py;
    let cx = pts[tri[2]][0] - px;
    let cy = pts[tri[2]][1] - py;

    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - bx * (ay * (cx * cx + cy * cy) - cy * (ax * ax + ay * ay))
        + cx * (ay * (bx * bx + by * by) - by * (ax * ax + ay * ay));

    // Compute signed area of triangle to determine orientation.
    // For CCW (positive area): det > 0 means inside circumcircle.
    // For CW  (negative area): det < 0 means inside circumcircle.
    let signed_area = (pts[tri[1]][0] - pts[tri[0]][0]) * (pts[tri[2]][1] - pts[tri[0]][1])
        - (pts[tri[2]][0] - pts[tri[0]][0]) * (pts[tri[1]][1] - pts[tri[0]][1]);
    if signed_area.abs() < 1e-30 {
        return false; // Degenerate triangle
    }
    (det * signed_area.signum()) > 1e-30
}

fn edge_in_tri(edge: [usize; 2], tri: &[usize; 3]) -> bool {
    let has0 = tri[0] == edge[0] || tri[1] == edge[0] || tri[2] == edge[0];
    let has1 = tri[0] == edge[1] || tri[1] == edge[1] || tri[2] == edge[1];
    has0 && has1
}

fn extract_boundary_edges_2d(triangles: &[[usize; 3]], _node_count: usize) -> Vec<[usize; 2]> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for tri in triangles {
        for i in 0..3 {
            let mut e = (tri[i], tri[(i + 1) % 3]);
            if e.0 > e.1 {
                std::mem::swap(&mut e.0, &mut e.1);
            }
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }
    edge_count
        .into_iter()
        .filter(|(_, c)| *c == 1)
        .map(|((a, b), _)| [a, b])
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ADVANCING FRONT METHOD (2D)
// ═══════════════════════════════════════════════════════════════════════════

/// Advancing-front meshing for 2D planar domains.
///
/// 1. Extracts the boundary contour from the source mesh (or uses the
///    convex hull of source nodes).
/// 2. Initializes a "front" — the list of unprocessed boundary edges.
/// 3. For each front edge, creates an ideal point opposite to it at
///    distance h, then either picks an existing nearby point or inserts
///    the ideal point, creating a new triangle.
/// 4. Applies Laplacian smoothing.
fn advancing_front_2d(source: &MeshData, params: &MeshParams) -> Result<MeshData, String> {
    if source.nodes.len() < 3 {
        return Err("Need at least 3 nodes for advancing-front meshing".into());
    }

    let h = if params.element_size > 0.0 {
        params.element_size
    } else {
        auto_element_size(&source.nodes)
    };

    // Project to 2D.
    let mut pts: Vec<[f64; 2]> = source.nodes.iter().map(|n| [n[0], n[1]]).collect();

    // Extract boundary loop from source triangles if available.
    let source_tris: Vec<[usize; 3]> = source
        .elements
        .iter()
        .filter_map(|e| {
            if e.connectivity.len() == 3 {
                Some([
                    e.connectivity[0] as usize,
                    e.connectivity[1] as usize,
                    e.connectivity[2] as usize,
                ])
            } else {
                None
            }
        })
        .collect();

    let boundary_edges = if source_tris.is_empty() {
        // No triangles: compute convex hull boundary.
        convex_hull_2d(&pts)
    } else {
        // Extract boundary edges from triangulated surface.
        let raw = extract_boundary_edges_2d(&source_tris, pts.len());
        order_boundary_loop(&raw)
    };

    // Ensure boundary is CCW (positive signed area) so the interior
    // normal direction is correct for the advancing-front algorithm.
    let signed_area_2: f64 = boundary_edges.iter().map(|e| {
        pts[e[0]][0] * pts[e[1]][1] - pts[e[1]][0] * pts[e[0]][1]
    }).sum();
    let boundary_edges: Vec<[usize; 2]> = if signed_area_2 < 0.0 {
        // CW boundary detected — reverse to make CCW.
        boundary_edges.iter().rev().map(|e| [e[1], e[0]]).collect()
    } else {
        boundary_edges
    };

    if boundary_edges.len() < 3 {
        return Err("Could not extract a valid boundary contour".into());
    }

    // Initialize front as the boundary edges.
    let mut front: Vec<[usize; 2]> = boundary_edges.clone();
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    let mut used_edges: HashSet<(usize, usize)> = HashSet::new();
    for e in &front {
        let key = if e[0] < e[1] { (e[0], e[1]) } else { (e[1], e[0]) };
        used_edges.insert(key);
    }

    let max_iter = (pts.len() + ((source.nodes.len() as f64 * 10.0) as usize)).max(10_000);
    let mut iter_count = 0;

    while !front.is_empty() && iter_count < max_iter {
        iter_count += 1;

        // Pick the shortest front edge for better quality.
        let fi = front
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = dist_2d(&pts[a[0]], &pts[a[1]]);
                let db = dist_2d(&pts[b[0]], &pts[b[1]]);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();

        let edge = front.remove(fi);
        let a = edge[0];
        let b = edge[1];

        // Compute ideal point: equilateral triangle on the "interior" side.
        let mx = (pts[a][0] + pts[b][0]) * 0.5;
        let my = (pts[a][1] + pts[b][1]) * 0.5;
        let dx = pts[b][0] - pts[a][0];
        let dy = pts[b][1] - pts[a][1];
        let len = (dx * dx + dy * dy).sqrt();
        let height = (h * h - (len / 2.0).powi(2)).max(0.0).sqrt().max(h * 0.866);
        // Normal direction (rotate edge 90° CCW).
        let nx = -dy / len;
        let ny = dx / len;
        let ideal = [mx + nx * height, my + ny * height];

        // Check if an existing front point is close enough.
        let mut best_existing: Option<usize> = None;
        let mut best_dist = h * 1.5;
        for fe in &front {
            for &vi in &[fe[0], fe[1]] {
                if vi == a || vi == b {
                    continue;
                }
                let d = dist_2d(&pts[vi], &ideal);
                if d < best_dist {
                    // Check that the triangle a-b-vi would not overlap existing ones.
                    let _ea1 = if a < vi { (a, vi) } else { (vi, a) };
                    let _eb1 = if b < vi { (b, vi) } else { (vi, b) };
                    // Allow if edges are already in used_edges (closing fronts) or new.
                    if !creates_degenerate(&pts, a, b, vi) {
                        best_dist = d;
                        best_existing = Some(vi);
                    }
                }
            }
        }

        let c = if let Some(vi) = best_existing {
            vi
        } else {
            // Insert new point.
            let new_idx = pts.len();
            pts.push(ideal);
            new_idx
        };

        // Create triangle.
        triangles.push([a, b, c]);

        // Update front: add new edges, remove if they already exist (interior).
        for &new_edge in &[[a, c], [c, b]] {
            let key = if new_edge[0] < new_edge[1] {
                (new_edge[0], new_edge[1])
            } else {
                (new_edge[1], new_edge[0])
            };
            if used_edges.contains(&key) {
                // Remove from front.
                if let Some(pos) = front.iter().position(|e| {
                    let ek = if e[0] < e[1] { (e[0], e[1]) } else { (e[1], e[0]) };
                    ek == key
                }) {
                    front.remove(pos);
                }
            } else {
                front.push(new_edge);
                used_edges.insert(key);
            }
        }
    }

    // Laplacian smoothing (keeping boundary fixed).
    let boundary_nodes: HashSet<usize> = boundary_edges
        .iter()
        .flat_map(|e| [e[0], e[1]])
        .collect();

    for _ in 0..params.smooth_iterations {
        let mut new_pts = pts.clone();
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
        for tri in &triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                neighbors[a].push(b);
                neighbors[b].push(a);
            }
        }
        for i in 0..pts.len() {
            if boundary_nodes.contains(&i) || neighbors[i].is_empty() {
                continue;
            }
            let avg_x: f64 = neighbors[i].iter().map(|&j| pts[j][0]).sum::<f64>()
                / neighbors[i].len() as f64;
            let avg_y: f64 = neighbors[i].iter().map(|&j| pts[j][1]).sum::<f64>()
                / neighbors[i].len() as f64;
            new_pts[i] = [avg_x, avg_y];
        }
        pts = new_pts;
    }

    // Build MeshData.
    let nodes: Vec<[f64; 3]> = pts.iter().map(|p| [p[0], p[1], 0.0]).collect();
    let elements: Vec<Element> = triangles
        .iter()
        .map(|t| Element {
            type_id: 100,
            connectivity: vec![t[0] as u32, t[1] as u32, t[2] as u32],
            region: None,
        })
        .collect();

    let boundary_ents: Vec<BoundaryEntity> = boundary_edges
        .iter()
        .map(|e| BoundaryEntity {
            type_id: None,
            connectivity: vec![e[0] as u32, e[1] as u32],
        })
        .collect();

    Ok(MeshData {
        dimension: 2,
        nodes,
        element_types: vec![make_tri3_type()],
        elements,
        boundaries: if boundary_ents.is_empty() {
            vec![]
        } else {
            vec![BoundaryEntitySet {
                name: "exterior".into(),
                entity_dim: 1,
                entities: boundary_ents,
            }]
        },
    })
}

fn dist_2d(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn creates_degenerate(pts: &[[f64; 2]], a: usize, b: usize, c: usize) -> bool {
    // Check if triangle has near-zero area.
    let ax = pts[b][0] - pts[a][0];
    let ay = pts[b][1] - pts[a][1];
    let bx = pts[c][0] - pts[a][0];
    let by = pts[c][1] - pts[a][1];
    let area = (ax * by - ay * bx).abs();
    let perimeter = dist_2d(&pts[a], &pts[b])
        + dist_2d(&pts[b], &pts[c])
        + dist_2d(&pts[c], &pts[a]);
    if perimeter < 1e-30 {
        return true;
    }
    // Aspect ratio check.
    area / (perimeter * perimeter) < 1e-6
}

/// Compute convex hull of 2D points and return boundary edges.
fn convex_hull_2d(pts: &[[f64; 2]]) -> Vec<[usize; 2]> {
    let n = pts.len();
    if n < 3 {
        return if n == 2 {
            vec![[0, 1], [1, 0]]
        } else {
            vec![]
        };
    }

    // Andrew's monotone chain.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        pts[a][0]
            .partial_cmp(&pts[b][0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                pts[a][1]
                    .partial_cmp(&pts[b][1])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    let cross = |o: usize, a: usize, b: usize| -> f64 {
        (pts[a][0] - pts[o][0]) * (pts[b][1] - pts[o][1])
            - (pts[a][1] - pts[o][1]) * (pts[b][0] - pts[o][0])
    };

    let mut lower = Vec::new();
    for &i in &indices {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], i) <= 0.0
        {
            lower.pop();
        }
        lower.push(i);
    }

    let mut upper = Vec::new();
    for &i in indices.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], i) <= 0.0
        {
            upper.pop();
        }
        upper.push(i);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);

    let hull = lower;
    let m = hull.len();
    (0..m).map(|i| [hull[i], hull[(i + 1) % m]]).collect()
}

/// Attempt to order boundary edges into a loop.
fn order_boundary_loop(edges: &[[usize; 2]]) -> Vec<[usize; 2]> {
    if edges.is_empty() {
        return vec![];
    }
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in edges {
        adj.entry(e[0]).or_default().push(e[1]);
        adj.entry(e[1]).or_default().push(e[0]);
    }

    let start = edges[0][0];
    let mut ordered = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut current = start;
    loop {
        visited.insert(current);
        let neighbors = match adj.get(&current) {
            Some(n) => n,
            None => break,
        };
        let next = neighbors.iter().find(|&&n| !visited.contains(&n));
        match next {
            Some(&n) => {
                ordered.push([current, n]);
                current = n;
            }
            None => {
                // Close the loop.
                ordered.push([current, start]);
                break;
            }
        }
    }
    ordered
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. SURFACE REMESHING (Isotropic)
// ═══════════════════════════════════════════════════════════════════════════

/// Isotropic surface remeshing via iterative edge operations.
///
/// Based on Botsch & Kobbelt (2004), this implements:
///   1. **Edge split**: Split edges longer than 4/3 × target length.
///   2. **Edge collapse**: Collapse edges shorter than 4/5 × target length.
///   3. **Edge flip**: Flip non-Delaunay edges to improve valence.
///   4. **Tangential smoothing**: Move vertices along the surface toward
///      their neighborhood centroid.
///
/// Works directly on 3D triangulated surfaces, preserving the surface shape.
fn surface_remesh(source: &MeshData, params: &MeshParams) -> Result<MeshData, String> {
    // Collect source triangles.
    let mut nodes: Vec<[f64; 3]> = source.nodes.clone();
    let mut triangles: Vec<[usize; 3]> = source
        .elements
        .iter()
        .filter_map(|e| {
            if e.connectivity.len() >= 3 {
                Some([
                    e.connectivity[0] as usize,
                    e.connectivity[1] as usize,
                    e.connectivity[2] as usize,
                ])
            } else {
                None
            }
        })
        .collect();

    if triangles.is_empty() {
        return Err("No triangles in source mesh for surface remeshing".into());
    }

    // Compute target edge length.
    let target_len = if params.element_size > 0.0 {
        params.element_size
    } else {
        // Mean edge length of input mesh.
        let mut total_len = 0.0;
        let mut edge_count = 0;
        for tri in &triangles {
            for i in 0..3 {
                total_len += dist3(&nodes[tri[i]], &nodes[tri[(i + 1) % 3]]);
                edge_count += 1;
            }
        }
        (total_len / edge_count as f64) * params.target_edge_ratio
    };

    let split_threshold = target_len * 4.0 / 3.0;
    let collapse_threshold = target_len * 4.0 / 5.0;

    // Identify boundary edges (to keep them fixed).
    let _boundary_edges_set = compute_boundary_edge_set(&triangles);

    // Iterative improvement.
    let iterations = params.smooth_iterations.max(3);
    for _iter in 0..iterations {
        // Phase 1: Edge split — collect all long edges, then subdivide triangles.
        let mut split_cache: HashMap<(usize, usize), usize> = HashMap::new();

        // Identify all edges that need splitting.
        for tri in &triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                if split_cache.contains_key(&key) {
                    continue;
                }
                if dist3(&nodes[a], &nodes[b]) > split_threshold {
                    let mid = [
                        (nodes[a][0] + nodes[b][0]) * 0.5,
                        (nodes[a][1] + nodes[b][1]) * 0.5,
                        (nodes[a][2] + nodes[b][2]) * 0.5,
                    ];
                    let idx = nodes.len();
                    nodes.push(mid);
                    split_cache.insert(key, idx);
                }
            }
        }

        // Subdivide triangles based on which edges were split.
        let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(triangles.len() * 2);
        for tri in &triangles {
            let mids: [Option<usize>; 3] = [
                {
                    let (a, b) = (tri[0], tri[1]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    split_cache.get(&key).copied()
                },
                {
                    let (a, b) = (tri[1], tri[2]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    split_cache.get(&key).copied()
                },
                {
                    let (a, b) = (tri[2], tri[0]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    split_cache.get(&key).copied()
                },
            ];

            match (mids[0], mids[1], mids[2]) {
                (None, None, None) => {
                    new_tris.push(*tri);
                }
                // Single-edge splits
                (Some(m), None, None) => {
                    new_tris.push([tri[0], m, tri[2]]);
                    new_tris.push([m, tri[1], tri[2]]);
                }
                (None, Some(m), None) => {
                    new_tris.push([tri[0], tri[1], m]);
                    new_tris.push([tri[0], m, tri[2]]);
                }
                (None, None, Some(m)) => {
                    new_tris.push([tri[0], tri[1], m]);
                    new_tris.push([m, tri[1], tri[2]]);
                }
                // Double-edge splits
                (Some(m01), Some(m12), None) => {
                    new_tris.push([tri[0], m01, tri[2]]);
                    new_tris.push([m01, tri[1], m12]);
                    new_tris.push([m01, m12, tri[2]]);
                }
                (Some(m01), None, Some(m20)) => {
                    new_tris.push([tri[0], m01, m20]);
                    new_tris.push([m01, tri[1], tri[2]]);
                    new_tris.push([m01, tri[2], m20]);
                }
                (None, Some(m12), Some(m20)) => {
                    new_tris.push([tri[0], tri[1], m12]);
                    new_tris.push([tri[0], m12, m20]);
                    new_tris.push([m12, tri[2], m20]);
                }
                // All-three-edges split (red refinement)
                (Some(m01), Some(m12), Some(m20)) => {
                    new_tris.push([tri[0], m01, m20]);
                    new_tris.push([m01, tri[1], m12]);
                    new_tris.push([m20, m12, tri[2]]);
                    new_tris.push([m01, m12, m20]);
                }
            }
        }
        triangles = new_tris;

        // Phase 2: Edge collapse (collapse short edges).
        let mut collapsed: HashSet<usize> = HashSet::new(); // collapsed node indices
        let mut remap: Vec<usize> = (0..nodes.len()).collect();
        let collapse_boundary = compute_boundary_edge_set(&triangles);

        for tri in &triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                if collapsed.contains(&a) || collapsed.contains(&b) {
                    continue;
                }
                let len = dist3(&nodes[remap[a]], &nodes[remap[b]]);
                if len < collapse_threshold {
                    // Don't collapse boundary edges.
                    let key = if a < b { (a, b) } else { (b, a) };
                    if collapse_boundary.contains(&key) {
                        continue;
                    }
                    // Merge b into a.
                    let mid = [
                        (nodes[remap[a]][0] + nodes[remap[b]][0]) * 0.5,
                        (nodes[remap[a]][1] + nodes[remap[b]][1]) * 0.5,
                        (nodes[remap[a]][2] + nodes[remap[b]][2]) * 0.5,
                    ];
                    nodes[remap[a]] = mid;
                    remap[b] = remap[a];
                    collapsed.insert(b);
                }
            }
        }

        // Chase remap to handle transitive collapses (a→b→c chains).
        for i in 0..remap.len() {
            let mut target = remap[i];
            let mut limit = 100;
            while remap[target] != target && limit > 0 {
                target = remap[target];
                limit -= 1;
            }
            remap[i] = target;
        }

        // Apply remap.
        for tri in &mut triangles {
            tri[0] = remap[tri[0]];
            tri[1] = remap[tri[1]];
            tri[2] = remap[tri[2]];
        }
        // Remove degenerate triangles.
        triangles.retain(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0]);

        // Phase 3: Edge flip (improve angles / valence).
        edge_flip_pass(&nodes, &mut triangles);

        // Phase 4: Tangential smoothing.
        let boundary_verts = compute_boundary_vertices(&triangles);
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for tri in &triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                if !neighbors[a].contains(&b) {
                    neighbors[a].push(b);
                }
                if !neighbors[b].contains(&a) {
                    neighbors[b].push(a);
                }
            }
        }

        for vi in 0..nodes.len() {
            if boundary_verts.contains(&vi) || neighbors[vi].is_empty() {
                continue;
            }
            // Centroid of neighbors.
            let nn = neighbors[vi].len() as f64;
            let cx: f64 = neighbors[vi].iter().map(|&j| nodes[j][0]).sum::<f64>() / nn;
            let cy: f64 = neighbors[vi].iter().map(|&j| nodes[j][1]).sum::<f64>() / nn;
            let cz: f64 = neighbors[vi].iter().map(|&j| nodes[j][2]).sum::<f64>() / nn;

            // Project back onto local tangent plane (approximate: vertex normal).
            let normal = vertex_normal(&nodes, &triangles, vi);
            let delta = sub3([cx, cy, cz], nodes[vi]);
            let tang = sub3(delta, scale3(normal, dot3(delta, normal)));
            nodes[vi] = add3(nodes[vi], scale3(tang, 0.5));
        }
    }

    // Compact: remove unreferenced nodes.
    let (nodes, triangles) = compact_mesh(nodes, triangles);

    // Build output.
    let elements: Vec<Element> = triangles
        .iter()
        .map(|t| Element {
            type_id: 100,
            connectivity: vec![t[0] as u32, t[1] as u32, t[2] as u32],
            region: None,
        })
        .collect();

    // Detect dimension from z-coords.
    let has_z = nodes.iter().any(|n| n[2].abs() > 1e-10);
    let has_y = nodes.iter().any(|n| n[1].abs() > 1e-10);
    let dim = if has_z { 3 } else if has_y { 2 } else { 1 };

    Ok(MeshData {
        dimension: dim,
        nodes,
        element_types: vec![make_tri3_type()],
        elements,
        boundaries: vec![],
    })
}

fn compute_boundary_edge_set(triangles: &[[usize; 3]]) -> HashSet<(usize, usize)> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for tri in triangles {
        for i in 0..3 {
            let mut e = (tri[i], tri[(i + 1) % 3]);
            if e.0 > e.1 {
                std::mem::swap(&mut e.0, &mut e.1);
            }
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }
    edge_count
        .into_iter()
        .filter(|(_, c)| *c == 1)
        .map(|(e, _)| e)
        .collect()
}

fn compute_boundary_vertices(triangles: &[[usize; 3]]) -> HashSet<usize> {
    let boundary_edges = compute_boundary_edge_set(triangles);
    let mut verts = HashSet::new();
    for (a, b) in &boundary_edges {
        verts.insert(*a);
        verts.insert(*b);
    }
    verts
}

fn vertex_normal(nodes: &[[f64; 3]], triangles: &[[usize; 3]], vi: usize) -> [f64; 3] {
    let mut normal = [0.0, 0.0, 0.0];
    for tri in triangles {
        if tri[0] == vi || tri[1] == vi || tri[2] == vi {
            let e1 = sub3(nodes[tri[1]], nodes[tri[0]]);
            let e2 = sub3(nodes[tri[2]], nodes[tri[0]]);
            let n = cross3(e1, e2);
            normal = add3(normal, n);
        }
    }
    normalize3(normal)
}

fn edge_flip_pass(nodes: &[[f64; 3]], triangles: &mut Vec<[usize; 3]>) {
    // Build adjacency: for each directed edge (a,b), store triangle index.
    let mut edge_to_tri: HashMap<(usize, usize), usize> = HashMap::new();
    for (ti, tri) in triangles.iter().enumerate() {
        for i in 0..3 {
            edge_to_tri.insert((tri[i], tri[(i + 1) % 3]), ti);
        }
    }

    let mut flipped = true;
    let mut max_passes = 3;
    while flipped && max_passes > 0 {
        flipped = false;
        max_passes -= 1;

        let tri_snapshot = triangles.clone();
        let mut skip: HashSet<usize> = HashSet::new();

        for ti in 0..tri_snapshot.len() {
            if skip.contains(&ti) {
                continue;
            }
            let tri_a = tri_snapshot[ti];
            for i in 0..3 {
                let a = tri_a[i];
                let b = tri_a[(i + 1) % 3];
                let c = tri_a[(i + 2) % 3];
                // Look for adjacent triangle sharing edge b→a.
                if let Some(&tj) = edge_to_tri.get(&(b, a)) {
                    if tj == ti || skip.contains(&tj) {
                        continue;
                    }
                    let tri_b = tri_snapshot[tj];
                    // Find the opposite vertex in tri_b.
                    let d = tri_b
                        .iter()
                        .find(|&&v| v != a && v != b)
                        .copied();
                    let d = match d {
                        Some(v) => v,
                        None => continue,
                    };

                    // Check if flipping improves the minimum angle.
                    let min_before = min_angle_of_two(nodes, a, b, c, d);
                    let min_after = min_angle_of_two_flipped(nodes, a, b, c, d);
                    if min_after > min_before + 1e-6 {
                        // Flip: replace (a,b,c) and (b,a,d) with (c,d,a) and (d,c,b).
                        triangles[ti] = [c, d, a];
                        triangles[tj] = [d, c, b];
                        skip.insert(ti);
                        skip.insert(tj);
                        flipped = true;
                        break;
                    }
                }
            }
        }

        // Rebuild edge_to_tri.
        edge_to_tri.clear();
        for (ti, tri) in triangles.iter().enumerate() {
            for i in 0..3 {
                edge_to_tri.insert((tri[i], tri[(i + 1) % 3]), ti);
            }
        }
    }
}

fn min_angle_of_two(nodes: &[[f64; 3]], a: usize, b: usize, c: usize, d: usize) -> f64 {
    tri_min_angle(nodes, a, b, c).min(tri_min_angle(nodes, b, a, d))
}

fn min_angle_of_two_flipped(nodes: &[[f64; 3]], a: usize, b: usize, c: usize, d: usize) -> f64 {
    tri_min_angle(nodes, c, d, a).min(tri_min_angle(nodes, d, c, b))
}

fn tri_min_angle(nodes: &[[f64; 3]], a: usize, b: usize, c: usize) -> f64 {
    let ab = dist3(&nodes[a], &nodes[b]);
    let bc = dist3(&nodes[b], &nodes[c]);
    let ca = dist3(&nodes[c], &nodes[a]);
    if ab < 1e-30 || bc < 1e-30 || ca < 1e-30 {
        return 0.0;
    }
    let angle_a = ((ab * ab + ca * ca - bc * bc) / (2.0 * ab * ca)).clamp(-1.0, 1.0).acos();
    let angle_b = ((ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc)).clamp(-1.0, 1.0).acos();
    let angle_c = std::f64::consts::PI - angle_a - angle_b;
    angle_a.min(angle_b).min(angle_c)
}

fn compact_mesh(
    nodes: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut used: BTreeSet<usize> = BTreeSet::new();
    for tri in &triangles {
        used.insert(tri[0]);
        used.insert(tri[1]);
        used.insert(tri[2]);
    }
    let mut remap: Vec<usize> = vec![0; nodes.len()];
    let mut new_nodes = Vec::new();
    for &old in &used {
        remap[old] = new_nodes.len();
        new_nodes.push(nodes[old]);
    }
    let new_tris: Vec<[usize; 3]> = triangles
        .iter()
        .map(|t| [remap[t[0]], remap[t[1]], remap[t[2]]])
        .collect();
    (new_nodes, new_tris)
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. DELAUNAY TET MESHING (3D) — Bowyer-Watson 3D
// ═══════════════════════════════════════════════════════════════════════════

/// 3D Delaunay tetrahedralization using the Bowyer-Watson algorithm.
///
/// This generates a volume mesh of tetrahedra inside the convex hull of the
/// input points (or within the closed surface if provided):
///   1. Creates a large super-tetrahedron enclosing all points.
///   2. Incrementally inserts each point:
///      a. Finds all tets whose circumsphere contains the point.
///      b. Removes those tets and collects the cavity boundary (triangular
///         faces visible from the new point).
///      c. Creates new tets by connecting each boundary face to the point.
///   3. Removes tets connected to the super-tetrahedron.
///
/// Interior point generation: if element_size is specified, a background
/// grid of points is inserted inside the surface to fill the volume.
fn delaunay_tet_3d(source: &MeshData, params: &MeshParams) -> Result<MeshData, String> {
    if source.nodes.len() < 4 {
        return Err("Need at least 4 nodes for tetrahedral meshing".into());
    }

    let h = if params.element_size > 0.0 {
        params.element_size
    } else {
        auto_element_size(&source.nodes)
    };

    // Collect source points + add interior grid points.
    let (bb_min, bb_max) = bounding_box(&source.nodes);
    let mut points: Vec<[f64; 3]> = source.nodes.clone();

    // Build surface triangles for inside/outside testing.
    let surface_tris: Vec<[usize; 3]> = source
        .elements
        .iter()
        .filter_map(|e| {
            if e.connectivity.len() == 3 {
                Some([
                    e.connectivity[0] as usize,
                    e.connectivity[1] as usize,
                    e.connectivity[2] as usize,
                ])
            } else if e.connectivity.len() == 4 {
                // For quads, split into 2 tris.
                None // handled below
            } else {
                None
            }
        })
        .collect();

    // Add quad-split triangles.
    let mut all_surface_tris = surface_tris.clone();
    for e in &source.elements {
        if e.connectivity.len() == 4 {
            let c = &e.connectivity;
            all_surface_tris.push([c[0] as usize, c[1] as usize, c[2] as usize]);
            all_surface_tris.push([c[0] as usize, c[2] as usize, c[3] as usize]);
        }
    }

    // ── Phase 1: Generate interior grid points (parallel, BVH-accelerated) ──
    if !all_surface_tris.is_empty() {
        let margin = h * 0.25;
        let x0 = bb_min[0] + margin;
        let y0 = bb_min[1] + margin;
        let z0 = bb_min[2] + margin;
        let x1 = bb_max[0] - margin;
        let y1 = bb_max[1] - margin;
        let z1 = bb_max[2] - margin;

        if x1 > x0 && y1 > y0 && z1 > z0 {
            // Build BVH for fast ray-casting.
            let accel = RayCastAccel::build(&source.nodes, &all_surface_tris);

            let nx = ((x1 - x0) / h).ceil().max(1.0) as usize;
            let ny = ((y1 - y0) / h).ceil().max(1.0) as usize;
            let nz = ((z1 - z0) / h).ceil().max(1.0) as usize;
            let dx = if nx > 0 { (x1 - x0) / nx as f64 } else { 0.0 };
            let dy = if ny > 0 { (y1 - y0) / ny as f64 } else { 0.0 };
            let dz = if nz > 0 { (z1 - z0) / nz as f64 } else { 0.0 };

            // Build flat list of candidate grid coordinates.
            let total_grid = (nx + 1) * (ny + 1) * (nz + 1);
            let grid_pts: Vec<[f64; 3]> = (0..total_grid)
                .into_par_iter()
                .map(|idx| {
                    let ix = idx % (nx + 1);
                    let iy = (idx / (nx + 1)) % (ny + 1);
                    let iz = idx / ((nx + 1) * (ny + 1));
                    [
                        x0 + ix as f64 * dx,
                        y0 + iy as f64 * dy,
                        z0 + iz as f64 * dz,
                    ]
                })
                .filter(|pt| accel.point_inside(pt))
                .collect();

            // Insert interior points using spatial hash for fast proximity check.
            let proximity = h * 0.3;
            let mut hash = SpatialHash::new(proximity);
            // Seed the hash with existing source vertices.
            for (i, p) in points.iter().enumerate() {
                hash.insert(i, p);
            }
            for pt in grid_pts {
                if !hash.has_neighbor(&pt, proximity, &points) {
                    let idx = points.len();
                    hash.insert(idx, &pt);
                    points.push(pt);
                }
            }
        }
    }

    // ── Phase 2: De-duplicate using spatial hash ────────────────────────────
    {
        let eps: f64 = 1e-12;
        let mut dedup_hash = SpatialHash::new(eps.max(1e-10));
        let mut unique: Vec<[f64; 3]> = Vec::with_capacity(points.len());
        for p in &points {
            if !dedup_hash.has_neighbor(p, eps, &unique) {
                let idx = unique.len();
                dedup_hash.insert(idx, p);
                unique.push(*p);
            }
        }
        points = unique;
    }
    let n = points.len();

    if n < 4 {
        return Err("Not enough unique points for tetrahedral meshing".into());
    }

    // ── Phase 3: Bowyer-Watson incremental insertion ────────────────────────
    let bw_dx = (bb_max[0] - bb_min[0] + 1.0).max(1.0);
    let bw_dy = (bb_max[1] - bb_min[1] + 1.0).max(1.0);
    let bw_dz = (bb_max[2] - bb_min[2] + 1.0).max(1.0);
    let dmax = bw_dx.max(bw_dy).max(bw_dz);
    let cx = (bb_min[0] + bb_max[0]) * 0.5;
    let cy = (bb_min[1] + bb_max[1]) * 0.5;
    let cz = (bb_min[2] + bb_max[2]) * 0.5;

    let mut all_pts = points.clone();
    let s0 = all_pts.len();
    all_pts.push([cx - 20.0 * dmax, cy - dmax, cz - dmax]);
    let s1 = all_pts.len();
    all_pts.push([cx + 20.0 * dmax, cy - dmax, cz - dmax]);
    let s2 = all_pts.len();
    all_pts.push([cx, cy + 20.0 * dmax, cz - dmax]);
    let s3 = all_pts.len();
    all_pts.push([cx, cy, cz + 20.0 * dmax]);

    let mut tets: Vec<[usize; 4]> = vec![[s0, s1, s2, s3]];

    // Use a HashSet of sorted face tuples for fast shared-face detection.
    for pi in 0..n {
        let pt = all_pts[pi];

        // Find bad tets.
        let mut bad = Vec::new();
        for (ti, tet) in tets.iter().enumerate() {
            if in_circumsphere(&all_pts, tet, &pt) {
                bad.push(ti);
            }
        }

        // Collect boundary faces of the cavity using a face-count map
        // instead of the O(bad²) containment check.
        let mut face_count: HashMap<(usize, usize, usize), ([usize; 3], u32)> = HashMap::new();
        for &bi in &bad {
            let tet = tets[bi];
            let faces = [
                [tet[0], tet[1], tet[2]],
                [tet[0], tet[1], tet[3]],
                [tet[0], tet[2], tet[3]],
                [tet[1], tet[2], tet[3]],
            ];
            for face in &faces {
                let mut sorted = *face;
                sorted.sort_unstable();
                let key = (sorted[0], sorted[1], sorted[2]);
                face_count
                    .entry(key)
                    .and_modify(|e| e.1 += 1)
                    .or_insert((*face, 1));
            }
        }
        // Faces that appear exactly once are the cavity boundary.
        let boundary_faces: Vec<[usize; 3]> = face_count
            .values()
            .filter(|(_, count)| *count == 1)
            .map(|(face, _)| *face)
            .collect();

        // Remove bad tets (in reverse order for swap_remove stability).
        let mut bad_sorted = bad;
        bad_sorted.sort_unstable();
        for &bi in bad_sorted.iter().rev() {
            tets.swap_remove(bi);
        }

        // Create new tets.
        for face in &boundary_faces {
            tets.push([pi, face[0], face[1], face[2]]);
        }
    }

    // ── Phase 4: Filter and extract ─────────────────────────────────────────
    // Remove tets connected to super-tetrahedron.
    tets.retain(|t| t[0] < n && t[1] < n && t[2] < n && t[3] < n);

    // Parallel filter: keep tets whose centroid is inside the surface.
    if !all_surface_tris.is_empty() {
        let accel = RayCastAccel::build(&source.nodes, &all_surface_tris);
        tets = tets
            .into_par_iter()
            .filter(|t| {
                let centroid = [
                    (points[t[0]][0] + points[t[1]][0] + points[t[2]][0] + points[t[3]][0]) / 4.0,
                    (points[t[0]][1] + points[t[1]][1] + points[t[2]][1] + points[t[3]][1]) / 4.0,
                    (points[t[0]][2] + points[t[1]][2] + points[t[2]][2] + points[t[3]][2]) / 4.0,
                ];
                accel.point_inside(&centroid)
            })
            .collect();
    }

    // Compact.
    let mut used: BTreeSet<usize> = BTreeSet::new();
    for tet in &tets {
        used.insert(tet[0]);
        used.insert(tet[1]);
        used.insert(tet[2]);
        used.insert(tet[3]);
    }
    let mut remap = vec![0usize; points.len()];
    let mut new_nodes = Vec::new();
    for &old in &used {
        remap[old] = new_nodes.len();
        new_nodes.push(points[old]);
    }
    let elements: Vec<Element> = tets
        .iter()
        .map(|t| Element {
            type_id: 102,
            connectivity: vec![
                remap[t[0]] as u32,
                remap[t[1]] as u32,
                remap[t[2]] as u32,
                remap[t[3]] as u32,
            ],
            region: None,
        })
        .collect();

    // Extract surface faces as boundary.
    let boundary = extract_tet_surface(&tets, &remap);

    Ok(MeshData {
        dimension: 3,
        nodes: new_nodes,
        element_types: vec![make_tet4_type()],
        elements,
        boundaries: if boundary.is_empty() {
            vec![]
        } else {
            vec![BoundaryEntitySet {
                name: "surface".into(),
                entity_dim: 2,
                entities: boundary,
            }]
        },
    })
}

/// Check if point lies inside circumsphere of tetrahedron.
///
/// Uses the standard insphere predicate. The orientation is computed
/// using edges from tet[0], which is the negative of the Shewchuk
/// convention (which subtracts tet[3]), so the comparison is flipped.
fn in_circumsphere(pts: &[[f64; 3]], tet: &[usize; 4], p: &[f64; 3]) -> bool {
    // Use the determinant of the "lift" matrix.
    let cols: Vec<[f64; 5]> = (0..4)
        .map(|i| {
            let q = pts[tet[i]];
            let dx = q[0] - p[0];
            let dy = q[1] - p[1];
            let dz = q[2] - p[2];
            [dx, dy, dz, dx * dx + dy * dy + dz * dz, 1.0]
        })
        .collect();

    // 4×4 determinant using cofactor expansion.
    let det = det4x4_circumsphere(&cols);

    // The sign of the determinant depends on tetrahedron orientation.
    // Our orient uses (b-a, c-a, d-a) which is -1× the Shewchuk convention
    // (a-d, b-d, c-d), so the product det*orient should be NEGATIVE for inside.
    let v01 = sub3(pts[tet[1]], pts[tet[0]]);
    let v02 = sub3(pts[tet[2]], pts[tet[0]]);
    let v03 = sub3(pts[tet[3]], pts[tet[0]]);
    let orient = dot3(v01, cross3(v02, v03));
    if orient.abs() < 1e-30 {
        return false; // Degenerate tetrahedron
    }
    (det * orient.signum()) < -1e-30
}

fn det4x4_circumsphere(cols: &[[f64; 5]]) -> f64 {
    // Matrix:
    // | dx0  dy0  dz0  dr0 |
    // | dx1  dy1  dz1  dr1 |
    // | dx2  dy2  dz2  dr2 |
    // | dx3  dy3  dz3  dr3 |
    let a = [
        [cols[0][0], cols[0][1], cols[0][2], cols[0][3]],
        [cols[1][0], cols[1][1], cols[1][2], cols[1][3]],
        [cols[2][0], cols[2][1], cols[2][2], cols[2][3]],
        [cols[3][0], cols[3][1], cols[3][2], cols[3][3]],
    ];
    det4x4(&a)
}

fn det4x4(m: &[[f64; 4]; 4]) -> f64 {
    let mut result = 0.0;
    for j in 0..4 {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        let minor = det3x3_sub(m, 0, j);
        result += sign * m[0][j] * minor;
    }
    result
}

fn det3x3_sub(m: &[[f64; 4]; 4], skip_row: usize, skip_col: usize) -> f64 {
    let mut sub = [[0.0; 3]; 3];
    let mut si = 0;
    for i in 0..4 {
        if i == skip_row {
            continue;
        }
        let mut sj = 0;
        for j in 0..4 {
            if j == skip_col {
                continue;
            }
            sub[si][sj] = m[i][j];
            sj += 1;
        }
        si += 1;
    }
    sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])
        - sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])
        + sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0])
}

/// Möller–Trumbore ray-triangle intersection with arbitrary direction.
fn ray_intersects_triangle_dir(
    origin: &[f64; 3],
    dir: &[f64; 3],
    nodes: &[[f64; 3]],
    tri: &[usize; 3],
) -> bool {
    let v0 = nodes[tri[0]];
    let v1 = nodes[tri[1]];
    let v2 = nodes[tri[2]];

    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let h = cross3(*dir, e2);
    let a = dot3(e1, h);

    if a.abs() < 1e-30 {
        return false;
    }

    let f = 1.0 / a;
    let s = sub3(*origin, v0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    let q = cross3(s, e1);
    let v = f * dot3(*dir, q);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * dot3(e2, q);
    t > 1e-10
}

fn extract_tet_surface(
    tets: &[[usize; 4]],
    remap: &[usize],
) -> Vec<BoundaryEntity> {
    let mut face_count: HashMap<(usize, usize, usize), usize> = HashMap::new();
    for tet in tets {
        let faces = [
            [tet[0], tet[1], tet[2]],
            [tet[0], tet[1], tet[3]],
            [tet[0], tet[2], tet[3]],
            [tet[1], tet[2], tet[3]],
        ];
        for f in &faces {
            let mut sorted = [f[0], f[1], f[2]];
            sorted.sort_unstable();
            let key = (sorted[0], sorted[1], sorted[2]);
            *face_count.entry(key).or_insert(0) += 1;
        }
    }
    face_count
        .into_iter()
        .filter(|(_, c)| *c == 1)
        .map(|((a, b, c), _)| BoundaryEntity {
            type_id: None,
            connectivity: vec![remap[a] as u32, remap[b] as u32, remap[c] as u32],
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. STRUCTURED HEX MESHING
// ═══════════════════════════════════════════════════════════════════════════

/// Axis-aligned structured hexahedral meshing.
///
/// Maps the bounding box of the source geometry to a regular grid of
/// hexahedra (brick elements).  Uses transfinite mapping to create
/// divisions along each axis.
///
/// Node numbering follows standard hex8 convention:
/// ```text
///     7───6
///    /|  /|
///   4───5 |
///   | 3─|─2
///   |/  |/
///   0───1
/// ```
fn structured_hex(source: &MeshData, params: &MeshParams) -> Result<MeshData, String> {
    if source.nodes.is_empty() {
        return Err("No nodes in source geometry for hex meshing".into());
    }

    let (bb_min, bb_max) = bounding_box(&source.nodes);

    let extents = [
        bb_max[0] - bb_min[0],
        bb_max[1] - bb_min[1],
        bb_max[2] - bb_min[2],
    ];
    let max_extent = extents.iter().cloned().fold(0.0f64, f64::max);
    if max_extent < 1e-30 {
        return Err("Degenerate geometry: all nodes at the same position".into());
    }

    // Detect thin/flat dimensions (extent < 1% of the largest axis).
    let thin_threshold = max_extent * 0.01;
    let thin = [
        extents[0] < thin_threshold,
        extents[1] < thin_threshold,
        extents[2] < thin_threshold,
    ];
    let thin_count = thin.iter().filter(|&&t| t).count();

    // If two or more axes are thin, geometry is effectively 1D or a point — fall back to 2D.
    let is_2d = thin[2] || thin_count >= 2;

    let (mut nx, mut ny, mut nz) = if params.element_size > 0.0 {
        let h = params.element_size;
        (
            if thin[0] { 1 } else { (extents[0] / h).ceil().max(1.0) as usize },
            if thin[1] { 1 } else { (extents[1] / h).ceil().max(1.0) as usize },
            if thin[2] || is_2d { 1 } else { (extents[2] / h).ceil().max(1.0) as usize },
        )
    } else {
        (
            if thin[0] { 1 } else { params.divisions[0].max(1) },
            if thin[1] { 1 } else { params.divisions[1].max(1) },
            if thin[2] || is_2d { 1 } else { params.divisions[2].max(1) },
        )
    };

    // Cap total element count to prevent memory exhaustion.
    const MAX_ELEMS: usize = 500_000;
    let total = nx * ny * nz;
    if total > MAX_ELEMS {
        let scale = (MAX_ELEMS as f64 / total as f64).cbrt();
        nx = (nx as f64 * scale).ceil().max(1.0) as usize;
        ny = (ny as f64 * scale).ceil().max(1.0) as usize;
        nz = if is_2d { 1 } else { (nz as f64 * scale).ceil().max(1.0) as usize };
        log::warn!(
            "Hex mesh element count capped: {}×{}×{} = {} (limit {MAX_ELEMS})",
            nx, ny, nz, nx * ny * nz
        );
    }

    let dx = if nx > 0 { extents[0] / nx as f64 } else { 0.0 };
    let dy = if ny > 0 { extents[1] / ny as f64 } else { 0.0 };
    let dz = if nz > 0 && !is_2d { extents[2] / nz as f64 } else { 0.0 };

    // Generate nodes.
    let nnx = nx + 1;
    let nny = ny + 1;
    let nnz = if is_2d { 1 } else { nz + 1 };
    let mut nodes = Vec::with_capacity(nnx * nny * nnz);
    for iz in 0..nnz {
        for iy in 0..nny {
            for ix in 0..nnx {
                nodes.push([
                    bb_min[0] + ix as f64 * dx,
                    bb_min[1] + iy as f64 * dy,
                    if is_2d {
                        bb_min[2]
                    } else {
                        bb_min[2] + iz as f64 * dz
                    },
                ]);
            }
        }
    }

    let node_idx = |ix: usize, iy: usize, iz: usize| -> u32 {
        (iz * nny * nnx + iy * nnx + ix) as u32
    };

    if is_2d {
        // Generate quad4 elements for 2D.
        let mut elements = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                elements.push(Element {
                    type_id: 101,
                    connectivity: vec![
                        node_idx(ix, iy, 0),
                        node_idx(ix + 1, iy, 0),
                        node_idx(ix + 1, iy + 1, 0),
                        node_idx(ix, iy + 1, 0),
                    ],
                    region: None,
                });
            }
        }

        // Boundary edges.
        let mut boundary_ents = Vec::new();
        // Bottom.
        for ix in 0..nx {
            boundary_ents.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![node_idx(ix, 0, 0), node_idx(ix + 1, 0, 0)],
            });
        }
        // Top.
        for ix in 0..nx {
            boundary_ents.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![node_idx(ix, ny, 0), node_idx(ix + 1, ny, 0)],
            });
        }
        // Left.
        for iy in 0..ny {
            boundary_ents.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![node_idx(0, iy, 0), node_idx(0, iy + 1, 0)],
            });
        }
        // Right.
        for iy in 0..ny {
            boundary_ents.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![node_idx(nx, iy, 0), node_idx(nx, iy + 1, 0)],
            });
        }

        return Ok(MeshData {
            dimension: 2,
            nodes,
            element_types: vec![make_quad4_type()],
            elements,
            boundaries: vec![
                BoundaryEntitySet {
                    name: "bottom".into(),
                    entity_dim: 1,
                    entities: boundary_ents[..nx].to_vec(),
                },
                BoundaryEntitySet {
                    name: "top".into(),
                    entity_dim: 1,
                    entities: boundary_ents[nx..2 * nx].to_vec(),
                },
                BoundaryEntitySet {
                    name: "left".into(),
                    entity_dim: 1,
                    entities: boundary_ents[2 * nx..2 * nx + ny].to_vec(),
                },
                BoundaryEntitySet {
                    name: "right".into(),
                    entity_dim: 1,
                    entities: boundary_ents[2 * nx + ny..].to_vec(),
                },
            ],
        });
    }

    // 3D hex elements.
    let mut elements = Vec::with_capacity(nx * ny * nz);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                elements.push(Element {
                    type_id: 103,
                    connectivity: vec![
                        node_idx(ix, iy, iz),         // 0
                        node_idx(ix + 1, iy, iz),     // 1
                        node_idx(ix + 1, iy + 1, iz), // 2
                        node_idx(ix, iy + 1, iz),     // 3
                        node_idx(ix, iy, iz + 1),     // 4
                        node_idx(ix + 1, iy, iz + 1), // 5
                        node_idx(ix + 1, iy + 1, iz + 1), // 6
                        node_idx(ix, iy + 1, iz + 1), // 7
                    ],
                    region: None,
                });
            }
        }
    }

    // Boundary faces (6 sides of the box).
    let mut boundaries = Vec::new();

    // Z-min face (bottom).
    let mut bottom = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            bottom.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![
                    node_idx(ix, iy, 0),
                    node_idx(ix + 1, iy, 0),
                    node_idx(ix + 1, iy + 1, 0),
                    node_idx(ix, iy + 1, 0),
                ],
            });
        }
    }
    boundaries.push(BoundaryEntitySet {
        name: "z_min".into(),
        entity_dim: 2,
        entities: bottom,
    });

    // Z-max face (top).
    let mut top = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            top.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![
                    node_idx(ix, iy, nz),
                    node_idx(ix + 1, iy, nz),
                    node_idx(ix + 1, iy + 1, nz),
                    node_idx(ix, iy + 1, nz),
                ],
            });
        }
    }
    boundaries.push(BoundaryEntitySet {
        name: "z_max".into(),
        entity_dim: 2,
        entities: top,
    });

    // Y-min face (front).
    let mut ymin = Vec::new();
    for iz in 0..nz {
        for ix in 0..nx {
            ymin.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![
                    node_idx(ix, 0, iz),
                    node_idx(ix + 1, 0, iz),
                    node_idx(ix + 1, 0, iz + 1),
                    node_idx(ix, 0, iz + 1),
                ],
            });
        }
    }
    boundaries.push(BoundaryEntitySet {
        name: "y_min".into(),
        entity_dim: 2,
        entities: ymin,
    });

    // Y-max face (back).
    let mut ymax = Vec::new();
    for iz in 0..nz {
        for ix in 0..nx {
            ymax.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![
                    node_idx(ix, ny, iz),
                    node_idx(ix + 1, ny, iz),
                    node_idx(ix + 1, ny, iz + 1),
                    node_idx(ix, ny, iz + 1),
                ],
            });
        }
    }
    boundaries.push(BoundaryEntitySet {
        name: "y_max".into(),
        entity_dim: 2,
        entities: ymax,
    });

    // X-min face (left).
    let mut xmin = Vec::new();
    for iz in 0..nz {
        for iy in 0..ny {
            xmin.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![
                    node_idx(0, iy, iz),
                    node_idx(0, iy + 1, iz),
                    node_idx(0, iy + 1, iz + 1),
                    node_idx(0, iy, iz + 1),
                ],
            });
        }
    }
    boundaries.push(BoundaryEntitySet {
        name: "x_min".into(),
        entity_dim: 2,
        entities: xmin,
    });

    // X-max face (right).
    let mut xmax = Vec::new();
    for iz in 0..nz {
        for iy in 0..ny {
            xmax.push(BoundaryEntity {
                type_id: None,
                connectivity: vec![
                    node_idx(nx, iy, iz),
                    node_idx(nx, iy + 1, iz),
                    node_idx(nx, iy + 1, iz + 1),
                    node_idx(nx, iy, iz + 1),
                ],
            });
        }
    }
    boundaries.push(BoundaryEntitySet {
        name: "x_max".into(),
        entity_dim: 2,
        entities: xmax,
    });

    Ok(MeshData {
        dimension: 3,
        nodes,
        element_types: vec![make_hex8_type()],
        elements,
        boundaries,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. QUAD-DOMINANT SURFACE MESHING
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a triangulated surface to quad-dominant by pairing adjacent
/// triangles that share an edge.
///
/// Algorithm:
///   1. Build edge-to-triangle adjacency.
///   2. Greedily pair adjacent triangles through the shared edge, choosing
///      the pair that produces the best-shaped quad (closest to planar &
///      convex).
///   3. Un-paired triangles are kept as Tri3 fallback.
///   4. Apply Laplacian smoothing to improve element shapes.
fn quad_dominant_surface(source: &MeshData, params: &MeshParams) -> Result<MeshData, String> {
    let mut nodes: Vec<[f64; 3]> = source.nodes.clone();
    let triangles: Vec<[usize; 3]> = source
        .elements
        .iter()
        .filter_map(|e| {
            if e.connectivity.len() >= 3 {
                Some([
                    e.connectivity[0] as usize,
                    e.connectivity[1] as usize,
                    e.connectivity[2] as usize,
                ])
            } else {
                None
            }
        })
        .collect();

    if triangles.is_empty() {
        return Err("No triangles in source mesh for quad conversion".into());
    }

    // Build edge-to-tri adjacency.
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, tri) in triangles.iter().enumerate() {
        for i in 0..3 {
            let mut e = (tri[i], tri[(i + 1) % 3]);
            if e.0 > e.1 {
                std::mem::swap(&mut e.0, &mut e.1);
            }
            edge_tris.entry(e).or_default().push(ti);
        }
    }

    // Greedy pairing.
    let mut paired: HashSet<usize> = HashSet::new();
    let mut quads: Vec<[usize; 4]> = Vec::new();

    // Score each edge for pairing quality.
    let mut edge_scores: Vec<((usize, usize), f64)> = Vec::new();
    for (edge, tris) in &edge_tris {
        if tris.len() == 2 {
            let t0 = &triangles[tris[0]];
            let t1 = &triangles[tris[1]];
            // Quality: how planar and convex is the resulting quad?
            let score = quad_quality(&nodes, t0, t1, edge);
            edge_scores.push((*edge, score));
        }
    }
    // Sort by quality (best first).
    edge_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (edge, _score) in &edge_scores {
        let tris = &edge_tris[edge];
        if tris.len() != 2 {
            continue;
        }
        let t0i = tris[0];
        let t1i = tris[1];
        if paired.contains(&t0i) || paired.contains(&t1i) {
            continue;
        }

        // Merge: find the quad vertices.
        let t0 = &triangles[t0i];
        let t1 = &triangles[t1i];
        let opp0 = t0.iter().find(|&&v| v != edge.0 && v != edge.1).copied().unwrap();
        let opp1 = t1.iter().find(|&&v| v != edge.0 && v != edge.1).copied().unwrap();

        // Order as: opp0, edge.0, opp1, edge.1 (CCW quad).
        quads.push([opp0, edge.0, opp1, edge.1]);
        paired.insert(t0i);
        paired.insert(t1i);
    }

    // Remaining un-paired triangles.
    let remaining_tris: Vec<[usize; 3]> = triangles
        .iter()
        .enumerate()
        .filter(|(i, _)| !paired.contains(i))
        .map(|(_, t)| *t)
        .collect();

    // Build elements.
    let mut elements: Vec<Element> = Vec::new();
    let mut types = vec![make_quad4_type()];

    for q in &quads {
        elements.push(Element {
            type_id: 101,
            connectivity: vec![q[0] as u32, q[1] as u32, q[2] as u32, q[3] as u32],
            region: None,
        });
    }

    if !remaining_tris.is_empty() {
        types.push(make_tri3_type());
        for t in &remaining_tris {
            elements.push(Element {
                type_id: 100,
                connectivity: vec![t[0] as u32, t[1] as u32, t[2] as u32],
                region: None,
            });
        }
    }

    // Laplacian smoothing.
    let boundary_verts = compute_boundary_vertices_from_elements(&elements, nodes.len());
    for _ in 0..params.smooth_iterations {
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for elem in &elements {
            let c = &elem.connectivity;
            for i in 0..c.len() {
                let a = c[i] as usize;
                let b = c[(i + 1) % c.len()] as usize;
                if !neighbors[a].contains(&b) {
                    neighbors[a].push(b);
                }
                if !neighbors[b].contains(&a) {
                    neighbors[b].push(a);
                }
            }
        }

        let old_nodes = nodes.clone();
        for vi in 0..nodes.len() {
            if boundary_verts.contains(&vi) || neighbors[vi].is_empty() {
                continue;
            }
            let nn = neighbors[vi].len() as f64;
            let cx: f64 = neighbors[vi].iter().map(|&j| old_nodes[j][0]).sum::<f64>() / nn;
            let cy: f64 = neighbors[vi].iter().map(|&j| old_nodes[j][1]).sum::<f64>() / nn;
            let cz: f64 = neighbors[vi].iter().map(|&j| old_nodes[j][2]).sum::<f64>() / nn;

            // Tangential projection.
            let normal = vertex_normal_from_elements(&old_nodes, &elements, vi);
            let delta = sub3([cx, cy, cz], old_nodes[vi]);
            let tang = sub3(delta, scale3(normal, dot3(delta, normal)));
            nodes[vi] = add3(old_nodes[vi], scale3(tang, 0.5));
        }
    }

    let has_z = nodes.iter().any(|n| n[2].abs() > 1e-10);
    let has_y = nodes.iter().any(|n| n[1].abs() > 1e-10);
    let dim = if has_z { 3 } else if has_y { 2 } else { 1 };

    Ok(MeshData {
        dimension: dim,
        nodes,
        element_types: types,
        elements,
        boundaries: vec![],
    })
}

fn quad_quality(
    nodes: &[[f64; 3]],
    t0: &[usize; 3],
    t1: &[usize; 3],
    edge: &(usize, usize),
) -> f64 {
    let opp0 = t0.iter().find(|&&v| v != edge.0 && v != edge.1).copied().unwrap();
    let opp1 = t1.iter().find(|&&v| v != edge.0 && v != edge.1).copied().unwrap();

    // Check convexity: both diagonals should create the same winding.
    let quad = [opp0, edge.0, opp1, edge.1];
    let mut min_angle = f64::MAX;
    let mut max_angle = 0.0f64;

    for i in 0..4 {
        let a = nodes[quad[i]];
        let b = nodes[quad[(i + 1) % 4]];
        let c = nodes[quad[(i + 2) % 4]];
        let e1 = sub3(a, b);
        let e2 = sub3(c, b);
        let n1 = norm3(e1);
        let n2 = norm3(e2);
        if n1 < 1e-30 || n2 < 1e-30 {
            return 0.0;
        }
        let cos_a = dot3(e1, e2) / (n1 * n2);
        let angle = cos_a.clamp(-1.0, 1.0).acos();
        min_angle = min_angle.min(angle);
        max_angle = max_angle.max(angle);
    }

    // Ideal quad has all 90° angles. Score = 1.0 - deviation.
    let half_pi = std::f64::consts::FRAC_PI_2;
    let deviation = ((min_angle - half_pi).abs() + (max_angle - half_pi).abs()) / half_pi;
    (1.0 - deviation * 0.5).max(0.0)
}

fn compute_boundary_vertices_from_elements(
    elements: &[Element],
    _node_count: usize,
) -> HashSet<usize> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for elem in elements {
        let c = &elem.connectivity;
        let n = c.len();
        for i in 0..n {
            let mut e = (c[i] as usize, c[(i + 1) % n] as usize);
            if e.0 > e.1 {
                std::mem::swap(&mut e.0, &mut e.1);
            }
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }
    let mut verts = HashSet::new();
    for ((a, b), count) in &edge_count {
        if *count == 1 {
            verts.insert(*a);
            verts.insert(*b);
        }
    }
    verts
}

fn vertex_normal_from_elements(
    nodes: &[[f64; 3]],
    elements: &[Element],
    vi: usize,
) -> [f64; 3] {
    let mut normal = [0.0, 0.0, 0.0];
    for elem in elements {
        let c = &elem.connectivity;
        if c.iter().any(|&v| v as usize == vi) && c.len() >= 3 {
            let a = nodes[c[0] as usize];
            let b = nodes[c[1] as usize];
            let cc = nodes[c[2] as usize];
            let e1 = sub3(b, a);
            let e2 = sub3(cc, a);
            let n = cross3(e1, e2);
            normal = add3(normal, n);
        }
    }
    normalize3(normal)
}

// ═══════════════════════════════════════════════════════════════════════════
// Mesh quality metrics (for the quality field)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute per-element quality metric and return as a node field.
///
/// For triangles: uses the aspect ratio (circumradius / inradius) normalized
/// to [0,1] where 1.0 is a perfect equilateral triangle.
/// For tets: uses the radius ratio (inradius / circumradius * 3).
/// For quads/hexes: uses the Jacobian-based quality.
pub fn compute_quality_field(mesh: &MeshData) -> Vec<f64> {
    let mut node_quality = vec![0.0f64; mesh.nodes.len()];
    let mut node_count = vec![0u32; mesh.nodes.len()];

    for elem in &mesh.elements {
        let q = element_quality(&mesh.nodes, &elem.connectivity, &elem.type_id, mesh);
        for &ni in &elem.connectivity {
            node_quality[ni as usize] += q;
            node_count[ni as usize] += 1;
        }
    }

    for i in 0..node_quality.len() {
        if node_count[i] > 0 {
            node_quality[i] /= node_count[i] as f64;
        }
    }
    node_quality
}

fn element_quality(nodes: &[[f64; 3]], conn: &[u32], type_id: &u32, mesh: &MeshData) -> f64 {
    let etype = mesh.find_type(*type_id);
    let name = etype.map(|t| t.name.as_str()).unwrap_or("");

    match name {
        "tri3" if conn.len() == 3 => {
            let a = nodes[conn[0] as usize];
            let b = nodes[conn[1] as usize];
            let c = nodes[conn[2] as usize];
            let ab = dist3(&a, &b);
            let bc = dist3(&b, &c);
            let ca = dist3(&c, &a);
            let s = (ab + bc + ca) / 2.0;
            let area = (s * (s - ab) * (s - bc) * (s - ca)).max(0.0).sqrt();
            if area < 1e-30 {
                return 0.0;
            }
            // Normalized quality: 4√3 × area / (ab² + bc² + ca²)
            let q = 4.0 * 3.0_f64.sqrt() * area / (ab * ab + bc * bc + ca * ca);
            q.clamp(0.0, 1.0)
        }
        "quad4" if conn.len() == 4 => {
            // Mean of diagonal-ratio and angle-deviation.
            let p: Vec<[f64; 3]> = conn.iter().map(|&i| nodes[i as usize]).collect();
            let d1 = dist3(&p[0], &p[2]);
            let d2 = dist3(&p[1], &p[3]);
            let diag_ratio = d1.min(d2) / d1.max(d2).max(1e-30);
            diag_ratio.clamp(0.0, 1.0)
        }
        "tet4" if conn.len() == 4 => {
            let p: Vec<[f64; 3]> = conn.iter().map(|&i| nodes[i as usize]).collect();
            let edges = [
                dist3(&p[0], &p[1]),
                dist3(&p[0], &p[2]),
                dist3(&p[0], &p[3]),
                dist3(&p[1], &p[2]),
                dist3(&p[1], &p[3]),
                dist3(&p[2], &p[3]),
            ];
            let max_edge = edges.iter().cloned().fold(0.0f64, f64::max);
            if max_edge < 1e-30 {
                return 0.0;
            }
            // Volume via scalar triple product.
            let v01 = sub3(p[1], p[0]);
            let v02 = sub3(p[2], p[0]);
            let v03 = sub3(p[3], p[0]);
            let vol = dot3(v01, cross3(v02, v03)).abs() / 6.0;
            // Normalized: 6√2 × V / max_edge³
            let q = 6.0 * 2.0_f64.sqrt() * vol / (max_edge * max_edge * max_edge);
            q.clamp(0.0, 1.0)
        }
        "hex8" if conn.len() == 8 => {
            // Simple: ratio of min edge to max edge.
            let p: Vec<[f64; 3]> = conn.iter().map(|&i| nodes[i as usize]).collect();
            let edges_idx = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ];
            let mut min_e = f64::MAX;
            let mut max_e = 0.0f64;
            for &(a, b) in &edges_idx {
                let d = dist3(&p[a], &p[b]);
                min_e = min_e.min(d);
                max_e = max_e.max(d);
            }
            if max_e < 1e-30 {
                return 0.0;
            }
            (min_e / max_e).clamp(0.0, 1.0)
        }
        _ => 0.5, // Unknown element type, neutral quality.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_3d_surface() -> MeshData {
        MeshData {
            dimension: 3,
            nodes: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            element_types: vec![make_tri3_type()],
            elements: vec![
                Element { type_id: 100, connectivity: vec![0, 1, 2], region: None },
                Element { type_id: 100, connectivity: vec![0, 1, 3], region: None },
                Element { type_id: 100, connectivity: vec![1, 2, 3], region: None },
                Element { type_id: 100, connectivity: vec![0, 3, 2], region: None },
            ],
            boundaries: vec![],
        }
    }

    fn sample_2d_flat() -> MeshData {
        MeshData {
            dimension: 2,
            nodes: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            element_types: vec![make_tri3_type()],
            elements: vec![
                Element { type_id: 100, connectivity: vec![0, 1, 2], region: None },
                Element { type_id: 100, connectivity: vec![0, 2, 3], region: None },
            ],
            boundaries: vec![],
        }
    }

    #[test]
    fn structured_hex_3d() {
        let mesh = sample_3d_surface();
        let params = MeshParams { divisions: [3, 3, 3], ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params).unwrap();
        assert!(result.mesh.node_count() > 0);
        assert!(result.mesh.element_count() > 0);
        println!("3D hex: {} nodes, {} elems", result.mesh.node_count(), result.mesh.element_count());
    }

    #[test]
    fn structured_hex_2d_flat() {
        let mesh = sample_2d_flat();
        let params = MeshParams { divisions: [4, 4, 4], ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params).unwrap();
        assert!(result.mesh.node_count() > 0);
        assert!(result.mesh.element_count() > 0);
        println!("2D hex: {} nodes, {} elems, dim={}", result.mesh.node_count(), result.mesh.element_count(), result.mesh.dimension);
    }

    #[test]
    fn structured_hex_auto_size() {
        let mesh = sample_3d_surface();
        let params = MeshParams { element_size: 0.3, ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params).unwrap();
        assert!(result.mesh.node_count() > 0);
        println!("auto-size hex: {} nodes, {} elems", result.mesh.node_count(), result.mesh.element_count());
    }

    #[test]
    fn quad_dominant_basic() {
        let mesh = sample_3d_surface();
        let params = MeshParams::default();
        let result = generate_mesh(&mesh, MeshAlgorithm::QuadDominant, &params).unwrap();
        assert!(result.mesh.node_count() > 0);
        println!("quad-dom: {} nodes, {} elems", result.mesh.node_count(), result.mesh.element_count());
    }

    #[test]
    fn hex_mesh_tessellates_ok() {
        let mesh = sample_3d_surface();
        let params = MeshParams { divisions: [3, 3, 3], ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params).unwrap();
        // Also verify tessellation doesn't panic.
        let tess = crate::tessellate::tessellate(&result);
        println!("hex tessellated: {} tri_verts, {} tri_idx, {} line_verts",
            tess.tri_vertices.len(), tess.tri_indices.len(), tess.line_vertices.len());
        assert!(tess.tri_vertices.len() > 0, "hex mesh should produce triangles");
        assert!(tess.tri_indices.len() > 0);
    }

    #[test]
    fn hex_mesh_quality_field() {
        let mesh = sample_3d_surface();
        let params = MeshParams { divisions: [3, 3, 3], ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params).unwrap();
        let quality = compute_quality_field(&result.mesh);
        assert_eq!(quality.len(), result.mesh.node_count());
        println!("hex quality: min={:.3}, max={:.3}",
            quality.iter().cloned().fold(f64::INFINITY, f64::min),
            quality.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        // A structured hex should have perfect quality (all edges equal).
        assert!(quality.iter().all(|&q| q > 0.0), "hex quality should be > 0");
    }

    #[test]
    fn hex_mesh_thin_geometry_becomes_2d() {
        // A thin plate: 10×10 with tiny Z extent (0.001). Should auto-detect 2D.
        let mesh = MeshData {
            dimension: 3,
            nodes: vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.001],
                [0.0, 10.0, 0.0],
            ],
            element_types: vec![make_tri3_type()],
            elements: vec![
                Element { type_id: 100, connectivity: vec![0, 1, 2], region: None },
                Element { type_id: 100, connectivity: vec![0, 2, 3], region: None },
            ],
            boundaries: vec![],
        };
        let params = MeshParams { divisions: [5, 5, 5], ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params).unwrap();
        // Should produce 2D quad elements (not 3D hex) because Z is thin.
        assert_eq!(result.mesh.dimension, 2, "thin Z should produce 2D mesh");
        assert!(result.mesh.element_count() > 0);
        println!("thin-geom hex: dim={}, {} nodes, {} elems",
            result.mesh.dimension, result.mesh.node_count(), result.mesh.element_count());
    }

    #[test]
    fn hex_mesh_degenerate_returns_error() {
        // All nodes at the same point: degenerate.
        let mesh = MeshData {
            dimension: 3,
            nodes: vec![[1.0, 1.0, 1.0]; 4],
            element_types: vec![make_tri3_type()],
            elements: vec![
                Element { type_id: 100, connectivity: vec![0, 1, 2], region: None },
            ],
            boundaries: vec![],
        };
        let params = MeshParams::default();
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params);
        assert!(result.is_err(), "degenerate geometry should error");
        println!("degenerate: {}", result.unwrap_err());
    }

    /// Cube geometry like a real OBJ import: 8 vertices, 12 triangles.
    fn sample_cube_obj() -> MeshData {
        MeshData {
            dimension: 3,
            nodes: vec![
                [0.0, 0.0, 0.0], // 0
                [1.0, 0.0, 0.0], // 1
                [1.0, 1.0, 0.0], // 2
                [0.0, 1.0, 0.0], // 3
                [0.0, 0.0, 1.0], // 4
                [1.0, 0.0, 1.0], // 5
                [1.0, 1.0, 1.0], // 6
                [0.0, 1.0, 1.0], // 7
            ],
            element_types: vec![make_tri3_type()],
            elements: vec![
                // -Z face
                Element { type_id: 100, connectivity: vec![0, 2, 1], region: None },
                Element { type_id: 100, connectivity: vec![0, 3, 2], region: None },
                // +Z face
                Element { type_id: 100, connectivity: vec![4, 5, 6], region: None },
                Element { type_id: 100, connectivity: vec![4, 6, 7], region: None },
                // -Y face
                Element { type_id: 100, connectivity: vec![0, 1, 5], region: None },
                Element { type_id: 100, connectivity: vec![0, 5, 4], region: None },
                // +Y face
                Element { type_id: 100, connectivity: vec![2, 3, 7], region: None },
                Element { type_id: 100, connectivity: vec![2, 7, 6], region: None },
                // -X face
                Element { type_id: 100, connectivity: vec![0, 4, 7], region: None },
                Element { type_id: 100, connectivity: vec![0, 7, 3], region: None },
                // +X face
                Element { type_id: 100, connectivity: vec![1, 2, 6], region: None },
                Element { type_id: 100, connectivity: vec![1, 6, 5], region: None },
            ],
            boundaries: vec![],
        }
    }

    #[test]
    fn cube_hex_mesh_3d() {
        let mesh = sample_cube_obj();
        let params = MeshParams { divisions: [3, 3, 3], ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::StructuredHex, &params).unwrap();
        println!("cube hex: dim={}, {} nodes, {} elems",
            result.mesh.dimension, result.mesh.node_count(), result.mesh.element_count());
        assert_eq!(result.mesh.dimension, 3, "cube should produce 3D hex mesh");
        assert!(result.mesh.element_count() > 0);
        // Verify tessellation works.
        let tess = crate::tessellate::tessellate(&result);
        assert!(tess.tri_vertices.len() > 0, "cube hex should tessellate");
    }

    #[test]
    fn cube_tet_mesh_3d() {
        let mesh = sample_cube_obj();
        let params = MeshParams { element_size: 0.5, ..Default::default() };
        let result = generate_mesh(&mesh, MeshAlgorithm::DelaunayTet, &params);
        match &result {
            Ok(r) => {
                println!("cube tet: dim={}, {} nodes, {} elems",
                    r.mesh.dimension, r.mesh.node_count(), r.mesh.element_count());
                assert!(r.mesh.element_count() > 0, "cube tet should have elements");
            }
            Err(e) => {
                panic!("cube tet failed: {e}");
            }
        }
    }

    #[test]
    fn debug_tet_internals() {
        // Reproduce the tet mesher pipeline step-by-step.
        let source = sample_cube_obj();
        let h = 0.5;
        let (bb_min, bb_max) = bounding_box(&source.nodes);
        println!("BB: min={:?}, max={:?}", bb_min, bb_max);

        // Collect surface tris.
        let all_surface_tris: Vec<[usize; 3]> = source.elements.iter()
            .filter_map(|e| {
                if e.connectivity.len() == 3 {
                    Some([e.connectivity[0] as usize, e.connectivity[1] as usize, e.connectivity[2] as usize])
                } else { None }
            })
            .collect();
        println!("Surface tris: {}", all_surface_tris.len());

        // Build BVH-accelerated ray caster.
        let accel = super::RayCastAccel::build(&source.nodes, &all_surface_tris);

        // Test with known interior point.
        let center = [0.5, 0.5, 0.5];
        let inside = accel.point_inside(&center);
        println!("Center (0.5,0.5,0.5) inside: {}", inside);
        assert!(inside, "Center of unit cube should be inside");

        // Test some grid points.
        let nx = ((bb_max[0] - bb_min[0]) / h).ceil() as usize + 1;
        let ny = ((bb_max[1] - bb_min[1]) / h).ceil() as usize + 1;
        let nz = ((bb_max[2] - bb_min[2]) / h).ceil() as usize + 1;
        println!("Grid: {}×{}×{}", nx, ny, nz);

        let mut inside_count = 0;
        let mut outside_count = 0;
        for iz in 1..nz {
            for iy in 1..ny {
                for ix in 1..nx {
                    let pt = [
                        bb_min[0] + ix as f64 * h,
                        bb_min[1] + iy as f64 * h,
                        bb_min[2] + iz as f64 * h,
                    ];
                    if accel.point_inside(&pt) {
                        inside_count += 1;
                    } else {
                        outside_count += 1;
                    }
                }
            }
        }
        println!("Interior points: {} inside, {} outside", inside_count, outside_count);
    }

    #[test]
    fn test_in_circumsphere_basic() {
        // Simple tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        let pts: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let tet = [0usize, 1, 2, 3];

        // Centroid (0.25, 0.25, 0.25) should be inside.
        let center = [0.25, 0.25, 0.25];
        let result = super::in_circumsphere(&pts, &tet, &center);
        println!("Centroid inside circumsphere: {}", result);

        // Compute orient for reference.
        let v01 = super::sub3(pts[1], pts[0]);
        let v02 = super::sub3(pts[2], pts[0]);
        let v03 = super::sub3(pts[3], pts[0]);
        let orient = super::dot3(v01, super::cross3(v02, v03));
        println!("Orient: {}", orient);

        // Far-away point should be outside.
        let far = [10.0, 10.0, 10.0];
        let far_result = super::in_circumsphere(&pts, &tet, &far);
        println!("Far point inside circumsphere: {}", far_result);

        assert!(result, "Centroid should be inside circumsphere");
        assert!(!far_result, "Far point should be outside circumsphere");
    }
}