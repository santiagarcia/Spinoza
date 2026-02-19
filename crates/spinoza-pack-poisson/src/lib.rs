//! spinoza-pack-poisson: Method pack for scalar Poisson problems.
//!
//! This pack registers:
//! - H1 scalar space support
//! - Poisson operator factory (assembled and matrixfree, 2D and 3D)
//! - Jacobi and MG preconditioner factories
//! - PoissonBuilder that composes the above from the registry
//!
//! **Note:** Solver factories (CG, GMRES) are provided by
//! `spinoza-pack-linear-solvers`, not by physics packs.
//!
//! Adding this crate as a dependency is sufficient for Spinoza to discover
//! it at link time via the inventory mechanism.

use spinoza_core::{
    BuiltOperator, BuiltProblem, Capability, CaseSpec, ComponentLookup, LinearOperator, MethodPack,
    MethodRegistry, OperatorFactory, OperatorKey, Preconditioner, PreconditionerFactory,
    PreconditionerKey, ProblemBuilder,
};
use spinoza_disc::{
    apply_constant_dirichlet_on_boundary, apply_constant_dirichlet_on_boundary_3d,
    apply_dirichlet_rhs_correction_2d, apply_dirichlet_rhs_correction_3d, assemble_poisson_system,
    assemble_poisson_system_3d, DirichletConstraints, PoissonLaplacianMF2D, PoissonLaplacianMF3D,
    StructuredHexMesh3D, StructuredQuadMesh2D,
};
use spinoza_solve::{
    JacobiPreconditioner, MultigridConfig, MultigridPreconditioner2D,
    MultigridPreconditioner3D,
};

// ---------------------------------------------------------------------------
// Pack definition
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct PoissonPack;

static CAPABILITIES: &[Capability] = &[
    Capability::SpaceH1Scalar,
    Capability::OperatorLaplacian,
    Capability::BcDirichlet,
    Capability::PrecondJacobi,
    Capability::PrecondMG,
];

impl MethodPack for PoissonPack {
    fn name(&self) -> &'static str {
        "poisson"
    }
    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
    fn capabilities(&self) -> &'static [Capability] {
        CAPABILITIES
    }
    fn register(&self, registry: &mut MethodRegistry) {
        registry.register_builder(Box::new(PoissonBuilder));
        registry.register_operator_factory(Box::new(PoissonOperatorFactory));
        registry.register_preconditioner_factory(Box::new(JacobiPrecondFactory));
        registry.register_preconditioner_factory(Box::new(MgPrecondFactory));
    }
}

spinoza_core::spinoza_register_pack!(PoissonPack);

// ---------------------------------------------------------------------------
// Operator factory
// ---------------------------------------------------------------------------

struct PoissonOperatorFactory;

impl OperatorFactory for PoissonOperatorFactory {
    fn name(&self) -> &str {
        "PoissonOperator"
    }

    fn supported_keys(&self) -> Vec<OperatorKey> {
        vec![
            OperatorKey {
                equation_family: "poisson".into(),
                backend: "assembled".into(),
                dimension: 2,
                space_signature: "H1Scalar".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            OperatorKey {
                equation_family: "poisson".into(),
                backend: "assembled".into(),
                dimension: 3,
                space_signature: "H1Scalar".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            OperatorKey {
                equation_family: "poisson".into(),
                backend: "matrixfree".into(),
                dimension: 2,
                space_signature: "H1Scalar".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            OperatorKey {
                equation_family: "poisson".into(),
                backend: "matrixfree".into(),
                dimension: 3,
                space_signature: "H1Scalar".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
        ]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::OperatorLaplacian]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[Capability::SpaceH1Scalar, Capability::BcDirichlet]
    }

    fn build(
        &self,
        key: &OperatorKey,
        spec: &CaseSpec,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<BuiltOperator, String> {
        let boundary_value = spec.bcs.last().map(|bc| bc.value).unwrap_or(0.0);
        let field_name = spec.fields[0].name.clone();

        match (key.dimension, key.backend.as_str()) {
            (2, "assembled") => build_op_2d_assembled(nx, ny, boundary_value, field_name),
            (2, "matrixfree") => build_op_2d_matrixfree(nx, ny, boundary_value, field_name),
            (3, "assembled") => build_op_3d_assembled(nx, ny, nz, boundary_value, field_name),
            (3, "matrixfree") => build_op_3d_matrixfree(nx, ny, nz, boundary_value, field_name),
            _ => Err(format!(
                "unsupported dimension/backend: dim={}, backend={}",
                key.dimension, key.backend
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Preconditioner factories
// ---------------------------------------------------------------------------

struct JacobiPrecondFactory;

impl PreconditionerFactory for JacobiPrecondFactory {
    fn name(&self) -> &str {
        "JacobiPreconditioner"
    }

    fn supported_keys(&self) -> Vec<PreconditionerKey> {
        vec![
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "assembled".into(),
                dimension: 2,
                operator_family: "poisson".into(),
                space_signature: "H1Scalar".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "assembled".into(),
                dimension: 3,
                operator_family: "poisson".into(),
                space_signature: "H1Scalar".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "matrixfree".into(),
                dimension: 2,
                operator_family: "poisson".into(),
                space_signature: "H1Scalar".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            PreconditionerKey {
                precond_type: "jacobi".into(),
                backend: "matrixfree".into(),
                dimension: 3,
                operator_family: "poisson".into(),
                space_signature: "H1Scalar".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
        ]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::PrecondJacobi]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[]
    }

    fn build(
        &self,
        _key: &PreconditionerKey,
        operator: &dyn LinearOperator,
        _spec: &CaseSpec,
        _nx: usize,
        _ny: usize,
        _nz: usize,
    ) -> Result<Box<dyn Preconditioner>, String> {
        Ok(Box::new(JacobiPreconditioner::from_operator(operator)?))
    }
}

struct MgPrecondFactory;

impl PreconditionerFactory for MgPrecondFactory {
    fn name(&self) -> &str {
        "MultigridPreconditioner"
    }

    fn supported_keys(&self) -> Vec<PreconditionerKey> {
        vec![
            PreconditionerKey {
                precond_type: "mg".into(),
                backend: "matrixfree".into(),
                dimension: 2,
                operator_family: "poisson".into(),
                space_signature: "H1Scalar".into(),
                element_family: "quad".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
            PreconditionerKey {
                precond_type: "mg".into(),
                backend: "matrixfree".into(),
                dimension: 3,
                operator_family: "poisson".into(),
                space_signature: "H1Scalar".into(),
                element_family: "hex".into(),
                order: 1,
                block_structure: "single_field".into(),
            },
        ]
    }

    fn capabilities_provided(&self) -> &[Capability] {
        &[Capability::PrecondMG]
    }

    fn capabilities_required(&self) -> &[Capability] {
        &[Capability::OperatorLaplacian]
    }

    fn build(
        &self,
        key: &PreconditionerKey,
        _operator: &dyn LinearOperator,
        spec: &CaseSpec,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Box<dyn Preconditioner>, String> {
        let boundary_value = spec.bcs.last().map(|bc| bc.value).unwrap_or(0.0);
        let mg_cfg = MultigridConfig::default();
        match key.dimension {
            2 => Ok(Box::new(
                MultigridPreconditioner2D::new(nx, ny, boundary_value, mg_cfg)
                    .map_err(|e| format!("mg preconditioner build failed: {e}"))?,
            )),
            3 => Ok(Box::new(
                MultigridPreconditioner3D::new(nx, ny, nz, boundary_value, mg_cfg)
                    .map_err(|e| format!("mg preconditioner build failed: {e}"))?,
            )),
            _ => Err(format!("MG not supported for dimension {}", key.dimension)),
        }
    }
}

// ---------------------------------------------------------------------------
// Problem builder (composes from registry)
// ---------------------------------------------------------------------------

struct PoissonBuilder;

impl ProblemBuilder for PoissonBuilder {
    fn name(&self) -> &str {
        "PoissonBuilder"
    }

    fn equation_type(&self) -> &str {
        "poisson"
    }

    fn supported_dimensions(&self) -> &[u8] {
        &[2, 3]
    }

    fn supported_backends(&self) -> &[&str] {
        &["assembled", "matrixfree"]
    }

    fn build(
        &self,
        spec: &CaseSpec,
        backend: &str,
        precond_hint: &str,
        nx: usize,
        ny: usize,
        nz: usize,
        registry: &MethodRegistry,
    ) -> Result<BuiltProblem, String> {
        let dim = spec.mesh.dimension;
        let elem = if dim == 2 { "quad" } else { "hex" };

        // 1. Request operator from registry.
        let op_key = OperatorKey {
            equation_family: "poisson".into(),
            backend: backend.into(),
            dimension: dim,
            space_signature: "H1Scalar".into(),
            element_family: elem.into(),
            order: 1,
            block_structure: "single_field".into(),
        };
        let built_op = match registry.find_operator(&op_key) {
            ComponentLookup::Found { component, .. } => {
                component.build(&op_key, spec, nx, ny, nz)?
            }
            ComponentLookup::NotFound => {
                return Err(format!(
                    "no operator factory for equation='poisson', backend='{backend}', dim={dim}"
                ));
            }
            ComponentLookup::Ambiguous(err) => return Err(err.to_string()),
        };

        // 2. Request preconditioner from registry.
        let pc_key = PreconditionerKey {
            precond_type: precond_hint.into(),
            backend: backend.into(),
            dimension: dim,
            operator_family: "poisson".into(),
            space_signature: "H1Scalar".into(),
            element_family: elem.into(),
            order: 1,
            block_structure: "single_field".into(),
        };
        let precond = match registry.find_preconditioner(&pc_key) {
            ComponentLookup::Found { component, .. } => {
                Some(component.build(&pc_key, built_op.operator.as_ref(), spec, nx, ny, nz)?)
            }
            ComponentLookup::NotFound => None,
            ComponentLookup::Ambiguous(err) => return Err(err.to_string()),
        };

        Ok(BuiltProblem {
            operator: built_op.operator,
            rhs: built_op.rhs,
            preconditioner: precond,
            field_names: built_op.field_names,
            field_sizes: built_op.field_sizes,
        })
    }
}

// ---------------------------------------------------------------------------
// Operator build helpers
// ---------------------------------------------------------------------------

fn build_op_2d_assembled(
    nx: usize,
    ny: usize,
    boundary_value: f64,
    field_name: String,
) -> Result<BuiltOperator, String> {
    let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
    let (mut matrix, mut rhs) = assemble_poisson_system(&mesh);
    apply_constant_dirichlet_on_boundary(&mesh, &mut matrix, &mut rhs, boundary_value);
    let csr = matrix.to_csr();
    let n = csr.size();
    Ok(BuiltOperator {
        operator: Box::new(csr),
        rhs,
        field_names: vec![field_name],
        field_sizes: vec![n],
    })
}

fn build_op_2d_matrixfree(
    nx: usize,
    ny: usize,
    boundary_value: f64,
    field_name: String,
) -> Result<BuiltOperator, String> {
    let mesh = StructuredQuadMesh2D::unit_square(nx, ny);
    let ndofs = mesh.num_nodes();
    let mut rhs = vec![0.0; ndofs];
    let bc_values: Vec<(usize, f64)> = mesh
        .boundary_nodes()
        .into_iter()
        .map(|idx| (idx, boundary_value))
        .collect();
    let constraints = DirichletConstraints::new(ndofs, &bc_values);
    apply_dirichlet_rhs_correction_2d(&mesh, &mut rhs, &constraints);
    let op = PoissonLaplacianMF2D::new(mesh, constraints);
    Ok(BuiltOperator {
        operator: Box::new(op),
        rhs,
        field_names: vec![field_name],
        field_sizes: vec![ndofs],
    })
}

fn build_op_3d_assembled(
    nx: usize,
    ny: usize,
    nz: usize,
    boundary_value: f64,
    field_name: String,
) -> Result<BuiltOperator, String> {
    let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
    let (mut matrix, mut rhs) = assemble_poisson_system_3d(&mesh);
    apply_constant_dirichlet_on_boundary_3d(&mesh, &mut matrix, &mut rhs, boundary_value);
    let csr = matrix.to_csr();
    let n = csr.size();
    Ok(BuiltOperator {
        operator: Box::new(csr),
        rhs,
        field_names: vec![field_name],
        field_sizes: vec![n],
    })
}

fn build_op_3d_matrixfree(
    nx: usize,
    ny: usize,
    nz: usize,
    boundary_value: f64,
    field_name: String,
) -> Result<BuiltOperator, String> {
    let mesh = StructuredHexMesh3D::unit_cube(nx, ny, nz);
    let ndofs = mesh.num_nodes();
    let mut rhs = vec![0.0; ndofs];
    let bc_values: Vec<(usize, f64)> = mesh
        .boundary_nodes()
        .into_iter()
        .map(|idx| (idx, boundary_value))
        .collect();
    let constraints = DirichletConstraints::new(ndofs, &bc_values);
    apply_dirichlet_rhs_correction_3d(&mesh, &mut rhs, &constraints);
    let op = PoissonLaplacianMF3D::new(mesh, constraints);
    Ok(BuiltOperator {
        operator: Box::new(op),
        rhs,
        field_names: vec![field_name],
        field_sizes: vec![ndofs],
    })
}

// Re-export mesh types for use by other packs.
pub use spinoza_disc::{
    CsrMatrix, DirichletConstraints as PoissonDirichletConstraints, PoissonLaplacianMF2D as MF2D,
    PoissonLaplacianMF3D as MF3D, SparseMatrixBuilder, StructuredHexMesh3D as HexMesh3D,
    StructuredQuadMesh2D as QuadMesh2D,
};
pub use spinoza_solve::{
    JacobiPreconditioner as JacobiPC, MultigridConfig as MGConfig,
    MultigridPreconditioner2D as MG2D, MultigridPreconditioner3D as MG3D,
};
