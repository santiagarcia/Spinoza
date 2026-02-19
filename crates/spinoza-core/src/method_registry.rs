//! Method registry: stores factories for operators, preconditioners, solvers,
//! and problem builders, keyed by typed descriptors.

use std::collections::BTreeMap;
use std::fmt;

use crate::capability::Capability;
use crate::{CaseSpec, LinearOperator, Preconditioner};

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

/// Legacy key describing what a factory provides or requires.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FactoryKey {
    pub equation_type: String,
    pub dimension: u8,
    pub backend: String,
}

/// Key for operator factory lookup.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct OperatorKey {
    pub equation_family: String,
    pub backend: String,
    pub dimension: u8,
    /// Function space signature, e.g. "H1Scalar", "H1Vector".
    pub space_signature: String,
    /// Element family, e.g. "quad" (2D), "hex" (3D).
    pub element_family: String,
    /// Polynomial order, e.g. 1 for Q1.
    pub order: u8,
    /// Block structure, e.g. "single_field" or "block2x2".
    pub block_structure: String,
}

/// Key for preconditioner factory lookup.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct PreconditionerKey {
    pub precond_type: String,
    pub backend: String,
    pub dimension: u8,
    /// The operator family this preconditioner is designed for.
    pub operator_family: String,
    /// Function space signature.
    pub space_signature: String,
    /// Element family.
    pub element_family: String,
    /// Polynomial order.
    pub order: u8,
    /// Block structure.
    pub block_structure: String,
}

/// Key for solver factory lookup.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct SolverKey {
    pub solver_type: String,
    pub spd: bool,
    pub complex: bool,
    /// Block structure, e.g. "single_field" or "block2x2".
    pub block_structure: String,
}

impl fmt::Display for OperatorKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "equation_family='{}', backend='{}', dimension={}, space='{}', element='{}', order={}, block='{}'",
            self.equation_family, self.backend, self.dimension,
            self.space_signature, self.element_family, self.order, self.block_structure,
        )
    }
}

impl fmt::Display for PreconditionerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "precond_type='{}', backend='{}', dimension={}, op_family='{}', space='{}', element='{}', order={}, block='{}'",
            self.precond_type, self.backend, self.dimension,
            self.operator_family, self.space_signature, self.element_family, self.order, self.block_structure,
        )
    }
}

impl fmt::Display for SolverKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "solver_type='{}', spd={}, complex={}, block='{}'",
            self.solver_type, self.spd, self.complex, self.block_structure,
        )
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of building a problem: operator, rhs, and optional preconditioner.
pub struct BuiltProblem {
    pub operator: Box<dyn LinearOperator>,
    pub rhs: Vec<f64>,
    pub preconditioner: Option<Box<dyn Preconditioner>>,
    pub field_names: Vec<String>,
    pub field_sizes: Vec<usize>,
}

/// Result from an operator factory: operator and rhs without preconditioner.
pub struct BuiltOperator {
    pub operator: Box<dyn LinearOperator>,
    pub rhs: Vec<f64>,
    pub field_names: Vec<String>,
    pub field_sizes: Vec<usize>,
}

/// Result from a solver factory.
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub solution: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Factory traits
// ---------------------------------------------------------------------------

/// A factory that produces linear operators for a given equation family,
/// backend, and dimension.
pub trait OperatorFactory: Send + Sync + 'static {
    /// Human readable factory name.
    fn name(&self) -> &str;

    /// Keys this factory can handle.
    fn supported_keys(&self) -> Vec<OperatorKey>;

    /// Capabilities this factory provides when used.
    fn capabilities_provided(&self) -> &[Capability];

    /// Capabilities that must already be available for this factory to work.
    fn capabilities_required(&self) -> &[Capability];

    /// Build the operator and rhs from a case spec.
    fn build(
        &self,
        key: &OperatorKey,
        spec: &CaseSpec,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<BuiltOperator, String>;
}

/// A factory that produces preconditioners for a given type, backend,
/// and dimension.
pub trait PreconditionerFactory: Send + Sync + 'static {
    /// Human readable factory name.
    fn name(&self) -> &str;

    /// Keys this factory can handle.
    fn supported_keys(&self) -> Vec<PreconditionerKey>;

    /// Capabilities this factory provides when used.
    fn capabilities_provided(&self) -> &[Capability];

    /// Capabilities that must already be available for this factory to work.
    fn capabilities_required(&self) -> &[Capability];

    /// Build a preconditioner given the assembled operator and case spec.
    fn build(
        &self,
        key: &PreconditionerKey,
        operator: &dyn LinearOperator,
        spec: &CaseSpec,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Box<dyn Preconditioner>, String>;
}

/// A factory that runs a linear or nonlinear solver.
pub trait SolverFactory: Send + Sync + 'static {
    /// Human readable factory name.
    fn name(&self) -> &str;

    /// Keys this factory can handle.
    fn supported_keys(&self) -> Vec<SolverKey>;

    /// Capabilities this factory provides when used.
    fn capabilities_provided(&self) -> &[Capability];

    /// Capabilities that must already be available for this factory to work.
    fn capabilities_required(&self) -> &[Capability];

    /// Solve the system and return the result.
    fn solve(
        &self,
        key: &SolverKey,
        operator: &dyn LinearOperator,
        rhs: &[f64],
        preconditioner: Option<&dyn Preconditioner>,
    ) -> Result<SolveResult, String>;
}

// ---------------------------------------------------------------------------
// ProblemBuilder trait
// ---------------------------------------------------------------------------

/// A factory that can translate a parsed CaseSpec into an executable problem.
///
/// ProblemBuilders coordinate component factories: they derive registry keys
/// from the spec, request an operator, preconditioner, and solver from the
/// registry, and compose them into a [`BuiltProblem`].
pub trait ProblemBuilder: Send + Sync + 'static {
    /// Human-readable builder name (e.g. "PoissonBuilder").
    fn name(&self) -> &str;

    /// The equation type this builder handles.
    fn equation_type(&self) -> &str;

    /// Dimensions this builder supports (2, 3, or both).
    fn supported_dimensions(&self) -> &[u8];

    /// Backends this builder supports ("assembled", "matrixfree").
    fn supported_backends(&self) -> &[&str];

    /// Build the problem from a spec, composing components from the registry.
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        spec: &CaseSpec,
        backend: &str,
        precond_hint: &str,
        nx: usize,
        ny: usize,
        nz: usize,
        registry: &MethodRegistry,
    ) -> Result<BuiltProblem, String>;
}

// ---------------------------------------------------------------------------
// ProblemBuilder selection and ambiguity (existing from Task 008)
// ---------------------------------------------------------------------------

/// Metadata about a selected builder, for traceability in solve output.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BuilderSelection {
    pub pack_name: String,
    pub builder_name: String,
}

/// Structured error when multiple builders match a query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AmbiguousBuilderError {
    pub equation_type: String,
    pub dimension: u8,
    pub backend: String,
    pub candidates: Vec<BuilderCandidate>,
}

/// One candidate in an ambiguity report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BuilderCandidate {
    pub pack_name: String,
    pub builder_name: String,
}

impl fmt::Display for AmbiguousBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ambiguous builder selection: {} builders match equation='{}', dim={}, backend='{}': ",
            self.candidates.len(),
            self.equation_type,
            self.dimension,
            self.backend,
        )?;
        for (i, c) in self.candidates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}::{}", c.pack_name, c.builder_name)?;
        }
        Ok(())
    }
}

/// Result of a builder lookup that detects conflicts.
pub enum BuilderLookup<'a> {
    /// Exactly one builder matched.
    Found {
        builder: &'a dyn ProblemBuilder,
        selection: BuilderSelection,
    },
    /// No builder matched.
    NotFound,
    /// Multiple builders matched — ambiguous.
    Ambiguous(AmbiguousBuilderError),
}

// ---------------------------------------------------------------------------
// Component selection and ambiguity (generic for operator/precond/solver)
// ---------------------------------------------------------------------------

/// Metadata about a selected component factory, for traceability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComponentSelection {
    pub pack_name: String,
    pub component_name: String,
    pub component_type: String,
}

/// One candidate in a component ambiguity report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComponentCandidate {
    pub pack_name: String,
    pub component_name: String,
}

/// Structured error when multiple component factories match a query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AmbiguousComponentError {
    pub component_type: String,
    pub key_description: String,
    pub candidates: Vec<ComponentCandidate>,
}

impl fmt::Display for AmbiguousComponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ambiguous {} selection: {} factories match {}: ",
            self.component_type,
            self.candidates.len(),
            self.key_description,
        )?;
        for (i, c) in self.candidates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}::{}", c.pack_name, c.component_name)?;
        }
        Ok(())
    }
}

/// Result of a component factory lookup with conflict detection.
pub enum ComponentLookup<'a, T: ?Sized> {
    /// Exactly one factory matched.
    Found {
        component: &'a T,
        selection: ComponentSelection,
    },
    /// No factory matched.
    NotFound,
    /// Multiple factories matched — ambiguous.
    Ambiguous(AmbiguousComponentError),
}

// ---------------------------------------------------------------------------
// Internal storage types
// ---------------------------------------------------------------------------

/// Internal storage: a builder paired with the pack that registered it.
struct RegisteredBuilder {
    builder: Box<dyn ProblemBuilder>,
    pack_name: String,
}

/// Internal storage: an operator factory paired with the pack that registered it.
struct RegisteredOperatorFactory {
    factory: Box<dyn OperatorFactory>,
    pack_name: String,
}

/// Internal storage: a preconditioner factory paired with pack name.
struct RegisteredPreconditionerFactory {
    factory: Box<dyn PreconditionerFactory>,
    pack_name: String,
}

/// Internal storage: a solver factory paired with pack name.
struct RegisteredSolverFactory {
    factory: Box<dyn SolverFactory>,
    pack_name: String,
}

// ---------------------------------------------------------------------------
// MethodRegistry
// ---------------------------------------------------------------------------

/// The method registry stores all factories discovered from packs.
pub struct MethodRegistry {
    entries: Vec<RegisteredBuilder>,
    operators: Vec<RegisteredOperatorFactory>,
    preconditioners: Vec<RegisteredPreconditionerFactory>,
    solvers: Vec<RegisteredSolverFactory>,
    capabilities: std::collections::BTreeSet<Capability>,
    /// Tracks which pack is currently being registered (set during `from_packs`).
    current_pack: Option<String>,
}

impl MethodRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            operators: Vec::new(),
            preconditioners: Vec::new(),
            solvers: Vec::new(),
            capabilities: std::collections::BTreeSet::new(),
            current_pack: None,
        }
    }

    /// Build the registry by iterating all registered method packs.
    pub fn from_packs() -> Self {
        let mut reg = Self::new();
        for pack in crate::method_pack::iter_packs() {
            for cap in pack.capabilities() {
                reg.capabilities.insert(*cap);
            }
            reg.set_current_pack(pack.name());
            pack.register(&mut reg);
            reg.clear_current_pack();
        }
        reg
    }

    /// Set the current pack name (used during registration).
    fn set_current_pack(&mut self, name: &str) {
        self.current_pack = Some(name.to_string());
    }

    /// Clear the current pack name.
    fn clear_current_pack(&mut self) {
        self.current_pack = None;
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register a problem builder.
    pub fn register_builder(&mut self, builder: Box<dyn ProblemBuilder>) {
        let pack_name = self
            .current_pack
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        self.entries.push(RegisteredBuilder { builder, pack_name });
    }

    /// Register an operator factory.
    pub fn register_operator_factory(&mut self, factory: Box<dyn OperatorFactory>) {
        let pack_name = self
            .current_pack
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        self.operators
            .push(RegisteredOperatorFactory { factory, pack_name });
    }

    /// Register a preconditioner factory.
    pub fn register_preconditioner_factory(&mut self, factory: Box<dyn PreconditionerFactory>) {
        let pack_name = self
            .current_pack
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        self.preconditioners
            .push(RegisteredPreconditionerFactory { factory, pack_name });
    }

    /// Register a solver factory.
    pub fn register_solver_factory(&mut self, factory: Box<dyn SolverFactory>) {
        let pack_name = self
            .current_pack
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        self.solvers
            .push(RegisteredSolverFactory { factory, pack_name });
    }

    /// Register a capability.
    pub fn register_capability(&mut self, cap: Capability) {
        self.capabilities.insert(cap);
    }

    // -----------------------------------------------------------------------
    // ProblemBuilder lookup
    // -----------------------------------------------------------------------

    /// Find a builder matching the given spec, backend, and dimension.
    ///
    /// Returns the first match. For conflict-aware lookup, use [`find_builder_checked`].
    pub fn find_builder(
        &self,
        equation_type: &str,
        dimension: u8,
        backend: &str,
    ) -> Option<&dyn ProblemBuilder> {
        self.entries.iter().find_map(|e| {
            if e.builder.equation_type() == equation_type
                && e.builder.supported_dimensions().contains(&dimension)
                && e.builder.supported_backends().contains(&backend)
            {
                Some(e.builder.as_ref())
            } else {
                None
            }
        })
    }

    /// Find a builder with conflict detection and traceability metadata.
    ///
    /// Returns [`BuilderLookup::Found`] with the builder and its selection
    /// metadata, [`BuilderLookup::NotFound`] when no match exists, or
    /// [`BuilderLookup::Ambiguous`] when multiple builders match.
    pub fn find_builder_checked(
        &self,
        equation_type: &str,
        dimension: u8,
        backend: &str,
    ) -> BuilderLookup<'_> {
        let matches: Vec<&RegisteredBuilder> = self
            .entries
            .iter()
            .filter(|e| {
                e.builder.equation_type() == equation_type
                    && e.builder.supported_dimensions().contains(&dimension)
                    && e.builder.supported_backends().contains(&backend)
            })
            .collect();

        match matches.len() {
            0 => BuilderLookup::NotFound,
            1 => {
                let entry = matches[0];
                BuilderLookup::Found {
                    builder: entry.builder.as_ref(),
                    selection: BuilderSelection {
                        pack_name: entry.pack_name.clone(),
                        builder_name: entry.builder.name().to_string(),
                    },
                }
            }
            _ => BuilderLookup::Ambiguous(AmbiguousBuilderError {
                equation_type: equation_type.to_string(),
                dimension,
                backend: backend.to_string(),
                candidates: matches
                    .iter()
                    .map(|e| BuilderCandidate {
                        pack_name: e.pack_name.clone(),
                        builder_name: e.builder.name().to_string(),
                    })
                    .collect(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Component factory lookup
    // -----------------------------------------------------------------------

    /// Find an operator factory matching the given key.
    pub fn find_operator(&self, key: &OperatorKey) -> ComponentLookup<'_, dyn OperatorFactory> {
        let matches: Vec<&RegisteredOperatorFactory> = self
            .operators
            .iter()
            .filter(|e| e.factory.supported_keys().contains(key))
            .collect();

        match matches.len() {
            0 => ComponentLookup::NotFound,
            1 => {
                let entry = &matches[0];
                ComponentLookup::Found {
                    component: entry.factory.as_ref(),
                    selection: ComponentSelection {
                        pack_name: entry.pack_name.clone(),
                        component_name: entry.factory.name().to_string(),
                        component_type: "operator_factory".to_string(),
                    },
                }
            }
            _ => ComponentLookup::Ambiguous(AmbiguousComponentError {
                component_type: "operator_factory".to_string(),
                key_description: key.to_string(),
                candidates: matches
                    .iter()
                    .map(|e| ComponentCandidate {
                        pack_name: e.pack_name.clone(),
                        component_name: e.factory.name().to_string(),
                    })
                    .collect(),
            }),
        }
    }

    /// Find a preconditioner factory matching the given key.
    pub fn find_preconditioner(
        &self,
        key: &PreconditionerKey,
    ) -> ComponentLookup<'_, dyn PreconditionerFactory> {
        let matches: Vec<&RegisteredPreconditionerFactory> = self
            .preconditioners
            .iter()
            .filter(|e| e.factory.supported_keys().contains(key))
            .collect();

        match matches.len() {
            0 => ComponentLookup::NotFound,
            1 => {
                let entry = &matches[0];
                ComponentLookup::Found {
                    component: entry.factory.as_ref(),
                    selection: ComponentSelection {
                        pack_name: entry.pack_name.clone(),
                        component_name: entry.factory.name().to_string(),
                        component_type: "preconditioner_factory".to_string(),
                    },
                }
            }
            _ => ComponentLookup::Ambiguous(AmbiguousComponentError {
                component_type: "preconditioner_factory".to_string(),
                key_description: key.to_string(),
                candidates: matches
                    .iter()
                    .map(|e| ComponentCandidate {
                        pack_name: e.pack_name.clone(),
                        component_name: e.factory.name().to_string(),
                    })
                    .collect(),
            }),
        }
    }

    /// Find a solver factory matching the given key.
    pub fn find_solver(&self, key: &SolverKey) -> ComponentLookup<'_, dyn SolverFactory> {
        let matches: Vec<&RegisteredSolverFactory> = self
            .solvers
            .iter()
            .filter(|e| e.factory.supported_keys().contains(key))
            .collect();

        match matches.len() {
            0 => ComponentLookup::NotFound,
            1 => {
                let entry = &matches[0];
                ComponentLookup::Found {
                    component: entry.factory.as_ref(),
                    selection: ComponentSelection {
                        pack_name: entry.pack_name.clone(),
                        component_name: entry.factory.name().to_string(),
                        component_type: "solver_factory".to_string(),
                    },
                }
            }
            _ => ComponentLookup::Ambiguous(AmbiguousComponentError {
                component_type: "solver_factory".to_string(),
                key_description: key.to_string(),
                candidates: matches
                    .iter()
                    .map(|e| ComponentCandidate {
                        pack_name: e.pack_name.clone(),
                        component_name: e.factory.name().to_string(),
                    })
                    .collect(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// All capabilities from all registered packs.
    pub fn capabilities(&self) -> &std::collections::BTreeSet<Capability> {
        &self.capabilities
    }

    /// List all registered pack names.
    pub fn pack_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> =
            crate::method_pack::iter_packs().map(|p| p.name()).collect();
        names.sort();
        names.dedup();
        names
    }

    /// Detailed pack info for the `list` command.
    pub fn pack_info(&self) -> Vec<PackInfo> {
        let mut infos: Vec<PackInfo> = crate::method_pack::iter_packs()
            .map(|p| PackInfo {
                name: p.name().to_string(),
                version: p.version().to_string(),
                capabilities: p.capabilities().iter().map(|c| c.to_string()).collect(),
            })
            .collect();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        infos
    }

    /// List all registered builders for introspection.
    pub fn builder_info(&self) -> Vec<BTreeMap<String, String>> {
        self.entries
            .iter()
            .map(|e| {
                let mut info = BTreeMap::new();
                info.insert("name".to_string(), e.builder.name().to_string());
                info.insert("pack".to_string(), e.pack_name.clone());
                info.insert(
                    "equation_type".to_string(),
                    e.builder.equation_type().to_string(),
                );
                info.insert(
                    "dimensions".to_string(),
                    format!("{:?}", e.builder.supported_dimensions()),
                );
                info.insert(
                    "backends".to_string(),
                    format!("{:?}", e.builder.supported_backends()),
                );
                info
            })
            .collect()
    }

    /// List all registered operator factories for introspection.
    pub fn operator_factory_info(&self) -> Vec<BTreeMap<String, String>> {
        self.operators
            .iter()
            .map(|e| {
                let mut info = BTreeMap::new();
                info.insert("name".to_string(), e.factory.name().to_string());
                info.insert("pack".to_string(), e.pack_name.clone());
                info.insert("type".to_string(), "operator_factory".to_string());
                let keys: Vec<String> = e
                    .factory
                    .supported_keys()
                    .iter()
                    .map(|k| format!("({}, {}, {}d, {}, {}, o{}, {})", k.equation_family, k.backend, k.dimension, k.space_signature, k.element_family, k.order, k.block_structure))
                    .collect();
                info.insert("keys".to_string(), keys.join(", "));
                info
            })
            .collect()
    }

    /// List all registered preconditioner factories for introspection.
    pub fn preconditioner_factory_info(&self) -> Vec<BTreeMap<String, String>> {
        self.preconditioners
            .iter()
            .map(|e| {
                let mut info = BTreeMap::new();
                info.insert("name".to_string(), e.factory.name().to_string());
                info.insert("pack".to_string(), e.pack_name.clone());
                info.insert("type".to_string(), "preconditioner_factory".to_string());
                let keys: Vec<String> = e
                    .factory
                    .supported_keys()
                    .iter()
                    .map(|k| format!("({}, {}, {}d, {}, {}, {}, o{}, {})", k.precond_type, k.backend, k.dimension, k.operator_family, k.space_signature, k.element_family, k.order, k.block_structure))
                    .collect();
                info.insert("keys".to_string(), keys.join(", "));
                info
            })
            .collect()
    }

    /// List all registered solver factories for introspection.
    pub fn solver_factory_info(&self) -> Vec<BTreeMap<String, String>> {
        self.solvers
            .iter()
            .map(|e| {
                let mut info = BTreeMap::new();
                info.insert("name".to_string(), e.factory.name().to_string());
                info.insert("pack".to_string(), e.pack_name.clone());
                info.insert("type".to_string(), "solver_factory".to_string());
                let keys: Vec<String> = e
                    .factory
                    .supported_keys()
                    .iter()
                    .map(|k| format!("({}, spd={}, complex={}, {})", k.solver_type, k.spd, k.complex, k.block_structure))
                    .collect();
                info.insert("keys".to_string(), keys.join(", "));
                info
            })
            .collect()
    }
}

/// Serializable pack description for the `list` command.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PackInfo {
    pub name: String,
    pub version: String,
    pub capabilities: Vec<String>,
}

impl Default for MethodRegistry {
    fn default() -> Self {
        Self::new()
    }
}
