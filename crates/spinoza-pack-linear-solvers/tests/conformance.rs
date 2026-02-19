//! Conformance test for the linear-solvers method pack.

#[test]
fn linear_solvers_pack_conforms() {
    // Anchor all packs.
    let _ = spinoza_pack_linear_solvers::LinearSolversPack;
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_elasticity::ElasticityPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;
    spinoza_core::conformance::assert_pack_conforms("linear_solvers");
}
