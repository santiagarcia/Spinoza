//! Conformance test for the elasticity method pack.

#[test]
fn elasticity_pack_conforms() {
    let _ = spinoza_pack_elasticity::ElasticityPack;
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;
    let _ = spinoza_pack_linear_solvers::LinearSolversPack;
    spinoza_core::conformance::assert_pack_conforms("elasticity");
}
