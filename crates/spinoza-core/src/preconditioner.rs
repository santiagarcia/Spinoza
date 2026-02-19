//! Preconditioner interfaces for iterative solvers.

/// A linear preconditioner that approximates `A^{-1}`.
///
/// Implementations must not allocate in [`Self::apply`].
pub trait Preconditioner {
    /// Apply the preconditioner as `z = M^{-1} r`.
    fn apply(&self, r: &[f64], z: &mut [f64]);
}
