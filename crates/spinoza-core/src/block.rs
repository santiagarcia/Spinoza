//! Block operator abstraction contracts.
//!
//! These minimal types allow method packs to describe and implement block
//! structured linear systems without requiring changes to core solvers.

/// Describes the layout of contiguous blocks within a single flat vector.
///
/// For a 2-field system with `n1` and `n2` DOFs, the layout is
/// `[field_0: 0..n1, field_1: n1..n1+n2]`.
#[derive(Debug, Clone)]
pub struct BlockLayout {
    /// Size of each block, in order.
    pub block_sizes: Vec<usize>,
}

impl BlockLayout {
    /// Total DOF count across all blocks.
    pub fn total_size(&self) -> usize {
        self.block_sizes.iter().sum()
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_sizes.len()
    }

    /// Return (offset, size) for block `i`.
    pub fn block_range(&self, i: usize) -> (usize, usize) {
        let offset: usize = self.block_sizes[..i].iter().sum();
        (offset, self.block_sizes[i])
    }
}
