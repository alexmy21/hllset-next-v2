//! `hllset-morphisms` — the two morphisms with their operational contracts.
//!
//! **Ingest (complete, single touch).** For each token, in one pass and
//! without leaving the module:
//!
//! 1. compute the 3 seeded hashes (n-seed encodings; the n-gram regime is
//!    the same shape with channel-selected seeds),
//! 2. set the atom in the corresponding HLLSet,
//! 3. insert the token into the corresponding LUT fiber,
//! 4. increment TF.
//!
//! **Materialize (LUT-first, TF only for ambiguity).** For each active bit
//! of the sketch, collect candidate tokens from **all pointed LUTs** across
//! all encodings; only when a bit resolves to more than one candidate does
//! the LUT's TF break the tie. TF is never the starting point — normally
//! people start with TF; this module never does.

pub mod ingest;
pub mod materialize;
pub mod tf;

pub use ingest::{Ingest, N_SEEDS, SEEDS};
pub use materialize::materialize;
pub use tf::TfTable;
