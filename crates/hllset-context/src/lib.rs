//! # hllset-context — conversation context over HLLSets
//!
//! This crate solves two problems for a user↔LLM conversation stored as
//! HLLSets:
//!
//! 1. **Context building** — each exchange is encoded as
//!    `1-HLLSet ∪ 2-HLLSet ∪ 3-HLLSet` (padded 2-/3-grams carry order);
//!    the conversation context is the top union of all exchanges plus a
//!    sparse follow-frequency adjacency matrix accumulated from the 2-gram
//!    layer.
//! 2. **LLM utilization** — the context is restored with a 3-gram
//!    De Bruijn walk (backed by `hllset-materialize`'s `InMemoryEngine`
//!    reverse LUT) and materialized into a conventional role-labelled text
//!    prompt that any LLM can complete.
//!
//! ```rust
//! use hllset_context::{ConversationContext, Exchange, Role};
//!
//! let ctx = ConversationContext::build(vec![
//!     Exchange::from_text(Role::User, "how does a cat sit"),
//!     Exchange::from_text(Role::Assistant, "a cat sits on the mat"),
//! ]);
//!
//! // The context HLLSet:
//! let popcount = ctx.top.hll.popcount();
//!
//! // Follow-frequency matrix (the 2-gram layer):
//! let row = hllset_context::ngrams::token_bit_position(b"cat");
//! let col = hllset_context::ngrams::token_bit_position(b"sits");
//! let follows = ctx.matrix.get(row, col);
//!
//! // Restore the exchanges from the union + LUT only:
//! let restored = ctx.restore();
//!
//! // And hand the whole thing to an LLM as a text prompt:
//! let prompt = ctx.to_prompt();               // live mode
//! let prompt = restored.to_prompt("what about a dog"); // restored mode
//! ```
//!
//! Conventions are bit-identical to `hllset-bridge`'s 3-gram fingerprinting
//! (`_START_` / `_END_` padding, NUL-joined n-grams).

pub mod context;
pub mod debruijn;
pub mod exchange;
pub mod matrix;
pub mod ngrams;
pub mod prompt;
pub mod roundtrip;

pub use context::{ContextTop, ConversationContext, SharedBitOrigin};
pub use debruijn::{DeBruijnRestorer, RestoredConversation, RestoredPath};
pub use exchange::{tokenize_text, BitAnchors, Exchange, Role};
pub use matrix::{SparseAdjacencyMatrix, TokenIndex, UnfoldedCell};
pub use ngrams::{END_MARKER, START_MARKER};
pub use prompt::{build_prompt, build_prompt_from_restored, build_prompt_with, PromptOptions};
pub use roundtrip::{roundtrip_exchange, roundtrip_exchange_with_extra_tokens, RoundTripReport};
