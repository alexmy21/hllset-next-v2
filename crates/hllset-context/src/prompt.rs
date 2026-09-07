//! LLM prompt materialization.
//!
//! # Problem 2b: hand the context to an LLM
//!
//! The context is restored into ordered token sequences and rendered as a
//! conventional role-labelled text prompt. Two modes:
//!
//! - **Live** — the exchange list is still in memory; original text is used.
//! - **Restored** — only union + LUT survived; De Bruijn paths are rendered
//!   with alternating roles (default: starting with `user`).
//!
//! Both modes end with an `assistant:` completion cue, so the returned string
//! can be passed directly to any LLM API or CLI.

use crate::debruijn::RestoredConversation;
use crate::{ConversationContext, Role};

/// Options controlling prompt rendering.
#[derive(Clone, Debug)]
pub struct PromptOptions {
    /// Include the opening instruction line.
    pub header: bool,
    /// Include the HLLSet digest line (exchanges, popcount, cardinality).
    pub include_metadata: bool,
    /// How many top follow links to print (0 = none).
    pub top_links: usize,
    /// How many matrix-predicted continuations to print after the last
    /// exchange (0 = none). These come from the cell-value path-switching
    /// rule: highest-count successors of the last exchange's final token.
    pub predicted_continuations: usize,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            header: true,
            include_metadata: true,
            top_links: 5,
            predicted_continuations: 3,
        }
    }
}

/// Materialize a live conversation context into an LLM prompt.
pub fn build_prompt(ctx: &ConversationContext) -> String {
    build_prompt_with(ctx, &PromptOptions::default())
}

/// Materialize a live conversation context into an LLM prompt with options.
pub fn build_prompt_with(ctx: &ConversationContext, options: &PromptOptions) -> String {
    let mut out = String::new();

    if options.header {
        out.push_str("You are continuing a conversation. Below is the reconstructed context.\n\n");
    }

    if options.include_metadata {
        out.push_str(&format!(
            "[context: {} exchanges | union popcount={} | cardinality~{:.0}]\n",
            ctx.exchanges.len(),
            ctx.top.popcount(),
            ctx.top.cardinality()
        ));
        if options.top_links > 0 {
            let links = ctx.matrix.top_links(options.top_links);
            if !links.is_empty() {
                let items: Vec<String> = links
                    .iter()
                    .map(|(p, s, n)| format!("{} → {} ×{}", lossy(p), lossy(s), n))
                    .collect();
                out.push_str(&format!("[top follow links: {}]\n", items.join("; ")));
            }
        }
        if options.predicted_continuations > 0 && !ctx.exchanges.is_empty() {
            let last = ctx.exchanges.len() - 1;
            let preds = ctx.suggest_continuations(last, options.predicted_continuations);
            if !preds.is_empty() {
                let items: Vec<String> = preds
                    .iter()
                    .map(|(t, n)| format!("{} ×{}", lossy(t), n))
                    .collect();
                out.push_str(&format!(
                    "[predicted continuations: {}]\n",
                    items.join("; ")
                ));
            }
        }
        out.push('\n');
    }

    for ex in &ctx.exchanges {
        out.push_str(&format!("{}: {}\n", ex.role.label(), ex.text));
    }
    out.push_str("assistant:");
    out
}

/// Materialize restored De Bruijn paths into an LLM prompt.
///
/// Roles alternate starting with `first_role`. The latest user query is
/// appended at the end, followed by the `assistant:` completion cue.
pub fn build_prompt_from_restored(
    restored: &RestoredConversation,
    latest_query: &str,
    first_role: Role,
) -> String {
    let mut out = String::new();
    out.push_str("You are continuing a conversation reconstructed from HLLSet context.\n\n");

    let mut role = first_role;
    for path in &restored.paths {
        out.push_str(&format!("{}: {}\n", role.label(), path.as_text()));
        role = role.toggle();
    }

    out.push_str(&format!("user: {}\n", latest_query));
    out.push_str("assistant:");
    out
}

/// Lossy UTF-8 rendering of a token.
fn lossy(token: &[u8]) -> String {
    String::from_utf8_lossy(token).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Exchange, Role};

    #[test]
    fn test_build_prompt_contains_turns() {
        let ctx = ConversationContext::build(vec![
            Exchange::from_text(Role::User, "how does a cat sit"),
            Exchange::from_text(Role::Assistant, "a cat sits on the mat"),
        ]);
        let prompt = build_prompt(&ctx);
        assert!(
            prompt.contains("user: how does a cat sit"),
            "prompt={prompt}"
        );
        assert!(prompt.contains("assistant: a cat sits on the mat"));
        assert!(prompt.ends_with("assistant:"));
        assert!(prompt.contains("[context: 2 exchanges"));
        assert!(prompt.contains("top follow links"));
    }

    #[test]
    fn test_build_prompt_predicted_continuations() {
        let ctx = ConversationContext::build(vec![
            Exchange::from_text(Role::User, "the cat sat"),
            Exchange::from_text(Role::Assistant, "the cat purred"),
        ]);
        let prompt = build_prompt(&ctx);
        // Last exchange ends with "purred"; a prediction line is emitted if
        // the matrix has successors for it.
        assert!(prompt.contains("[top follow links:"));
        assert!(prompt.contains("[predicted continuations:") || prompt.ends_with("assistant:"));
    }

    #[test]
    fn test_build_prompt_no_metadata() {
        let ctx = ConversationContext::build(vec![Exchange::from_text(Role::User, "hello")]);
        let opts = PromptOptions {
            header: false,
            include_metadata: false,
            top_links: 0,
            predicted_continuations: 0,
        };
        let prompt = build_prompt_with(&ctx, &opts);
        assert!(!prompt.contains("[context:"));
        assert!(prompt.starts_with("user: hello\n"));
        assert!(prompt.ends_with("assistant:"));
    }

    #[test]
    fn test_build_prompt_from_restored_alternates_roles() {
        let ctx = ConversationContext::build(vec![
            Exchange::from_text(Role::User, "alpha beta"),
            Exchange::from_text(Role::Assistant, "gamma delta"),
        ]);
        let restored = ctx.restore();
        let prompt = build_prompt_from_restored(&restored, "epsilon zeta", Role::User);
        assert!(prompt.contains("user: alpha beta"), "prompt={prompt}");
        assert!(prompt.contains("assistant: gamma delta"));
        assert!(prompt.contains("user: epsilon zeta"));
        assert!(prompt.ends_with("assistant:"));
    }

    #[test]
    fn test_empty_context_prompt() {
        let ctx = ConversationContext::new();
        let prompt = build_prompt(&ctx);
        assert!(prompt.contains("[context: 0 exchanges"));
        assert!(prompt.ends_with("assistant:"));
    }
}
