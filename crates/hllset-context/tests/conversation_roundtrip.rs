//! End-to-end conversation roundtrip:
//! build context → inspect matrix → drop exchanges (union + LUT only) →
//! De Bruijn restore → materialize prompt.

use hllset_context::{build_prompt_from_restored, ngrams, ConversationContext, Exchange, Role};

#[test]
fn full_conversation_roundtrip() {
    let exchanges = vec![
        Exchange::from_text(Role::User, "how does a cat sit"),
        Exchange::from_text(Role::Assistant, "a cat sits on the mat"),
        Exchange::from_text(Role::User, "what about a dog"),
        Exchange::from_text(Role::Assistant, "a dog sits on the floor"),
    ];
    let ctx = ConversationContext::build(exchanges);

    // Context building: top union must equal the union of exchanges.
    let mut manual = ctx.exchanges[0].hll.clone();
    for ex in &ctx.exchanges[1..] {
        manual = manual.union(&ex.hll);
    }
    assert_eq!(ctx.top.hll.popcount(), manual.popcount());
    assert!(ctx.top.hll_2.popcount() > 0);
    assert!(ctx.top.hll_3.popcount() > 0);

    // Adjacency matrix: "a → cat" occurs in two exchanges.
    let row_a = ngrams::token_bit_position(b"a");
    let col_cat = ngrams::token_bit_position(b"cat");
    assert_eq!(ctx.matrix.get(row_a, col_cat), 2);
    // "cat → sit" (exchange 1) and "cat → sits" (exchange 2) each once.
    let row_cat = ngrams::token_bit_position(b"cat");
    assert_eq!(
        ctx.matrix.get(row_cat, ngrams::token_bit_position(b"sit")),
        1
    );
    assert_eq!(
        ctx.matrix.get(row_cat, ngrams::token_bit_position(b"sits")),
        1
    );
    // "sits" follows "dog" in one exchange.
    let row_dog = ngrams::token_bit_position(b"dog");
    assert_eq!(
        ctx.matrix.get(row_dog, ngrams::token_bit_position(b"sits")),
        1
    );

    // Restore paths from the union + LUT only.
    let restored = ctx.restore();
    assert!(
        restored.len() >= 4,
        "expected 4 restored paths, got {}",
        restored.len()
    );

    let texts: Vec<String> = restored.paths.iter().map(|p| p.as_text()).collect();
    assert!(
        texts.contains(&"how does a cat sit".to_string()),
        "texts={texts:?}"
    );
    assert!(texts.contains(&"a cat sits on the mat".to_string()));
    assert!(texts.contains(&"what about a dog".to_string()));
    assert!(texts.contains(&"a dog sits on the floor".to_string()));

    // Confidence is bounded and non-trivial.
    assert!((0.0..=1.0).contains(&restored.confidence));
    assert!(restored.confidence > 0.0);

    // Prompts in both modes.
    let live = ctx.to_prompt();
    assert!(live.contains("user: how does a cat sit"));
    assert!(live.ends_with("assistant:"));

    let restored_prompt =
        build_prompt_from_restored(&restored, "and what about a bird", Role::User);
    assert!(restored_prompt.contains("user: and what about a bird"));
    assert!(restored_prompt.ends_with("assistant:"));
}
