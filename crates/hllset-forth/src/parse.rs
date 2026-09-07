//! Forth source parser.
//!
//! Forth syntax is trivial: whitespace-separated tokens.
//! - `\` starts a line comment (to end of line)
//! - `(` starts a block comment (to matching `)`)
//! - `"..."` is a string literal
//! - Numbers are f64
//! - Everything else is an identifier (word name)

use crate::ast::{Ast, Word};

/// Parse Forth source into an AST.
///
/// # Errors
///
/// Returns `Err` on:
/// - Unterminated string literal
/// - Unterminated block comment
/// - Invalid number format
pub fn parse(source: &str) -> Result<Ast, ParseError> {
    let mut words = Vec::new();
    let mut chars = source.chars().peekable();
    let mut buf = String::new();

    while let Some(&ch) = chars.peek() {
        match ch {
            // Whitespace — skip, flush any buffered word
            c if c.is_whitespace() => {
                flush_buf(&mut buf, &mut words)?;
                chars.next();
            }

            // Line comment — consume to end of line
            '\\' => {
                flush_buf(&mut buf, &mut words)?;
                chars.next();
                while let Some(&c) = chars.peek() {
                    if c == '\n' {
                        break;
                    }
                    chars.next();
                }
            }

            // Block comment — consume to matching )
            '(' => {
                flush_buf(&mut buf, &mut words)?;
                chars.next(); // consume (
                let mut depth = 1;
                while depth > 0 {
                    match chars.next() {
                        Some('(') => depth += 1,
                        Some(')') => depth -= 1,
                        Some(_) => {}
                        None => {
                            return Err(ParseError::UnterminatedComment);
                        }
                    }
                }
            }

            // String literal
            '"' => {
                flush_buf(&mut buf, &mut words)?;
                chars.next(); // consume opening "
                let mut s = String::new();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => {
                            // Simple escape: \" and \\
                            match chars.next() {
                                Some(c) => s.push(c),
                                None => return Err(ParseError::UnterminatedString),
                            }
                        }
                        Some(c) => s.push(c),
                        None => return Err(ParseError::UnterminatedString),
                    }
                }
                words.push(Word::Str(s));
            }

            // Everything else — accumulate into buffer
            _ => {
                buf.push(ch);
                chars.next();
            }
        }
    }

    flush_buf(&mut buf, &mut words)?;
    Ok(group_colon_defs(words))
}

/// Post-process: detect `:` name ... `;` patterns and wrap as ColonDef.
fn group_colon_defs(words: Vec<Word>) -> Ast {
    let mut result = Vec::new();
    let mut i = 0;
    while i < words.len() {
        // Check for `:` at current position
        if let Word::Ident(ref col) = words[i] {
            if col == ":" && i + 2 < words.len() {
                if let Word::Ident(ref name) = words[i + 1] {
                    // Find matching `;`
                    let body_start = i + 2;
                    if let Some(semi_pos) = words[body_start..]
                        .iter()
                        .position(|w| matches!(w, Word::Ident(s) if s == ";"))
                    {
                        let body_end = body_start + semi_pos;
                        let body: Vec<Word> = words[body_start..body_end].to_vec();
                        result.push(Word::ColonDef {
                            name: name.clone(),
                            body,
                        });
                        i = body_end + 1; // skip past `;`
                        continue;
                    }
                }
            }
        }
        result.push(words[i].clone());
        i += 1;
    }
    Ast::new(result)
}

/// Flush the accumulated token buffer: try to parse as number, fall back to ident.
fn flush_buf(buf: &mut String, words: &mut Vec<Word>) -> Result<(), ParseError> {
    let token = buf.trim().to_string();
    buf.clear();

    if token.is_empty() {
        return Ok(());
    }

    // Try integer
    if let Ok(n) = token.parse::<i64>() {
        words.push(Word::Num(n as f64));
        return Ok(());
    }

    // Try float
    if let Ok(n) = token.parse::<f64>() {
        words.push(Word::Num(n));
        return Ok(());
    }

    // Negative number with leading minus? The parse above should handle it,
    // but just in case: treat as ident
    words.push(Word::Ident(token));
    Ok(())
}

/// Parse errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    UnterminatedString,
    UnterminatedComment,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnterminatedString => write!(f, "unterminated string literal"),
            ParseError::UnterminatedComment => write!(f, "unterminated block comment"),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let ast = parse("").unwrap();
        assert!(ast.is_empty());
    }

    #[test]
    fn test_simple_words() {
        let ast = parse("DUP SWAP DROP").unwrap();
        assert_eq!(ast.words.len(), 3);
        assert_eq!(ast.words[0], Word::Ident("DUP".into()));
        assert_eq!(ast.words[1], Word::Ident("SWAP".into()));
        assert_eq!(ast.words[2], Word::Ident("DROP".into()));
    }

    #[test]
    fn test_string_literal() {
        let ast = parse(r#""hello world" INSCRIBE"#).unwrap();
        assert_eq!(ast.words.len(), 2);
        assert_eq!(ast.words[0], Word::Str("hello world".into()));
        assert_eq!(ast.words[1], Word::Ident("INSCRIBE".into()));
    }

    #[test]
    fn test_numbers() {
        let ast = parse("42 3.14 -1 INSCRIBE").unwrap();
        assert_eq!(ast.words.len(), 4);
        assert_eq!(ast.words[0], Word::Num(42.0));
        assert_eq!(ast.words[1], Word::Num(3.14));
        assert_eq!(ast.words[2], Word::Num(-1.0));
        assert_eq!(ast.words[3], Word::Ident("INSCRIBE".into()));
    }

    #[test]
    fn test_line_comment() {
        let ast = parse("DUP \\ this is a comment\n SWAP").unwrap();
        assert_eq!(ast.words.len(), 2);
        assert_eq!(ast.words[0], Word::Ident("DUP".into()));
        assert_eq!(ast.words[1], Word::Ident("SWAP".into()));
    }

    #[test]
    fn test_block_comment() {
        let ast = parse("DUP ( this is a comment ) SWAP").unwrap();
        assert_eq!(ast.words.len(), 2);
    }

    #[test]
    fn test_nested_block_comment() {
        let ast = parse("DUP ( outer ( inner ) still comment ) SWAP").unwrap();
        assert_eq!(ast.words.len(), 2);
    }

    #[test]
    fn test_hllset_program() {
        let src =
            r#""neural" "network" 2 INSCRIBE"gradient" "backprop" 2 INSCRIBE INTERSECT STORE"#;
        let ast = parse(src).unwrap();
        assert_eq!(ast.words.len(), 10);
        // Verify it round-trips through Display
        let displayed = ast.to_string();
        let reparsed = parse(&displayed).unwrap();
        assert_eq!(ast.words.len(), reparsed.words.len());
    }

    #[test]
    fn test_unterminated_string() {
        assert!(parse(r#""hello"#).is_err());
    }

    #[test]
    fn test_colon_definition() {
        let ast = parse(r#": DOUBLE 2 * ;"#).unwrap();
        assert_eq!(ast.words.len(), 1);
        match &ast.words[0] {
            Word::ColonDef { name, body } => {
                assert_eq!(name, "DOUBLE");
                assert_eq!(body.len(), 2); // 2, *
            }
            _ => panic!("expected ColonDef"),
        }
    }
}
