//! Abstract Syntax Tree for Forth source.
//!
//! The AST is the canonical representation of HLLSet operations.
//! Every backend lowers from this tree.

/// A Forth word — the fundamental unit of computation.
#[derive(Debug, Clone, PartialEq)]
pub enum Word {
    /// Push a string literal onto the stack.
    Str(String),
    /// Push a number onto the stack.
    Num(f64),
    /// A named word (operation or user-defined).
    Ident(String),
    /// A sequence of words (for blocks, definitions).
    Seq(Vec<Word>),
    /// A colon definition: `: NAME body... ;`
    ColonDef { name: String, body: Vec<Word> },
}

/// A complete Forth program: a sequence of words.
#[derive(Debug, Clone, PartialEq)]
pub struct Ast {
    pub words: Vec<Word>,
}

impl Ast {
    pub fn new(words: Vec<Word>) -> Self {
        Self { words }
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

impl From<Vec<Word>> for Ast {
    fn from(words: Vec<Word>) -> Self {
        Self::new(words)
    }
}

impl Word {
    /// If this is an Ident, return the name.
    pub fn as_ident(&self) -> Option<&str> {
        match self {
            Word::Ident(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// If this is a Str, return the string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Word::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// If this is a Num, return the value.
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Word::Num(n) => Some(*n),
            _ => None,
        }
    }
}

impl std::fmt::Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Word::Str(s) => write!(f, "\"{}\"", s),
            Word::Num(n) => write!(f, "{}", n),
            Word::Ident(s) => write!(f, "{}", s),
            Word::Seq(words) => {
                write!(f, "( ")?;
                for w in words {
                    write!(f, "{} ", w)?;
                }
                write!(f, ")")
            }
            Word::ColonDef { name, body } => {
                write!(f, ": {} ", name)?;
                for w in body {
                    write!(f, "{} ", w)?;
                }
                write!(f, ";")
            }
        }
    }
}

impl std::fmt::Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for w in &self.words {
            write!(f, "{} ", w)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        let ast = Ast::new(vec![
            Word::Str("hello".into()),
            Word::Str("world".into()),
            Word::Num(2.0),
            Word::Ident("INSCRIBE".into()),
        ]);
        assert_eq!(ast.to_string(), "\"hello\" \"world\" 2 INSCRIBE ");
    }
}
