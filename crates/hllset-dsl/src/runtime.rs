//! DSL runtime — Lua VM with HLLSet algebra bindings.
//!
//! `DslRuntime` wraps an `mlua` Lua VM and registers the global `hllset` table
//! with `inscribe()` and `LatticeElement` userdata. Lua scripts can use
//! `+`, `*`, `-` operators and `#` for cardinality.

use crate::lattice::LatticeElement;
use crate::tokenizer::Tokenizer;
use hllset_storage::{MemoryStorage, Storage};
use mlua::prelude::*;
use mlua::{FromLua, FromLuaMulti, MetaMethod, Table, UserData, UserDataMethods, Value};
use std::collections::HashMap;
use std::rc::Rc;

/// The DSL runtime: a Lua VM with HLLSet algebra bindings.
///
/// # Example
///
/// ```rust
/// use hllset_dsl::DslRuntime;
///
/// let mut rt = DslRuntime::new().unwrap();
/// let count = rt.eval::<f64>(r#"
///     local a = hllset.inscribe({"hello", "world", "lua"})
///     return #a
/// "#).unwrap();
/// assert!(count > 0.0);
/// ```
pub struct DslRuntime {
    lua: Lua,
    registry: HashMap<String, LatticeElement>,
    storage: Rc<dyn Storage>,
}

impl DslRuntime {
    /// Access the storage backend.
    pub fn storage(&self) -> &dyn Storage {
        &*self.storage
    }

    /// Create a new DSL runtime with all bindings registered.
    pub fn new() -> LuaResult<Self> {
        let lua = Lua::new();

        // Register the global `hllset` table
        let hllset_table = lua.create_table()?;
        let storage: Rc<dyn Storage> = Rc::new(MemoryStorage::new());

        // hllset.inscribe(tokens) -> LatticeElement
        let inscribe_fn = lua.create_function(|_lua, tokens: Table| {
            let mut elems: Vec<String> = Vec::new();
            for pair in tokens.pairs::<Value, Value>() {
                let (_, v) = pair?;
                if let Value::String(s) = v {
                    elems.push(s.to_str()?.to_string());
                } else if let Value::Integer(n) = v {
                    elems.push(n.to_string());
                } else if let Value::Number(n) = v {
                    elems.push(n.to_string());
                }
            }
            Ok(LatticeElement::from_tokens(&elems))
        })?;
        hllset_table.set("inscribe", inscribe_fn)?;

        // hllset.empty() -> LatticeElement (the lattice bottom ⊥)
        let empty_fn = lua.create_function(|_lua, ()| Ok(LatticeElement::empty()))?;
        hllset_table.set("empty", empty_fn)?;

        // hllset.tokenize(text) -> LatticeElement (default word tokenizer)
        use crate::tokenizer::Tokenizer;
        let tokenize_fn = lua.create_function(|_lua, text: String| {
            let tok = Tokenizer::new().lowercase();
            Ok(tok.apply(text.as_bytes()))
        })?;
        hllset_table.set("tokenize", tokenize_fn)?;

        // hllset.tokenizer() -> Tokenizer (builder)
        let tokenizer_factory = lua.create_function(|_lua, ()| {
            let tok = Tokenizer::new();
            // Wrap in UserData and return
            Ok(tok)
        })?;
        hllset_table.set("tokenizer", tokenizer_factory)?;

        // Storage bindings — use storage Rc cloned into closures
        let storage_lua = Rc::clone(&storage);
        let store_fn = lua.create_function(move |_, elem: LatticeElement| {
            let data = elem.hllset().to_bytes();
            storage_lua
                .store(elem.key(), &data)
                .map_err(|e| LuaError::external(e.to_string()))?;
            Ok(())
        })?;
        hllset_table.set("store", store_fn)?;

        let storage_lua = Rc::clone(&storage);
        let load_fn = lua.create_function(move |_, key: String| {
            match storage_lua
                .load(&key)
                .map_err(|e| LuaError::external(e.to_string()))?
            {
                Some(data) => {
                    let hllset = hllset_core::HLLSet::from_bytes(&data)
                        .ok_or_else(|| LuaError::external("invalid HLLSet data"))?;
                    Ok(Some(LatticeElement::new(hllset)))
                }
                None => Ok(None),
            }
        })?;
        hllset_table.set("load", load_fn)?;

        let storage_lua = Rc::clone(&storage);
        let exists_fn = lua.create_function(move |_, key: String| {
            storage_lua
                .exists(&key)
                .map_err(|e| LuaError::external(e.to_string()))
        })?;
        hllset_table.set("exists", exists_fn)?;

        let storage_lua = Rc::clone(&storage);
        let list_fn = lua.create_function(move |_, prefix: String| {
            storage_lua
                .list(&prefix)
                .map_err(|e| LuaError::external(e.to_string()))
        })?;
        hllset_table.set("list", list_fn)?;

        let storage_lua = Rc::clone(&storage);
        let pin_fn = lua.create_function(move |_, key: String| {
            storage_lua
                .pin(&key)
                .map_err(|e| LuaError::external(e.to_string()))
        })?;
        hllset_table.set("pin", pin_fn)?;

        let storage_lua = Rc::clone(&storage);
        let unpin_fn = lua.create_function(move |_, key: String| {
            storage_lua
                .unpin(&key)
                .map_err(|e| LuaError::external(e.to_string()))
        })?;
        hllset_table.set("unpin", unpin_fn)?;

        let storage_lua = Rc::clone(&storage);
        let gc_fn = lua.create_function(move |_, ()| {
            storage_lua
                .gc()
                .map_err(|e| LuaError::external(e.to_string()))
        })?;
        hllset_table.set("gc", gc_fn)?;

        // ── Temporal bindings ────────────────────────────────────────

        let storage_lua = Rc::clone(&storage);
        let get_tmp_fn = lua.create_function(move |_, key: String| {
            storage_lua
                .get_tmp(&key)
                .map_err(|e| LuaError::external(e.to_string()))
        })?;
        hllset_table.set("get_tmp", get_tmp_fn)?;

        let storage_lua = Rc::clone(&storage);
        let put_tmp_fn = lua.create_function(move |_, (key, val): (String, mlua::String)| {
            let val_bytes = val.as_bytes();
            storage_lua
                .put_tmp(&key, &val_bytes)
                .map_err(|e| LuaError::external(e.to_string()))
        })?;
        hllset_table.set("put_tmp", put_tmp_fn)?;

        let storage_lua = Rc::clone(&storage);
        let cas_tmp_fn = lua.create_function(
            move |_, (key, old, new): (String, mlua::String, mlua::String)| {
                let old_bytes = old.as_bytes();
                let new_bytes = new.as_bytes();
                storage_lua
                    .cas_tmp(&key, &old_bytes, &new_bytes)
                    .map_err(|e| LuaError::external(e.to_string()))
            },
        )?;
        hllset_table.set("cas_tmp", cas_tmp_fn)?;

        lua.globals().set("hllset", hllset_table)?;

        Ok(Self {
            lua,
            registry: HashMap::new(),
            storage,
        })
    }

    /// Evaluate a Lua expression and return the deserialized result.
    ///
    /// ```rust
    /// # use hllset_dsl::DslRuntime;
    /// # let rt = DslRuntime::new().unwrap();
    /// let sum: f64 = rt.eval("return 1 + 2").unwrap();
    /// assert_eq!(sum, 3.0);
    /// ```
    pub fn eval<T: FromLuaMulti>(&self, script: &str) -> LuaResult<T> {
        self.lua.load(script).eval()
    }

    /// Execute a Lua script (no return value).
    pub fn exec(&self, script: &str) -> LuaResult<()> {
        self.lua.load(script).exec()
    }

    /// Store a LatticeElement in the runtime registry by name.
    pub fn store(&mut self, name: &str, elem: LatticeElement) {
        self.registry.insert(name.to_string(), elem);
    }

    /// Look up a stored LatticeElement.
    pub fn get(&self, name: &str) -> Option<&LatticeElement> {
        self.registry.get(name)
    }

    /// Access the raw Lua VM (for advanced use).
    pub fn lua(&self) -> &Lua {
        &self.lua
    }

    /// Run a script that produces a LatticeElement as the return value.
    ///
    /// The script must `return` a LatticeElement userdata.
    pub fn eval_element(&self, script: &str) -> LuaResult<LatticeElement> {
        self.lua.load(script).eval()
    }
}

// ── Lua UserData for LatticeElement ─────────────────────────────────────────

impl UserData for LatticeElement {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        // ── Accessors ──

        // key() -> string: content-addressable key, e.g. "h:a3f82c..."
        methods.add_method("key", |_, this, ()| Ok(this.key().to_string()));

        // content_hash() -> string: SHA-1 hex (without prefix)
        methods.add_method("content_hash", |_, this, ()| Ok(this.content_hash()));

        // card() -> number: estimated cardinality
        methods.add_method("card", |_, this, ()| Ok(this.cardinality()));

        // popcount() -> integer: number of bits set
        methods.add_method("popcount", |_, this, ()| Ok(this.popcount()));

        // is_empty() -> boolean
        methods.add_method("is_empty", |_, this, ()| Ok(this.is_empty()));

        // ── Set algebra: methods ──

        // union(b) -> LatticeElement: a:union(b) (same as a + b)
        methods.add_method("union", |_, this, other: LatticeElement| {
            Ok(this.union(&other))
        });

        // intersection(b) -> LatticeElement: a:intersection(b) (same as a * b)
        methods.add_method("intersection", |_, this, other: LatticeElement| {
            Ok(this.intersection(&other))
        });

        // difference(b) -> LatticeElement: a:difference(b) (same as a - b)
        methods.add_method("difference", |_, this, other: LatticeElement| {
            Ok(this.difference(&other))
        });

        // ── BSS morphisms ──

        // bss_inclusion(b) -> number: BSSτ = |A∩B| / |B|
        methods.add_method("bss_inclusion", |_, this, other: LatticeElement| {
            Ok(this.bss_inclusion(&other))
        });

        // bss_exclusion(b) -> number: BSSρ = |A\B| / |B|
        methods.add_method("bss_exclusion", |_, this, other: LatticeElement| {
            Ok(this.bss_exclusion(&other))
        });

        // morph_to(b, tau_min, rho_max) -> { inclusion, exclusion, holds }
        methods.add_method(
            "morph_to",
            |lua, this, (other, tau_min, rho_max): (LatticeElement, f64, f64)| {
                let result = this.morph_to(&other, tau_min, rho_max);
                let tbl = lua.create_table()?;
                tbl.set("inclusion", result.inclusion)?;
                tbl.set("exclusion", result.exclusion)?;
                tbl.set("holds", result.morphism_holds)?;
                Ok(tbl)
            },
        );

        // ── Jaccard ──

        // jaccard(b) -> number: |A ∩ B| / |A ∪ B|
        methods.add_method("jaccard", |_, this, other: LatticeElement| {
            Ok(this.jaccard_similarity(&other))
        });

        // ── Subset relations ──

        // is_subset_of(b) -> boolean
        methods.add_method("is_subset_of", |_, this, other: LatticeElement| {
            Ok(this.is_subset_of(&other))
        });

        // is_superset_of(b) -> boolean
        methods.add_method("is_superset_of", |_, this, other: LatticeElement| {
            Ok(this.is_superset_of(&other))
        });

        // ── Meta-methods: operator overloading ──

        // #a — cardinality (Lua length operator)
        methods.add_meta_method(MetaMethod::Len, |_, this, ()| Ok(this.cardinality()));

        // a + b — union (lattice join)
        methods.add_meta_method(MetaMethod::Add, |_, this, other: LatticeElement| {
            Ok(this.union(&other))
        });

        // a * b — intersection (lattice meet)
        methods.add_meta_method(MetaMethod::Mul, |_, this, other: LatticeElement| {
            Ok(this.intersection(&other))
        });

        // a - b — set difference
        methods.add_meta_method(MetaMethod::Sub, |_, this, other: LatticeElement| {
            Ok(this.difference(&other))
        });

        // a == b — checks if keys are equal (not structural equality)
        methods.add_meta_method(MetaMethod::Eq, |_, this, other: LatticeElement| {
            Ok(this.key() == other.key())
        });

        // tostring(a) — human-readable representation
        methods.add_meta_method(MetaMethod::ToString, |_, this, ()| {
            Ok(format!(
                "LatticeElement(key={}, card={:.1}, popcount={})",
                this.key(),
                this.cardinality(),
                this.popcount()
            ))
        });
    }
}

// ── FromLua impl for LatticeElement ────────────────────────────────────────

impl FromLua for LatticeElement {
    fn from_lua(value: Value, _lua: &Lua) -> LuaResult<Self> {
        if let Value::UserData(ud) = value {
            Ok(ud.borrow::<Self>()?.clone())
        } else {
            Err(LuaError::FromLuaConversionError {
                from: value.type_name(),
                to: "LatticeElement".to_string(),
                message: None,
            })
        }
    }
}

// ── Lua UserData for Tokenizer ──────────────────────────────────────────────

impl UserData for Tokenizer {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        // word_pattern(): set pattern to default word matcher
        methods.add_method_mut("word_pattern", |_, this, ()| {
            *this = this.clone().word_pattern();
            Ok(())
        });

        // lowercase(): add lowercase normalizer
        methods.add_method_mut("lowercase", |_, this, ()| {
            *this = this.clone().lowercase();
            Ok(())
        });

        // ngrams(min, max): set n-gram range
        methods.add_method_mut("ngrams", |_, this, (min, max): (usize, usize)| {
            *this = this.clone().ngrams(min, max);
            Ok(())
        });

        // pad(start, end): add boundary tokens
        methods.add_method_mut("pad", |_, this, (start, end): (String, String)| {
            let s = start.into_bytes();
            let e = end.into_bytes();
            *this = this.clone().pad(&s, &e);
            Ok(())
        });

        // tokenize(text) -> table of byte strings
        methods.add_method("tokenize", |lua, this, text: String| {
            let tokens = this.tokenize(text.as_bytes());
            let tbl = lua.create_table()?;
            for (i, token) in tokens.iter().enumerate() {
                let s = lua.create_string(token)?;
                tbl.set(i + 1, s)?;
            }
            Ok(tbl)
        });

        // apply(text) -> LatticeElement
        methods.add_method("apply", |_, this, text: String| {
            Ok(this.apply(text.as_bytes()))
        });
    }
}

// ── FromLua impl for Tokenizer ─────────────────────────────────────────────

impl FromLua for Tokenizer {
    fn from_lua(value: Value, _lua: &Lua) -> LuaResult<Self> {
        if let Value::UserData(ud) = value {
            Ok(ud.borrow::<Self>()?.clone())
        } else {
            Err(LuaError::FromLuaConversionError {
                from: value.type_name(),
                to: "Tokenizer".to_string(),
                message: None,
            })
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_runtime() {
        let rt = DslRuntime::new().unwrap();
        let result: f64 = rt.eval("return 2 + 2").unwrap();
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_inscribe_and_cardinality() {
        let rt = DslRuntime::new().unwrap();
        let result: f64 = rt
            .eval(
                r#"
            local a = hllset.inscribe({"hello", "world", "lua"})
            return #a
        "#,
            )
            .unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_empty_cardinality_zero() {
        let rt = DslRuntime::new().unwrap();
        let result: f64 = rt
            .eval(
                r#"
            local e = hllset.empty()
            return #e
        "#,
            )
            .unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_union_operator() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local a = hllset.inscribe({"a", "b"})
            local b = hllset.inscribe({"b", "c"})
            local c = a + b
            return #c, #a, #b
        "#;
        let (c_card, a_card, b_card): (f64, f64, f64) = rt.eval(script).unwrap();
        assert!(c_card >= a_card);
        assert!(c_card >= b_card);
    }

    #[test]
    fn test_intersection_operator() {
        let rt = DslRuntime::new().unwrap();
        let result: f64 = rt
            .eval(
                r#"
            local a = hllset.inscribe({"x", "y", "z"})
            local b = hllset.inscribe({"y", "z", "w"})
            local c = a * b
            return #c
        "#,
            )
            .unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_difference_operator() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local a = hllset.inscribe({"keep", "remove"})
            local b = hllset.inscribe({"remove"})
            local c = a - b
            return #c
        "#;
        let result: f64 = rt.eval(script).unwrap();
        assert!(result >= 0.0);
    }

    #[test]
    fn test_operator_chaining() {
        let rt = DslRuntime::new().unwrap();
        let result: f64 = rt
            .eval(
                r#"
            local a = hllset.inscribe({"a", "b", "c"})
            local b = hllset.inscribe({"b", "c", "d"})
            local c = hllset.inscribe({"c", "d", "e"})
            local chain = a + b * c - a
            return #chain
        "#,
            )
            .unwrap();
        assert!(result >= 0.0);
    }

    #[test]
    fn test_bss_inclusion_method() {
        let rt = DslRuntime::new().unwrap();
        let tau: f64 = rt
            .eval(
                r#"
            local a = hllset.inscribe({"shared", "unique_a"})
            local b = hllset.inscribe({"shared", "unique_b"})
            return a:bss_inclusion(b)
        "#,
            )
            .unwrap();
        assert!(tau >= 0.0 && tau <= 1.0, "tau={tau}");
    }

    #[test]
    fn test_bss_exclusion_self() {
        let rt = DslRuntime::new().unwrap();
        let rho: f64 = rt
            .eval(
                r#"
            local a = hllset.inscribe({"test"})
            return a:bss_exclusion(a)
        "#,
            )
            .unwrap();
        assert!(rho < 0.01, "rho={rho}");
    }

    #[test]
    fn test_morph_to_method() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local a = hllset.inscribe({"shared", "only_a"})
            local b = hllset.inscribe({"shared", "only_b"})
            local result = a:morph_to(b, 0.3, 0.9)
            return result.inclusion, result.exclusion, result.holds
        "#;
        let (tau, rho, holds): (f64, f64, bool) = rt.eval(script).unwrap();
        assert!(tau >= 0.0 && tau <= 1.0, "tau={tau}");
        assert!(rho >= 0.0, "rho={rho}");
        let _ = holds;
    }

    #[test]
    fn test_key_is_deterministic() {
        let rt = DslRuntime::new().unwrap();
        let key1: String = rt
            .eval(
                r#"
            local a = hllset.inscribe({"same", "tokens"})
            return a:key()
        "#,
            )
            .unwrap();
        let key2: String = rt
            .eval(
                r#"
            local a = hllset.inscribe({"same", "tokens"})
            return a:key()
        "#,
            )
            .unwrap();
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_eval_element() {
        let rt = DslRuntime::new().unwrap();
        let elem = rt
            .eval_element(
                r#"
            local a = hllset.inscribe({"hello", "dsl"})
            return a
        "#,
            )
            .unwrap();
        assert!(elem.key().starts_with("h:"));
        assert!(elem.cardinality() > 0.0);
    }

    #[test]
    fn test_jaccard_same_is_one() {
        let rt = DslRuntime::new().unwrap();
        let j: f64 = rt
            .eval(
                r#"
            local a = hllset.inscribe({"a", "b"})
            return a:jaccard(a)
        "#,
            )
            .unwrap();
        assert!((j - 1.0).abs() < 0.02, "jaccard={j}");
    }

    #[test]
    fn test_is_subset() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local a = hllset.inscribe({"x", "y"})
            return a:is_subset_of(a)
        "#;
        let result: bool = rt.eval(script).unwrap();
        assert!(result);
    }

    #[test]
    fn test_tostring() {
        let rt = DslRuntime::new().unwrap();
        let s: String = rt
            .eval(
                r#"
            local a = hllset.inscribe({"hello"})
            return tostring(a)
        "#,
            )
            .unwrap();
        assert!(s.starts_with("LatticeElement(key=h:"));
        assert!(s.contains("popcount="));
    }

    #[test]
    fn test_store_and_get() {
        let mut rt = DslRuntime::new().unwrap();
        let elem = LatticeElement::from_tokens(&["stored"]);
        rt.store("test_elem", elem.clone());
        let retrieved = rt.get("test_elem").unwrap();
        assert_eq!(retrieved.key(), elem.key());
    }

    // ── Tokenizer Lua tests ──────────────────────────────────────────

    #[test]
    fn test_tokenize_lua_text() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local e = hllset.tokenize("The Cat Sat On The Mat")
            return #e, e:key()
        "#;
        let (card, key): (f64, String) = rt.eval(script).unwrap();
        assert!(card > 0.0);
        assert!(key.starts_with("h:"));
    }

    #[test]
    fn test_tokenize_lua_deterministic() {
        let rt = DslRuntime::new().unwrap();
        let key1: String = rt
            .eval(r#"local e = hllset.tokenize("hello world"); return e:key()"#)
            .unwrap();
        let key2: String = rt
            .eval(r#"local e = hllset.tokenize("hello world"); return e:key()"#)
            .unwrap();
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_tokenizer_factory_lua() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local tok = hllset.tokenizer()
            tok:lowercase()
            tok:word_pattern()
            local e = tok:apply("Hello Lua World")
            return #e
        "#;
        let card: f64 = rt.eval(script).unwrap();
        assert!(card > 0.0);
    }

    #[test]
    fn test_tokenizer_ngrams_lua() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local tok = hllset.tokenizer()
            tok:lowercase()
            tok:ngrams(1, 2)
            local e = tok:apply("the cat sat")
            return #e
        "#;
        let card: f64 = rt.eval(script).unwrap();
        assert!(card > 0.0);
    }

    #[test]
    fn test_tokenizer_pad_lua() {
        let rt = DslRuntime::new().unwrap();
        let script = r#"
            local tok = hllset.tokenizer()
            tok:lowercase()
            tok:pad("<S>", "</S>")
            tok:ngrams(2, 2)
            local e = tok:apply("hello world")
            return #e
        "#;
        let card: f64 = rt.eval(script).unwrap();
        assert!(card > 0.0);
    }

    #[test]
    fn test_tokenizer_tokenize_returns_tokens() {
        let rt = DslRuntime::new().unwrap();
        // tokenize() should return a table of tokens
        let script = r#"
            local tok = hllset.tokenizer()
            tok:lowercase()
            local tokens = tok:tokenize("hello world")
            return #tokens, tokens[1], tokens[2]
        "#;
        let (n, t1, t2): (usize, String, String) = rt.eval(script).unwrap();
        assert!(n >= 2, "expected >= 2 tokens, got {n}");
        assert_eq!(t1, "hello");
        assert_eq!(t2, "world");
    }
}
