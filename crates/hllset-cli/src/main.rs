//! hllset — HLLSet Algebra DSL shell (v0.2 — ipfrs-native, mesh)
//!
//! Reads Lua scripts from stdin (or -e/--eval) and outputs JSON results.
//! With --repl or no arguments (TTY), enters interactive mode.
//! Mesh commands (--mesh-*) replace the ROS 2 Python integration.
//!
//! ## Usage
//!
//! ```sh
//! echo 'return hllset.inscribe({"hello","world"}):key()' | hllset
//! hllset -e 'return hllset.tokenize("hello world"):key()'
//! hllset --repl     # interactive REPL with shared runtime
//! hllset --mesh-algebra  # start algebra mesh node
//! ```

use hllset_dsl::DslRuntime;
use mlua::Value as LuaValue;
use std::io::{self, BufRead, IsTerminal, Read, Write};

fn main() {
    let mut script = String::new();
    let args: Vec<String> = std::env::args().collect();

    let repl_mode = args.len() >= 2 && (args[1] == "--repl" || args[1] == "-r")
        || (args.len() == 1 && std::io::stdin().is_terminal());

    if repl_mode {
        run_repl();
        return;
    }

    if args.len() >= 2 && args[1] == "--mesh-algebra" {
        run_mesh_algebra();
        return;
    } else if args.len() >= 2 && args[1] == "--mesh-worker" {
        let worker_id = if args.len() > 2 { &args[2] } else { "worker-0" };
        run_mesh_worker(worker_id);
        return;
    } else if args.len() >= 2 && args[1] == "--mesh-noether" {
        let threshold: i64 = if args.len() > 2 {
            args[2].parse().unwrap_or(5)
        } else {
            5
        };
        run_mesh_noether(threshold);
        return;
    }

    let forth_debug = args.len() >= 2 && args[1] == "--forth-debug";

    if forth_debug {
        let forth_src = args[2..].join(" ");
        let ast = hllset_forth::parse(&forth_src).unwrap_or_else(|e| {
            eprintln!("Forth parse error: {}", e);
            std::process::exit(1);
        });
        let lua_code = hllset_forth::compile_to_lua(&ast);
        eprintln!("--- Generated Lua ---");
        eprintln!("{}", lua_code);
        eprintln!("--- End Lua ---");
        script = lua_code;
    } else if args.len() >= 3 && args[1] == "-e" {
        script = args[2..].join(" ");
    } else if args.len() >= 3 && args[1] == "--forth" {
        let forth_src = args[2..].join(" ");
        let ast = hllset_forth::parse(&forth_src).unwrap_or_else(|e| {
            eprintln!("Forth parse error: {}", e);
            std::process::exit(1);
        });
        script = hllset_forth::compile_to_lua(&ast);
    } else if args.len() >= 3 && args[1] == "-f" {
        let path = &args[2];
        script = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Error reading file '{}': {}", path, e);
            std::process::exit(1);
        });
    } else if args.len() >= 2 && (args[1] == "-h" || args[1] == "--help") {
        print_help();
        return;
    } else {
        io::stdin().read_to_string(&mut script).unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            std::process::exit(1);
        });
    }

    if script.trim().is_empty() {
        eprintln!("Error: no script provided");
        std::process::exit(1);
    }

    let rt = DslRuntime::new().unwrap_or_else(|e| {
        eprintln!("Error initializing Lua: {}", e);
        std::process::exit(1);
    });

    match rt.eval::<LuaValue>(&script) {
        Ok(val) => {
            println!("{}", serde_json::to_string(&lua_to_json(val)).unwrap());
        }
        Err(e) => {
            eprintln!("{}", serde_json::json!({"error": format!("{}", e)}));
            std::process::exit(1);
        }
    }
}

fn run_repl() {
    let rt = DslRuntime::new().unwrap_or_else(|e| {
        eprintln!("Error initializing Lua: {}", e);
        std::process::exit(1);
    });

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut line = String::new();

    eprintln!("hllang REPL — HLLSet Algebra DSL (v0.2 ipfrs-native)");
    eprintln!("Type 'exit' or Ctrl-D to quit. Lua variables persist across lines.");
    eprintln!("Lines with 'return' print JSON; others execute silently.");
    eprintln!();

    loop {
        line.clear();
        eprint!("hllang> ");
        let _ = stdout.flush();

        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                eprintln!();
                break;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "exit" || trimmed == "quit" {
                    break;
                }

                match rt.eval::<LuaValue>(trimmed) {
                    Ok(val) => match val {
                        LuaValue::Nil => {}
                        _ => {
                            println!("{}", serde_json::to_string(&lua_to_json(val)).unwrap());
                        }
                    },
                    Err(e) => {
                        eprintln!("error: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("read error: {}", e);
                break;
            }
        }
    }

    eprintln!("bye.");
}

fn print_help() {
    eprintln!("hllset — HLLSet Algebra DSL shell (v0.2 — ipfrs-native)\n");
    eprintln!("Usage:");
    eprintln!("  hllset -e '<lua script>'      Evaluate inline Lua script");
    eprintln!("  hllset --forth '<forth>'      Compile+run Forth DSL");
    eprintln!("  hllset -f <file.lua>          Evaluate script file");
    eprintln!("  echo '...' | hllset           Evaluate from stdin");
    eprintln!("  hllset --repl                 Interactive REPL (shared runtime)");
    eprintln!("  hllset                        Enter REPL (if TTY, no pipe)");
    eprintln!();
    eprintln!("Mesh commands (replaces ROS 2 pub/sub):");
    eprintln!("  hllset --mesh-algebra         Start algebra node (ingest -> HLLSet)");
    eprintln!("  hllset --mesh-worker [id]     Start stateless worker node");
    eprintln!("  hllset --mesh-noether [thr]   Start Noether flux controller");
    eprintln!();
    eprintln!("Storage: ipfrs-core (CID via sled) — no Go IPFS daemon required.");
    eprintln!("Messaging: in-process tokio bus — no ROS 2 / rclpy required.");
    eprintln!();
    eprintln!("In REPL mode, Lua variables persist across lines.");
    eprintln!("hllset.store() / hllset.load() share the same runtime.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  hllset -e 'return hllset.inscribe({{\"hello\",\"world\"}}):key()'");
    eprintln!("  hllset -e 'return #hllset.tokenize(\"hello world\")'");
    eprintln!("  hllset --repl");
    eprintln!("  hllset --mesh-algebra  # start algebra node");
}

// ── Mesh stubs ──────────────────────────────────────────────────────

fn run_mesh_algebra() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        use hllset_mesh::{AlgebraNode, InProcessBus};
        use std::sync::Arc;

        let bus = Arc::new(InProcessBus::new(64));
        let _algebra = AlgebraNode::new(bus);
        eprintln!("[mesh] Algebra node started on in-process bus");
        eprintln!("[mesh] Call algebra.ingest_text(\"hello world\") to test");
        eprintln!("[mesh] Press Ctrl-C to stop");

        let _ = tokio::signal::ctrl_c().await;
        eprintln!("\n[mesh] Shutting down...");
    });
}

fn run_mesh_worker(worker_id: &str) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        use hllset_mesh::{InProcessBus, WorkerNode};
        use std::sync::Arc;

        let bus = Arc::new(InProcessBus::new(64));
        let worker = WorkerNode::new(worker_id, bus);
        eprintln!(
            "[mesh] Worker '{}' started on in-process bus",
            worker.worker_id()
        );
        eprintln!("[mesh] Press Ctrl-C to stop");

        let _ = tokio::signal::ctrl_c().await;
        eprintln!("\n[mesh] Shutting down...");
    });
}

fn run_mesh_noether(threshold: i64) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        use hllset_mesh::{InProcessBus, NoetherController};
        use std::sync::Arc;

        let bus = Arc::new(InProcessBus::new(64));
        let controller = NoetherController::new(bus, threshold);
        controller.start().await;
        eprintln!(
            "[mesh] Noether controller started (threshold={})",
            threshold
        );
        eprintln!("[mesh] Press Ctrl-C to stop");

        let _ = tokio::signal::ctrl_c().await;
        controller.stop().await;
        eprintln!("\n[mesh] Shutting down...");
    });
}

fn lua_to_json(v: LuaValue) -> serde_json::Value {
    match v {
        LuaValue::Nil => serde_json::Value::Null,
        LuaValue::Boolean(b) => serde_json::Value::Bool(b),
        LuaValue::Integer(n) => serde_json::Value::Number(serde_json::Number::from(n)),
        LuaValue::Number(n) => serde_json::Number::from_f64(n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        LuaValue::String(s) => {
            let s = s.to_str().map(|b| b.to_string()).unwrap_or_default();
            serde_json::Value::String(s)
        }
        LuaValue::Table(t) => {
            let mut is_array = true;
            let mut arr = Vec::new();
            for pair in t.pairs::<LuaValue, LuaValue>() {
                if let Ok((k, v)) = pair {
                    if let LuaValue::Integer(i) = k {
                        if i >= 1 {
                            let idx = (i - 1) as usize;
                            if idx >= arr.len() {
                                arr.resize(idx + 1, serde_json::Value::Null);
                            }
                            arr[idx] = lua_to_json(v);
                            continue;
                        }
                    }
                    is_array = false;
                }
            }
            if is_array && !arr.is_empty() {
                serde_json::Value::Array(arr)
            } else {
                let mut map = serde_json::Map::new();
                for pair in t.pairs::<LuaValue, LuaValue>() {
                    if let Ok((k, v)) = pair {
                        let key = match k {
                            LuaValue::String(s) => {
                                s.to_str().map(|b| b.to_string()).unwrap_or_default()
                            }
                            LuaValue::Integer(n) => n.to_string(),
                            _ => format!("{:?}", k),
                        };
                        map.insert(key, lua_to_json(v));
                    }
                }
                serde_json::Value::Object(map)
            }
        }
        _ => serde_json::Value::String(format!("{:?}", v)),
    }
}
