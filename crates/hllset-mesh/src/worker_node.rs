//! Worker node — Rust-native replacement for ROS 2 `worker_node.py`.
//!
//! Stateless computation node. Subscribes to compute requests, performs
//! HLLSet operations, publishes results. No state between requests.

use crate::bus::MeshBus;
use crate::{ComputeRequest, ComputeResult, Message};
use hllset_dsl::LatticeElement;
use std::sync::Arc;
use tracing::debug;

/// Stateless worker that processes HLLSet computation requests.
pub struct WorkerNode {
    worker_id: String,
    bus: Arc<dyn MeshBus>,
    request_count: std::sync::atomic::AtomicU64,
}

impl WorkerNode {
    /// Create a new worker attached to the given mesh bus.
    pub fn new(worker_id: impl Into<String>, bus: Arc<dyn MeshBus>) -> Self {
        Self {
            worker_id: worker_id.into(),
            bus,
            request_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Process a single compute request and return the result.
    pub async fn handle_request(&self, req: ComputeRequest) -> Result<ComputeResult, String> {
        let count = self
            .request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        debug!(
            "[{}] op={} args={:?} (req #{})",
            req.request_id, req.op, req.args, count
        );

        let result: serde_json::Value = match req.op.as_str() {
            "tokenize" => {
                let text = req.args.as_str().unwrap_or_default();
                let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
                let elem = LatticeElement::from_tokens(&tokens);
                serde_json::Value::String(elem.key().to_string())
            }
            "inscribe" => {
                let tokens: Vec<String> = req
                    .args
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let elem = LatticeElement::from_tokens(&tokens);
                serde_json::Value::String(elem.key().to_string())
            }
            "union" => {
                let (a, b) = self.extract_two(&req.args)?;
                serde_json::Value::String(a.union(&b).key().to_string())
            }
            "intersect" => {
                let (a, b) = self.extract_two(&req.args)?;
                serde_json::Value::String(a.intersection(&b).key().to_string())
            }
            "bss" => {
                let (a, b) = self.extract_two(&req.args)?;
                let val = serde_json::Number::from_f64(a.bss_inclusion(&b))
                    .unwrap_or(serde_json::Number::from(0));
                serde_json::Value::Number(val)
            }
            "cardinality" => {
                let tokens: Vec<String> = req
                    .args
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let elem = LatticeElement::from_tokens(&tokens);
                let val = serde_json::Number::from_f64(elem.cardinality())
                    .unwrap_or(serde_json::Number::from(0));
                serde_json::Value::Number(val)
            }
            other => return Err(format!("unknown operation: {other}")),
        };

        Ok(ComputeResult {
            op: req.op,
            request_id: req.request_id,
            result,
            worker: self.worker_id.clone(),
        })
    }

    /// Extract two LatticeElements from args: [tokens_a, tokens_b].
    fn extract_two(
        &self,
        args: &serde_json::Value,
    ) -> Result<(LatticeElement, LatticeElement), String> {
        let arr = args
            .as_array()
            .ok_or("args must be array of two token arrays")?;
        let tokens_a: Vec<String> = arr
            .first()
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let tokens_b: Vec<String> = arr
            .get(1)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok((
            LatticeElement::from_tokens(&tokens_a),
            LatticeElement::from_tokens(&tokens_b),
        ))
    }

    /// Return the worker id.
    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }
}
