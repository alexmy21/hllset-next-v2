//! Module vocabulary and port signatures — the soldered "bag of modules"
//! (TRANSITION §2, §5; session-1 decisions Q1/Q6).
//!
//! [`ModuleKind`] is shared by the DSL (app-facing), the host interface
//! (wire-level), and the module implementations. Port counts are fixed per
//! kind; changing them is a soldered-contract change.

/// Node identifier shared by DSL declarations and wire commands.
pub type NodeId = u32;

/// Port identifier (direction-specific, 0-based index).
pub type PortId = u8;

/// The module-node kinds of the "bag of modules".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ModuleKind {
    /// InLUT set recovery → `W[T(H)]` rows (tensor stays host-side, Q6).
    #[default]
    Slice,
    /// Grounded softmax in logit space (renormalize once).
    Renorm,
    /// Exact-LUT vocabulary gate (nanoLM Phase 2/5 finding).
    Gate,
    /// ContextMatrix + τ/ρ report.
    Ground,
    /// Expert gate union + logit mixture.
    Experts,
    /// CRDT pointwise-max context merge.
    Merge,
}

/// All soldered module kinds, in declaration order.
pub const MODULE_KINDS: [ModuleKind; 6] = [
    ModuleKind::Slice,
    ModuleKind::Renorm,
    ModuleKind::Gate,
    ModuleKind::Ground,
    ModuleKind::Experts,
    ModuleKind::Merge,
];

/// Input/output port counts of one module kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModulePorts {
    pub inputs: PortId,
    pub outputs: PortId,
}

/// The soldered port signature of a module kind.
///
/// | Kind | in → out | Rationale |
/// |---|---|---|
/// | Slice | 1 → 1 | token set in, slice index out |
/// | Renorm | 1 → 1 | logits in, grounded distribution out |
/// | Gate | 1 → 1 | ids in, filtered ids out |
/// | Ground | 1 → 2 | recent ids in; prior tokens-out + τ/ρ report out |
/// | Experts | 2 → 1 | two expert logit streams in, mixture out |
/// | Merge | 2 → 1 | two contexts in, merged context out |
pub const fn module_ports(kind: ModuleKind) -> ModulePorts {
    match kind {
        ModuleKind::Slice => ModulePorts {
            inputs: 1,
            outputs: 1,
        },
        ModuleKind::Renorm => ModulePorts {
            inputs: 1,
            outputs: 1,
        },
        ModuleKind::Gate => ModulePorts {
            inputs: 1,
            outputs: 1,
        },
        ModuleKind::Ground => ModulePorts {
            inputs: 1,
            outputs: 2,
        },
        ModuleKind::Experts => ModulePorts {
            inputs: 2,
            outputs: 1,
        },
        ModuleKind::Merge => ModulePorts {
            inputs: 2,
            outputs: 1,
        },
    }
}

/// True if `port` is a valid input port of `kind`.
pub const fn input_port_in_range(kind: ModuleKind, port: PortId) -> bool {
    port < module_ports(kind).inputs
}

/// True if `port` is a valid output port of `kind`.
pub const fn output_port_in_range(kind: ModuleKind, port: PortId) -> bool {
    port < module_ports(kind).outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_table_is_pinned() {
        assert_eq!(
            module_ports(ModuleKind::Slice),
            ModulePorts {
                inputs: 1,
                outputs: 1
            }
        );
        assert_eq!(
            module_ports(ModuleKind::Renorm),
            ModulePorts {
                inputs: 1,
                outputs: 1
            }
        );
        assert_eq!(
            module_ports(ModuleKind::Gate),
            ModulePorts {
                inputs: 1,
                outputs: 1
            }
        );
        assert_eq!(
            module_ports(ModuleKind::Ground),
            ModulePorts {
                inputs: 1,
                outputs: 2
            }
        );
        assert_eq!(
            module_ports(ModuleKind::Experts),
            ModulePorts {
                inputs: 2,
                outputs: 1
            }
        );
        assert_eq!(
            module_ports(ModuleKind::Merge),
            ModulePorts {
                inputs: 2,
                outputs: 1
            }
        );
    }

    #[test]
    fn port_range_helpers_match_table() {
        for kind in MODULE_KINDS {
            let ports = module_ports(kind);
            assert!(!input_port_in_range(kind, ports.inputs));
            assert!(!output_port_in_range(kind, ports.outputs));
            assert!(input_port_in_range(kind, ports.inputs - 1));
            assert!(output_port_in_range(kind, ports.outputs - 1));
        }
    }
}
