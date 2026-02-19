//! Conformance harness for method pack verification.
//!
//! Pack authors include this module in their test suites to verify:
//! 1. The pack is discoverable via inventory.
//! 2. Capabilities are deterministic across repeated calls.
//! 3. Registered builders are queryable by the declared equation types.
//! 4. No duplicate or conflicting builder keys exist within the pack.
//!
//! # Example
//!
//! ```ignore
//! #[test]
//! fn pack_conforms() {
//!     let _ = my_pack::MyPack; // anchor
//!     spinoza_core::conformance::verify_pack("my_pack").unwrap();
//! }
//! ```

use std::collections::BTreeSet;
use std::fmt;

use crate::method_pack::iter_packs;
use crate::method_registry::MethodRegistry;

/// A single conformance check result.
#[derive(Debug)]
pub struct CheckResult {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

/// Aggregate result of all conformance checks.
#[derive(Debug)]
pub struct ConformanceReport {
    pub pack_name: String,
    pub checks: Vec<CheckResult>,
}

impl ConformanceReport {
    /// Returns `true` when every check passed.
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Returns only the failed checks.
    pub fn failures(&self) -> Vec<&CheckResult> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }
}

impl fmt::Display for ConformanceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Conformance report for pack '{}':", self.pack_name)?;
        for check in &self.checks {
            let status = if check.passed { "PASS" } else { "FAIL" };
            writeln!(f, "  [{status}] {}: {}", check.name, check.detail)?;
        }
        if self.all_passed() {
            writeln!(f, "All checks passed.")?;
        } else {
            let n = self.failures().len();
            writeln!(f, "{n} check(s) failed.")?;
        }
        Ok(())
    }
}

/// Run all conformance checks for the named pack.
///
/// The pack crate must be linked (anchored) before calling this function.
/// Returns a [`ConformanceReport`] with individual check results.
pub fn verify_pack(pack_name: &str) -> ConformanceReport {
    let mut checks = Vec::new();

    // 1. Discovery check
    let found = iter_packs().any(|p| p.name() == pack_name);
    checks.push(CheckResult {
        name: "discovery",
        passed: found,
        detail: if found {
            format!("pack '{pack_name}' found via inventory")
        } else {
            format!(
                "pack '{}' NOT found; available: {:?}",
                pack_name,
                iter_packs().map(|p| p.name()).collect::<Vec<_>>()
            )
        },
    });

    if !found {
        return ConformanceReport {
            pack_name: pack_name.to_string(),
            checks,
        };
    }

    let entry = iter_packs().find(|p| p.name() == pack_name).unwrap();

    // 2. Deterministic capabilities
    let caps_a: Vec<String> = entry.capabilities().iter().map(|c| c.to_string()).collect();
    let caps_b: Vec<String> = entry.capabilities().iter().map(|c| c.to_string()).collect();
    let deterministic = caps_a == caps_b;
    checks.push(CheckResult {
        name: "capabilities_deterministic",
        passed: deterministic,
        detail: if deterministic {
            format!("{} capabilities returned consistently", caps_a.len())
        } else {
            "capabilities differ across calls".to_string()
        },
    });

    // 3. Non-empty capabilities
    let has_caps = !caps_a.is_empty();
    checks.push(CheckResult {
        name: "capabilities_nonempty",
        passed: has_caps,
        detail: if has_caps {
            format!("pack declares {} capabilities", caps_a.len())
        } else {
            "pack declares zero capabilities".to_string()
        },
    });

    // 4. Version non-empty
    let version = entry.version();
    let has_version = !version.is_empty();
    checks.push(CheckResult {
        name: "version_nonempty",
        passed: has_version,
        detail: if has_version {
            format!("version = '{version}'")
        } else {
            "version is empty".to_string()
        },
    });

    // 5. Builders are queryable
    let registry = MethodRegistry::from_packs();
    let builder_infos = registry.builder_info();
    let pack_builders: Vec<_> = builder_infos
        .iter()
        .filter(|info| info.get("pack").map(|s| s.as_str()) == Some(pack_name))
        .collect();

    let has_builders = !pack_builders.is_empty();
    checks.push(CheckResult {
        name: "builders_registered",
        passed: has_builders,
        detail: if has_builders {
            format!(
                "{} builder(s) registered: {}",
                pack_builders.len(),
                pack_builders
                    .iter()
                    .filter_map(|b| b.get("name"))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            "no builders registered by this pack".to_string()
        },
    });

    // 6. No duplicate builder keys within the same pack
    let mut seen_keys = BTreeSet::new();
    let mut duplicates = Vec::new();
    for info in &pack_builders {
        let key = format!(
            "{}|{}|{}",
            info.get("equation_type").unwrap_or(&String::new()),
            info.get("dimensions").unwrap_or(&String::new()),
            info.get("backends").unwrap_or(&String::new()),
        );
        if !seen_keys.insert(key.clone()) {
            duplicates.push(key);
        }
    }
    let no_dupes = duplicates.is_empty();
    checks.push(CheckResult {
        name: "no_duplicate_keys",
        passed: no_dupes,
        detail: if no_dupes {
            "no duplicate builder keys within pack".to_string()
        } else {
            format!("duplicate builder keys: {duplicates:?}")
        },
    });

    // 7. Operator factories registered
    let op_infos = registry.operator_factory_info();
    let pack_ops: Vec<_> = op_infos
        .iter()
        .filter(|info| info.get("pack").map(|s| s.as_str()) == Some(pack_name))
        .collect();
    checks.push(CheckResult {
        name: "operator_factories",
        passed: true,
        detail: format!(
            "{} operator factory(ies) registered{}",
            pack_ops.len(),
            if pack_ops.is_empty() {
                String::new()
            } else {
                format!(
                    ": {}",
                    pack_ops
                        .iter()
                        .filter_map(|f| f.get("name"))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        ),
    });

    // 8. Preconditioner factories registered
    let pc_infos = registry.preconditioner_factory_info();
    let pack_pcs: Vec<_> = pc_infos
        .iter()
        .filter(|info| info.get("pack").map(|s| s.as_str()) == Some(pack_name))
        .collect();
    checks.push(CheckResult {
        name: "preconditioner_factories",
        passed: true,
        detail: format!(
            "{} preconditioner factory(ies) registered{}",
            pack_pcs.len(),
            if pack_pcs.is_empty() {
                String::new()
            } else {
                format!(
                    ": {}",
                    pack_pcs
                        .iter()
                        .filter_map(|f| f.get("name"))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        ),
    });

    // 9. Solver factories registered
    let slv_infos = registry.solver_factory_info();
    let pack_slvs: Vec<_> = slv_infos
        .iter()
        .filter(|info| info.get("pack").map(|s| s.as_str()) == Some(pack_name))
        .collect();
    checks.push(CheckResult {
        name: "solver_factories",
        passed: true,
        detail: format!(
            "{} solver factory(ies) registered{}",
            pack_slvs.len(),
            if pack_slvs.is_empty() {
                String::new()
            } else {
                format!(
                    ": {}",
                    pack_slvs
                        .iter()
                        .filter_map(|f| f.get("name"))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        ),
    });

    // 10. No duplicate operator factory keys within the pack
    {
        let mut seen = BTreeSet::new();
        let mut dupes = Vec::new();
        for info in &pack_ops {
            let k = info.get("keys").cloned().unwrap_or_default();
            if !seen.insert(k.clone()) {
                dupes.push(k);
            }
        }
        let ok = dupes.is_empty();
        checks.push(CheckResult {
            name: "no_duplicate_operator_factory_keys",
            passed: ok,
            detail: if ok {
                "no duplicate operator factory keys".to_string()
            } else {
                format!("duplicate operator factory keys: {dupes:?}")
            },
        });
    }

    // 11. No duplicate solver factory keys within the pack
    {
        let mut seen = BTreeSet::new();
        let mut dupes = Vec::new();
        for info in &pack_slvs {
            let k = info.get("keys").cloned().unwrap_or_default();
            if !seen.insert(k.clone()) {
                dupes.push(k);
            }
        }
        let ok = dupes.is_empty();
        checks.push(CheckResult {
            name: "no_duplicate_solver_factory_keys",
            passed: ok,
            detail: if ok {
                "no duplicate solver factory keys".to_string()
            } else {
                format!("duplicate solver factory keys: {dupes:?}")
            },
        });
    }

    // 12. Physics packs must not register single_field solver factories.
    //     Only dedicated solver packs should own generic solver registrations.
    //     Block-specific solvers (block_structure != "single_field") are allowed.
    {
        let is_physics_pack = !["linear_solvers"].contains(&pack_name);
        if is_physics_pack {
            let single_field_solvers: Vec<String> = pack_slvs
                .iter()
                .filter(|info| {
                    info.get("keys")
                        .map(|k| k.contains("single_field"))
                        .unwrap_or(false)
                })
                .filter_map(|info| info.get("name").cloned())
                .collect();
            let ok = single_field_solvers.is_empty();
            checks.push(CheckResult {
                name: "no_single_field_solver_in_physics_pack",
                passed: ok,
                detail: if ok {
                    "physics pack correctly delegates single_field solvers to solver packs"
                        .to_string()
                } else {
                    format!(
                        "physics pack registers single_field solver factories \
                         (move to a solver pack): {}",
                        single_field_solvers.join(", ")
                    )
                },
            });
        }
    }

    ConformanceReport {
        pack_name: pack_name.to_string(),
        checks,
    }
}

/// Convenience: run all conformance checks and panic if any fail.
///
/// Suitable for use directly in a `#[test]` function.
pub fn assert_pack_conforms(pack_name: &str) {
    let report = verify_pack(pack_name);
    if !report.all_passed() {
        panic!("Pack conformance failed:\n{report}");
    }
}
