//! Golden style integration tests for the audit workflow.
//!
//! These tests exercise `audit_spec` against the example case spec files
//! shipped in the repository and compare the rendered output to expected
//! golden strings.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;
use spinoza_core::CapabilityRegistry;
use spinoza_devtools::audit_spec;

fn spec_dir() -> PathBuf {
    // The workspace root is three levels up from this test file's directory.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("spec")
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

fn run_audit_json(case_path: &str) -> (i32, Value) {
    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let output = Command::new(exe)
        .arg("audit")
        .arg(case_path)
        .arg("--format")
        .arg("json")
        .current_dir(workspace_root())
        .output()
        .expect("failed to execute cargo run");

    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8(output.stdout).expect("stdout must be utf8");
    assert!(
        output.stderr.is_empty(),
        "json mode must not print to stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed =
        serde_json::from_str::<Value>(&stdout).expect("stdout must be a single json object");
    (code, parsed)
}

fn assert_json_matches_golden(mut actual: Value, golden_file: &str) {
    let expected_path = golden_dir().join(golden_file);
    let expected_text =
        std::fs::read_to_string(&expected_path).expect("failed to read expected json file");
    let expected: Value =
        serde_json::from_str(&expected_text).expect("expected json file is invalid");

    let case_path_is_absolute = actual
        .get("case_path")
        .and_then(Value::as_str)
        .map(|p| std::path::Path::new(p).is_absolute())
        .unwrap_or(false);

    if case_path_is_absolute {
        actual
            .as_object_mut()
            .expect("actual must be a json object")
            .remove("case_path");
        let mut expected_obj = expected
            .as_object()
            .expect("expected must be a json object")
            .clone();
        expected_obj.remove("case_path");
        assert_eq!(actual, Value::Object(expected_obj));
    } else {
        assert_eq!(actual, expected);
    }
}

// -----------------------------------------------------------------------
// Golden: valid spec passes audit with implemented capabilities
// -----------------------------------------------------------------------

#[test]
fn golden_valid_spec_ok() {
    let path = spec_dir().join("poisson_minimal_ok.toml");
    let registry = CapabilityRegistry::default_registry();
    let report = audit_spec(&path, &registry).expect("spec should parse and validate");

    assert!(report.is_ok(), "expected all capabilities available");

    let rendered = report.render();

    let expected = concat!(
        "Problem: Poisson 2D minimal\n",
        "Required capabilities:\n",
        "  BC_Dirichlet\n",
        "  Operator_Laplacian\n",
        "  Precond_Jacobi\n",
        "  Solver_CG\n",
        "  Space_H1Scalar\n",
        "All capabilities available.\n",
    );

    assert_eq!(rendered, expected, "golden output mismatch:\n{rendered}");
}

// -----------------------------------------------------------------------
// Golden: invalid spec fails validation
// -----------------------------------------------------------------------

#[test]
fn golden_invalid_spec_fails_validation() {
    let path = spec_dir().join("poisson_invalid.toml");
    let registry = CapabilityRegistry::default_registry();
    let result = audit_spec(&path, &registry);

    assert!(result.is_err(), "expected validation error");
    let err = result.unwrap_err();
    let msg = err.to_string();

    // The first validation error should fire on dimension.
    assert!(
        msg.contains("dimension must be 2 or 3"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn golden_json_valid_spec_ok() {
    let (code, json) = run_audit_json("spec/poisson_minimal_ok.toml");
    assert_eq!(code, 0, "expected success exit code 0");
    assert_json_matches_golden(json, "poisson_minimal_ok.json");
}

#[test]
fn golden_json_invalid_spec_validation_error() {
    let (code, json) = run_audit_json("spec/poisson_invalid.toml");
    assert_eq!(code, 2, "expected validation exit code 2");
    assert_json_matches_golden(json, "poisson_invalid.json");
}
