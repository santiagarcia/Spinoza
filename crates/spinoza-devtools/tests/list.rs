//! Integration tests for the `list` subcommand.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

#[test]
fn list_text_output_contains_pack_names() {
    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let output = Command::new(exe)
        .arg("list")
        .current_dir(workspace_root())
        .output()
        .expect("failed to execute list command");

    assert!(output.status.success(), "list command should succeed");
    let stdout = String::from_utf8(output.stdout).expect("stdout must be utf8");
    assert!(stdout.contains("poisson"), "should list the poisson pack");
    assert!(stdout.contains("block2x2"), "should list the block2x2 pack");
    assert!(stdout.contains("Discovered packs:"), "should have header");
}

#[test]
fn list_json_output_has_correct_shape() {
    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let output = Command::new(exe)
        .arg("list")
        .arg("--format")
        .arg("json")
        .current_dir(workspace_root())
        .output()
        .expect("failed to execute list --json command");

    assert!(
        output.status.success(),
        "list --json command should succeed"
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout must be utf8");
    let json: Value = serde_json::from_str(&stdout).expect("stdout must be valid json");

    let packs = json
        .get("packs")
        .and_then(Value::as_array)
        .expect("should have packs array");
    assert!(packs.len() >= 2, "should have at least 2 packs");

    // Check that each pack has name, version, capabilities.
    for pack in packs {
        assert!(pack.get("name").is_some(), "pack should have name");
        assert!(pack.get("version").is_some(), "pack should have version");
        assert!(
            pack.get("capabilities").is_some(),
            "pack should have capabilities"
        );
    }

    // Verify determinism: run again and compare.
    let output2 = Command::new(exe)
        .arg("list")
        .arg("--format")
        .arg("json")
        .current_dir(workspace_root())
        .output()
        .expect("second list execution failed");
    let stdout2 = String::from_utf8(output2.stdout).expect("stdout2 must be utf8");
    assert_eq!(stdout, stdout2, "list output must be deterministic");
}
