//! Integration tests for the `new-pack` subcommand.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn cleanup_dir(path: &std::path::Path) {
    if path.exists() {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn new_pack_generates_valid_scaffold() {
    let tmp_dir = std::env::temp_dir().join("spinoza_test_newpack_scaffold");
    cleanup_dir(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let output = Command::new(exe)
        .arg("new-pack")
        .arg("test_heat")
        .arg("--path")
        .arg(&tmp_dir)
        .current_dir(workspace_root())
        .output()
        .expect("failed to execute new-pack command");

    assert!(
        output.status.success(),
        "new-pack command should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout must be utf8");
    assert!(
        stdout.contains("spinoza-pack-test_heat"),
        "should report created crate"
    );

    // Verify generated files exist.
    let crate_dir = tmp_dir.join("spinoza-pack-test_heat");
    assert!(crate_dir.join("Cargo.toml").exists(), "Cargo.toml exists");
    assert!(crate_dir.join("src/lib.rs").exists(), "src/lib.rs exists");
    assert!(
        crate_dir.join("tests/conformance.rs").exists(),
        "tests/conformance.rs exists"
    );
    assert!(crate_dir.join("README.md").exists(), "README.md exists");

    // Verify Cargo.toml content.
    let cargo_toml = std::fs::read_to_string(crate_dir.join("Cargo.toml")).unwrap();
    assert!(cargo_toml.contains("spinoza-pack-test_heat"));
    assert!(cargo_toml.contains("spinoza-core"));

    // Verify lib.rs has the pack struct.
    let lib_rs = std::fs::read_to_string(crate_dir.join("src/lib.rs")).unwrap();
    assert!(lib_rs.contains("TestHeatPack"));
    assert!(lib_rs.contains("TestHeatBuilder"));
    assert!(lib_rs.contains("spinoza_register_pack!"));
    assert!(lib_rs.contains("impl MethodPack"));
    assert!(lib_rs.contains("impl ProblemBuilder"));

    // Verify conformance test.
    let test_rs = std::fs::read_to_string(crate_dir.join("tests/conformance.rs")).unwrap();
    assert!(test_rs.contains("pack_conforms"));
    assert!(test_rs.contains("test_heat"));

    cleanup_dir(&tmp_dir);
}

#[test]
fn new_pack_rejects_invalid_name() {
    let tmp_dir = std::env::temp_dir().join("spinoza_test_newpack_invalid");
    cleanup_dir(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let output = Command::new(exe)
        .arg("new-pack")
        .arg("MyBadName")
        .arg("--path")
        .arg(&tmp_dir)
        .current_dir(workspace_root())
        .output()
        .expect("failed to execute new-pack command");

    assert!(
        !output.status.success(),
        "new-pack should fail for invalid name"
    );
    let stderr = String::from_utf8(output.stderr).expect("stderr must be utf8");
    assert!(
        stderr.contains("lowercase"),
        "should mention lowercase requirement"
    );

    cleanup_dir(&tmp_dir);
}

#[test]
fn new_pack_rejects_existing_directory() {
    let tmp_dir = std::env::temp_dir().join("spinoza_test_newpack_exists");
    cleanup_dir(&tmp_dir);
    let existing = tmp_dir.join("spinoza-pack-existing");
    std::fs::create_dir_all(&existing).expect("create existing dir");

    let exe = env!("CARGO_BIN_EXE_spinoza-devtools");
    let output = Command::new(exe)
        .arg("new-pack")
        .arg("existing")
        .arg("--path")
        .arg(&tmp_dir)
        .current_dir(workspace_root())
        .output()
        .expect("failed to execute new-pack command");

    assert!(
        !output.status.success(),
        "new-pack should fail for existing dir"
    );

    cleanup_dir(&tmp_dir);
}
