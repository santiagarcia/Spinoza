//! CLI entry point for spinoza-devtools.

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand, ValueEnum};
use spinoza_core::{CapabilityRegistry, MethodRegistry};

// Re-use library logic.
use spinoza_devtools::{
    audit_spec, build_list_json, render_list_text, solve_case_file, AuditJsonOutput, SolveBackend,
    SolvePreconditioner,
};

#[derive(Parser)]
#[command(name = "spinoza-devtools", about = "Developer utilities for Spinoza")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Audit a case spec file against the current capability registry.
    Audit {
        /// Path to the TOML case spec file.
        path: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Assemble and solve a minimal Poisson problem from a case spec.
    Solve {
        /// Path to the TOML case spec file.
        path: PathBuf,
        /// Output JSON path.
        #[arg(long)]
        out: PathBuf,
        /// Solve backend.
        #[arg(long, value_enum, default_value_t = SolveBackendArg::Assembled)]
        backend: SolveBackendArg,
        /// Preconditioner.
        #[arg(long, value_enum, default_value_t = SolvePrecondArg::Jacobi)]
        precond: SolvePrecondArg,
    },
    /// List all discovered method packs and their capabilities.
    List {
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Generate a new method pack crate scaffold.
    NewPack {
        /// Pack name (e.g. "my_diffusion"). Used as the crate suffix: spinoza-pack-<name>.
        name: String,
        /// Directory in which to create the pack crate.
        #[arg(long, default_value = "crates")]
        path: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SolveBackendArg {
    Assembled,
    Matrixfree,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SolvePrecondArg {
    Jacobi,
    Mg,
    BlockJacobi,
}

impl From<SolveBackendArg> for SolveBackend {
    fn from(value: SolveBackendArg) -> Self {
        match value {
            SolveBackendArg::Assembled => SolveBackend::Assembled,
            SolveBackendArg::Matrixfree => SolveBackend::MatrixFree,
        }
    }
}

impl From<SolvePrecondArg> for SolvePreconditioner {
    fn from(value: SolvePrecondArg) -> Self {
        match value {
            SolvePrecondArg::Jacobi => SolvePreconditioner::Jacobi,
            SolvePrecondArg::Mg => SolvePreconditioner::Mg,
            SolvePrecondArg::BlockJacobi => SolvePreconditioner::BlockJacobi,
        }
    }
}

fn main() {
    // Anchor pack crates so the linker keeps them (enables inventory discovery).
    let _ = spinoza_pack_poisson::PoissonPack;
    let _ = spinoza_pack_block2x2::Block2x2Pack;

    let cli = Cli::parse();

    match cli.command {
        Commands::Audit { path, format } => {
            let registry = CapabilityRegistry::default_registry();
            match audit_spec(&path, &registry) {
                Ok(report) => match format {
                    OutputFormat::Text => {
                        print!("{}", report.render());
                        if !report.is_ok() {
                            process::exit(1);
                        }
                    }
                    OutputFormat::Json => {
                        let json = AuditJsonOutput::from_success(&path, &report);
                        let payload = serde_json::to_string_pretty(&json)
                            .expect("json serialization must succeed");
                        println!("{payload}");
                        process::exit(json.exit_code);
                    }
                },
                Err(e) => match format {
                    OutputFormat::Text => {
                        eprintln!("Error: {e}");
                        process::exit(2);
                    }
                    OutputFormat::Json => {
                        let json = AuditJsonOutput::from_validation_error(&path, e.to_string());
                        let payload = serde_json::to_string_pretty(&json)
                            .expect("json serialization must succeed");
                        println!("{payload}");
                        process::exit(json.exit_code);
                    }
                },
            }
        }
        Commands::Solve {
            path,
            out,
            backend,
            precond,
        } => match solve_case_file(&path, backend.into(), precond.into()) {
            Ok(output) => {
                let payload =
                    serde_json::to_string_pretty(&output).expect("json serialization must succeed");
                if let Err(e) = std::fs::write(&out, payload) {
                    eprintln!("Error: failed to write output json: {e}");
                    process::exit(2);
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
                process::exit(2);
            }
        },
        Commands::List { format } => {
            let registry = MethodRegistry::from_packs();
            match format {
                OutputFormat::Text => {
                    print!("{}", render_list_text(&registry));
                }
                OutputFormat::Json => {
                    let json = build_list_json(&registry);
                    let payload = serde_json::to_string_pretty(&json)
                        .expect("json serialization must succeed");
                    println!("{payload}");
                }
            }
        }
        Commands::NewPack { name, path } => {
            match spinoza_devtools::generate_new_pack(&name, &path) {
                Ok(crate_dir) => {
                    println!("Created pack crate at {}", crate_dir.display());
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(2);
                }
            }
        }
    }
}
