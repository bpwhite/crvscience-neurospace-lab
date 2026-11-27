// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![deny(unsafe_code)]

use clap::{Parser, Subcommand};
use neurospace_lab::datasets::DatasetId;
use neurospace_lab::engine::NeuralSpaceSim;
use neurospace_lab::gui::{render_headless_screenshot, NeuroSpaceApp};
use neurospace_lab::models::{ModelArch, RenderMode};
use neurospace_lab::persistence::WorksetEnvelope;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "neurospace-lab",
    version = "1.6.0",
    about = "Native Rust 3D Neural Network Topology, Plasticity & Cerebral Cortex Wetware Lab"
)]
struct Cli {
    /// Render a 1600x1000 offscreen GPU PNG screenshot to this path and exit
    #[arg(long, value_name = "PNG_PATH")]
    headless_screenshot: Option<PathBuf>,

    /// View preset for `--headless-screenshot` (1 = M13 Cerebral Cortex, 2 = M11 Cortical Columns, 3 = M14 Frontier LLM)
    #[arg(long, default_value_t = 1)]
    screenshot_view: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the interactive 3D Neural Space GUI workstation (default)
    Gui,
    /// Lock a dataset and benchmark all 15 neural network models headlessly
    BenchDataset {
        /// Dataset code (D1..D7)
        #[arg(short, long, default_value = "D4")]
        dataset: String,
        /// Training steps per model
        #[arg(short, long, default_value_t = 60)]
        steps: usize,
    },
    /// Lock a neural network model and benchmark all 7 datasets headlessly
    BenchModel {
        /// Model code (M0..M14)
        #[arg(short, long, default_value = "M13")]
        model: String,
        /// Training steps per dataset
        #[arg(short, long, default_value_t = 60)]
        steps: usize,
    },
    /// Emit a portable SHA-256 verified .neuroworkset.json snapshot to stdout
    ExportSampleWorkset,
    /// Run structural plasticity parameter-sweep & network-efficiency/firing-rate bifurcation digest
    PlasticitySweep {
        /// Model code (M0..M14)
        #[arg(short, long, default_value = "M13")]
        model: String,
        /// Dataset code (D1..D7)
        #[arg(short, long, default_value = "D4")]
        dataset: String,
        /// Simulation steps per sweep point
        #[arg(short, long, default_value_t = 25)]
        steps: usize,
        /// Output format: table, json, or csv
        #[arg(long, default_value = "table")]
        format: String,
        /// Optional path to write JSON or CSV output
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn attach_parent_console_if_cli() {
    if std::env::args_os().len() > 1 {
        const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
        extern "system" {
            fn AttachConsole(dw_process_id: u32) -> i32;
        }
        // SAFETY: Called once at process startup on Windows when CLI arguments are present
        // so stdout/stderr attach to the parent terminal under `windows_subsystem = "windows"`.
        unsafe {
            let _ = AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(windows)]
    attach_parent_console_if_cli();

    let cli = Cli::parse();

    if let Some(out_path) = cli.headless_screenshot {
        render_headless_screenshot(&out_path, cli.screenshot_view)?;
        return Ok(());
    }

    match cli.command.unwrap_or(Commands::Gui) {
        Commands::Gui => {
            let native_options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_inner_size([1440.0, 900.0])
                    .with_min_inner_size([1024.0, 680.0])
                    .with_icon(std::sync::Arc::new(neurospace_lab::icon::app_egui_icon()))
                    .with_title("NeuroSpace Lab v1.6.0 — 3D Neural Topology & Cerebral Cortex Grey-Matter Workstation"),
                ..Default::default()
            };
            eframe::run_native(
                "NeuroSpace Lab v1.6.0",
                native_options,
                Box::new(|cc| Ok(Box::new(NeuroSpaceApp::new(cc)))),
            )?;
        }
        Commands::BenchDataset { dataset, steps } => {
            let ds = DatasetId::from_code(&dataset)?;
            let mut sim = NeuralSpaceSim::new(ModelArch::M1PerceptronMlp, ds, 2026);
            sim.dataset_locked = true;
            sim.run_locked_sweep(steps);
            println!(
                "=== Locked Dataset Benchmark: {} ({}) ===",
                ds.code(),
                ds.title()
            );
            println!(
                "{:<6} {:<46} {:>10} {:>10} {:>12}",
                "Code", "Architecture", "Score(%)", "Loss", "Neurons/Syn"
            );
            for rec in &sim.comparison_ledger {
                println!(
                    "{:<6} {:<46} {:>9.2}% {:>10.4} {:>5}/{:<5}",
                    rec.model.code(),
                    rec.model.name(),
                    rec.primary_score,
                    rec.loss,
                    rec.neuron_count,
                    rec.synapse_count
                );
            }
        }
        Commands::BenchModel { model, steps } => {
            let arch = ModelArch::from_code(&model)?;
            let mut sim = NeuralSpaceSim::new(arch, DatasetId::D1ImageClassification, 2026);
            sim.model_locked = true;
            sim.run_locked_sweep(steps);
            println!(
                "=== Locked Model Benchmark: {} ({}) ===",
                arch.code(),
                arch.name()
            );
            println!(
                "{:<6} {:<44} {:>10} {:>10} {:>12}",
                "Data", "Domain", "Score(%)", "Loss", "Neurons/Syn"
            );
            for rec in &sim.comparison_ledger {
                println!(
                    "{:<6} {:<44} {:>9.2}% {:>10.4} {:>5}/{:<5}",
                    rec.dataset.code(),
                    rec.dataset.title(),
                    rec.primary_score,
                    rec.loss,
                    rec.neuron_count,
                    rec.synapse_count
                );
            }
        }
        Commands::ExportSampleWorkset => {
            let mut sim = NeuralSpaceSim::new(
                ModelArch::M13CerebralCortexGreyMatter,
                DatasetId::D7MolecularProperties,
                2026,
            );
            for _ in 0..25 {
                sim.step_simulation();
            }
            let env = WorksetEnvelope::from_sim(&sim, RenderMode::AutoAdaptive, 1.0);
            println!("{}", serde_json::to_string_pretty(&env)?);
        }
        Commands::PlasticitySweep {
            model,
            dataset,
            steps,
            format,
            out,
        } => {
            let arch = ModelArch::from_code(&model)?;
            let ds = DatasetId::from_code(&dataset)?;
            let sim = NeuralSpaceSim::new(arch, ds, 2026);
            let digest = sim.run_plasticity_bifurcation_sweep(steps);
            let text = match format.to_ascii_lowercase().as_str() {
                "json" => digest.to_json_pretty()?,
                "csv" => digest.to_csv(),
                "table" => {
                    let mut buf = format!(
                        "=== Plasticity Bifurcation Digest: {} / {} (critical_prune={:.3}, peak_eff={:.4}) ===\n",
                        digest.model.code(),
                        digest.dataset.code(),
                        digest.critical_prune_threshold,
                        digest.peak_efficiency_score
                    );
                    buf.push_str(&format!(
                        "{:<7} {:<7} {:<20} {:>6} {:>6} {:>9} {:>9} {:>8}\n",
                        "Prune", "Growth", "Regime", "Neur", "Syn", "FireRate", "NetEff", "Score%"
                    ));
                    for p in &digest.points {
                        buf.push_str(&format!(
                            "{:<7.3} {:<7.2} {:<20} {:>6} {:>6} {:>9.4} {:>9.4} {:>7.2}%\n",
                            p.prune_threshold,
                            p.plasticity_growth,
                            p.regime_label,
                            p.active_neurons,
                            p.active_synapses,
                            p.mean_firing_rate,
                            p.network_efficiency,
                            p.primary_score
                        ));
                    }
                    buf
                }
                other => {
                    return Err(format!("Unsupported format '{other}'; expected table, json, or csv").into());
                }
            };
            if let Some(out_path) = out {
                std::fs::write(&out_path, &text)?;
                println!(
                    "Wrote plasticity bifurcation digest ({format}) to {}",
                    out_path.display()
                );
            } else {
                println!("{text}");
            }
        }
    }
    Ok(())
}
