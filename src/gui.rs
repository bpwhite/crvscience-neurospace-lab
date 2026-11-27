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

use crate::datasets::DatasetId;
use crate::engine::{NeuralSpaceSim, NeuronRole, OptoClamp, PlasticityBifurcationDigest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeTool {
    Inspect,
    AblateLesion,
    ChR2Excite,
    NpHRSilence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaletteAction {
    SelectModel(ModelArch),
    FocusCorticalRole(NeuronRole),
    ApplyPlasticityProtocol(&'static str),
    ApplyStimulationPreset(&'static str),
    SetCameraPreset(u8),
    ToggleAutoOrbit,
    SetRenderMode(RenderMode),
    ToggleBottomMatrix,
    RunPlasticitySweep,
    ExportBifurcationJson,
    ExportBifurcationCsv,
    ExportWorksetJson,
    QuickSaveF5,
    QuickLoadF9,
}

#[derive(Debug, Clone)]
pub struct PaletteEntry {
    pub category: String,
    pub title: String,
    pub subtitle: String,
    pub action: PaletteAction,
    search_blob: String,
}

pub fn build_palette_entries() -> Vec<PaletteEntry> {
    let mut entries = Vec::new();
    let mut push = |cat: &str, title: String, sub: String, act: PaletteAction| {
        let blob = format!(
            "{} {} {}",
            cat.to_ascii_lowercase(),
            title.to_ascii_lowercase(),
            sub.to_ascii_lowercase()
        );
        entries.push(PaletteEntry {
            category: cat.to_string(),
            title,
            subtitle: sub,
            action: act,
            search_blob: blob,
        });
    };

    for arch in ModelArch::ALL {
        push(
            "Architecture",
            format!("{} — {}", arch.code(), arch.name()),
            arch.learning_rule_summary().to_string(),
            PaletteAction::SelectModel(arch),
        );
    }

    for (role, title, sub) in [
        (NeuronRole::SensoryInput, "Cortical Region: Thalamus / Sensory Input", "Focus afferent sensory / thalamic relay soma"),
        (NeuronRole::SpinyStellateL4, "Cortical Region: Layer 4 Spiny Stellate", "Focus granular thalamocortical recipient lamina"),
        (NeuronRole::PyramidalL23, "Cortical Region: Layer 2/3 Association Pyramidal", "Focus supragranular cortico-cortical pyramidal neuron"),
        (NeuronRole::ThickPyramidalL5, "Cortical Region: Layer 5 Thick-Tufted Pyramidal", "Focus subcortical burst-coupled pyramidal projection"),
        (NeuronRole::CorticothalamicL6, "Cortical Region: Layer 6 Corticothalamic", "Focus deep feedback corticothalamic soma"),
        (NeuronRole::MolecularInterneuronL1, "Cortical Region: Layer 1 Apical Interneuron", "Focus molecular layer apical tuft modulation"),
        (NeuronRole::InhibitoryInterneuron, "Cortical Region: GABAergic Basket/Martinotti", "Focus inhibitory microcircuit interneuron"),
        (NeuronRole::AstrocyteGlia, "Cortical Region: Protoplasmic Astrocyte", "Focus tripartite astrocytic calcium glia"),
        (NeuronRole::MoeRouter, "Region: Top-2 Sparse MoE Router", "Focus Frontier LLM mixture-of-experts gating router"),
    ] {
        push(
            "Cortical Region",
            title.to_string(),
            sub.to_string(),
            PaletteAction::FocusCorticalRole(role),
        );
    }

    for (id, title, sub) in [
        ("balanced", "Plasticity: Homeostatic Balanced Regime", "Set prune_threshold=0.045, plasticity_growth=0.35"),
        ("neurogenesis", "Plasticity: High Neurogenesis & Axonal Sprouting", "Set prune_threshold=0.025, plasticity_growth=0.85"),
        ("pruning", "Plasticity: Aggressive Synaptic Pruning", "Set prune_threshold=0.140, plasticity_growth=0.12"),
        ("frozen", "Plasticity: Static / Frozen Structural Graph", "Set prune_threshold=0.000, plasticity_growth=0.00"),
    ] {
        push(
            "Plasticity",
            title.to_string(),
            sub.to_string(),
            PaletteAction::ApplyPlasticityProtocol(id),
        );
    }
    push(
        "Plasticity",
        "Run Structural Plasticity Bifurcation Sweep".to_string(),
        "Sweep prune/growth parameters and inspect efficiency & firing-rate bifurcations".to_string(),
        PaletteAction::RunPlasticitySweep,
    );

    for (id, title, sub) in [
        ("chr2_drive", "Stimulation: ChR2 Blue-Light Cortical Excitation", "Arm ChR2 tool (drive=1.55) and photostimulate hidden/L4 soma"),
        ("nphr_silence", "Stimulation: NpHR Yellow-Light Photoinhibition", "Arm NpHR tool and hyperpolarize representative cortical soma"),
        ("focal_stroke", "Stimulation / Lesion: Focal Cluster Stroke (25%)", "Ablate 25% spatial cluster to probe peri-infarct recovery"),
        ("sever_feedback", "Stimulation / Lesion: Sever Top-Down Feedback", "Cut reverse L1/predictive-error feedback tracts"),
        ("heal_all", "Stimulation: Heal All Lesions, Tracts & Clamps", "Restore full intact structural connectome and clear clamps"),
    ] {
        push(
            "Stimulation",
            title.to_string(),
            sub.to_string(),
            PaletteAction::ApplyStimulationPreset(id),
        );
    }

    for (preset, title, sub) in [
        (1, "View: Camera Isometric 3/4 (1)", "Smoothly reset to default 3/4 orbital framing"),
        (2, "View: Camera Sagittal X-Y (2)", "Align camera to sagittal laminar cross-section"),
        (3, "View: Camera Coronal Y-Z (3)", "Align camera to coronal transverse slice"),
        (4, "View: Camera Top-Down X-Z (4)", "Align camera to dorsal top-down projection"),
    ] {
        push(
            "View",
            title.to_string(),
            sub.to_string(),
            PaletteAction::SetCameraPreset(preset),
        );
    }
    push(
        "View",
        "View: Toggle Camera Auto-Orbit (A)".to_string(),
        "Pause or resume continuous 3D orbital rotation".to_string(),
        PaletteAction::ToggleAutoOrbit,
    );
    push(
        "View",
        "View: Toggle Bottom Lock Matrix & Sparklines".to_string(),
        "Show or collapse telemetry sparklines and comparison ledger".to_string(),
        PaletteAction::ToggleBottomMatrix,
    );
    for rm in RenderMode::ALL {
        push(
            "Render",
            format!("Render: {}", rm.label()),
            "Switch 3D soma/dendrite morphology continuum".to_string(),
            PaletteAction::SetRenderMode(rm),
        );
    }

    push(
        "Export",
        "Export: Plasticity Bifurcation Digest (.json)".to_string(),
        "Write structural plasticity sweep & bifurcation metrics to JSON".to_string(),
        PaletteAction::ExportBifurcationJson,
    );
    push(
        "Export",
        "Export: Plasticity Bifurcation Digest (.csv)".to_string(),
        "Write structural plasticity sweep & bifurcation metrics to CSV".to_string(),
        PaletteAction::ExportBifurcationCsv,
    );
    push(
        "Export",
        "Export: Verified .neuroworkset.json Snapshot (Ctrl+S)".to_string(),
        "Save SHA-256 verified network topology and weights".to_string(),
        PaletteAction::ExportWorksetJson,
    );
    push(
        "Vault",
        "Vault: SQLite QuickSave (F5)".to_string(),
        "Persist current simulation state into quicksave_f5 slot".to_string(),
        PaletteAction::QuickSaveF5,
    );
    push(
        "Vault",
        "Vault: SQLite QuickLoad (F9)".to_string(),
        "Restore saved quicksave_f5 slot from SQLite WAL".to_string(),
        PaletteAction::QuickLoadF9,
    );

    entries
}

pub fn search_palette_entries<'a>(
    entries: &'a [PaletteEntry],
    query: &str,
    limit: usize,
) -> Vec<&'a PaletteEntry> {
    let trimmed = query.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return entries.iter().take(limit.max(1)).collect();
    }
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let mut scored: Vec<(i32, &PaletteEntry)> = Vec::new();
    for entry in entries {
        let mut matched = true;
        let mut score = 0_i32;
        let title_lc = entry.title.to_ascii_lowercase();
        for &tk in &tokens {
            if let Some(pos) = entry.search_blob.find(tk) {
                if title_lc.starts_with(tk) {
                    score += 35;
                } else if pos < 18 {
                    score += 18;
                } else {
                    score += 6;
                }
            } else {
                matched = false;
                break;
            }
        }
        if matched {
            scored.push((score, entry));
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored
        .into_iter()
        .take(limit.max(1))
        .map(|(_, e)| e)
        .collect()
}
use crate::models::{ModelArch, ModelTier, OutputGoal, RenderMode};
use crate::persistence::{LabStore, WorksetEnvelope};
use crate::renderer_3d::{draw_neural_space_3d, OrbitCamera3D};
use egui::{Color32, RichText, ScrollArea, Sense};

pub struct NeuroSpaceApp {
    pub sim: NeuralSpaceSim,
    pub camera: OrbitCamera3D,
    pub render_mode: RenderMode,
    pub manual_organic_blend: f32,
    pub show_group_highlights: bool,
    pub running: bool,
    pub steps_per_frame: usize,
    pub hovered_neuron: Option<usize>,
    pub selected_neuron: Option<usize>,
    pub probe_tool: ProbeTool,
    pub show_bottom_matrix: bool,
    pub show_about_modal: bool,
    pub cmd_palette_open: bool,
    pub cmd_query: String,
    pub cmd_selected_idx: usize,
    pub cmd_entries: Vec<PaletteEntry>,
    pub show_bifurcation_modal: bool,
    pub last_bifurcation_digest: Option<PlasticityBifurcationDigest>,
    pub status_banner: String,
    store: Option<LabStore>,
}

impl NeuroSpaceApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let (store, err_msg) = match LabStore::open("neurospace_lab.sqlite") {
            Ok(st) => (Some(st), None),
            Err(e) => (
                None,
                Some(format!(
                    "SQLite WAL store unavailable ({e}); running in-memory."
                )),
            ),
        };
        let mut app = Self::new_with_store(store);
        if let Some(msg) = err_msg {
            app.status_banner = msg;
        }
        app
    }

    pub fn new_headless() -> Self {
        Self::new_with_store(None)
    }

    fn new_with_store(store: Option<LabStore>) -> Self {
        let sim = NeuralSpaceSim::new(
            ModelArch::M13CerebralCortexGreyMatter,
            DatasetId::D4PlantSpecies,
            2026,
        );
        Self {
            sim,
            camera: OrbitCamera3D::default(),
            render_mode: RenderMode::AutoAdaptive,
            manual_organic_blend: 0.5,
            show_group_highlights: true,
            running: true,
            steps_per_frame: 1,
            hovered_neuron: None,
            selected_neuron: None,
            probe_tool: ProbeTool::Inspect,
            show_bottom_matrix: true,
            show_about_modal: false,
            cmd_palette_open: false,
            cmd_query: String::new(),
            cmd_selected_idx: 0,
            cmd_entries: build_palette_entries(),
            show_bifurcation_modal: false,
            last_bifurcation_digest: None,
            status_banner: "Ready — Drag 3D Neural Space to orbit, Ctrl+K for command palette, F5/F9 to save/load."
                .to_string(),
            store,
        }
    }

    fn apply_workset_envelope(&mut self, env: WorksetEnvelope, source_desc: &str) {
        let render_mode = env.render_mode;
        let manual_blend = env.manual_organic_blend;
        match self.sim.apply_workset_envelope(&env) {
            Ok(()) => {
                self.render_mode = render_mode;
                self.manual_organic_blend = manual_blend;
                self.selected_neuron = None;
                self.hovered_neuron = None;
                self.status_banner = format!(
                    "Restored verified {source_desc} (step {}, {} somata, {} syn).",
                    self.sim.step_count,
                    self.sim.neurons.len(),
                    self.sim.synapses.len()
                );
            }
            Err(e) => {
                self.status_banner = format!("Failed to restore {source_desc}: {e}");
            }
        }
    }

    fn quick_save(&mut self) {
        let env = WorksetEnvelope::from_sim(&self.sim, self.render_mode, self.manual_organic_blend);
        if let Some(st) = &self.store {
            match st.save_slot("quicksave_f5", &env) {
                Ok(()) => {
                    let prefix = &env.sha256_digest[..10.min(env.sha256_digest.len())];
                    self.status_banner = format!("Saved slot 'quicksave_f5' (SHA-256 {prefix}...)");
                }
                Err(e) => {
                    self.status_banner = format!("QuickSave failed for 'quicksave_f5': {e}");
                }
            }
        } else {
            self.status_banner = "QuickSave unavailable: SQLite store is not active.".to_string();
        }
    }

    fn quick_load(&mut self) {
        let loaded = if let Some(st) = &self.store {
            st.load_slot("quicksave_f5")
        } else {
            self.status_banner = "QuickLoad unavailable: SQLite store is not active.".to_string();
            return;
        };
        match loaded {
            Ok(Some(env)) => {
                if !env.verify_digest() {
                    self.status_banner =
                        "Checksum verification failed for slot 'quicksave_f5'.".to_string();
                    return;
                }
                self.apply_workset_envelope(env, "slot 'quicksave_f5'");
            }
            Ok(None) => {
                self.status_banner = "No saved 'quicksave_f5' slot found.".to_string();
            }
            Err(e) => {
                self.status_banner = format!("QuickLoad failed for 'quicksave_f5': {e}");
            }
        }
    }

    fn export_workset_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("NeuroSpace Workset", &["neuroworkset.json", "json"])
            .set_file_name("session.neuroworkset.json")
            .save_file()
        {
            let env =
                WorksetEnvelope::from_sim(&self.sim, self.render_mode, self.manual_organic_blend);
            match env.save_to_file(&path) {
                Ok(()) => {
                    let prefix = &env.sha256_digest[..10.min(env.sha256_digest.len())];
                    self.status_banner = format!(
                        "Exported workset '{}' (SHA-256 {prefix}...).",
                        path.display()
                    );
                }
                Err(e) => {
                    self.status_banner = e;
                }
            }
        }
    }

    fn import_workset_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("NeuroSpace Workset", &["json"])
            .pick_file()
        {
            match WorksetEnvelope::load_from_file(&path) {
                Ok(env) => {
                    let desc = format!("workset '{}'", path.display());
                    self.apply_workset_envelope(env, &desc);
                }
                Err(e) => {
                    self.status_banner = e;
                }
            }
        }
    }

    pub fn ensure_bifurcation_digest(&mut self, steps_per_point: usize) -> &PlasticityBifurcationDigest {
        let needs_compute = self
            .last_bifurcation_digest
            .as_ref()
            .map_or(true, |d| d.model != self.sim.model || d.dataset != self.sim.dataset_id);
        if needs_compute {
            self.last_bifurcation_digest = Some(self.sim.run_plasticity_bifurcation_sweep(steps_per_point));
        }
        self.last_bifurcation_digest
            .get_or_insert_with(|| self.sim.run_plasticity_bifurcation_sweep(steps_per_point))
    }

    fn export_bifurcation_digest_dialog(&mut self, is_csv: bool) {
        let digest = self.ensure_bifurcation_digest(20).clone();
        let (default_name, filter_lbl, ext, payload_res) = if is_csv {
            (
                "neurospace_plasticity_bifurcation.csv",
                "CSV Table",
                "csv",
                Ok(digest.to_csv()),
            )
        } else {
            (
                "neurospace_plasticity_bifurcation.json",
                "JSON Digest",
                "json",
                digest
                    .to_json_pretty()
                    .map_err(|e| format!("JSON serialize error: {e}")),
            )
        };
        let payload = match payload_res {
            Ok(txt) => txt,
            Err(e) => {
                self.status_banner = e;
                return;
            }
        };
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(filter_lbl, &[ext])
            .set_file_name(default_name)
            .save_file()
        {
            match std::fs::write(&path, payload) {
                Ok(()) => {
                    self.status_banner = format!(
                        "Exported plasticity bifurcation digest ({}) to {}",
                        ext.to_uppercase(),
                        path.display()
                    );
                }
                Err(e) => self.status_banner = format!("Digest export failed: {e}"),
            }
        }
    }

    pub fn execute_palette_action(&mut self, action: PaletteAction) {
        match action {
            PaletteAction::SelectModel(arch) => {
                self.selected_neuron = None;
                self.hovered_neuron = None;
                let was_locked = self.sim.model_locked;
                self.sim.model_locked = false;
                self.sim.select_model(arch);
                self.sim.model_locked = was_locked;
                self.status_banner = format!("Switched architecture to {} ({})", arch.code(), arch.name());
            }
            PaletteAction::FocusCorticalRole(role) => {
                if let Some(idx) = self.sim.neurons.iter().position(|n| n.role == role && !n.ablated) {
                    self.selected_neuron = Some(idx);
                    self.status_banner = format!(
                        "Focused cortical region/soma #{} [{}]",
                        self.sim.neurons[idx].id,
                        role.label_for_model(self.sim.model, self.sim.neurons[idx].layer)
                    );
                } else {
                    self.selected_neuron = self.sim.neurons.first().map(|n| n.id);
                    self.status_banner = format!(
                        "Role '{}' not present in {}; focused primary soma.",
                        role.label(),
                        self.sim.model.code()
                    );
                }
            }
            PaletteAction::ApplyPlasticityProtocol(proto) => {
                match proto {
                    "neurogenesis" => {
                        self.sim.params.prune_threshold = 0.025;
                        self.sim.params.plasticity_growth = 0.85;
                    }
                    "pruning" => {
                        self.sim.params.prune_threshold = 0.14;
                        self.sim.params.plasticity_growth = 0.12;
                    }
                    "frozen" => {
                        self.sim.params.prune_threshold = 0.0;
                        self.sim.params.plasticity_growth = 0.0;
                    }
                    _ => {
                        self.sim.params.prune_threshold = 0.045;
                        self.sim.params.plasticity_growth = 0.35;
                    }
                }
                self.status_banner = format!(
                    "Applied plasticity protocol '{proto}' (prune={:.3}, growth={:.2}).",
                    self.sim.params.prune_threshold, self.sim.params.plasticity_growth
                );
            }
            PaletteAction::ApplyStimulationPreset(preset) => match preset {
                "chr2_drive" => {
                    self.probe_tool = ProbeTool::ChR2Excite;
                    self.sim.opto_clamp_drive = 1.55;
                    if let Some(idx) = self.sim.neurons.iter().position(|n| !n.ablated && n.layer >= 1) {
                        self.sim.set_neuron_clamp(idx, OptoClamp::ChR2Excite);
                        self.selected_neuron = Some(idx);
                    }
                    self.status_banner = "Applied ChR2 blue-light cortical photostimulation (drive=1.55).".to_string();
                }
                "nphr_silence" => {
                    self.probe_tool = ProbeTool::NpHRSilence;
                    if let Some(idx) = self.sim.neurons.iter().position(|n| !n.ablated && n.layer >= 1) {
                        self.sim.set_neuron_clamp(idx, OptoClamp::NpHRSilence);
                        self.selected_neuron = Some(idx);
                    }
                    self.status_banner = "Applied NpHR yellow-light photoinhibition clamp.".to_string();
                }
                "focal_stroke" => {
                    let n = self.sim.induce_focal_stroke();
                    self.status_banner = format!("Focal cluster stroke ablated {n} somata.");
                }
                "sever_feedback" => {
                    let n = self.sim.sever_feedback_tracts();
                    self.status_banner = format!("Severed {n} top-down feedback tracts.");
                }
                _ => {
                    self.sim.heal_all_interventions();
                    self.status_banner = "Healed all lesions, severed tracts & opto-clamps.".to_string();
                }
            },
            PaletteAction::SetCameraPreset(id) => match id {
                2 => {
                    self.camera.set_preset(0.0, 0.0, 25.0, glam::Vec3::ZERO, false);
                    self.status_banner = "Camera preset: Sagittal X-Y (2)".to_string();
                }
                3 => {
                    self.camera.set_preset(std::f32::consts::FRAC_PI_2, 0.04, 25.0, glam::Vec3::ZERO, false);
                    self.status_banner = "Camera preset: Coronal Y-Z (3)".to_string();
                }
                4 => {
                    self.camera.set_preset(0.0, 1.35, 25.0, glam::Vec3::ZERO, false);
                    self.status_banner = "Camera preset: Top-Down X-Z (4)".to_string();
                }
                _ => {
                    self.camera.set_preset(-0.52, 0.24, 25.0, glam::Vec3::ZERO, false);
                    self.status_banner = "Camera preset: Isometric (1)".to_string();
                }
            },
            PaletteAction::ToggleAutoOrbit => {
                self.camera.auto_orbit = !self.camera.auto_orbit;
                self.status_banner = format!(
                    "Camera auto-orbit {} (A)",
                    if self.camera.auto_orbit { "enabled" } else { "paused" }
                );
            }
            PaletteAction::SetRenderMode(rm) => {
                self.render_mode = rm;
                self.status_banner = format!("Render mode: {}", rm.label());
            }
            PaletteAction::ToggleBottomMatrix => {
                self.show_bottom_matrix = !self.show_bottom_matrix;
            }
            PaletteAction::RunPlasticitySweep => {
                let digest = self.sim.run_plasticity_bifurcation_sweep(20);
                self.status_banner = format!(
                    "Plasticity sweep complete (critical prune={:.3}, peak eff={:.4}).",
                    digest.critical_prune_threshold, digest.peak_efficiency_score
                );
                self.last_bifurcation_digest = Some(digest);
                self.show_bifurcation_modal = true;
            }
            PaletteAction::ExportBifurcationJson => self.export_bifurcation_digest_dialog(false),
            PaletteAction::ExportBifurcationCsv => self.export_bifurcation_digest_dialog(true),
            PaletteAction::ExportWorksetJson => self.export_workset_dialog(),
            PaletteAction::QuickSaveF5 => self.quick_save(),
            PaletteAction::QuickLoadF9 => self.quick_load(),
        }
    }
}

impl NeuroSpaceApp {
    pub fn ui_step(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.cmd_palette_open {
                self.cmd_palette_open = false;
            } else if self.show_bifurcation_modal {
                self.show_bifurcation_modal = false;
            } else if self.show_about_modal {
                self.show_about_modal = false;
            } else if self.selected_neuron.is_some() {
                self.selected_neuron = None;
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            self.show_about_modal = !self.show_about_modal;
        }
        if ctx.input(|i| (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::K)) {
            self.cmd_palette_open = !self.cmd_palette_open;
            self.cmd_query.clear();
            self.cmd_selected_idx = 0;
        }
        if ctx.input(|i| (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::O)) {
            self.import_workset_dialog();
        }
        if ctx.input(|i| (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::S)) {
            self.export_workset_dialog();
        }
        if !ctx.wants_keyboard_input() {
            let no_mod = ctx.input(|i| !i.modifiers.ctrl && !i.modifiers.command && !i.modifiers.alt);
            if no_mod {
                if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                    self.running = !self.running;
                }
                if ctx.input(|i| i.key_pressed(egui::Key::F5)) {
                    self.quick_save();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::F9)) {
                    self.quick_load();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
                    self.camera.set_preset(-0.52, 0.24, 25.0, glam::Vec3::ZERO, false);
                    self.status_banner = "Camera preset: Isometric (1)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Num2)) {
                    self.camera.set_preset(0.0, 0.0, 25.0, glam::Vec3::ZERO, false);
                    self.status_banner = "Camera preset: Sagittal X-Y (2)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Num3)) {
                    self.camera.set_preset(
                        std::f32::consts::FRAC_PI_2,
                        0.04,
                        25.0,
                        glam::Vec3::ZERO,
                        false,
                    );
                    self.status_banner = "Camera preset: Coronal Y-Z (3)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Num4)) {
                    self.camera.set_preset(0.0, 1.35, 25.0, glam::Vec3::ZERO, false);
                    self.status_banner = "Camera preset: Top-Down X-Z (4)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                    self.camera.reset();
                    self.status_banner = "Camera reset to Isometric + Auto-Orbit (R)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::A)) {
                    self.camera.auto_orbit = !self.camera.auto_orbit;
                    self.status_banner = format!(
                        "Camera auto-orbit {} (A)",
                        if self.camera.auto_orbit { "enabled" } else { "paused" }
                    );
                }
                if ctx.input(|i| i.key_pressed(egui::Key::N) || i.key_pressed(egui::Key::Period)) {
                    self.running = false;
                    self.sim.step_simulation();
                    self.status_banner =
                        format!("Single-stepped to tick {} (N / .)", self.sim.step_count);
                }
                if ctx.input(|i| i.key_pressed(egui::Key::I)) {
                    self.probe_tool = ProbeTool::Inspect;
                    self.status_banner = "Probe tool: Inspect (I)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::X)) {
                    self.probe_tool = ProbeTool::AblateLesion;
                    self.status_banner = "Probe tool: Ablate / Lesion (X)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::E)) {
                    self.probe_tool = ProbeTool::ChR2Excite;
                    self.status_banner = "Probe tool: Optogenetic ChR2 Excite (E)".to_string();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::S)) {
                    self.probe_tool = ProbeTool::NpHRSilence;
                    self.status_banner = "Probe tool: Optogenetic NpHR Silence (S)".to_string();
                }
            }
        }

        let sweep_active = self.sim.active_sweep.is_some();
        if sweep_active {
            let still_running = self.sim.advance_locked_sweep(8);
            if !still_running {
                if let Some(msg) = self.sim.plasticity_log.first() {
                    self.status_banner = msg.clone();
                }
            }
            ctx.request_repaint();
        } else if self.running {
            for _ in 0..self.steps_per_frame {
                self.sim.step_simulation();
            }
            if self.camera.auto_orbit {
                self.camera.yaw += 0.0038;
            }
            ctx.request_repaint();
        }

        let dt = ctx.input(|i| i.predicted_dt);
        if self.camera.damp_step(dt) {
            ctx.request_repaint();
        }

        // Top Bar (2 clean rows so KPI labels never clip or split mid-phrase)
        egui::TopBottomPanel::top("top_hud_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(
                    RichText::new("NEUROSPACE 3D LAB v1.6.0")
                        .color(Color32::from_rgb(56, 189, 248)),
                );
                ui.separator();
                if ui
                    .button(if self.running { "|| Pause" } else { "▶ Run" })
                    .clicked()
                {
                    self.running = !self.running;
                }
                if ui
                    .add_enabled(self.sim.active_sweep.is_none(), egui::Button::new("⏭ Step"))
                    .clicked()
                {
                    self.running = false;
                    self.sim.step_simulation();
                }
                if ui
                    .add_enabled(
                        self.sim.active_sweep.is_none(),
                        egui::Button::new("↺ Rebuild Topology"),
                    )
                    .clicked()
                {
                    self.selected_neuron = None;
                    self.hovered_neuron = None;
                    self.sim.rebuild_topology(false);
                }
                ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=8).text("xSpeed"));

                ui.separator();
                if ui.button("📸 Snapshot Run").clicked() {
                    self.sim.record_snapshot("ManualPin");
                    if let (Some(st), Some(last)) =
                        (&self.store, self.sim.comparison_ledger.last())
                    {
                        if let Err(e) = st.append_comparison(last) {
                            self.status_banner =
                                format!("Snapshot pinned in memory, SQLite append failed: {e}");
                        }
                    }
                }
                if let Some(prog) = self.sim.sweep_progress() {
                    ui.add(
                        egui::ProgressBar::new(prog.fraction())
                            .desired_width(210.0)
                            .text(prog.description),
                    );
                    if ui.button("✖ Cancel").clicked() {
                        self.sim.cancel_locked_sweep();
                        self.status_banner = "Cancelled locked batch sweep.".to_string();
                    }
                } else {
                    let sweep_label = if self.sim.dataset_locked && !self.sim.model_locked {
                        "⚡ Sweep All 15 Models (Locked Dataset)"
                    } else {
                        "⚡ Sweep All 7 Datasets (Locked Model)"
                    };
                    if ui.button(sweep_label).clicked() {
                        self.selected_neuron = None;
                        self.hovered_neuron = None;
                        self.sim.start_locked_sweep(35);
                        self.status_banner = "Running incremental non-blocking sweep...".to_string();
                    }
                }

                ui.separator();
                if ui.button("Jump / Search (Ctrl+K)").clicked() {
                    self.cmd_palette_open = !self.cmd_palette_open;
                    self.cmd_query.clear();
                    self.cmd_selected_idx = 0;
                }
                if ui.button("Export Workset...").clicked() {
                    self.export_workset_dialog();
                }
                if ui.button("Import Workset...").clicked() {
                    self.import_workset_dialog();
                }
            });

            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.show_group_highlights, "Layer/Lamina Zones");
                ui.checkbox(&mut self.show_bottom_matrix, "📊 Lock Matrix");
                ui.separator();

                let kpi_col = Color32::from_rgb(167, 243, 208);
                ui.label(RichText::new(format!("Step {}", self.sim.step_count)).color(kpi_col));
                ui.label("|");
                ui.label(
                    RichText::new(format!("Somata {}", self.sim.neurons.len())).color(kpi_col),
                );
                ui.label("|");
                ui.label(
                    RichText::new(format!("Synapses {}", self.sim.synapses.len())).color(kpi_col),
                );
                ui.label("|");
                ui.label(
                    RichText::new(format!(
                        "{}: {:.1}%",
                        self.sim.dataset_id.task_kind().primary_metric_label(),
                        self.sim.primary_metric
                    ))
                    .strong()
                    .color(Color32::from_rgb(52, 211, 153)),
                );
                ui.label("|");
                ui.label(
                    RichText::new(format!(
                        "Adv Margin (ε={:.2}): {:.1}%",
                        self.sim.params.adversarial_eps, self.sim.adversarial_margin
                    ))
                    .color(Color32::from_rgb(125, 211, 252)),
                );
                ui.label("|");
                ui.label(
                    RichText::new(format!(
                        "{}: {:.3}",
                        self.sim.dataset_id.task_kind().loss_label(),
                        self.sim.current_loss
                    ))
                    .strong()
                    .color(Color32::from_rgb(251, 191, 36)),
                );

                let ablated = self.sim.ablated_count();
                let severed = self.sim.severed_count();
                if ablated > 0 || severed > 0 {
                    ui.label("|");
                    ui.label(
                        RichText::new(format!("Ablated {ablated} | Severed {severed}"))
                            .color(Color32::from_rgb(251, 113, 133)),
                    );
                }
            });
        });

        // Left Control Column
        egui::SidePanel::left("left_control_column")
            .default_width(342.0)
            .resizable(true)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    // 1. Model Selector + Model Lock
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.strong("1. Neural Architecture (Tiers I–V)");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.toggle_value(&mut self.sim.model_locked, "🔒 Lock Model");
                            });
                        });
                        ui.small(self.sim.model.tier().label());

                        egui::ComboBox::from_id_salt("model_arch_combo")
                            .width(302.0)
                            .selected_text(format!(
                                "{} — {}",
                                self.sim.model.code(),
                                self.sim.model.name()
                            ))
                            .show_ui(ui, |ui| {
                                for tier in [
                                    ModelTier::Tier1Mechanistic,
                                    ModelTier::Tier2FoundationLlm,
                                    ModelTier::Tier2AdversarialEvolutionary,
                                    ModelTier::Tier3SpikingDynamical,
                                    ModelTier::Tier4BiologicalTheoretical,
                                    ModelTier::Tier5HistologicalGreyMatter,
                                ] {
                                    ui.label(
                                        RichText::new(tier.label())
                                            .strong()
                                            .color(Color32::from_rgb(125, 211, 252)),
                                    );
                                    for arch in ModelArch::ALL.into_iter().filter(|a| a.tier() == tier) {
                                        let label = format!("{} · {}", arch.code(), arch.name());
                                        if ui
                                            .add_enabled(
                                                !self.sim.model_locked || arch == self.sim.model,
                                                egui::Button::new(label).selected(arch == self.sim.model),
                                            )
                                            .clicked()
                                        {
                                            self.selected_neuron = None;
                                            self.hovered_neuron = None;
                                            self.sim.select_model(arch);
                                        }
                                    }
                                    ui.separator();
                                }
                            });
                        ui.small(self.sim.model.learning_rule_summary());
                        ui.checkbox(
                            &mut self.sim.params.continual_transfer,
                            "Retain Hidden Core Weights When Switching Datasets",
                        );
                    });

                    ui.add_space(6.0);

                    // 2. Dataset Selector + Dataset Lock
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.strong("2. Benchmark & Scientific Dataset");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.toggle_value(&mut self.sim.dataset_locked, "🔒 Lock Data");
                            });
                        });

                        for ds in DatasetId::ALL {
                            let selected = ds == self.sim.dataset_id;
                            let row_txt = format!("{} · {}", ds.code(), ds.title());
                            if ui
                                .add_enabled(
                                    !self.sim.dataset_locked || selected,
                                    egui::Button::new(row_txt).selected(selected),
                                )
                                .clicked()
                            {
                                self.selected_neuron = None;
                                self.hovered_neuron = None;
                                self.sim.select_dataset(ds, 2026);
                            }
                        }
                        ui.small(self.sim.dataset_id.domain_summary());
                    });

                    ui.add_space(6.0);

                    // 3. Output Goal Selector
                    ui.group(|ui| {
                        ui.strong("3. Output Goal & Optimization Pressure");
                        egui::ComboBox::from_id_salt("output_goal_combo")
                            .width(302.0)
                            .selected_text(self.sim.goal.label())
                            .show_ui(ui, |ui| {
                                for g in OutputGoal::ALL {
                                    ui.selectable_value(&mut self.sim.goal, g, g.label());
                                }
                            });
                    });

                    ui.add_space(6.0);

                    // 4. Parameter Determinants
                    ui.group(|ui| {
                        ui.strong("4. Parameter Determinants & Flow Sample");
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.learning_rate, 0.002..=0.15)
                                .text("Learning Rate η"),
                        )
                        .on_hover_text(
                            "Step size η controlling synaptic weight update magnitude per simulation tick.",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.momentum, 0.0..=0.96)
                                .text("Momentum β"),
                        )
                        .on_hover_text(
                            "Exponential velocity damping β smoothing gradient descent trajectories.",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.prune_threshold, 0.0..=0.22)
                                .text("Synaptic Prune |w|"),
                        )
                        .on_hover_text(
                            "Structural pruning threshold: synapses with |w| below this value are eliminated.",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.plasticity_growth, 0.0..=1.0)
                                .text("Neurogenesis / Sprout"),
                        )
                        .on_hover_text(
                            "Rate of activity-dependent synaptogenesis and compensatory new-soma sprouting.",
                        );
                        let prev_flow = self.sim.params.flow_sample_rate;
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.flow_sample_rate, 0.0..=1.0)
                                .text("River Stream Intensity"),
                        )
                        .on_hover_text(
                            "Fraction of active pathways carrying calm macroscopic signal stream wisps.",
                        );
                        if (self.sim.params.flow_sample_rate - prev_flow).abs() > 0.01 {
                            self.sim.refresh_flow_sample();
                        }
                        let prev_scale = self.sim.params.hidden_scale;
                        ui.add_enabled(
                            self.sim.active_sweep.is_none(),
                            egui::Slider::new(&mut self.sim.params.hidden_scale, 0.5..=2.0)
                                .text("Cortical Capacity"),
                        )
                        .on_hover_text(
                            "Scales hidden/cortical layer soma count and rebuilds network topology.",
                        );
                        if (self.sim.params.hidden_scale - prev_scale).abs() > 0.04 {
                            self.selected_neuron = None;
                            self.sim.rebuild_topology(false);
                        }
                        let prev_eps = self.sim.params.adversarial_eps;
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.adversarial_eps, 0.0..=0.45)
                                .text("Adversarial ε"),
                        )
                        .on_hover_text(
                            "Perturbation radius ε used to probe FGSM/PGD adversarial margin and robust training.",
                        );
                        if (self.sim.params.adversarial_eps - prev_eps).abs() > 0.004 {
                            self.sim.evaluate_validation_core();
                        }
                        ui.horizontal_wrapped(|ui| {
                            if ui.small_button("🔬 Plasticity Bifurcation Sweep").clicked() {
                                self.execute_palette_action(PaletteAction::RunPlasticitySweep);
                            }
                            if ui.small_button("Export CSV").clicked() {
                                self.export_bifurcation_digest_dialog(true);
                            }
                            if ui.small_button("Export JSON").clicked() {
                                self.export_bifurcation_digest_dialog(false);
                            }
                        });
                    });

                    ui.add_space(6.0);

                    // 5. Adaptive Rendering Continuum
                    ui.group(|ui| {
                        ui.strong("5. Adaptive & Histological 3D Renderer");
                        egui::ComboBox::from_id_salt("render_mode_combo")
                            .width(302.0)
                            .selected_text(self.render_mode.label())
                            .show_ui(ui, |ui| {
                                for rm in RenderMode::ALL {
                                    ui.selectable_value(&mut self.render_mode, rm, rm.label());
                                }
                            });
                        if self.render_mode == RenderMode::HybridCyberManifold {
                            ui.add(
                                egui::Slider::new(&mut self.manual_organic_blend, 0.35..=0.65)
                                    .text("Hybrid Blend"),
                            );
                        }
                        let eff_morph = self
                            .render_mode
                            .resolve_organic_factor(self.sim.model, self.manual_organic_blend);
                        let mode_tag = if eff_morph > 1.08 {
                            "3D Histological Grey Matter (Pyramidal & Stellate)".to_string()
                        } else {
                            format!("{:.0}% Organic Continuum", eff_morph * 100.0)
                        };
                        ui.small(format!("Active Morphology: {}", mode_tag));
                        ui.checkbox(&mut self.show_group_highlights, "Highlight Layer / Lamina Zones");
                        ui.checkbox(&mut self.camera.auto_orbit, "Auto-Orbit 3D Neural Space (A)");
                        ui.horizontal_wrapped(|ui| {
                            if ui.small_button("1:Iso").on_hover_text("Isometric 3/4 camera preset (1)").clicked() {
                                self.camera.set_preset(-0.52, 0.24, 25.0, glam::Vec3::ZERO, false);
                            }
                            if ui.small_button("2:Sag").on_hover_text("Sagittal X-Y camera preset (2)").clicked() {
                                self.camera.set_preset(0.0, 0.0, 25.0, glam::Vec3::ZERO, false);
                            }
                            if ui.small_button("3:Cor").on_hover_text("Coronal Y-Z camera preset (3)").clicked() {
                                self.camera.set_preset(
                                    std::f32::consts::FRAC_PI_2,
                                    0.04,
                                    25.0,
                                    glam::Vec3::ZERO,
                                    false,
                                );
                            }
                            if ui.small_button("4:Top").on_hover_text("Top-Down X-Z camera preset (4)").clicked() {
                                self.camera.set_preset(0.0, 1.35, 25.0, glam::Vec3::ZERO, false);
                            }
                            if ui.small_button("R:Reset").on_hover_text("Reset camera target & auto-orbit (R)").clicked() {
                                self.camera.reset();
                            }
                        });
                    });

                    ui.add_space(6.0);

                    // 6. Causal Lesion, Tract-Severing & Optogenetics Wetbench
                    ui.group(|ui| {
                        ui.strong("6. Causal Lesion & Optogenetics Lab");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::Inspect, "🔍 Inspect (I)")
                                .on_hover_text("Inspect soma coordinates, membrane potential, activation, and gradient in 3D (Hotkey: I).");
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::AblateLesion, "✂ Ablate (X)")
                                .on_hover_text("Click a 3D soma to toggle focal lesion ablation (silences outgoing signals) (Hotkey: X).");
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::ChR2Excite, "[+] ChR2 (E)")
                                .on_hover_text("Channelrhodopsin-2 photostimulation: click a 3D soma to clamp-excite it (Hotkey: E).");
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::NpHRSilence, "[-] NpHR (S)")
                                .on_hover_text("Halorhodopsin photoinhibition: click a 3D soma to optogenetically silence it (Hotkey: S).");
                        });
                        ui.add(
                            egui::Slider::new(&mut self.sim.opto_clamp_drive, 0.2..=1.8)
                                .text("ChR2 Laser Drive"),
                        )
                        .on_hover_text("Optical depolarization drive applied to ChR2-clamped neurons.");
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .button("✂ Sever Feedback")
                                .on_hover_text("Sever reverse top-down feedback tracts to observe predictive-coding disruption.")
                                .clicked()
                            {
                                let n = self.sim.sever_feedback_tracts();
                                self.status_banner = format!("Severed {} reverse feedback tracts.", n);
                            }
                            if ui
                                .button("✂ Sever Skip/Lat")
                                .on_hover_text("Sever lateral and skip-connection axons across cortical/columnar zones.")
                                .clicked()
                            {
                                let n = self.sim.sever_lateral_skip_tracts();
                                self.status_banner = format!("Severed {} lateral/skip axons.", n);
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .button("⚡ Focal Stroke (25%)")
                                .on_hover_text("Ablate 25% of somata in a localized cluster to observe acute drop and plastic recovery.")
                                .clicked()
                            {
                                let n = self.sim.induce_focal_stroke();
                                self.status_banner = format!("Focal lesion ablated {} somata.", n);
                            }
                            if ui
                                .button("♻ Heal All")
                                .on_hover_text("Restore all ablated somata, severed tracts, and optogenetic clamps.")
                                .clicked()
                            {
                                self.sim.heal_all_interventions();
                                self.status_banner = "Healed all lesions, tracts & clamps.".to_string();
                            }
                        });
                        if let Some(pre) = self.sim.pre_lesion_metric {
                            let acute = self.sim.acute_lesion_metric.unwrap_or(pre);
                            ui.small(format!(
                                "Lesion Recovery: Pre {:.1}% -> Acute {:.1}% -> Now {:.1}%",
                                pre, acute, self.sim.primary_metric
                            ));
                            draw_lesion_recovery_bar(ui, pre, acute, self.sim.primary_metric);
                        }
                    });

                    ui.add_space(6.0);
                    ui.small(&self.status_banner);
                });
            });

        // Always-visible slim bottom footer bar (Copyright + About modal trigger + status)
        egui::TopBottomPanel::bottom("copyright_footer_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.small(&self.status_banner);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("About").clicked() {
                        self.show_about_modal = true;
                    }
                    ui.hyperlink_to("crvscience.com", "https://crvscience.com");
                    ui.small("Copyright CRV Science Network | GPL-3.0-or-later");
                });
            });
        });

        // Bottom Panel
        if self.show_bottom_matrix {
            egui::TopBottomPanel::bottom("bottom_lock_matrix")
                .default_height(232.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.columns(3, |cols| {
                        cols[0].strong(format!(
                            "Live Sample Inspector — {} ({})",
                            self.sim.dataset_id.code(),
                            self.sim.active_sample.sample_tag
                        ));
                        let out_labels = self.sim.dataset_id.output_labels();
                        for (i, &pred) in self.sim.last_prediction.iter().enumerate() {
                            let target = self.sim.active_sample.targets.get(i).copied().unwrap_or(0.0);
                            let lbl = out_labels.get(i).copied().unwrap_or("Out");
                            cols[0].horizontal(|ui| {
                                ui.label(format!("{:<14}", lbl));
                                ui.add(
                                    egui::ProgressBar::new(pred.clamp(0.0, 1.0))
                                        .desired_width(145.0)
                                        .text(format!("pred {:.2} | target {:.2}", pred, target)),
                                );
                            });
                        }
                        cols[0].separator();
                        cols[0].small(format!(
                            "Structural Adaptation Totals — +{} Sprouted | -{} Pruned | +{} Neurogenesis",
                            self.sim.sprouted_total, self.sim.pruned_total, self.sim.neurogenesis_total
                        ));
                        if let Some(msg) = self.sim.plasticity_log.first() {
                            cols[0].small(RichText::new(msg).color(Color32::from_rgb(125, 211, 252)));
                        }

                        cols[1].strong("Live Telemetry Sparklines (Metric, Loss & Lesion)");
                        draw_sparkline(
                            &mut cols[1],
                            self.sim.dataset_id.task_kind().primary_metric_label(),
                            &self.sim.metric_history,
                            Color32::from_rgb(52, 211, 153),
                            true,
                            self.sim.pre_lesion_metric,
                            self.sim.acute_lesion_metric,
                            &self.sim.lesion_events,
                            format!("{:.1}%", self.sim.primary_metric),
                        );
                        cols[1].add_space(4.0);
                        draw_sparkline(
                            &mut cols[1],
                            &format!(
                                "{} (Syn: {})",
                                self.sim.dataset_id.task_kind().loss_label(),
                                self.sim.synapses.len()
                            ),
                            &self.sim.loss_history,
                            Color32::from_rgb(251, 191, 36),
                            false,
                            None,
                            None,
                            &[],
                            format!("{:.3}", self.sim.current_loss),
                        );

                        cols[2].strong("Locked Dataset / Locked Model Comparison Ledger");
                        ScrollArea::vertical().max_height(168.0).show(&mut cols[2], |ui| {
                            egui::Grid::new("comparison_ledger_grid")
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.strong("Context");
                                    ui.strong("Model");
                                    ui.strong("Data");
                                    ui.strong("Score");
                                    ui.strong("Loss");
                                    ui.strong("N / Syn");
                                    ui.end_row();

                                    for rec in self.sim.comparison_ledger.iter().rev() {
                                        ui.label(&rec.lock_context);
                                        ui.label(rec.model.code());
                                        ui.label(rec.dataset.code());
                                        ui.label(format!("{:.1}%", rec.primary_score));
                                        ui.label(format!("{:.3}", rec.loss));
                                        ui.label(format!(
                                            "{} / {}",
                                            rec.neuron_count, rec.synapse_count
                                        ));
                                        ui.end_row();
                                    }
                                });
                        });
                    });
                });
        }

        // Primary Center Panel: Interactive 3D Neural Space
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_rect_before_wrap();
            let resp = ui.allocate_rect(avail, Sense::click_and_drag());

            if resp.dragged_by(egui::PointerButton::Primary) {
                let delta = resp.drag_delta();
                self.camera.apply_orbit_delta(delta.x, delta.y);
                ctx.request_repaint();
            } else if resp.dragged_by(egui::PointerButton::Secondary)
                || resp.dragged_by(egui::PointerButton::Middle)
            {
                let delta = resp.drag_delta();
                self.camera.apply_pan_delta(delta.x, delta.y, avail.height());
                ctx.request_repaint();
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if resp.hovered() && scroll.abs() > 0.1 {
                self.camera.apply_zoom_delta(scroll);
                ctx.request_repaint();
            }

            let painter = ui.painter_at(avail);
            let mut hovered = None;
            draw_neural_space_3d(
                &painter,
                avail,
                &self.sim,
                &self.camera,
                self.render_mode,
                self.manual_organic_blend,
                self.show_group_highlights,
                &mut hovered,
                resp.hover_pos(),
                self.selected_neuron,
            );
            self.hovered_neuron = hovered;
            if resp.clicked() {
                let shift = ui.input(|i| i.modifiers.shift);
                let alt = ui.input(|i| i.modifiers.alt);
                if let Some(h_idx) = self.hovered_neuron {
                    if shift || self.probe_tool == ProbeTool::AblateLesion {
                        self.sim.toggle_ablate_neuron(h_idx);
                    } else if alt || self.probe_tool == ProbeTool::ChR2Excite {
                        self.sim.set_neuron_clamp(h_idx, OptoClamp::ChR2Excite);
                    } else if self.probe_tool == ProbeTool::NpHRSilence {
                        self.sim.set_neuron_clamp(h_idx, OptoClamp::NpHRSilence);
                    }
                    self.selected_neuron = Some(h_idx);
                } else {
                    self.selected_neuron = None;
                }
            }

            if let Some(n_idx) = self.selected_neuron.or(self.hovered_neuron) {
                if let Some(n) = self.sim.neurons.get(n_idx) {
                    let card_rect = egui::Rect::from_min_size(
                        avail.min + egui::vec2(14.0, 14.0),
                        egui::vec2(285.0, 96.0),
                    );
                    painter.rect_filled(
                        card_rect,
                        6.0,
                        Color32::from_rgba_unmultiplied(15, 23, 42, 222),
                    );
                    painter.text(
                        card_rect.min + egui::vec2(10.0, 10.0),
                        egui::Align2::LEFT_TOP,
                        format!(
                            "Soma #{} [{}] (L{})\nPos: ({:+.1}, {:+.1}, {:+.1}) | Ca2+: {:.2}\nAct: {:+.3} | Vm: {:+.3} | Bias: {:+.3}\nUtility: {:.3} | Grad: {:+.3}",
                            n.id,
                            n.role.label_for_model(self.sim.model, n.layer),
                            n.layer,
                            n.pos.x,
                            n.pos.y,
                            n.pos.z,
                            n.apical_ca_burst,
                            n.activation,
                            n.membrane_v,
                            n.bias,
                            n.utility,
                            n.grad
                        ),
                        egui::FontId::monospace(11.2),
                        Color32::from_rgb(226, 232, 240),
                    );
                }
            }

            painter.text(
                avail.max - egui::vec2(12.0, 10.0),
                egui::Align2::RIGHT_BOTTOM,
                "Left-Drag: Orbit | Right-Drag: Pan | Scroll: Zoom | 1-4: Views | R: Reset | F1: Help",
                egui::FontId::monospace(10.0),
                Color32::from_rgba_unmultiplied(148, 163, 184, 185),
            );
        });

        if self.show_about_modal {
            let mut open = self.show_about_modal;
            egui::Window::new("About & Field Guide — NeuroSpace Lab")
                .open(&mut open)
                .collapsible(false)
                .resizable(true)
                .default_width(580.0)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ScrollArea::vertical().max_height(480.0).show(ui, |ui| {
                        ui.heading(
                            RichText::new("NeuroSpace Lab v1.6.0")
                                .color(Color32::from_rgb(56, 189, 248)),
                        );
                        ui.label("Safe-Rust 3D Neural Network Topology, Plasticity & Cerebral Cortex Grey-Matter Workstation");
                        ui.separator();

                        ui.strong("Keyboard Shortcuts & 3D Camera Navigation");
                        egui::Grid::new("help_shortcuts_grid")
                            .striped(true)
                            .spacing([14.0, 4.0])
                            .show(ui, |ui| {
                                for (key, desc) in [
                                    ("Space", "Play / Pause continuous simulation"),
                                    ("N or .", "Pause and single-step simulation by 1 tick"),
                                    ("Left-Drag", "Orbit 3D Neural Space (with smooth inertial damping)"),
                                    ("Right-Drag / Middle-Drag", "Pan 3D camera focal target"),
                                    ("Scroll Wheel", "Zoom camera distance [8.5 .. 54.0]"),
                                    ("1 / 2 / 3 / 4", "Camera Presets: Isometric / Sagittal / Coronal / Top-Down"),
                                    ("R", "Reset camera to default Isometric + Auto-Orbit"),
                                    ("A", "Toggle camera auto-orbit"),
                                    ("I / X / E / S", "Probe Tools: Inspect / Ablate / ChR2 (+) / NpHR (-)"),
                                    ("Shift + Click Soma", "Quick toggle focal ablation on hovered soma"),
                                    ("Alt + Click Soma", "Quick toggle ChR2 photostimulation on hovered soma"),
                                    ("F5 / F9", "SQLite WAL QuickSave / QuickLoad ('quicksave_f5')"),
                                    ("Ctrl/Cmd + K", "Quick-Jump Command Palette (regions, protocols, exports)"),
                                    ("Ctrl/Cmd+S / Ctrl/Cmd+O", "Export / Import SHA-256 .neuroworkset.json"),
                                    ("F1 / Esc", "Toggle this Help / Close Modal or Clear Selection"),
                                ] {
                                    ui.label(RichText::new(key).monospace().strong().color(Color32::from_rgb(125, 211, 252)));
                                    ui.label(desc);
                                    ui.end_row();
                                }
                            });

                        ui.separator();
                        ui.strong("Causal Lesion, Tractotomy & Optogenetics Guide");
                        ui.small("• Inspect (I): Displays Soma ID, laminar role (L1–L6 / MoE / Attention), 3D coordinates, apical dendritic Ca2+ plateau, activation, membrane Vm, bias, utility, and backprop gradient.");
                        ui.small("• Focal Ablation (X or Shift+Click): Silences individual somata without removing downstream structures, engaging homeostatic gain (1.45x) and peri-infarct bypass sprouting.");
                        ui.small("• ChR2 Photostimulation (E or Alt+Click): Channelrhodopsin-2 blue-light depolarizing clamp driving soma activation to ChR2 Laser Drive [0.2 .. 1.8] and triggering spikes in M7/M13.");
                        ui.small("• NpHR Photoinhibition (S): Halorhodopsin yellow-light hyperpolarizing clamp silencing soma output (activation = 0.0) reversibly without structural ablation.");
                        ui.small("• Tractotomy & Stroke: 'Sever Feedback' isolates top-down apical L1/predictive-error paths; 'Sever Skip/Lat' severs lateral & skip connections; 'Focal Stroke (25%)' lesions a local spatial cluster.");

                        ui.separator();
                        ui.strong("Copyright CRV Science Network");
                        ui.small("Licensed under GNU General Public License v3.0 or later (GPL-3.0-or-later).");
                        ui.small("Distributed in the hope that it will be useful, WITHOUT ANY WARRANTY.");
                        ui.add_space(4.0);
                        ui.hyperlink_to("CRV Science Network — https://crvscience.com", "https://crvscience.com");
                        ui.hyperlink_to("CRV Science Public Apps — https://crvscience.com/apps", "https://crvscience.com/apps");
                    });
                    ui.add_space(8.0);
                    if ui.button("Close (Esc)").clicked() {
                        self.show_about_modal = false;
                    }
                });
            if !open {
                self.show_about_modal = false;
            }
        }

        if self.cmd_palette_open {
            let mut open = self.cmd_palette_open;
            let mut chosen_action: Option<PaletteAction> = None;
            egui::Window::new("Quick-Jump Command Palette (Ctrl+K)")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .default_width(500.0)
                .anchor(egui::Align2::CENTER_TOP, [0.0, 48.0])
                .show(ctx, |ui| {
                    let resp = ui.text_edit_singleline(&mut self.cmd_query);
                    if resp.changed() {
                        self.cmd_selected_idx = 0;
                    }
                    resp.request_focus();
                    let hits = search_palette_entries(&self.cmd_entries, &self.cmd_query, 18);
                    if !hits.is_empty() {
                        self.cmd_selected_idx = self.cmd_selected_idx.min(hits.len() - 1);
                        if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                            self.cmd_selected_idx = (self.cmd_selected_idx + 1).min(hits.len() - 1);
                        }
                        if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                            self.cmd_selected_idx = self.cmd_selected_idx.saturating_sub(1);
                        }
                        if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                            chosen_action = Some(hits[self.cmd_selected_idx].action.clone());
                        }
                    } else {
                        self.cmd_selected_idx = 0;
                    }
                    ui.small("Use Up/Down to select, Enter to execute, Esc to close.");
                    ui.separator();
                    ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                        for (idx, entry) in hits.iter().enumerate() {
                            let is_sel = idx == self.cmd_selected_idx;
                            let lbl = format!("[{}] {} — {}", entry.category, entry.title, entry.subtitle);
                            if ui.selectable_label(is_sel, lbl).clicked() {
                                chosen_action = Some(entry.action.clone());
                            }
                        }
                    });
                });
            self.cmd_palette_open = open;
            if let Some(act) = chosen_action {
                self.cmd_palette_open = false;
                self.execute_palette_action(act);
            }
        }

        if self.show_bifurcation_modal {
            let mut open = self.show_bifurcation_modal;
            let mut export_req: Option<bool> = None;
            let mut rerun_req = false;
            let digest_opt = self.last_bifurcation_digest.clone();
            egui::Window::new("Structural Plasticity Bifurcation Digest")
                .open(&mut open)
                .collapsible(false)
                .resizable(true)
                .default_width(640.0)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    if let Some(digest) = &digest_opt {
                        ui.label(RichText::new(format!(
                            "{} ({}) on {} | Critical Prune Boundary: {:.3} | Peak Efficiency: {:.4}",
                            digest.model.code(),
                            digest.model.name(),
                            digest.dataset.code(),
                            digest.critical_prune_threshold,
                            digest.peak_efficiency_score
                        )).strong());
                        ui.separator();
                        egui::Grid::new("bifurcation_digest_grid").striped(true).show(ui, |ui| {
                            ui.strong("Prune");
                            ui.strong("Growth");
                            ui.strong("Regime");
                            ui.strong("Neur/Syn");
                            ui.strong("FireRate");
                            ui.strong("NetEff");
                            ui.strong("Score");
                            ui.end_row();
                            for p in &digest.points {
                                ui.label(format!("{:.3}", p.prune_threshold));
                                ui.label(format!("{:.2}", p.plasticity_growth));
                                ui.label(&p.regime_label);
                                ui.label(format!("{}/{}", p.active_neurons, p.active_synapses));
                                ui.label(format!("{:.3}", p.mean_firing_rate));
                                ui.label(format!("{:.4}", p.network_efficiency));
                                ui.label(format!("{:.1}%", p.primary_score));
                                ui.end_row();
                            }
                        });
                    } else {
                        ui.label("No plasticity sweep computed yet.");
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Re-Run Sweep").clicked() {
                            rerun_req = true;
                        }
                        if ui.button("Export JSON...").clicked() {
                            export_req = Some(false);
                        }
                        if ui.button("Export CSV...").clicked() {
                            export_req = Some(true);
                        }
                        if ui.button("Close (Esc)").clicked() {
                            self.show_bifurcation_modal = false;
                        }
                    });
                });
            if !open {
                self.show_bifurcation_modal = false;
            }
            if rerun_req {
                self.execute_palette_action(PaletteAction::RunPlasticitySweep);
            }
            if let Some(is_csv) = export_req {
                self.export_bifurcation_digest_dialog(is_csv);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_sparkline(
    ui: &mut egui::Ui,
    title: &str,
    data: &[f32],
    color: Color32,
    is_percent: bool,
    pre_lesion: Option<f32>,
    acute_lesion: Option<f32>,
    lesion_events: &[(usize, String)],
    current_text: String,
) {
    let avail_w = ui.available_width().max(180.0);
    ui.horizontal(|ui| {
        ui.small(RichText::new(title).strong().color(Color32::from_rgb(203, 213, 225)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.small(RichText::new(current_text).strong().color(color));
        });
    });

    let (rect, resp) = ui.allocate_exact_size(egui::vec2(avail_w, 68.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(10, 16, 28));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(71, 85, 105, 140)),
        egui::StrokeKind::Outside,
    );

    let has_top_band = pre_lesion.is_some() || acute_lesion.is_some() || !lesion_events.is_empty();
    let top_pad = if has_top_band { 14.0 } else { 6.0 };
    let plot = egui::Rect::from_min_max(
        egui::Pos2::new(rect.min.x + 36.0, rect.min.y + top_pad),
        egui::Pos2::new(rect.max.x - 6.0, rect.max.y - 13.0),
    );
    painter.line_segment(
        [
            egui::Pos2::new(plot.min.x, plot.min.y),
            egui::Pos2::new(plot.min.x, plot.max.y),
        ],
        egui::Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(71, 85, 105, 110)),
    );
    painter.line_segment(
        [
            egui::Pos2::new(plot.min.x, plot.max.y),
            egui::Pos2::new(plot.max.x, plot.max.y),
        ],
        egui::Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(71, 85, 105, 110)),
    );
    painter.line_segment(
        [
            egui::Pos2::new(plot.min.x, plot.center().y),
            egui::Pos2::new(plot.max.x, plot.center().y),
        ],
        egui::Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(71, 85, 105, 80)),
    );

    if data.is_empty() {
        painter.text(
            plot.center(),
            egui::Align2::CENTER_CENTER,
            "Collecting telemetry...",
            egui::FontId::monospace(10.0),
            Color32::from_rgb(148, 163, 184),
        );
        return;
    }

    let mut min_v = data.iter().copied().fold(f32::INFINITY, f32::min);
    let mut max_v = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if let Some(p) = pre_lesion {
        min_v = min_v.min(p);
        max_v = max_v.max(p);
    }
    if let Some(a) = acute_lesion {
        min_v = min_v.min(a);
        max_v = max_v.max(a);
    }
    if is_percent {
        min_v = min_v.min(0.0);
        max_v = (max_v * 1.05).max(min_v + 10.0).min(100.0);
    } else if (max_v - min_v).abs() < 0.05 {
        min_v = (min_v - 0.05).max(0.0);
        max_v = min_v + 0.1;
    }
    let span = (max_v - min_v).max(1e-4);

    let y_of = |v: f32| -> f32 {
        let norm = ((v - min_v) / span).clamp(0.0, 1.0);
        plot.max.y - norm * plot.height()
    };

    // Pre / Acute lesion reference lines + compact top legend
    let pre_col = Color32::from_rgb(56, 189, 248);
    let acute_col = Color32::from_rgb(244, 63, 94);
    let mut legend_x = plot.max.x - 2.0;
    if let Some(acute) = acute_lesion {
        let y = y_of(acute);
        painter.line_segment(
            [egui::Pos2::new(plot.min.x, y), egui::Pos2::new(plot.max.x, y)],
            egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(244, 63, 94, 165)),
        );
        let lbl = format!("Acute {:.1}%", acute);
        let lw = lbl.len() as f32 * 5.4 + 12.0;
        painter.line_segment(
            [
                egui::Pos2::new(legend_x - lw, rect.min.y + 6.5),
                egui::Pos2::new(legend_x - lw + 8.0, rect.min.y + 6.5),
            ],
            egui::Stroke::new(1.8_f32, acute_col),
        );
        painter.text(
            egui::Pos2::new(legend_x, rect.min.y + 2.0),
            egui::Align2::RIGHT_TOP,
            lbl,
            egui::FontId::monospace(8.5),
            acute_col,
        );
        legend_x -= lw + 8.0;
    }
    if let Some(pre) = pre_lesion {
        let y = y_of(pre);
        painter.line_segment(
            [egui::Pos2::new(plot.min.x, y), egui::Pos2::new(plot.max.x, y)],
            egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(56, 189, 248, 165)),
        );
        let lbl = format!("Pre {:.1}%", pre);
        let lw = lbl.len() as f32 * 5.4 + 12.0;
        painter.line_segment(
            [
                egui::Pos2::new(legend_x - lw, rect.min.y + 6.5),
                egui::Pos2::new(legend_x - lw + 8.0, rect.min.y + 6.5),
            ],
            egui::Stroke::new(1.8_f32, pre_col),
        );
        painter.text(
            egui::Pos2::new(legend_x, rect.min.y + 2.0),
            egui::Align2::RIGHT_TOP,
            lbl,
            egui::FontId::monospace(8.5),
            pre_col,
        );
    }

    let n = data.len();
    let x_of = |i: usize| -> f32 {
        if n <= 1 {
            plot.center().x
        } else {
            plot.min.x + (i as f32 / (n - 1) as f32) * plot.width()
        }
    };

    // Render lesion event vertical markers + tag labels cleanly
    let mut last_tag_x = f32::NEG_INFINITY;
    for (ev_idx, tag) in lesion_events {
        if *ev_idx < n {
            let ex = x_of(*ev_idx);
            painter.line_segment(
                [egui::Pos2::new(ex, plot.min.y), egui::Pos2::new(ex, plot.max.y)],
                egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(244, 63, 94, 185)),
            );
            let tag_w = (tag.len() as f32 * 5.2 + 6.0).clamp(24.0, 64.0);
            let tx = ex.clamp(plot.min.x + tag_w * 0.5, (plot.max.x - tag_w * 0.5).max(plot.min.x));
            if (tx - last_tag_x).abs() >= tag_w * 0.85 {
                let tag_rect = egui::Rect::from_center_size(
                    egui::Pos2::new(tx, rect.min.y + 6.5),
                    egui::vec2(tag_w, 11.0),
                );
                painter.rect_filled(tag_rect, 2.0, Color32::from_rgba_unmultiplied(15, 23, 42, 225));
                painter.text(
                    tag_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    tag,
                    egui::FontId::monospace(8.2),
                    Color32::from_rgb(253, 164, 175),
                );
                last_tag_x = tx;
            }
        }
    }

    let pts: Vec<egui::Pos2> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| egui::Pos2::new(x_of(i), y_of(v)))
        .collect();
    if pts.len() >= 2 {
        painter.add(egui::Shape::line(pts.clone(), egui::Stroke::new(1.8_f32, color)));
    }
    if let Some(&last_pt) = pts.last() {
        painter.circle_filled(last_pt, 2.6, color);
    }

    // Y-axis min / max labels
    let min_lbl = if is_percent {
        format!("{min_v:.0}%")
    } else {
        format!("{min_v:.2}")
    };
    let max_lbl = if is_percent {
        format!("{max_v:.0}%")
    } else {
        format!("{max_v:.2}")
    };
    painter.text(
        egui::Pos2::new(rect.min.x + 32.0, plot.min.y),
        egui::Align2::RIGHT_TOP,
        max_lbl,
        egui::FontId::monospace(9.0),
        Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::Pos2::new(rect.min.x + 32.0, plot.max.y),
        egui::Align2::RIGHT_BOTTOM,
        min_lbl,
        egui::FontId::monospace(9.0),
        Color32::from_rgb(148, 163, 184),
    );

    // X-axis sample/step ticks along bottom axis
    let tick_indices: Vec<usize> = if n <= 1 {
        vec![0]
    } else if n < 5 {
        vec![0, n - 1]
    } else {
        vec![0, n / 2, n - 1]
    };
    for (ti, &idx) in tick_indices.iter().enumerate() {
        let tx = x_of(idx);
        painter.line_segment(
            [
                egui::Pos2::new(tx, plot.max.y),
                egui::Pos2::new(tx, plot.max.y + 3.0),
            ],
            egui::Stroke::new(0.9_f32, Color32::from_rgb(100, 116, 139)),
        );
        let align = if ti == 0 {
            egui::Align2::LEFT_BOTTOM
        } else if ti + 1 == tick_indices.len() {
            egui::Align2::RIGHT_BOTTOM
        } else {
            egui::Align2::CENTER_BOTTOM
        };
        painter.text(
            egui::Pos2::new(tx, rect.max.y - 1.0),
            align,
            format!("#{}", idx + 1),
            egui::FontId::monospace(8.2),
            Color32::from_rgb(148, 163, 184),
        );
    }

    if let Some(hover_pos) = resp.hover_pos() {
        let idx = if n <= 1 || plot.width() <= 1.0 {
            0
        } else {
            let t = ((hover_pos.x - plot.min.x) / plot.width()).clamp(0.0, 1.0);
            (t * (n - 1) as f32).round() as usize
        }
        .min(n - 1);
        let hx = x_of(idx);
        let hy = y_of(data[idx]);
        painter.line_segment(
            [egui::Pos2::new(hx, plot.min.y), egui::Pos2::new(hx, plot.max.y)],
            egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(226, 232, 240, 140)),
        );
        painter.circle_filled(egui::Pos2::new(hx, hy), 3.2, Color32::WHITE);
        let val_str = if is_percent {
            format!("{:.1}%", data[idx])
        } else {
            format!("{:.4}", data[idx])
        };
        resp.on_hover_text(format!("{title}\nSample #{}: {val_str}", idx + 1));
    }
}

fn draw_lesion_recovery_bar(ui: &mut egui::Ui, pre: f32, acute: f32, now: f32) {
    let avail_w = ui.available_width().max(160.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(avail_w, 34.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_rgb(12, 19, 34));
    let plot = rect.shrink2(egui::vec2(18.0, 6.0));
    let min_v = pre.min(acute).min(now).min(0.0);
    let max_v = pre.max(acute).max(now).max(min_v + 10.0).min(100.0);
    let span = (max_v - min_v).max(1e-4);
    let y_of = |v: f32| plot.max.y - ((v - min_v) / span).clamp(0.0, 1.0) * plot.height();
    let pts = vec![
        egui::Pos2::new(plot.min.x, y_of(pre)),
        egui::Pos2::new(plot.center().x, y_of(acute)),
        egui::Pos2::new(plot.max.x, y_of(now)),
    ];
    painter.add(egui::Shape::line(
        pts.clone(),
        egui::Stroke::new(1.6_f32, Color32::from_rgb(56, 189, 248)),
    ));
    painter.circle_filled(pts[0], 3.0, Color32::from_rgb(56, 189, 248));
    painter.circle_filled(pts[1], 3.0, Color32::from_rgb(244, 63, 94));
    painter.circle_filled(pts[2], 3.0, Color32::from_rgb(52, 211, 153));
}

impl eframe::App for NeuroSpaceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui_step(ctx);
    }
}

fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    struct NoopWake;
    impl std::task::Wake for NoopWake {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(NoopWake));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn write_rgba_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected {
        return Err(format!("Unexpected RGBA buffer length {} (expected {})", rgba.len(), expected).into());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}

/// Render an offscreen 1600x1000 RGBA PNG screenshot without opening any OS GUI window.
pub fn render_headless_screenshot(
    out_path: &std::path::Path,
    view: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    use eframe::egui_wgpu::{wgpu, Renderer, ScreenDescriptor};

    let width: u32 = 1600;
    let height: u32 = 1000;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("No GPU adapter available for offscreen rendering")?;
    let (device, queue) = block_on_future(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("neurospace-offscreen-device"),
            ..Default::default()
        },
        None,
    ))
    .map_err(|e| format!("Failed to request offscreen GPU device: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("neurospace-offscreen-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bytes_per_row = width * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neurospace-offscreen-staging"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut renderer = Renderer::new(&device, format, None, 1, false);
    let egui_ctx = egui::Context::default();
    egui_ctx.set_pixels_per_point(1.0);
    egui_ctx.set_visuals(egui::Visuals::dark());

    let mut app = NeuroSpaceApp::new_headless();
    app.running = false;
    match view {
        2 => {
            app.sim = NeuralSpaceSim::new(
                ModelArch::M11CorticalColumnsHtm,
                DatasetId::D6RoboticsNavigation,
                2026,
            );
            app.sim.dataset_locked = true;
            app.sim.run_locked_sweep(45);
            app.camera.yaw = 0.72;
            app.camera.pitch = 0.28;
            app.status_banner = "Preset 2 — M11 Cortical Columns (HTM) on LiDAR-12 Autonomous Nav".to_string();
        }
        3 => {
            app.sim = NeuralSpaceSim::new(
                ModelArch::M14FrontierLlmMoe,
                DatasetId::D7MolecularProperties,
                2026,
            );
            app.sim.model_locked = true;
            app.sim.run_locked_sweep(45);
            app.camera.yaw = -0.58;
            app.camera.pitch = 0.34;
            app.status_banner = "Preset 3 — M14 Frontier LLM (GQA + Sparse MoE) on ChemGraph QSAR".to_string();
        }
        _ => {
            app.sim = NeuralSpaceSim::new(
                ModelArch::M13CerebralCortexGreyMatter,
                DatasetId::D4PlantSpecies,
                2026,
            );
            app.sim.dataset_locked = true;
            app.sim.run_locked_sweep(55);
            app.selected_neuron = app.sim.neurons.first().map(|n| n.id);
            app.camera.yaw = 0.48;
            app.camera.pitch = 0.24;
            app.status_banner = "Preset 1 — Tier V M13 6-Layered Cerebral Cortex Grey Matter (L1–L6)".to_string();
        }
    }
    for _ in 0..60 {
        app.sim.step_simulation();
    }

    for frame_idx in 0..4 {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            viewports: std::iter::once((
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(1.0),
                    ..Default::default()
                },
            ))
            .collect(),
            time: Some(frame_idx as f64 / 60.0),
            predicted_dt: 1.0 / 60.0,
            ..Default::default()
        };
        let full_output = egui_ctx.run(raw_input, |ctx| {
            app.ui_step(ctx);
        });
        for (id, image_delta) in &full_output.textures_delta.set {
            renderer.update_texture(&device, &queue, *id, image_delta);
        }
        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: 1.0,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("neurospace-offscreen-encoder"),
        });
        let user_cmd_bufs = renderer.update_buffers(
            &device,
            &queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("neurospace-offscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let mut render_pass = render_pass.forget_lifetime();
            renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }
        if frame_idx == 3 {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(user_cmd_bufs.into_iter().chain(std::iter::once(encoder.finish())));
        device.poll(wgpu::Maintain::Wait);
        for id in &full_output.textures_delta.free {
            renderer.free_texture(id);
        }
    }

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("GPU readback channel error: {e}"))?
        .map_err(|e| format!("GPU buffer map failed: {e}"))?;
    let mapped = slice.get_mapped_range();
    write_rgba_png(out_path, width, height, &mapped)?;
    drop(mapped);
    staging.unmap();
    Ok(())
}
