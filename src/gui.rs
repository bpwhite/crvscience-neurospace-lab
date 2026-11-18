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
use crate::engine::{NeuralSpaceSim, OptoClamp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeTool {
    Inspect,
    AblateLesion,
    ChR2Excite,
    NpHRSilence,
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
    pub status_banner: String,
    store: Option<LabStore>,
}

impl NeuroSpaceApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
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
            status_banner: "Ready — Drag 3D Neural Space to orbit, scroll to zoom, F5/F9 to save/load."
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
}

impl NeuroSpaceApp {
    pub fn ui_step(&mut self, ctx: &egui::Context) {
        ctx.set_visuals(egui::Visuals::dark());

        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.running = !self.running;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F5)) {
            self.quick_save();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F9)) {
            self.quick_load();
        }

        if self.running {
            for _ in 0..self.steps_per_frame {
                self.sim.step_simulation();
            }
            if self.camera.auto_orbit {
                self.camera.yaw += 0.0038;
            }
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
                if ui.button("⏭ Step").clicked() {
                    self.sim.step_simulation();
                }
                if ui.button("↺ Rebuild Topology").clicked() {
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
                let sweep_label = if self.sim.dataset_locked && !self.sim.model_locked {
                    "⚡ Sweep All 15 Models (Locked Dataset)"
                } else {
                    "⚡ Sweep All 7 Datasets (Locked Model)"
                };
                if ui.button(sweep_label).clicked() {
                    self.sim.run_locked_sweep(35);
                }

                ui.separator();
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
                        );
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.momentum, 0.0..=0.96)
                                .text("Momentum β"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.prune_threshold, 0.0..=0.22)
                                .text("Synaptic Prune |w|"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.plasticity_growth, 0.0..=1.0)
                                .text("Neurogenesis / Sprout"),
                        );
                        let prev_flow = self.sim.params.flow_sample_rate;
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.flow_sample_rate, 0.0..=1.0)
                                .text("River Stream Intensity"),
                        );
                        if (self.sim.params.flow_sample_rate - prev_flow).abs() > 0.01 {
                            self.sim.refresh_flow_sample();
                        }
                        let prev_scale = self.sim.params.hidden_scale;
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.hidden_scale, 0.5..=2.0)
                                .text("Cortical Capacity"),
                        );
                        if (self.sim.params.hidden_scale - prev_scale).abs() > 0.04 {
                            self.selected_neuron = None;
                            self.sim.rebuild_topology(false);
                        }
                        ui.add(
                            egui::Slider::new(&mut self.sim.params.adversarial_eps, 0.0..=0.45)
                                .text("Adversarial ε"),
                        );
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
                        ui.checkbox(&mut self.camera.auto_orbit, "Auto-Orbit 3D Neural Space");
                    });

                    ui.add_space(6.0);

                    // 6. Causal Lesion, Tract-Severing & Optogenetics Wetbench
                    ui.group(|ui| {
                        ui.strong("6. Causal Lesion & Optogenetics Lab");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::Inspect, "🔍 Inspect");
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::AblateLesion, "✂ Ablate");
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::ChR2Excite, "[+] ChR2 (+)");
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::NpHRSilence, "[-] NpHR (-)");
                        });
                        ui.add(
                            egui::Slider::new(&mut self.sim.opto_clamp_drive, 0.2..=1.8)
                                .text("ChR2 Laser Drive"),
                        );
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("✂ Sever Feedback").clicked() {
                                let n = self.sim.sever_feedback_tracts();
                                self.status_banner = format!("Severed {} reverse feedback tracts.", n);
                            }
                            if ui.button("✂ Sever Skip/Lat").clicked() {
                                let n = self.sim.sever_lateral_skip_tracts();
                                self.status_banner = format!("Severed {} lateral/skip axons.", n);
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("⚡ Focal Stroke (25%)").clicked() {
                                let n = self.sim.induce_focal_stroke();
                                self.status_banner = format!("Focal lesion ablated {} somata.", n);
                            }
                            if ui.button("♻ Heal All").clicked() {
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
                    ui.small("Copyright CRV Science Network | GPL-3.0-or-later");
                });
            });
        });

        // Bottom Panel
        if self.show_bottom_matrix {
            egui::TopBottomPanel::bottom("bottom_lock_matrix")
                .default_height(210.0)
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

            if resp.dragged() {
                let delta = resp.drag_delta();
                self.camera.auto_orbit = false;
                self.camera.yaw -= delta.x * 0.008;
                self.camera.pitch = (self.camera.pitch + delta.y * 0.008).clamp(-1.35, 1.35);
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if resp.hovered() && scroll.abs() > 0.1 {
                self.camera.distance = (self.camera.distance - scroll * 0.035).clamp(8.5, 54.0);
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
        });

        if self.show_about_modal {
            let mut open = self.show_about_modal;
            egui::Window::new("About NeuroSpace Lab")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.heading(
                        RichText::new("NeuroSpace Lab v1.6.0")
                            .color(Color32::from_rgb(56, 189, 248)),
                    );
                    ui.label("3D Neural Network Topology, Plasticity & Cerebral Cortex Wetware Lab");
                    ui.separator();
                    ui.strong("Copyright CRV Science Network");
                    ui.small(
                        "Licensed under GNU General Public License v3.0 or later (GPL-3.0-or-later).",
                    );
                    ui.small("Distributed in the hope that it will be useful, WITHOUT ANY WARRANTY.");
                    ui.add_space(6.0);
                    ui.hyperlink_to("CRV Science Network — https://crvscience.com", "https://crvscience.com");
                    ui.hyperlink_to("CRV Science Public Apps — https://crvscience.com/apps", "https://crvscience.com/apps");
                    ui.add_space(8.0);
                    if ui.button("Close").clicked() {
                        self.show_about_modal = false;
                    }
                });
            if !open {
                self.show_about_modal = false;
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

    let (rect, _) = ui.allocate_exact_size(egui::vec2(avail_w, 56.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(10, 16, 28));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(71, 85, 105, 140)),
        egui::StrokeKind::Outside,
    );

    let plot = egui::Rect::from_min_max(
        egui::Pos2::new(rect.min.x + 36.0, rect.min.y + 5.0),
        egui::Pos2::new(rect.max.x - 6.0, rect.max.y - 5.0),
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

    // Optional Pre / Acute lesion reference lines
    if let Some(pre) = pre_lesion {
        let y = y_of(pre);
        painter.line_segment(
            [egui::Pos2::new(plot.min.x, y), egui::Pos2::new(plot.max.x, y)],
            egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(56, 189, 248, 160)),
        );
    }
    if let Some(acute) = acute_lesion {
        let y = y_of(acute);
        painter.line_segment(
            [egui::Pos2::new(plot.min.x, y), egui::Pos2::new(plot.max.x, y)],
            egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(244, 63, 94, 160)),
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

    for (ev_idx, _tag) in lesion_events {
        if *ev_idx < n {
            let ex = x_of(*ev_idx);
            painter.line_segment(
                [egui::Pos2::new(ex, plot.min.y), egui::Pos2::new(ex, plot.max.y)],
                egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(244, 63, 94, 175)),
            );
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
