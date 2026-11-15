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
    pub status_banner: String,
    store: Option<LabStore>,
}

impl NeuroSpaceApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::new_with_store(LabStore::open("neurospace_lab.sqlite").ok())
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
            status_banner: "Ready — Drag 3D Neural Space to orbit, scroll to zoom, F5/F9 to save/load."
                .to_string(),
            store,
        }
    }

    fn quick_save(&mut self) {
        let env = WorksetEnvelope::from_sim(&self.sim, self.render_mode, self.manual_organic_blend);
        if let Some(st) = &self.store {
            if st.save_slot("quicksave_f5", &env).is_ok() {
                self.status_banner = format!(
                    "Saved slot 'quicksave_f5' (SHA-256 {}…)",
                    &env.sha256_digest[..10]
                );
            }
        }
    }

    fn quick_load(&mut self) {
        if let Some(st) = &self.store {
            if let Ok(Some(env)) = st.load_slot("quicksave_f5") {
                if env.verify_digest() {
                    self.sim.dataset_locked = false;
                    self.sim.model_locked = false;
                    self.sim.params = env.params;
                    self.sim.goal = env.goal;
                    self.sim.select_dataset(env.dataset, 2026);
                    self.sim.select_model(env.model);
                    self.sim.dataset_locked = env.dataset_locked;
                    self.sim.model_locked = env.model_locked;
                    self.sim.comparison_ledger = env.comparison_ledger;
                    self.render_mode = env.render_mode;
                    self.manual_organic_blend = env.manual_organic_blend;
                    self.status_banner = "Restored verified 'quicksave_f5' workset.".to_string();
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

        // Top Bar
        egui::TopBottomPanel::top("top_hud_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(
                    RichText::new("NEUROSPACE 3D LAB v1.6.0").color(Color32::from_rgb(56, 189, 248)),
                );
                ui.separator();
                if ui
                    .button(if self.running { "⏸ Pause" } else { "▶ Run" })
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
                    if let (Some(st), Some(last)) = (&self.store, self.sim.comparison_ledger.last()) {
                        let _ = st.append_comparison(last);
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
                ui.checkbox(&mut self.show_group_highlights, "🔲 Layer/Lamina Zones");
                ui.checkbox(&mut self.show_bottom_matrix, "📊 Lock Matrix");
                ui.label(
                    RichText::new(format!(
                        "Step {} | Somata {} | Synapses {} | {}: {:.1}% | {}: {:.3}",
                        self.sim.step_count,
                        self.sim.neurons.len(),
                        self.sim.synapses.len(),
                        self.sim.dataset_id.task_kind().primary_metric_label(),
                        self.sim.primary_metric,
                        self.sim.dataset_id.task_kind().loss_label(),
                        self.sim.current_loss
                    ))
                    .color(Color32::from_rgb(167, 243, 208)),
                );
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
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::ChR2Excite, "🔆 ChR2 (+)");
                            ui.selectable_value(&mut self.probe_tool, ProbeTool::NpHRSilence, "🟣 NpHR (-)");
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
                        }
                    });

                    ui.add_space(6.0);
                    ui.small(&self.status_banner);
                });
            });

        // Bottom Panel
        if self.show_bottom_matrix {
            egui::TopBottomPanel::bottom("bottom_lock_matrix")
                .default_height(185.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.columns(2, |cols| {
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

                        cols[1].strong("Locked Dataset / Locked Model Comparison Ledger");
                        ScrollArea::vertical().max_height(145.0).show(&mut cols[1], |ui| {
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
                            n.role.label(),
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
    }
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
    for _ in 0..35 {
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
