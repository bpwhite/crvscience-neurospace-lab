use crate::datasets::{BuiltDataset, DatasetId, SampleRecord, TaskKind};
use crate::models::{HyperParams, ModelArch, OutputGoal};
use glam::Vec3;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Causal optogenetic clamp state applied to an individual soma in 3D Neural Space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptoClamp {
    None,
    ChR2Excite,
    NpHRSilence,
}

/// Functional & cytoarchitectural identity of a neuron inside 3D Neural Space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronRole {
    SensoryInput,
    ExcitatoryHidden,
    InhibitoryInterneuron,
    ReadoutOutput,
    GeneratorUnit,
    DiscriminatorUnit,
    PredictionErrorUnit,
    CorticalMiniColumn,
    AstrocyteGlia,
    MolecularInterneuronL1,
    PyramidalL23,
    SpinyStellateL4,
    ThickPyramidalL5,
    CorticothalamicL6,
}

impl NeuronRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::SensoryInput => "Sensory / Thalamic Input",
            Self::ExcitatoryHidden => "Excitatory Hidden",
            Self::InhibitoryInterneuron => "GABAergic Basket/Martinotti",
            Self::ReadoutOutput => "Readout / Efferent Tract",
            Self::GeneratorUnit => "Generator Manifold",
            Self::DiscriminatorUnit => "Discriminator Critic",
            Self::PredictionErrorUnit => "Prediction-Error Unit",
            Self::CorticalMiniColumn => "Cortical Mini-Column",
            Self::AstrocyteGlia => "Protoplasmic Astrocyte",
            Self::MolecularInterneuronL1 => "L1 Apical/Horizontal Cell",
            Self::PyramidalL23 => "L2/3 Association Pyramidal",
            Self::SpinyStellateL4 => "L4 Spiny Stellate Granule",
            Self::ThickPyramidalL5 => "L5 Thick-Tufted Pyramidal",
            Self::CorticothalamicL6 => "L6 Corticothalamic Fusiform",
        }
    }

    pub fn is_pyramidal(self) -> bool {
        matches!(self, Self::PyramidalL23 | Self::ThickPyramidalL5)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronNode {
    pub id: usize,
    pub layer: usize,
    pub role: NeuronRole,
    pub pos: Vec3,
    pub rest_pos: Vec3,
    pub vel: Vec3,
    pub activation: f32,
    pub smooth_activation: f32,
    pub prev_activation: f32,
    pub membrane_v: f32,
    pub apical_ca_burst: f32,
    pub spike_trace: f32,
    pub spiked: bool,
    pub ablated: bool,
    pub opto_clamp: OptoClamp,
    pub bias: f32,
    pub tau: f32,
    pub grad: f32,
    pub utility: f32,
    pub soma_scale: f32,
    pub phase_seed: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
    pub velocity: f32,
    pub eligibility: f32,
    pub pulse: f32,
    pub sampled_tracer: bool,
    pub is_skip_or_lateral: bool,
    pub is_feedback: bool,
    pub severed: bool,
    pub smooth_flux: f32,
    pub smooth_grad_flux: f32,
    pub curve_offset: Vec3,
    pub age_ticks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRecord {
    pub timestamp_step: u64,
    pub model: ModelArch,
    pub dataset: DatasetId,
    pub goal: OutputGoal,
    pub lock_context: String,
    pub primary_score: f32,
    pub loss: f32,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub metabolic_cost: f32,
}

#[derive(Debug, Clone)]
pub struct NeuralSpaceSim {
    pub model: ModelArch,
    pub dataset_id: DatasetId,
    pub dataset: BuiltDataset,
    pub goal: OutputGoal,
    pub params: HyperParams,
    pub dataset_locked: bool,
    pub model_locked: bool,

    pub neurons: Vec<NeuronNode>,
    pub synapses: Vec<SynapseEdge>,
    pub input_ids: Vec<usize>,
    pub output_ids: Vec<usize>,

    pub step_count: u64,
    pub epoch_count: u64,
    pub current_loss: f32,
    pub primary_metric: f32,
    pub metabolic_flux: f32,
    pub adversarial_margin: f32,

    pub pruned_total: u64,
    pub sprouted_total: u64,
    pub neurogenesis_total: u64,
    pub apoptosis_total: u64,
    pub opto_clamp_drive: f32,
    pub pre_lesion_metric: Option<f32>,
    pub acute_lesion_metric: Option<f32>,

    pub loss_history: Vec<f32>,
    pub metric_history: Vec<f32>,
    pub synapse_history: Vec<usize>,
    pub plasticity_log: Vec<String>,
    pub comparison_ledger: Vec<ComparisonRecord>,

    pub active_sample: SampleRecord,
    pub last_prediction: Vec<f32>,

    rng: ChaCha8Rng,
    next_neuron_uid: usize,
}

impl NeuralSpaceSim {
    pub fn new(model: ModelArch, dataset_id: DatasetId, seed: u64) -> Self {
        let params = HyperParams::default();
        let dataset = BuiltDataset::generate(dataset_id, seed, params.input_noise);
        let active_sample = dataset.val[0].clone();
        let out_dim = dataset_id.output_dim();

        let mut sim = Self {
            model,
            dataset_id,
            dataset,
            goal: OutputGoal::MaxTaskFidelity,
            params,
            dataset_locked: false,
            model_locked: false,
            neurons: Vec::new(),
            synapses: Vec::new(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            step_count: 0,
            epoch_count: 0,
            current_loss: 0.65,
            primary_metric: 22.0,
            metabolic_flux: 0.35,
            adversarial_margin: 0.50,
            pruned_total: 0,
            sprouted_total: 0,
            neurogenesis_total: 0,
            apoptosis_total: 0,
            opto_clamp_drive: 1.10,
            pre_lesion_metric: None,
            acute_lesion_metric: None,
            loss_history: Vec::with_capacity(240),
            metric_history: Vec::with_capacity(240),
            synapse_history: Vec::with_capacity(240),
            plasticity_log: Vec::new(),
            comparison_ledger: Vec::new(),
            active_sample,
            last_prediction: vec![0.0; out_dim],
            rng: ChaCha8Rng::seed_from_u64(seed),
            next_neuron_uid: 0,
        };
        sim.rebuild_topology(false);
        sim
    }

    pub fn record_snapshot(&mut self, context_tag: &str) {
        if self.step_count == 0
            && self.comparison_ledger.iter().any(|r| {
                r.model == self.model && r.dataset == self.dataset_id && r.lock_context == context_tag
            })
        {
            return;
        }
        self.comparison_ledger.push(ComparisonRecord {
            timestamp_step: self.step_count,
            model: self.model,
            dataset: self.dataset_id,
            goal: self.goal,
            lock_context: context_tag.to_string(),
            primary_score: self.primary_metric,
            loss: self.current_loss,
            neuron_count: self.neurons.len(),
            synapse_count: self.synapses.len(),
            metabolic_cost: self.metabolic_flux,
        });
        if self.comparison_ledger.len() > 48 {
            self.comparison_ledger.remove(0);
        }
    }

    pub fn select_model(&mut self, new_model: ModelArch) {
        if self.model_locked || new_model == self.model {
            return;
        }
        if self.dataset_locked && self.step_count > 0 {
            let tag = format!("DatasetLocked[{}]", self.dataset_id.code());
            self.record_snapshot(&tag);
        }
        self.model = new_model;
        self.rebuild_topology(false);
        self.push_log(format!(
            "Topology morphed -> {} ({})",
            new_model.code(),
            new_model.name()
        ));
    }

    pub fn select_dataset(&mut self, new_dataset: DatasetId, seed: u64) {
        if self.dataset_locked || new_dataset == self.dataset_id {
            return;
        }
        if self.model_locked && self.step_count > 0 {
            let tag = format!("ModelLocked[{}]", self.model.code());
            self.record_snapshot(&tag);
        }
        self.dataset_id = new_dataset;
        self.dataset = BuiltDataset::generate(new_dataset, seed, self.params.input_noise);
        self.active_sample = self.dataset.val[0].clone();
        self.last_prediction = vec![0.0; new_dataset.output_dim()];
        let preserve_core = self.model_locked && self.params.continual_transfer;
        self.rebuild_topology(preserve_core);
        self.push_log(format!(
            "Dataset switched -> {} ({})",
            new_dataset.code(),
            if preserve_core { "Continual Latent Transfer" } else { "Fresh Init" }
        ));
    }

    pub fn run_locked_sweep(&mut self, steps_per_entry: usize) {
        if self.dataset_locked && !self.model_locked {
            let orig_model = self.model;
            let locked_ds = self.dataset_id;
            for arch in ModelArch::ALL {
                self.model = arch;
                self.rebuild_topology(false);
                for _ in 0..steps_per_entry {
                    self.step_simulation();
                }
                let tag = format!("Sweep:Dataset[{}]", locked_ds.code());
                self.record_snapshot(&tag);
            }
            self.model = orig_model;
            self.rebuild_topology(false);
            for _ in 0..(steps_per_entry / 2).max(10) {
                self.step_simulation();
            }
            self.push_log(format!(
                "Completed 13-model sweep on locked {}",
                locked_ds.code()
            ));
        } else {
            let locked_model = self.model;
            let orig_ds = self.dataset_id;
            let prev_ds_lock = self.dataset_locked;
            self.dataset_locked = false;
            for ds in DatasetId::ALL {
                self.dataset_id = ds;
                self.dataset = BuiltDataset::generate(ds, 42, self.params.input_noise);
                self.active_sample = self.dataset.val[0].clone();
                self.rebuild_topology(self.params.continual_transfer);
                for _ in 0..steps_per_entry {
                    self.step_simulation();
                }
                let tag = format!("Sweep:Model[{}]", locked_model.code());
                self.record_snapshot(&tag);
            }
            self.dataset_id = orig_ds;
            self.dataset = BuiltDataset::generate(orig_ds, 42, self.params.input_noise);
            self.active_sample = self.dataset.val[0].clone();
            self.rebuild_topology(false);
            self.dataset_locked = prev_ds_lock;
            self.push_log(format!(
                "Completed 7-dataset sweep on locked {}",
                locked_model.code()
            ));
        }
    }

    fn push_log(&mut self, msg: String) {
        self.plasticity_log.insert(0, format!("[t={}] {}", self.step_count, msg));
        if self.plasticity_log.len() > 28 {
            self.plasticity_log.pop();
        }
    }

    fn alloc_neuron(&mut self, layer: usize, role: NeuronRole, pos: Vec3) -> usize {
        let id = self.next_neuron_uid;
        self.next_neuron_uid += 1;
        let tau = match self.model {
            ModelArch::M8LiquidTimeConstant => 0.25 + self.rng.gen::<f32>() * 1.65,
            ModelArch::M7SpikingLifStdp | ModelArch::M13CerebralCortexGreyMatter => {
                0.45 + self.rng.gen::<f32>() * 0.55
            }
            _ => 1.0,
        };
        let bias = (self.rng.gen::<f32>() - 0.5) * 0.25;
        let phase_seed = self.rng.gen::<f32>() * std::f32::consts::TAU;
        self.neurons.push(NeuronNode {
            id,
            layer,
            role,
            pos,
            rest_pos: pos,
            vel: Vec3::ZERO,
            activation: 0.0,
            smooth_activation: 0.18,
            prev_activation: 0.0,
            membrane_v: 0.0,
            apical_ca_burst: 0.0,
            spike_trace: 0.0,
            spiked: false,
            ablated: false,
            opto_clamp: OptoClamp::None,
            bias,
            tau,
            grad: 0.0,
            utility: 0.5,
            soma_scale: match role {
                NeuronRole::AstrocyteGlia => 1.45,
                NeuronRole::ThickPyramidalL5 => 1.25,
                _ => 1.0,
            },
            phase_seed,
        });
        self.neurons.len() - 1
    }

    /// Re-elects a calm, sparse representative sample of synapses (~10%, capped) so the user
    /// sees smooth macroscopic directional currents rather than every synapse firing at once.
    pub fn refresh_flow_sample(&mut self) {
        let n_syn = self.synapses.len();
        if n_syn == 0 {
            return;
        }
        let rate = self.params.flow_sample_rate.clamp(0.0, 0.45);
        let budget = ((n_syn as f32 * rate).round() as usize).clamp(
            if rate > 0.005 { 3 } else { 0 },
            22,
        );

        for syn in &mut self.synapses {
            syn.sampled_tracer = false;
        }
        if budget == 0 {
            return;
        }

        // Deterministically stride + weight-rank across synapses so all layers get calm representation
        let mut candidates: Vec<(usize, f32)> = self
            .synapses
            .iter()
            .enumerate()
            .filter(|(_, s)| s.weight.abs() > 0.14)
            .map(|(i, s)| (i, s.weight.abs() + s.eligibility * 0.6))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let pool_len = candidates.len().max(1);
        let stride = (pool_len / budget.max(1)).max(1);
        let phase_shift = (self.step_count as usize / 90) % stride.max(1);
        let mut chosen = 0usize;
        let mut idx = phase_shift;
        while chosen < budget && idx < candidates.len() {
            let syn_i = candidates[idx].0;
            self.synapses[syn_i].sampled_tracer = true;
            chosen += 1;
            idx += stride;
        }
    }

    pub fn rebuild_topology(&mut self, preserve_hidden_weights: bool) {
        let old_weights: Vec<f32> = if preserve_hidden_weights {
            self.synapses.iter().map(|s| s.weight).collect()
        } else {
            Vec::new()
        };

        self.neurons.clear();
        self.synapses.clear();
        self.input_ids.clear();
        self.output_ids.clear();
        self.next_neuron_uid = 0;
        self.step_count = 0;
        self.epoch_count = 0;
        self.pruned_total = 0;
        self.sprouted_total = 0;
        self.neurogenesis_total = 0;
        self.apoptosis_total = 0;
        self.pre_lesion_metric = None;
        self.acute_lesion_metric = None;
        self.loss_history.clear();
        self.metric_history.clear();
        self.synapse_history.clear();

        let in_dim = self.dataset_id.input_dim();
        let out_dim = self.dataset_id.output_dim();
        let scale = self.params.hidden_scale.clamp(0.5, 2.0);

        if self.model == ModelArch::M0MinimalPerceptron {
            // Minimal Rosenblatt Perceptron: at most 4 pooled input receptors wired directly to readout!
            let min_in = in_dim.min(4);
            for i in 0..min_in {
                let y = (i as f32 - (min_in as f32 - 1.0) * 0.5) * 2.1;
                let idx = self.alloc_neuron(0, NeuronRole::SensoryInput, Vec3::new(-5.8, y, 0.0));
                self.input_ids.push(idx);
            }
            for o in 0..out_dim {
                let y = if out_dim == 1 {
                    0.0
                } else {
                    (o as f32 / (out_dim - 1) as f32 - 0.5) * 4.6
                };
                let oid = self.alloc_neuron(1, NeuronRole::ReadoutOutput, Vec3::new(5.8, y, 0.0));
                self.output_ids.push(oid);
            }
            let in_c = self.input_ids.clone();
            let out_c = self.output_ids.clone();
            for (ii, &u) in in_c.iter().enumerate() {
                for (oi, &v) in out_c.iter().enumerate() {
                    if out_c.len() <= 3 || (ii + oi) % 2 == 0 || ii == oi % in_c.len() {
                        let w = (self.rng.gen::<f32>() * 2.0 - 1.0) * 0.65;
                        self.add_synapse(u, v, w, false);
                    }
                }
            }
            self.refresh_flow_sample();
            self.evaluate_validation_metrics();
            return;
        }

        if self.model == ModelArch::M13CerebralCortexGreyMatter {
            self.build_cerebral_cortex_grey_matter(in_dim, out_dim, scale, &old_weights);
            self.refresh_flow_sample();
            self.evaluate_validation_metrics();
            return;
        }

        // 1. Sensory Input Manifold at X = -9.5
        for i in 0..in_dim {
            let pos = match self.model {
                ModelArch::M2VolumetricCnn | ModelArch::M1PerceptronMlp => {
                    let cols = (in_dim as f32).sqrt().ceil() as usize;
                    let r = i / cols;
                    let c = i % cols;
                    Vec3::new(
                        -9.5,
                        (r as f32 - (cols as f32 - 1.0) * 0.5) * 1.45,
                        (c as f32 - (cols as f32 - 1.0) * 0.5) * 1.45,
                    )
                }
                _ => {
                    let angle = (i as f32 / in_dim as f32) * std::f32::consts::TAU;
                    let radius = 3.2 + (i % 2) as f32 * 0.65;
                    Vec3::new(-9.5, angle.sin() * radius, angle.cos() * radius)
                }
            };
            let idx = self.alloc_neuron(0, NeuronRole::SensoryInput, pos);
            self.input_ids.push(idx);
        }

        // 2. Hidden / Cortical / Adversarial / Reservoir Interior
        let mut hidden_layers: Vec<Vec<usize>> = Vec::new();
        match self.model {
            ModelArch::M0MinimalPerceptron => {}
            ModelArch::M1PerceptronMlp => {
                let h1 = ((12.0 * scale).round() as usize).max(5);
                let h2 = ((10.0 * scale).round() as usize).max(4);
                for (l_idx, (&count, &x_pos)) in [h1, h2].iter().zip(&[-3.2f32, 3.2f32]).enumerate() {
                    let mut layer_vec = Vec::with_capacity(count);
                    let cols = 4usize;
                    for k in 0..count {
                        let r = k / cols;
                        let c = k % cols;
                        let pos = Vec3::new(
                            x_pos,
                            (r as f32 - 1.2) * 1.65,
                            (c as f32 - 1.5) * 1.65,
                        );
                        layer_vec.push(self.alloc_neuron(l_idx + 1, NeuronRole::ExcitatoryHidden, pos));
                    }
                    hidden_layers.push(layer_vec);
                }
            }
            ModelArch::M2VolumetricCnn => {
                let counts = [
                    ((16.0 * scale).round() as usize).max(8),
                    ((12.0 * scale).round() as usize).max(6),
                    ((8.0 * scale).round() as usize).max(4),
                ];
                let xs = [-4.6f32, 0.0f32, 4.4f32];
                for (l_idx, (&cnt, &x_pos)) in counts.iter().zip(&xs).enumerate() {
                    let mut lv = Vec::with_capacity(cnt);
                    for k in 0..cnt {
                        let plane = (k % 2) as f32 - 0.5;
                        let row = ((k / 2) % 3) as f32 - 1.0;
                        let col = (k / 6) as f32 - 0.5;
                        let spread = 1.6 - l_idx as f32 * 0.25;
                        let pos = Vec3::new(
                            x_pos + plane * 0.75,
                            row * spread * 1.35,
                            col * spread * 1.65 + plane * 1.2,
                        );
                        lv.push(self.alloc_neuron(l_idx + 1, NeuronRole::ExcitatoryHidden, pos));
                    }
                    hidden_layers.push(lv);
                }
            }
            ModelArch::M3SparseTransformer => {
                let heads = ((14.0 * scale).round() as usize).max(6);
                let ffn = ((10.0 * scale).round() as usize).max(5);
                let mut ring1 = Vec::new();
                for k in 0..heads {
                    let theta = (k as f32 / heads as f32) * std::f32::consts::TAU;
                    let head_group = (k % 3) as f32 - 1.0;
                    let pos = Vec3::new(-2.8 + head_group * 0.65, theta.sin() * 4.0, theta.cos() * 4.0);
                    ring1.push(self.alloc_neuron(1, NeuronRole::ExcitatoryHidden, pos));
                }
                let mut ring2 = Vec::new();
                for k in 0..ffn {
                    let theta = (k as f32 / ffn as f32) * std::f32::consts::TAU + 0.3;
                    let pos = Vec3::new(3.4, theta.sin() * 2.9, theta.cos() * 2.9);
                    ring2.push(self.alloc_neuron(2, NeuronRole::ExcitatoryHidden, pos));
                }
                hidden_layers.push(ring1);
                hidden_layers.push(ring2);
            }
            ModelArch::M14FrontierLlmMoe => {
                // Stage 1: Causal GQA Attention Heads (Y > 0) + Persistent KV-Cache Ring (Y < 0)
                let n_gqa = ((8.0 * scale).round() as usize).max(4);
                let n_kv = ((6.0 * scale).round() as usize).max(4);
                let mut gqa_kv = Vec::new();
                for k in 0..n_gqa {
                    let a = (k as f32 / n_gqa as f32) * std::f32::consts::PI;
                    gqa_kv.push(self.alloc_neuron(
                        1,
                        NeuronRole::ExcitatoryHidden,
                        Vec3::new(-4.8, 1.5 + a.sin() * 2.4, (k as f32 - n_gqa as f32 * 0.5) * 1.15),
                    ));
                }
                for k in 0..n_kv {
                    let a = (k as f32 / n_kv as f32) * std::f32::consts::TAU;
                    gqa_kv.push(self.alloc_neuron(
                        1,
                        NeuronRole::CorticalMiniColumn,
                        Vec3::new(-4.8, -2.6 + a.sin() * 0.8, a.cos() * 2.6),
                    ));
                }
                // Stage 2: Top-2 MoE Gating Router + Residual Stream Spine
                let n_router = ((5.0 * scale).round() as usize).max(4);
                let mut router = Vec::new();
                for k in 0..n_router {
                    let a = (k as f32 / n_router as f32) * std::f32::consts::TAU;
                    router.push(self.alloc_neuron(
                        2,
                        NeuronRole::GeneratorUnit,
                        Vec3::new(-0.4, a.sin() * 1.5, a.cos() * 1.5),
                    ));
                }
                // Stage 3: 4 Distinct SwiGLU Mixture-of-Experts Sub-Blocks in YZ Quadrants
                let per_exp = ((4.0 * scale).round() as usize).max(3);
                let mut experts = Vec::new();
                let quads = [(2.6f32, 2.6f32), (2.6, -2.6), (-2.6, 2.6), (-2.6, -2.6)];
                for &(qy, qz) in &quads {
                    for e in 0..per_exp {
                        let a = (e as f32 / per_exp as f32) * std::f32::consts::TAU;
                        experts.push(self.alloc_neuron(
                            3,
                            NeuronRole::ExcitatoryHidden,
                            Vec3::new(4.4, qy + a.sin() * 0.85, qz + a.cos() * 0.85),
                        ));
                    }
                }
                hidden_layers.push(gqa_kv);
                hidden_layers.push(router);
                hidden_layers.push(experts);
            }
            ModelArch::M4MinimaxGan => {
                let n_gen = ((11.0 * scale).round() as usize).max(5);
                let n_disc = ((11.0 * scale).round() as usize).max(5);
                let mut gen_layer = Vec::new();
                for k in 0..n_gen {
                    let t = k as f32 / n_gen.max(1) as f32;
                    let angle = t * std::f32::consts::TAU * 1.5;
                    let pos = Vec3::new(
                        -4.0 + t * 6.0,
                        angle.sin() * (1.5 + t * 1.8),
                        -3.2 + angle.cos() * 1.2,
                    );
                    gen_layer.push(self.alloc_neuron(1, NeuronRole::GeneratorUnit, pos));
                }
                let mut disc_layer = Vec::new();
                for k in 0..n_disc {
                    let t = k as f32 / n_disc.max(1) as f32;
                    let angle = t * std::f32::consts::TAU * 1.5;
                    let pos = Vec3::new(
                        -3.0 + t * 6.5,
                        angle.cos() * (3.0 - t * 1.4),
                        3.2 + angle.sin() * 1.2,
                    );
                    disc_layer.push(self.alloc_neuron(2, NeuronRole::DiscriminatorUnit, pos));
                }
                hidden_layers.push(gen_layer);
                hidden_layers.push(disc_layer);
            }
            ModelArch::M5AdversarialDefensePgd => {
                let shell = ((14.0 * scale).round() as usize).max(6);
                let bottle = ((6.0 * scale).round() as usize).max(3);
                let verify = ((10.0 * scale).round() as usize).max(5);
                let mut l1 = Vec::new();
                for k in 0..shell {
                    let a = (k as f32 / shell as f32) * std::f32::consts::TAU;
                    l1.push(self.alloc_neuron(
                        1,
                        NeuronRole::InhibitoryInterneuron,
                        Vec3::new(-4.8, a.sin() * 4.2, a.cos() * 4.2),
                    ));
                }
                let mut l2 = Vec::new();
                for k in 0..bottle {
                    let a = (k as f32 / bottle as f32) * std::f32::consts::TAU;
                    l2.push(self.alloc_neuron(
                        2,
                        NeuronRole::ExcitatoryHidden,
                        Vec3::new(0.0, a.sin() * 1.35, a.cos() * 1.35),
                    ));
                }
                let mut l3 = Vec::new();
                for k in 0..verify {
                    let a = (k as f32 / verify as f32) * std::f32::consts::TAU;
                    l3.push(self.alloc_neuron(
                        3,
                        NeuronRole::ExcitatoryHidden,
                        Vec3::new(4.6, a.sin() * 2.8, a.cos() * 2.8),
                    ));
                }
                hidden_layers.push(l1);
                hidden_layers.push(l2);
                hidden_layers.push(l3);
            }
            ModelArch::M6GeneticGannNeat => {
                let h1 = ((7.0 * scale).round() as usize).max(4);
                let h2 = ((7.0 * scale).round() as usize).max(4);
                let mut l1 = Vec::new();
                let mut l2 = Vec::new();
                for k in 0..h1 {
                    let y = (k as f32 - h1 as f32 * 0.5) * 1.35;
                    let z = ((k * 3) as f32).sin() * 2.4;
                    l1.push(self.alloc_neuron(1, NeuronRole::ExcitatoryHidden, Vec3::new(-3.0, y, z)));
                }
                for k in 0..h2 {
                    let y = (k as f32 - h2 as f32 * 0.5) * 1.35;
                    let z = ((k * 5 + 1) as f32).cos() * 2.4;
                    l2.push(self.alloc_neuron(2, NeuronRole::ExcitatoryHidden, Vec3::new(2.8, y, z)));
                }
                hidden_layers.push(l1);
                hidden_layers.push(l2);
            }
            ModelArch::M7SpikingLifStdp => {
                let n_exc = ((16.0 * scale).round() as usize).max(8);
                let n_inh = ((5.0 * scale).round() as usize).max(2);
                let mut l_exc = Vec::new();
                for k in 0..n_exc {
                    let band = if k < n_exc / 2 { -2.4 } else { 2.4 };
                    let a = (k as f32 / (n_exc / 2).max(1) as f32) * std::f32::consts::TAU;
                    let pos = Vec3::new(band, a.sin() * 3.1, a.cos() * 3.1);
                    l_exc.push(self.alloc_neuron(1, NeuronRole::ExcitatoryHidden, pos));
                }
                let mut l_inh = Vec::new();
                for k in 0..n_inh {
                    let a = (k as f32 / n_inh as f32) * std::f32::consts::TAU;
                    let pos = Vec3::new(0.0, a.sin() * 4.6, a.cos() * 4.6);
                    l_inh.push(self.alloc_neuron(2, NeuronRole::InhibitoryInterneuron, pos));
                }
                hidden_layers.push(l_exc);
                hidden_layers.push(l_inh);
            }
            ModelArch::M8LiquidTimeConstant => {
                let n_liq = ((18.0 * scale).round() as usize).max(8);
                let mut gang1 = Vec::new();
                let mut gang2 = Vec::new();
                for k in 0..n_liq {
                    let phi = std::f32::consts::PI * (3.0 - 5.0f32.sqrt()) * k as f32;
                    let y = 1.0 - (k as f32 / (n_liq - 1).max(1) as f32) * 2.0;
                    let rad = (1.0 - y * y).sqrt() * 3.5;
                    let pos = Vec3::new(phi.cos() * rad * 1.2, y * 3.4, phi.sin() * rad);
                    if k < n_liq / 2 {
                        gang1.push(self.alloc_neuron(1, NeuronRole::ExcitatoryHidden, pos));
                    } else {
                        let role = if k % 4 == 0 {
                            NeuronRole::InhibitoryInterneuron
                        } else {
                            NeuronRole::ExcitatoryHidden
                        };
                        gang2.push(self.alloc_neuron(2, role, pos));
                    }
                }
                hidden_layers.push(gang1);
                hidden_layers.push(gang2);
            }
            ModelArch::M9EchoStateReservoir => {
                let n_res = ((26.0 * scale).round() as usize).max(12);
                let mut res_a = Vec::new();
                let mut res_b = Vec::new();
                for k in 0..n_res {
                    let phi = std::f32::consts::PI * (3.0 - 5.0f32.sqrt()) * k as f32;
                    let y = 1.0 - (k as f32 / (n_res - 1).max(1) as f32) * 2.0;
                    let r_shell = 4.2 * (0.72 + 0.28 * ((k % 3) as f32 / 2.0));
                    let rad = (1.0 - y * y).sqrt() * r_shell;
                    let pos = Vec3::new(phi.cos() * rad, y * r_shell, phi.sin() * rad);
                    let role = if k % 5 == 0 {
                        NeuronRole::InhibitoryInterneuron
                    } else {
                        NeuronRole::ExcitatoryHidden
                    };
                    if pos.x < 0.0 {
                        res_a.push(self.alloc_neuron(1, role, pos));
                    } else {
                        res_b.push(self.alloc_neuron(2, role, pos));
                    }
                }
                if res_a.is_empty() {
                    if let Some(moved) = res_b.pop() {
                        res_a.push(moved);
                    }
                }
                if res_b.is_empty() {
                    if let Some(moved) = res_a.pop() {
                        res_b.push(moved);
                    }
                }
                hidden_layers.push(res_a);
                hidden_layers.push(res_b);
            }
            ModelArch::M10PredictiveCoding => {
                let pairs = ((9.0 * scale).round() as usize).max(4);
                let mut l_val = Vec::new();
                let mut l_err = Vec::new();
                for k in 0..pairs {
                    let x = -3.6 + (k % 3) as f32 * 3.6;
                    let z = ((k / 3) as f32 - 1.0) * 2.4;
                    l_val.push(self.alloc_neuron(
                        1,
                        NeuronRole::ExcitatoryHidden,
                        Vec3::new(x, 2.2 + (k % 2) as f32 * 0.7, z),
                    ));
                    l_err.push(self.alloc_neuron(
                        2,
                        NeuronRole::PredictionErrorUnit,
                        Vec3::new(x, -2.2 - (k % 2) as f32 * 0.7, z),
                    ));
                }
                hidden_layers.push(l_val);
                hidden_layers.push(l_err);
            }
            ModelArch::M11CorticalColumnsHtm => {
                let cols = ((16.0 * scale).round() as usize).max(8);
                let mut c1 = Vec::new();
                let mut c2 = Vec::new();
                for k in 0..cols {
                    let ring = if k < cols / 2 { 0 } else { 1 };
                    let idx_in_ring = if ring == 0 { k } else { k - cols / 2 };
                    let count_ring = if ring == 0 { cols / 2 } else { cols - cols / 2 };
                    let a = (idx_in_ring as f32 / count_ring.max(1) as f32) * std::f32::consts::TAU;
                    let x = if ring == 0 { -2.6 } else { 2.6 };
                    let y = a.sin() * 3.4;
                    let z = a.cos() * 3.4;
                    let n_id = self.alloc_neuron(ring + 1, NeuronRole::CorticalMiniColumn, Vec3::new(x, y, z));
                    if ring == 0 {
                        c1.push(n_id);
                    } else {
                        c2.push(n_id);
                    }
                }
                hidden_layers.push(c1);
                hidden_layers.push(c2);
            }
            ModelArch::M12ConnectomicOrganoid | ModelArch::M13CerebralCortexGreyMatter => {
                let n_soma = ((20.0 * scale).round() as usize).max(10);
                let n_glia = ((4.0 * scale).round() as usize).max(2);
                let mut organoid_core = Vec::new();
                let mut organoid_shell = Vec::new();
                for k in 0..n_soma {
                    let u = self.rng.gen::<f32>() * std::f32::consts::TAU;
                    let v = (self.rng.gen::<f32>() * 2.0 - 1.0).acos();
                    let r = 1.6 + self.rng.gen::<f32>() * 2.8;
                    let pos = Vec3::new(r * v.sin() * u.cos(), r * v.cos(), r * v.sin() * u.sin());
                    let role = if k % 5 == 0 {
                        NeuronRole::InhibitoryInterneuron
                    } else {
                        NeuronRole::ExcitatoryHidden
                    };
                    if pos.x <= 0.0 {
                        organoid_core.push(self.alloc_neuron(1, role, pos));
                    } else {
                        organoid_shell.push(self.alloc_neuron(2, role, pos));
                    }
                }
                for g in 0..n_glia {
                    let a = (g as f32 / n_glia as f32) * std::f32::consts::TAU;
                    let pos = Vec3::new(0.0, a.sin() * 1.9, a.cos() * 1.9);
                    organoid_core.push(self.alloc_neuron(1, NeuronRole::AstrocyteGlia, pos));
                }
                if organoid_shell.is_empty() {
                    if let Some(n) = organoid_core.pop() {
                        organoid_shell.push(n);
                    }
                }
                hidden_layers.push(organoid_core);
                hidden_layers.push(organoid_shell);
            }
        }

        // 3. Task Readout Manifold at X = +9.5
        for o in 0..out_dim {
            let y = if out_dim == 1 {
                0.0
            } else {
                (o as f32 / (out_dim - 1) as f32 - 0.5) * 5.2
            };
            let z = ((o as f32) * 1.2).sin() * 0.9;
            let idx = self.alloc_neuron(
                hidden_layers.len() + 1,
                NeuronRole::ReadoutOutput,
                Vec3::new(9.5, y, z),
            );
            self.output_ids.push(idx);
        }

        // 4. Wire Synapses
        let mut all_chains = Vec::with_capacity(hidden_layers.len() + 2);
        all_chains.push(self.input_ids.clone());
        for hl in &hidden_layers {
            all_chains.push(hl.clone());
        }
        all_chains.push(self.output_ids.clone());

        let connect_prob = match self.model {
            ModelArch::M1PerceptronMlp => 0.92,
            ModelArch::M6GeneticGannNeat => 0.45,
            ModelArch::M9EchoStateReservoir => 0.38,
            ModelArch::M11CorticalColumnsHtm => 0.55,
            _ => 0.72,
        };

        let mut weight_cursor = 0usize;
        for w_idx in 0..(all_chains.len() - 1) {
            let src_group = &all_chains[w_idx];
            let dst_group = &all_chains[w_idx + 1];
            let fan_in = src_group.len().max(1) as f32;
            let std_init = (1.6 / fan_in).sqrt();

            for &u in src_group {
                for &v in dst_group {
                    if self.rng.gen::<f32>() <= connect_prob || dst_group.len() <= 2 {
                        let w = if !old_weights.is_empty() && w_idx > 0 && w_idx + 2 < all_chains.len() {
                            let reused = old_weights[weight_cursor % old_weights.len()];
                            weight_cursor += 1;
                            reused
                        } else {
                            let raw = (self.rng.gen::<f32>() * 2.0 - 1.0) * std_init;
                            if self.neurons[u].role == NeuronRole::InhibitoryInterneuron {
                                -raw.abs()
                            } else {
                                raw
                            }
                        };
                        self.add_synapse(u, v, w, false);
                    }
                }
            }
        }

        if self.model == ModelArch::M14FrontierLlmMoe && !hidden_layers.is_empty() {
            let out_c = self.output_ids.clone();
            for &uo in &out_c {
                for &v_kv in hidden_layers[0].iter().skip(hidden_layers[0].len() / 2) {
                    let w = (self.rng.gen::<f32>() - 0.35) * 0.34;
                    self.add_synapse(uo, v_kv, w, true);
                    if let Some(s) = self.synapses.last_mut() { s.is_feedback = true; }
                }
            }
        }

        if !matches!(self.model, ModelArch::M0MinimalPerceptron | ModelArch::M1PerceptronMlp) {
            for hl in &hidden_layers {
                for i in 0..hl.len() {
                    let u = hl[i];
                    let v = hl[(i + 1) % hl.len()];
                    if u != v && self.rng.gen::<f32>() < 0.55 {
                        let w = (self.rng.gen::<f32>() - 0.5) * 0.35;
                        self.add_synapse(u, v, w, true);
                    }
                }
            }
            if hidden_layers.len() >= 2 {
                let out_targets: Vec<usize> = self.output_ids.iter().take(3).copied().collect();
                for &u in hidden_layers[0].iter().take(4) {
                    for &v in &out_targets {
                        let w = (self.rng.gen::<f32>() - 0.5) * 0.28;
                        self.add_synapse(u, v, w, true);
                    }
                }
                // Explicit Reverse Inter-Layer Feedback Trunks (L_{k+1} -> L_k) for bidirectional models
                let rev_prob = match self.model {
                    ModelArch::M4MinimaxGan => 0.48,         // Discriminator Critic -> Generator feedback
                    ModelArch::M7SpikingLifStdp => 0.55,     // Inhibitory Surround -> Excitatory feedback
                    ModelArch::M8LiquidTimeConstant => 0.45, // Recurrent Liquid ODE feedback
                    ModelArch::M9EchoStateReservoir => 0.42, // Chaotic Reservoir return loops
                    ModelArch::M10PredictiveCoding => 0.65,  // Top-down Generative Prediction -> Error/Lower
                    ModelArch::M11CorticalColumnsHtm => 0.45,// Distal context feedback between column rings
                    ModelArch::M12ConnectomicOrganoid => 0.50,// Rich-club organoid shell -> core feedback
                    _ => 0.0,
                };
                if rev_prob > 0.0 {
                    for l_idx in (1..hidden_layers.len()).rev() {
                        let upper = hidden_layers[l_idx].clone();
                        let lower = hidden_layers[l_idx - 1].clone();
                        for &u_hi in &upper {
                            for &v_lo in &lower {
                                if self.rng.gen::<f32>() < rev_prob {
                                    let sign = if self.neurons[u_hi].role == NeuronRole::InhibitoryInterneuron {
                                        -1.0
                                    } else {
                                        1.0
                                    };
                                    let w = sign * (0.14 + self.rng.gen::<f32>() * 0.34);
                                    self.add_synapse(u_hi, v_lo, w, true);
                                    if let Some(last_s) = self.synapses.last_mut() {
                                        last_s.is_feedback = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.refresh_flow_sample();
        self.evaluate_validation_metrics();
    }

    /// Constructs the 6-layered 3D Cerebral Cortex Grey Matter slab (`L1` Pia Mater top -> `L6` &
    /// Subcortical White Matter bottom) with canonical Douglas-Martin neocortical microcircuitry:
    /// Thalamic Input -> L4 Spiny Stellate -> L2/3 Pyramidal -> L5 Thick-Tufted Pyramidal -> L6 & Output,
    /// plus L1 apical tuft feedback & Basket/Martinotti GABAergic interneurons.
    fn build_cerebral_cortex_grey_matter(
        &mut self,
        in_dim: usize,
        out_dim: usize,
        scale: f32,
        old_weights: &[f32],
    ) {
        // Subcortical Thalamic Relay Input (bottom-left below L6, Y = -5.4, X in [-8.2 .. -3.2])
        for i in 0..in_dim {
            let t = i as f32 / (in_dim.max(2) - 1) as f32;
            let x = -8.2 + t * 5.0;
            let z = ((i % 3) as f32 - 1.0) * 1.9;
            let nid = self.alloc_neuron(0, NeuronRole::SensoryInput, Vec3::new(x, -5.4, z));
            self.input_ids.push(nid);
        }

        // L4 Internal Granular Layer (Spiny Stellate thalamocortical recipients, Y = +0.2)
        let n_l4 = ((10.0 * scale).round() as usize).max(6);
        let mut l4_ids = Vec::with_capacity(n_l4);
        for k in 0..n_l4 {
            let a = (k as f32 / n_l4 as f32) * std::f32::consts::TAU;
            let r = 2.4 + (k % 2) as f32 * 1.6;
            let pos = Vec3::new(
                a.cos() * r * 1.25,
                0.2 + (self.rng.gen::<f32>() - 0.5) * 0.45,
                a.sin() * r,
            );
            l4_ids.push(self.alloc_neuron(1, NeuronRole::SpinyStellateL4, pos));
        }

        // L2/3 Supragranular Association Pyramidal Layer (Y = +2.4, apical tufts reaching L1 Y = +4.7)
        let n_l23 = ((10.0 * scale).round() as usize).max(6);
        let mut l23_ids = Vec::with_capacity(n_l23);
        for k in 0..n_l23 {
            let a = (k as f32 / n_l23 as f32) * std::f32::consts::TAU + 0.3;
            let r = 2.6 + (k % 2) as f32 * 1.7;
            let pos = Vec3::new(
                a.cos() * r * 1.3,
                2.4 + (self.rng.gen::<f32>() - 0.5) * 0.5,
                a.sin() * r,
            );
            l23_ids.push(self.alloc_neuron(2, NeuronRole::PyramidalL23, pos));
        }

        // L1 Molecular Layer Horizontal / Cajal-Retzius Interneurons (Y = +4.6, near Pia Mater)
        let n_l1 = ((4.0 * scale).round() as usize).max(2);
        let mut l1_ids = Vec::with_capacity(n_l1);
        for k in 0..n_l1 {
            let a = (k as f32 / n_l1 as f32) * std::f32::consts::TAU;
            let pos = Vec3::new(a.cos() * 3.4, 4.6, a.sin() * 2.8);
            l1_ids.push(self.alloc_neuron(2, NeuronRole::MolecularInterneuronL1, pos));
        }

        // L5 Infragranular Thick-Tufted Pyramidal Output Neurons (Y = -1.9, long apical trunks to L1!)
        let n_l5 = ((9.0 * scale).round() as usize).max(5);
        let mut l5_ids = Vec::with_capacity(n_l5);
        for k in 0..n_l5 {
            let a = (k as f32 / n_l5 as f32) * std::f32::consts::TAU + 0.6;
            let r = 2.2 + (k % 2) as f32 * 1.8;
            let pos = Vec3::new(
                a.cos() * r * 1.25,
                -1.9 + (self.rng.gen::<f32>() - 0.5) * 0.4,
                a.sin() * r,
            );
            l5_ids.push(self.alloc_neuron(3, NeuronRole::ThickPyramidalL5, pos));
        }

        // GABAergic Basket & Martinotti Interneurons + Protoplasmic Grey-Matter Astrocytes
        let n_inh = ((4.0 * scale).round() as usize).max(3);
        let mut inh_ids = Vec::with_capacity(n_inh);
        for k in 0..n_inh {
            let y = -1.2 + (k as f32 / n_inh.max(1) as f32) * 3.6;
            let a = k as f32 * 2.1;
            inh_ids.push(self.alloc_neuron(
                2,
                NeuronRole::InhibitoryInterneuron,
                Vec3::new(a.cos() * 4.4, y, a.sin() * 3.8),
            ));
        }
        for g in 0..2 {
            let z = if g == 0 { -1.6 } else { 1.6 };
            let _ = self.alloc_neuron(2, NeuronRole::AstrocyteGlia, Vec3::new(0.0, 0.6, z));
        }

        // L6 Corticothalamic Multiform Layer (Y = -3.7)
        let n_l6 = ((6.0 * scale).round() as usize).max(4);
        let mut l6_ids = Vec::with_capacity(n_l6);
        for k in 0..n_l6 {
            let a = (k as f32 / n_l6 as f32) * std::f32::consts::TAU;
            let pos = Vec3::new(a.cos() * 3.4, -3.7, a.sin() * 2.9);
            l6_ids.push(self.alloc_neuron(4, NeuronRole::CorticothalamicL6, pos));
        }

        // Subcortical White-Matter Efferent Readout Tract (Y = -5.4, X in [+3.2 .. +8.2])
        for o in 0..out_dim {
            let t = if out_dim == 1 {
                0.5
            } else {
                o as f32 / (out_dim - 1) as f32
            };
            let x = 3.2 + t * 5.0;
            let z = ((o % 3) as f32 - 1.0) * 1.8;
            let oid = self.alloc_neuron(5, NeuronRole::ReadoutOutput, Vec3::new(x, -5.4, z));
            self.output_ids.push(oid);
        }

        // Canonical Cortical Microcircuit Synapses
        let mut wc = 0usize;
        let mut next_w = |rng: &mut ChaCha8Rng, scale_std: f32, neg: bool| -> f32 {
            if !old_weights.is_empty() {
                let v = old_weights[wc % old_weights.len()];
                wc += 1;
                v
            } else {
                let r = (rng.gen::<f32>() * 2.0 - 1.0) * scale_std;
                if neg { -r.abs().max(0.08) } else { r }
            }
        };

        // (a) Thalamocortical Afferents: Input -> L4 (primary) & L6 (collateral)
        let in_copy = self.input_ids.clone();
        for &u in &in_copy {
            for &v in &l4_ids {
                if self.rng.gen::<f32>() < 0.72 {
                    let w = next_w(&mut self.rng, 0.48, false);
                    self.add_synapse(u, v, w, false);
                }
            }
        }
        // (b) Ascending Granular-to-Supragranular: L4 -> L2/3
        for &u in &l4_ids {
            for &v in &l23_ids {
                if self.rng.gen::<f32>() < 0.75 {
                    let w = next_w(&mut self.rng, 0.45, false);
                    self.add_synapse(u, v, w, false);
                }
            }
        }
        // (c) Supragranular-to-Infragranular: L2/3 -> L5 Thick-Tufted Pyramidals & L1 Plexus
        for &u in &l23_ids {
            for &v in &l5_ids {
                if self.rng.gen::<f32>() < 0.78 {
                    let w = next_w(&mut self.rng, 0.46, false);
                    self.add_synapse(u, v, w, false);
                }
            }
            for &v1 in &l1_ids {
                if self.rng.gen::<f32>() < 0.45 {
                    let w = next_w(&mut self.rng, 0.30, false);
                    self.add_synapse(u, v1, w, true);
                }
            }
        }
        // (d) Top-Down & Deep-to-Superficial Cortical Reverse Feedback Loops:
        //     L1 -> L5 & L2/3 (apical tuft modulation), L5 -> L2/3 & L1, and L6 -> L4 (corticothalamic gate)
        for &u1 in &l1_ids {
            for &v_pyr in l5_ids.iter().chain(l23_ids.iter().take(4)) {
                if self.rng.gen::<f32>() < 0.55 {
                    let w = next_w(&mut self.rng, 0.28, false);
                    self.add_synapse(u1, v_pyr, w, true);
                    if let Some(s) = self.synapses.last_mut() { s.is_feedback = true; }
                }
            }
        }
        for &u5 in l5_ids.iter().take(5) {
            for &v23 in l23_ids.iter().take(5) {
                if self.rng.gen::<f32>() < 0.42 {
                    let w = next_w(&mut self.rng, 0.26, false);
                    self.add_synapse(u5, v23, w, true);
                    if let Some(s) = self.synapses.last_mut() { s.is_feedback = true; }
                }
            }
        }
        for &u6 in &l6_ids {
            for &v4 in l4_ids.iter().take(6) {
                if self.rng.gen::<f32>() < 0.50 {
                    let w = next_w(&mut self.rng, 0.30, false);
                    self.add_synapse(u6, v4, w, true);
                    if let Some(s) = self.synapses.last_mut() { s.is_feedback = true; }
                }
            }
        }
        // (e) Infragranular Output & Corticothalamic Loop: L5 -> Output + L6; L6 -> Output
        let out_copy = self.output_ids.clone();
        for &u5 in &l5_ids {
            for &vo in &out_copy {
                let w = next_w(&mut self.rng, 0.52, false);
                self.add_synapse(u5, vo, w, false);
            }
            for &v6 in l6_ids.iter().take(3) {
                if self.rng.gen::<f32>() < 0.55 {
                    let w = next_w(&mut self.rng, 0.34, false);
                    self.add_synapse(u5, v6, w, false);
                }
            }
        }
        for &u6 in &l6_ids {
            for &vo in &out_copy {
                if self.rng.gen::<f32>() < 0.75 {
                    let w = next_w(&mut self.rng, 0.38, false);
                    self.add_synapse(u6, vo, w, false);
                }
            }
        }
        // (f) Basket / Martinotti Inhibitory Stabilization
        for &ui in &inh_ids {
            for &vp in l23_ids.iter().chain(l5_ids.iter()) {
                if self.rng.gen::<f32>() < 0.42 {
                    let w = next_w(&mut self.rng, 0.36, true);
                    self.add_synapse(ui, vp, w, true);
                }
            }
        }
    }

    fn add_synapse(&mut self, from: usize, to: usize, weight: f32, is_skip_or_lateral: bool) {
        if from == to || self.synapses.iter().any(|s| s.from == from && s.to == to) {
            return;
        }
        let curve_offset = Vec3::new(
            (self.rng.gen::<f32>() - 0.5) * 0.8,
            (self.rng.gen::<f32>() - 0.5) * 1.4,
            (self.rng.gen::<f32>() - 0.5) * 1.4,
        );
        self.synapses.push(SynapseEdge {
            from,
            to,
            weight: weight.clamp(-2.5, 2.5),
            velocity: 0.0,
            eligibility: 0.0,
            // Spread initial pulse phases across [0 .. 2.4] so sampled tracers are calmly staggered
            pulse: self.rng.gen::<f32>() * 2.4,
            sampled_tracer: false,
            is_skip_or_lateral,
            is_feedback: self.neurons[from].layer > self.neurons[to].layer
                || (self.model == ModelArch::M13CerebralCortexGreyMatter && self.neurons[from].pos.y < self.neurons[to].pos.y && self.neurons[from].role != NeuronRole::SensoryInput && self.neurons[from].role != NeuronRole::SpinyStellateL4),
            curve_offset,
            severed: false,
            smooth_flux: 0.25,
            smooth_grad_flux: 0.15,
            age_ticks: 0,
        });
    }

    pub fn forward_sample(&mut self, features: &[f32], apply_adv_noise: bool) -> Vec<f32> {
        for n in &mut self.neurons {
            n.prev_activation = n.activation;
            n.spiked = false;
        }

        let eps = if apply_adv_noise
            || matches!(
                self.model,
                ModelArch::M4MinimaxGan | ModelArch::M5AdversarialDefensePgd
            )
            || self.goal == OutputGoal::AdversarialImmunity
        {
            self.params.adversarial_eps
        } else {
            0.0
        };

        let n_in = self.input_ids.len().max(1);
        for (k, &nid) in self.input_ids.iter().enumerate() {
            let base = if self.model == ModelArch::M0MinimalPerceptron && features.len() > n_in {
                let chunk = features.len().div_ceil(n_in);
                let slice = &features[(k * chunk).min(features.len())..((k + 1) * chunk).min(features.len())];
                if slice.is_empty() { 0.0 } else { slice.iter().sum::<f32>() / slice.len() as f32 }
            } else {
                features.get(k).copied().unwrap_or(0.0)
            };
            let perturb = if eps > 0.0 {
                ((self.step_count as f32 * 0.37 + k as f32).sin()) * eps
            } else {
                0.0
            };
            self.neurons[nid].activation = (base + perturb).clamp(-2.2, 2.2);
        }

        let passes = match self.model {
            ModelArch::M13CerebralCortexGreyMatter => 4,
            ModelArch::M8LiquidTimeConstant
            | ModelArch::M9EchoStateReservoir
            | ModelArch::M10PredictiveCoding
            | ModelArch::M12ConnectomicOrganoid => 3,
            _ => 2,
        };

        let clamp_drive = self.opto_clamp_drive;
        for n in &mut self.neurons {
            if n.ablated {
                n.activation = 0.0;
                n.membrane_v = 0.0;
            } else if n.opto_clamp == OptoClamp::ChR2Excite {
                n.activation = clamp_drive;
            } else if n.opto_clamp == OptoClamp::NpHRSilence {
                n.activation = 0.0;
            }
        }

        for _ in 0..passes {
            let mut currents = vec![0.0f32; self.neurons.len()];
            for syn in &self.synapses {
                if syn.severed || self.neurons[syn.from].ablated || self.neurons[syn.to].ablated {
                    continue;
                }
                currents[syn.to] += self.neurons[syn.from].activation * syn.weight;
            }

            for (idx, n) in self.neurons.iter_mut().enumerate() {
                if n.ablated {
                    n.activation = 0.0;
                    continue;
                }
                match n.opto_clamp {
                    OptoClamp::ChR2Excite => {
                        n.activation = clamp_drive;
                        continue;
                    }
                    OptoClamp::NpHRSilence => {
                        n.activation = 0.0;
                        continue;
                    }
                    OptoClamp::None => {}
                }
                if n.role == NeuronRole::SensoryInput {
                    continue;
                }
                let drive = currents[idx] + n.bias;
                n.activation = match self.model {
                    ModelArch::M13CerebralCortexGreyMatter if n.role != NeuronRole::ReadoutOutput => {
                        // Two-compartment Larkum pyramidal apical-basal coincidence integration
                        n.membrane_v = n.membrane_v * 0.72 + drive * 0.52;
                        if n.role.is_pyramidal() {
                            let ca_gain = 1.0 + 0.25 * n.apical_ca_burst;
                            let act = (n.membrane_v * ca_gain).tanh();
                            n.apical_ca_burst = (n.apical_ca_burst * 0.85 + act.abs() * 0.18).clamp(0.0, 1.0);
                            if act > 0.68 {
                                n.spiked = true;
                            }
                            act
                        } else {
                            drive.tanh()
                        }
                    }
                    ModelArch::M7SpikingLifStdp if n.role != NeuronRole::ReadoutOutput => {
                        n.membrane_v = n.membrane_v * 0.76 + drive * 0.48;
                        if n.membrane_v > 0.62 {
                            n.spiked = true;
                            n.membrane_v = -0.25;
                            n.spike_trace = 1.0;
                            1.0
                        } else {
                            n.spike_trace *= 0.86;
                            (n.membrane_v.max(0.0) * 0.75 + n.spike_trace * 0.35).clamp(0.0, 1.0)
                        }
                    }
                    ModelArch::M8LiquidTimeConstant if n.role != NeuronRole::ReadoutOutput => {
                        let gate = 1.0 / (1.0 + (-drive).exp());
                        let dt = 0.28;
                        let dx = -(1.0 / n.tau + gate) * n.activation + gate * drive.tanh();
                        (n.activation + dt * dx).clamp(-1.5, 1.5)
                    }
                    ModelArch::M11CorticalColumnsHtm if n.role == NeuronRole::CorticalMiniColumn => {
                        (drive - 0.18).max(0.0).tanh()
                    }
                    ModelArch::M12ConnectomicOrganoid if n.role == NeuronRole::AstrocyteGlia => {
                        (n.prev_activation * 0.90 + drive.tanh() * 0.22).clamp(-1.0, 1.0)
                    }
                    _ => {
                        if n.role == NeuronRole::ReadoutOutput {
                            drive
                        } else {
                            (0.65 * drive.tanh() + 0.35 * drive.max(0.1 * drive)).clamp(-2.2, 2.2)
                        }
                    }
                };
            }
        }

        let raw_out: Vec<f32> = self
            .output_ids
            .iter()
            .map(|&oid| self.neurons[oid].activation)
            .collect();

        let preds = match self.dataset_id.task_kind() {
            TaskKind::MultiClassClassification => {
                let max_logit = raw_out
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = raw_out.iter().map(|&z| (z - max_logit).exp()).collect();
                let sum: f32 = exps.iter().sum::<f32>().max(1e-6);
                exps.into_iter().map(|e| e / sum).collect()
            }
            TaskKind::ContinuousRegression
            | TaskKind::KinematicControl
            | TaskKind::MultiPropertyQSAR => raw_out
                .into_iter()
                .map(|z| 1.0 / (1.0 + (-z).exp()))
                .collect(),
        };

        for (&oid, &p) in self.output_ids.iter().zip(&preds) {
            self.neurons[oid].activation = p;
        }
        preds
    }

    pub fn step_simulation(&mut self) {
        self.step_count += 1;
        let n_train = self.dataset.train.len().max(1);
        let bsize = self.params.batch_size.clamp(4, 32);
        if (self.step_count as usize * bsize) % n_train < bsize {
            self.epoch_count += 1;
        }

        let lr = self.params.learning_rate;
        let mom = self.params.momentum;
        let l1_reg = match self.goal {
            OutputGoal::SparseEdgeEfficiency => 0.0045,
            OutputGoal::MetabolicBioEconomy => 0.0035,
            _ => 0.0008,
        };

        let mut grad_syn = vec![0.0f32; self.synapses.len()];
        let mut grad_bias = vec![0.0f32; self.neurons.len()];
        let mut batch_loss = 0.0f32;

        for b in 0..bsize {
            let s_idx = ((self.step_count as usize * bsize) + b) % n_train;
            let sample = self.dataset.train[s_idx].clone();
            let preds = self.forward_sample(&sample.features, false);
            if b == 0 && (self.step_count == 1 || self.step_count % 150 == 0) {
                let v_idx = ((self.step_count / 150) as usize) % self.dataset.val.len().max(1);
                self.active_sample = self.dataset.val[v_idx].clone();
            }

            for n in &mut self.neurons {
                n.grad = 0.0;
            }
            for (o_i, &oid) in self.output_ids.iter().enumerate() {
                let y = sample.targets.get(o_i).copied().unwrap_or(0.0);
                let p = preds[o_i];
                let diff = p - y;
                batch_loss += match self.dataset_id.task_kind() {
                    TaskKind::MultiClassClassification => -y * p.clamp(1e-5, 1.0).ln(),
                    _ => diff * diff,
                };
                self.neurons[oid].grad = diff;
            }

            for pass in 0..3 {
                for syn in self.synapses.iter().rev() {
                    if syn.severed || self.neurons[syn.from].ablated || self.neurons[syn.to].ablated {
                        continue;
                    }
                    if syn.is_skip_or_lateral && pass > 0 {
                        continue;
                    }
                    let dst_grad = self.neurons[syn.to].grad;
                    let src_act = self.neurons[syn.from].activation;
                    let deriv = (1.0 - src_act.tanh().powi(2)).max(0.08);
                    self.neurons[syn.from].grad =
                        (self.neurons[syn.from].grad + dst_grad * syn.weight * deriv).clamp(-3.0, 3.0);
                }
            }

            let bio_factor = self.model.intrinsic_organic_factor().min(1.0);
            for (s_i, syn) in self.synapses.iter_mut().enumerate() {
                if syn.severed || self.neurons[syn.from].ablated || self.neurons[syn.to].ablated {
                    syn.smooth_flux *= 0.90;
                    syn.smooth_grad_flux *= 0.90;
                    continue;
                }
                let pre = self.neurons[syn.from].activation;
                let post = self.neurons[syn.to].activation;
                let post_grad = self.neurons[syn.to].grad;

                let hebbian_term = -post_grad * pre * 0.85
                    + if matches!(
                        self.model,
                        ModelArch::M7SpikingLifStdp
                            | ModelArch::M12ConnectomicOrganoid
                            | ModelArch::M13CerebralCortexGreyMatter
                    ) {
                        (pre * post - 0.12 * post * post * syn.weight) * 0.08
                    } else {
                        0.0
                    };
                let delta_g = (1.0 - 0.25 * bio_factor) * (post_grad * pre) - 0.25 * bio_factor * hebbian_term;
                grad_syn[s_i] += delta_g;
                syn.eligibility = syn.eligibility * 0.92 + (pre * post).abs() * 0.08;
                syn.smooth_flux = syn.smooth_flux * 0.96 + (pre * syn.weight).abs().min(1.8) * 0.04;
                syn.smooth_grad_flux = syn.smooth_grad_flux * 0.96 + (post_grad * syn.weight).abs().min(1.8) * 0.04;
            }

            for (n_i, n) in self.neurons.iter_mut().enumerate() {
                grad_bias[n_i] += n.grad;
                n.utility = n.utility * 0.96 + (n.activation.abs() * 0.5 + n.grad.abs()).min(1.5) * 0.04;
            }
        }

        let inv_b = 1.0 / bsize as f32;
        let has_damage = self.ablated_count() > 0 || self.severed_count() > 0;
        let homeo_boost = if has_damage { 1.45 } else { 1.0 };
        for (s_i, syn) in self.synapses.iter_mut().enumerate() {
            if syn.severed || self.neurons[syn.from].ablated || self.neurons[syn.to].ablated {
                continue;
            }
            let g = grad_syn[s_i] * inv_b + l1_reg * syn.weight.signum();
            let lr = lr * homeo_boost;
            syn.velocity = mom * syn.velocity - lr * g;
            syn.weight = (syn.weight + syn.velocity).clamp(-2.8, 2.8);
            if self.neurons[syn.from].role == NeuronRole::InhibitoryInterneuron
                && matches!(
                    self.model,
                    ModelArch::M7SpikingLifStdp | ModelArch::M13CerebralCortexGreyMatter
                )
            {
                syn.weight = -syn.weight.abs().max(0.02);
            }
            // Calm, gentle macroscopic flow wave progression with refractory pause [0.0 .. 2.5]
            syn.pulse = (syn.pulse + 0.0065 + syn.eligibility * 0.005) % 2.8;
            syn.age_ticks = syn.age_ticks.saturating_add(1);
        }

        for (n_i, n) in self.neurons.iter_mut().enumerate() {
            if n.role != NeuronRole::SensoryInput {
                n.bias = (n.bias - lr * 0.6 * grad_bias[n_i] * inv_b).clamp(-1.8, 1.8);
            }
            n.smooth_activation = n.smooth_activation * 0.985 + n.activation.abs().min(1.2) * 0.015;
        }

        let held_feats = self.active_sample.features.clone();
        self.last_prediction = self.forward_sample(&held_feats, false);

        self.current_loss = (batch_loss * inv_b).clamp(0.005, 3.5);
        self.animate_neural_space();

        if self.step_count % 90 == 0 {
            self.step_structural_plasticity();
        }
        if self.step_count % 75 == 0 {
            self.refresh_flow_sample();
        }
        if self.step_count % 5 == 0 {
            self.evaluate_validation_metrics();
        }
    }

    fn animate_neural_space(&mut self) {
        let organic = self.model.intrinsic_organic_factor().min(1.0);
        let is_cortex = self.model == ModelArch::M13CerebralCortexGreyMatter;
        let t = self.step_count as f32 * 0.008;

        let mut forces = vec![Vec3::ZERO; self.neurons.len()];
        if organic > 0.20 && !is_cortex {
            for syn in &self.synapses {
                let diff = self.neurons[syn.to].pos - self.neurons[syn.from].pos;
                let dist = diff.length().max(0.5);
                let pull = (syn.weight.abs() - 0.35) * 0.015 * organic;
                let dir = diff / dist;
                forces[syn.from] += dir * pull;
                forces[syn.to] -= dir * pull;
            }
        }

        for (i, n) in self.neurons.iter_mut().enumerate() {
            if n.role == NeuronRole::SensoryInput || n.role == NeuronRole::ReadoutOutput {
                n.soma_scale = 1.0;
                continue;
            }
            // Preserve laminar cortical depth (Y) strictly in M13 Grey Matter while allowing subtle tissue breathing
            let drift = if is_cortex {
                Vec3::new(
                    (t + n.phase_seed).sin() * 0.07,
                    (t * 1.1 + n.phase_seed).cos() * 0.03,
                    (t * 0.85 + n.phase_seed * 1.7).sin() * 0.07,
                )
            } else {
                Vec3::new(
                    (t + n.phase_seed).sin() * 0.14 * organic,
                    (t * 1.3 + n.phase_seed).cos() * 0.18 * organic,
                    (t * 0.9 + n.phase_seed * 1.7).sin() * 0.18 * organic,
                )
            };
            let spring = (n.rest_pos + drift - n.pos) * 0.14;
            n.vel = (n.vel + spring + forces[i]) * 0.76;
            n.pos += n.vel;
            n.pos.x = n.pos.x.clamp(-8.2, 8.2);
            n.pos.y = n.pos.y.clamp(-6.2, 6.2);
            n.pos.z = n.pos.z.clamp(-6.2, 6.2);
            n.soma_scale = match n.role {
                NeuronRole::ThickPyramidalL5 => 1.18,
                NeuronRole::AstrocyteGlia => 1.35,
                _ => 1.0,
            };
        }
    }

    fn step_structural_plasticity(&mut self) {
        if self.model == ModelArch::M0MinimalPerceptron {
            return;
        }
        let prune_thresh = self.params.prune_threshold * match self.goal {
            OutputGoal::SparseEdgeEfficiency | OutputGoal::MetabolicBioEconomy => 1.45,
            _ => 1.0,
        };

        let min_synapses = (self.input_ids.len() + self.output_ids.len() * 2).max(24);
        let mut changed = false;
        if self.synapses.len() > min_synapses && prune_thresh > 0.005 {
            let mut pruned_now = 0usize;
            self.synapses.retain(|s| {
                if pruned_now < 2 && s.age_ticks > 22 && s.weight.abs() < prune_thresh {
                    pruned_now += 1;
                    false
                } else {
                    true
                }
            });
            if pruned_now > 0 {
                self.pruned_total += pruned_now as u64;
                changed = true;
                self.push_log(format!("Pruned {} weak synapse(s) (|w| < {:.3})", pruned_now, prune_thresh));
            }
        }

        let max_synapses = 420usize;
        if self.synapses.len() < max_synapses
            && self.rng.gen::<f32>() < self.params.plasticity_growth * 0.55
            && self.neurons.len() > 4
        {
            let u = self.rng.gen_range(0..self.neurons.len());
            let v = self.rng.gen_range(0..self.neurons.len());
            let valid_dir = self.neurons[u].layer < self.neurons[v].layer
                || (self.model.intrinsic_organic_factor() > 0.5
                    && self.neurons[v].role != NeuronRole::SensoryInput
                    && self.neurons[u].role != NeuronRole::ReadoutOutput);
            if u != v && valid_dir && !self.synapses.iter().any(|s| s.from == u && s.to == v) {
                let w_init = (self.rng.gen::<f32>() - 0.45) * 0.24;
                let lateral = self.neurons[u].layer >= self.neurons[v].layer;
                self.add_synapse(u, v, w_init, lateral);
                self.sprouted_total += 1;
                changed = true;
                self.push_log(format!(
                    "Sprouted axon N{} -> N{} (w={:+.2})",
                    self.neurons[u].id, self.neurons[v].id, w_init
                ));
            }
        }

        let max_neurons = 98usize;
        let growth_prob = self.params.plasticity_growth
            * if matches!(
                self.model,
                ModelArch::M6GeneticGannNeat
                    | ModelArch::M10PredictiveCoding
                    | ModelArch::M11CorticalColumnsHtm
                    | ModelArch::M12ConnectomicOrganoid
                    | ModelArch::M13CerebralCortexGreyMatter
            ) {
                0.28
            } else {
                0.10
            };

        if self.neurons.len() < max_neurons
            && self.current_loss > 0.16
            && self.rng.gen::<f32>() < growth_prob
            && !self.synapses.is_empty()
        {
            let split_idx = self.rng.gen_range(0..self.synapses.len());
            let u = self.synapses[split_idx].from;
            let v = self.synapses[split_idx].to;
            let old_w = self.synapses[split_idx].weight;
            if self.neurons[u].layer < self.neurons[v].layer {
                let mid = (self.neurons[u].rest_pos + self.neurons[v].rest_pos) * 0.5
                    + Vec3::new(
                        (self.rng.gen::<f32>() - 0.5) * 1.1,
                        (self.rng.gen::<f32>() - 0.5) * 0.6,
                        (self.rng.gen::<f32>() - 0.5) * 1.1,
                    );
                let new_layer = ((self.neurons[u].layer + self.neurons[v].layer) / 2).max(1);
                let role = if self.model == ModelArch::M13CerebralCortexGreyMatter {
                    if mid.y > 1.2 {
                        NeuronRole::PyramidalL23
                    } else if mid.y > -0.6 {
                        NeuronRole::SpinyStellateL4
                    } else {
                        NeuronRole::ThickPyramidalL5
                    }
                } else if self.model == ModelArch::M12ConnectomicOrganoid && self.rng.gen::<f32>() < 0.2 {
                    NeuronRole::InhibitoryInterneuron
                } else {
                    NeuronRole::ExcitatoryHidden
                };
                let new_idx = self.alloc_neuron(new_layer, role, mid);
                let uid = self.neurons[new_idx].id;
                self.add_synapse(u, new_idx, 0.82, false);
                self.add_synapse(new_idx, v, old_w, false);
                self.neurogenesis_total += 1;
                changed = true;
                self.push_log(format!("Neurogenesis: budded {} #{}", role.label(), uid));
            }
        }

        if self.ablated_count() > 0 && self.synapses.len() < 440 {
            if self.sprout_compensatory_bypass(2) > 0 {
                changed = true;
            }
        }
        if changed {
            self.refresh_flow_sample();
        }
    }

    pub fn ablated_count(&self) -> usize {
        self.neurons.iter().filter(|n| n.ablated).count()
    }

    pub fn severed_count(&self) -> usize {
        self.synapses.iter().filter(|s| s.severed).count()
    }

    pub fn clamped_count(&self) -> usize {
        self.neurons.iter().filter(|n| n.opto_clamp != OptoClamp::None).count()
    }

    /// Toggle structural ablation (lesion) on a soma; protects final surviving input/output node.
    pub fn toggle_ablate_neuron(&mut self, idx: usize) {
        if idx >= self.neurons.len() {
            return;
        }
        let role = self.neurons[idx].role;
        if !self.neurons[idx].ablated {
            if role == NeuronRole::SensoryInput
                && self.input_ids.iter().filter(|&&i| !self.neurons[i].ablated).count() <= 1
            {
                return;
            }
            if role == NeuronRole::ReadoutOutput
                && self.output_ids.iter().filter(|&&i| !self.neurons[i].ablated).count() <= 1
            {
                return;
            }
            if self.pre_lesion_metric.is_none() {
                self.pre_lesion_metric = Some(self.primary_metric);
            }
            self.neurons[idx].ablated = true;
            self.neurons[idx].opto_clamp = OptoClamp::None;
            self.neurons[idx].activation = 0.0;
            self.neurons[idx].smooth_activation = 0.0;
            let uid = self.neurons[idx].id;
            self.evaluate_validation_metrics();
            self.acute_lesion_metric = Some(self.primary_metric);
            self.push_log(format!("Ablated Soma #{} ({})", uid, role.label()));
        } else {
            self.neurons[idx].ablated = false;
            let uid = self.neurons[idx].id;
            self.evaluate_validation_metrics();
            self.push_log(format!("Re-innervated Soma #{}", uid));
        }
    }

    /// Apply or cycle optogenetic ChR2 excitation / NpHR silencing clamp on a soma.
    pub fn set_neuron_clamp(&mut self, idx: usize, clamp: OptoClamp) {
        if let Some(n) = self.neurons.get_mut(idx) {
            if n.ablated {
                return;
            }
            n.opto_clamp = if n.opto_clamp == clamp { OptoClamp::None } else { clamp };
            let uid = n.id;
            let state_now = n.opto_clamp;
            self.evaluate_validation_metrics();
            self.push_log(format!("Opto-clamp Soma #{} -> {:?}", uid, state_now));
        }
    }

    /// Surgical feedback tractotomy: severs all inter-layer reverse feedback pathways (`is_feedback`).
    pub fn sever_feedback_tracts(&mut self) -> usize {
        if self.pre_lesion_metric.is_none() {
            self.pre_lesion_metric = Some(self.primary_metric);
        }
        let mut count = 0usize;
        for syn in &mut self.synapses {
            if syn.is_feedback && !syn.severed {
                syn.severed = true;
                syn.smooth_flux = 0.0;
                count += 1;
            }
        }
        if count > 0 {
            self.evaluate_validation_metrics();
            self.acute_lesion_metric = Some(self.primary_metric);
            self.push_log(format!("Feedback tractotomy: severed {} reverse axons", count));
        }
        count
    }

    /// Severs lateral / residual skip pathways to test skip-stream causal necessity.
    pub fn sever_lateral_skip_tracts(&mut self) -> usize {
        if self.pre_lesion_metric.is_none() {
            self.pre_lesion_metric = Some(self.primary_metric);
        }
        let mut count = 0usize;
        for syn in &mut self.synapses {
            if syn.is_skip_or_lateral && !syn.is_feedback && !syn.severed {
                syn.severed = true;
                syn.smooth_flux = 0.0;
                count += 1;
            }
        }
        if count > 0 {
            self.evaluate_validation_metrics();
            self.acute_lesion_metric = Some(self.primary_metric);
            self.push_log(format!("Lateral/skip tractotomy: severed {} axons", count));
        }
        count
    }

    /// Induces a focal 3D micro-lesion (stroke) ablating ~25% of hidden/cortical somata near a locus,
    /// triggering compensatory homeostatic bypass sprouting around the damaged territory.
    pub fn induce_focal_stroke(&mut self) -> usize {
        let candidates: Vec<usize> = self
            .neurons
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                !n.ablated
                    && n.role != NeuronRole::SensoryInput
                    && n.role != NeuronRole::ReadoutOutput
            })
            .map(|(i, _)| i)
            .collect();
        if candidates.is_empty() {
            return 0;
        }
        if self.pre_lesion_metric.is_none() {
            self.pre_lesion_metric = Some(self.primary_metric);
        }
        let center_idx = candidates[self.rng.gen_range(0..candidates.len())];
        let center_pos = self.neurons[center_idx].pos;
        let target_kill = (candidates.len() / 4).clamp(1, 10);

        let mut dists: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|i| (i, self.neurons[i].pos.distance(center_pos)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut hit = 0usize;
        for (idx, _) in dists.into_iter().take(target_kill) {
            self.neurons[idx].ablated = true;
            self.neurons[idx].opto_clamp = OptoClamp::None;
            self.neurons[idx].activation = 0.0;
            self.neurons[idx].smooth_activation = 0.0;
            hit += 1;
        }
        // Immediately sprout 3-5 compensatory peri-infarct bypass axons between surviving somata
        self.sprout_compensatory_bypass(4);
        self.evaluate_validation_metrics();
        self.acute_lesion_metric = Some(self.primary_metric);
        self.push_log(format!(
            "Focal 3D lesion: ablated {} somata + sprouted bypass collateral",
            hit
        ));
        hit
    }

    /// Sprouts compensatory bypass axons bridging surviving upstream -> downstream somata around lesions.
    pub fn sprout_compensatory_bypass(&mut self, max_new: usize) -> usize {
        let alive: Vec<usize> = self
            .neurons
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.ablated)
            .map(|(i, _)| i)
            .collect();
        if alive.len() < 3 {
            return 0;
        }
        let mut added = 0usize;
        for _ in 0..(max_new * 8) {
            if added >= max_new {
                break;
            }
            let u = alive[self.rng.gen_range(0..alive.len())];
            let v = alive[self.rng.gen_range(0..alive.len())];
            if u != v
                && self.neurons[u].layer < self.neurons[v].layer
                && !self.synapses.iter().any(|s| s.from == u && s.to == v && !s.severed)
            {
                let w = (self.rng.gen::<f32>() - 0.35) * 0.42;
                self.add_synapse(u, v, w, true);
                self.sprouted_total += 1;
                added += 1;
            }
        }
        added
    }

    /// Restores all ablated somata, reconnects severed tracts, and clears optogenetic clamps.
    pub fn heal_all_interventions(&mut self) {
        for n in &mut self.neurons {
            n.ablated = false;
            n.opto_clamp = OptoClamp::None;
        }
        for s in &mut self.synapses {
            s.severed = false;
        }
        self.pre_lesion_metric = None;
        self.acute_lesion_metric = None;
        self.evaluate_validation_metrics();
        self.push_log("Restored all ablated somata, severed tracts & opto-clamps".to_string());
    }

    pub fn clear_all_clamps(&mut self) {
        for n in &mut self.neurons {
            n.opto_clamp = OptoClamp::None;
        }
    }

    pub fn evaluate_validation_metrics(&mut self) {
        let n_val = self.dataset.val.len().min(48).max(1);
        let mut score_acc = 0.0f32;
        for i in 0..n_val {
            let sample = self.dataset.val[i].clone();
            let preds = self.forward_sample(&sample.features, false);
            match self.dataset_id.task_kind() {
                TaskKind::MultiClassClassification => {
                    let argmax = preds
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    if argmax == sample.class_index {
                        score_acc += 100.0;
                    }
                }
                TaskKind::ContinuousRegression
                | TaskKind::KinematicControl
                | TaskKind::MultiPropertyQSAR => {
                    let mae: f32 = preds
                        .iter()
                        .zip(&sample.targets)
                        .map(|(p, y)| (p - y).abs())
                        .sum::<f32>()
                        / preds.len().max(1) as f32;
                    score_acc += (1.0 - mae * 1.65).clamp(0.0, 1.0) * 100.0;
                }
            }
        }
        self.primary_metric = score_acc / n_val as f32;

        let act_energy: f32 = self.neurons.iter().map(|n| n.activation.abs()).sum::<f32>()
            / self.neurons.len().max(1) as f32;
        let syn_load = self.synapses.len() as f32 / 250.0;
        self.metabolic_flux = (0.55 * act_energy + 0.45 * syn_load).clamp(0.05, 2.5);
        self.adversarial_margin = (100.0 - self.current_loss * 38.0).clamp(10.0, 99.0);

        self.loss_history.push(self.current_loss);
        if self.loss_history.len() > 180 {
            self.loss_history.remove(0);
        }
        self.metric_history.push(self.primary_metric);
        if self.metric_history.len() > 180 {
            self.metric_history.remove(0);
        }
        self.synapse_history.push(self.synapses.len());
        if self.synapse_history.len() > 180 {
            self.synapse_history.remove(0);
        }
    }
}
