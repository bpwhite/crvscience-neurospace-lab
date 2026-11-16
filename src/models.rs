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

use serde::{Deserialize, Serialize};

/// Five evolutionary tiers spanning mechanistic perceptrons up to 3D histological cerebral cortex grey matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    Tier1Mechanistic,
    Tier2FoundationLlm,
    Tier2AdversarialEvolutionary,
    Tier3SpikingDynamical,
    Tier4BiologicalTheoretical,
    Tier5HistologicalGreyMatter,
}

impl ModelTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Tier1Mechanistic => "Tier I: Classical & Mechanistic",
            Self::Tier2FoundationLlm => "Tier II: Frontier Large Language Models (LLM / MoE)",
            Self::Tier2AdversarialEvolutionary => "Tier II: Adversarial & Evolutionary (GAN / GANN)",
            Self::Tier3SpikingDynamical => "Tier III: Spiking & Continuous Dynamical",
            Self::Tier4BiologicalTheoretical => "Tier IV: Biological & Cortical Algorithms",
            Self::Tier5HistologicalGreyMatter => "Tier V: Pure 3D Histological Wetware (Grey Matter)",
        }
    }
}

/// Thirteen neural architectures culminating in 6-layered 3D Neocortical Grey Matter (L1-L6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArch {
    M0MinimalPerceptron,
    M1PerceptronMlp,
    M2VolumetricCnn,
    M3SparseTransformer,
    M14FrontierLlmMoe,
    M4MinimaxGan,
    M5AdversarialDefensePgd,
    M6GeneticGannNeat,
    M7SpikingLifStdp,
    M8LiquidTimeConstant,
    M9EchoStateReservoir,
    M10PredictiveCoding,
    M11CorticalColumnsHtm,
    M12ConnectomicOrganoid,
    M13CerebralCortexGreyMatter,
}

impl ModelArch {
    pub const ALL: [ModelArch; 15] = [
        Self::M0MinimalPerceptron,
        Self::M1PerceptronMlp,
        Self::M2VolumetricCnn,
        Self::M3SparseTransformer,
        Self::M14FrontierLlmMoe,
        Self::M4MinimaxGan,
        Self::M5AdversarialDefensePgd,
        Self::M6GeneticGannNeat,
        Self::M7SpikingLifStdp,
        Self::M8LiquidTimeConstant,
        Self::M9EchoStateReservoir,
        Self::M10PredictiveCoding,
        Self::M11CorticalColumnsHtm,
        Self::M12ConnectomicOrganoid,
        Self::M13CerebralCortexGreyMatter,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::M0MinimalPerceptron => "M0",
            Self::M1PerceptronMlp => "M1",
            Self::M2VolumetricCnn => "M2",
            Self::M3SparseTransformer => "M3",
            Self::M14FrontierLlmMoe => "M14",
            Self::M4MinimaxGan => "M4",
            Self::M5AdversarialDefensePgd => "M5",
            Self::M6GeneticGannNeat => "M6",
            Self::M7SpikingLifStdp => "M7",
            Self::M8LiquidTimeConstant => "M8",
            Self::M9EchoStateReservoir => "M9",
            Self::M10PredictiveCoding => "M10",
            Self::M11CorticalColumnsHtm => "M11",
            Self::M12ConnectomicOrganoid => "M12",
            Self::M13CerebralCortexGreyMatter => "M13",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::M0MinimalPerceptron => "Minimal Single-Layer Perceptron (Direct Gate)",
            Self::M1PerceptronMlp => "Deep Multi-Layer Perceptron (MLP)",
            Self::M2VolumetricCnn => "Volumetric Residual ConvNet (3D-CNN)",
            Self::M3SparseTransformer => "Multi-Head Sparse Attention Transformer",
            Self::M14FrontierLlmMoe => "Frontier Autoregressive LLM (GQA + Sparse MoE + KV)",
            Self::M4MinimaxGan => "Generative Adversarial Network (Minimax GAN)",
            Self::M5AdversarialDefensePgd => "Adversarial Robust Network (PGD-AT)",
            Self::M6GeneticGannNeat => "Genetic Algorithm Neural Net (GANN / NEAT)",
            Self::M7SpikingLifStdp => "Leaky Integrate-and-Fire Spiking (LIF + STDP)",
            Self::M8LiquidTimeConstant => "Liquid Time-Constant ODE Network (LTC)",
            Self::M9EchoStateReservoir => "Chaotic Echo-State Cortical Reservoir",
            Self::M10PredictiveCoding => "Hierarchical Predictive Coding (Free Energy)",
            Self::M11CorticalColumnsHtm => "Thousand-Brains Cortical Columns (HTM)",
            Self::M12ConnectomicOrganoid => "Connectomic Organoid & Astrocyte Tripartite",
            Self::M13CerebralCortexGreyMatter => "Cerebral Cortex Grey Matter (3D L1-L6 Laminae)",
        }
    }

    pub fn tier(self) -> ModelTier {
        match self {
            Self::M0MinimalPerceptron | Self::M1PerceptronMlp | Self::M2VolumetricCnn | Self::M3SparseTransformer => {
                ModelTier::Tier1Mechanistic
            }
            Self::M14FrontierLlmMoe => ModelTier::Tier2FoundationLlm,
            Self::M4MinimaxGan | Self::M5AdversarialDefensePgd | Self::M6GeneticGannNeat => {
                ModelTier::Tier2AdversarialEvolutionary
            }
            Self::M7SpikingLifStdp | Self::M8LiquidTimeConstant | Self::M9EchoStateReservoir => {
                ModelTier::Tier3SpikingDynamical
            }
            Self::M10PredictiveCoding
            | Self::M11CorticalColumnsHtm
            | Self::M12ConnectomicOrganoid => ModelTier::Tier4BiologicalTheoretical,
            Self::M13CerebralCortexGreyMatter => ModelTier::Tier5HistologicalGreyMatter,
        }
    }

    /// Natural position along Mechanistic (0.0) -> Organic (1.0) -> Pure 3D Histological Grey Matter (1.25).
    pub fn intrinsic_organic_factor(self) -> f32 {
        match self {
            Self::M0MinimalPerceptron => 0.00,
            Self::M1PerceptronMlp => 0.04,
            Self::M2VolumetricCnn => 0.12,
            Self::M3SparseTransformer => 0.22,
            Self::M14FrontierLlmMoe => 0.26,
            Self::M4MinimaxGan => 0.36,
            Self::M5AdversarialDefensePgd => 0.42,
            Self::M6GeneticGannNeat => 0.56,
            Self::M7SpikingLifStdp => 0.72,
            Self::M8LiquidTimeConstant => 0.78,
            Self::M9EchoStateReservoir => 0.82,
            Self::M10PredictiveCoding => 0.88,
            Self::M11CorticalColumnsHtm => 0.94,
            Self::M12ConnectomicOrganoid => 1.00,
            Self::M13CerebralCortexGreyMatter => 1.25,
        }
    }

    pub fn learning_rule_summary(self) -> &'static str {
        match self {
            Self::M0MinimalPerceptron => "1958 single-layer direct linear threshold gate (minimal nodes & links, zero hidden layers).",
            Self::M1PerceptronMlp => "Backprop + AdamW moment updates over dense linear threshold slabs.",
            Self::M2VolumetricCnn => "Local 3D receptive-field kernels + residual skip highways.",
            Self::M3SparseTransformer => "Dynamic Query-Key dot-product routing & sparse top-k attention heads.",
            Self::M14FrontierLlmMoe => "Causal GQA + rolling KV-Cache + Top-2 Router dispatching tokens across 4 SwiGLU MoE Experts + autoregressive loop.",
            Self::M4MinimaxGan => "Dual Generator vs. Discriminator Nash game with feature-matching head.",
            Self::M5AdversarialDefensePgd => "Inner-loop FGSM/PGD perturbation maximization + outer-loop robust descent.",
            Self::M6GeneticGannNeat => "Neuroevolutionary weight mutation, node speciation & cross-layer synaptogenesis.",
            Self::M7SpikingLifStdp => "Discrete action potentials, Dale's Law E/I balance & asymmetric STDP trace.",
            Self::M8LiquidTimeConstant => "Input-dependent synaptic time constants dx/dt = -[1/tau + f(x)]x + f(x)A.",
            Self::M9EchoStateReservoir => "Edge-of-chaos recurrent sphere (spectral radius ~0.98) + plastic readout.",
            Self::M10PredictiveCoding => "Local Hebbian prediction-error minimization across cortical value/error pairs.",
            Self::M11CorticalColumnsHtm => "Sparse Distributed Representations (k-WTA) + distal lateral dendritic voting.",
            Self::M12ConnectomicOrganoid => "Small-world wetware hubs, dopamine-modulated Hebbian & astrocytic Ca2+ waves.",
            Self::M13CerebralCortexGreyMatter => {
                "6-layer neocortical grey matter (Thalamus->L4 Stellate->L2/3 Pyramidal->L5 Thick-Tufted->L6) with apical L1 tufts & burst-dependent plasticity."
            }
        }
    }
}

/// Optimization goal shaping the composite objective and structural plasticity pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputGoal {
    MaxTaskFidelity,
    SparseEdgeEfficiency,
    AdversarialImmunity,
    FreeEnergyCalibration,
    MetabolicBioEconomy,
}

impl OutputGoal {
    pub const ALL: [OutputGoal; 5] = [
        Self::MaxTaskFidelity,
        Self::SparseEdgeEfficiency,
        Self::AdversarialImmunity,
        Self::FreeEnergyCalibration,
        Self::MetabolicBioEconomy,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::MaxTaskFidelity => "Max Predictive Fidelity (Acc / R²)",
            Self::SparseEdgeEfficiency => "Topological Sparsity & Low Latency",
            Self::AdversarialImmunity => "Adversarial Perturbation Robustness",
            Self::FreeEnergyCalibration => "Variational Free-Energy & Uncertainty",
            Self::MetabolicBioEconomy => "Metabolic ATP / Spike Efficiency",
        }
    }
}

/// Controls how the 3D Neural Space renders somata, apical/basal dendrites, axons, and scaffolds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenderMode {
    AutoAdaptive,
    MechanisticLattice,
    HybridCyberManifold,
    OrganicCorticalArbor,
    HistologicalGreyMatter3D,
}

impl RenderMode {
    pub const ALL: [RenderMode; 5] = [
        Self::AutoAdaptive,
        Self::MechanisticLattice,
        Self::HybridCyberManifold,
        Self::OrganicCorticalArbor,
        Self::HistologicalGreyMatter3D,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::AutoAdaptive => "Auto-Adaptive (Follows Model Tier)",
            Self::MechanisticLattice => "Mechanistic (Geometric Polyhedra & Bus-Bars)",
            Self::HybridCyberManifold => "Hybrid Cyber-Cortical (Splines & Manifolds)",
            Self::OrganicCorticalArbor => "Organic Wetware (Somata & Radial Dendrites)",
            Self::HistologicalGreyMatter3D => {
                "Pure 3D Histological Grey Matter (Pyramidal Somata & Apical Tufts)"
            }
        }
    }

    /// Returns morphology factor in `[0.0 .. 1.25]`; values `> 1.05` engage full 3D anatomical
    /// pyramidal somata, ascending apical trunks, L1 apical tufts, and basal dendritic skirts.
    pub fn resolve_organic_factor(self, arch: ModelArch, manual_blend: f32) -> f32 {
        match self {
            Self::AutoAdaptive => arch.intrinsic_organic_factor(),
            Self::MechanisticLattice => 0.0,
            Self::HybridCyberManifold => manual_blend.clamp(0.35, 0.65),
            Self::OrganicCorticalArbor => 0.96,
            Self::HistologicalGreyMatter3D => 1.25,
        }
    }
}

/// Left-column model parameter determinants governing real-time training, plasticity, and flow sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    pub learning_rate: f32,
    pub momentum: f32,
    pub prune_threshold: f32,
    pub plasticity_growth: f32,
    pub hidden_scale: f32,
    pub input_noise: f32,
    pub adversarial_eps: f32,
    pub flow_sample_rate: f32,
    pub batch_size: usize,
    pub continual_transfer: bool,
}

impl Default for HyperParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.035,
            momentum: 0.85,
            prune_threshold: 0.045,
            plasticity_growth: 0.35,
            hidden_scale: 1.0,
            input_noise: 0.08,
            adversarial_eps: 0.12,
            flow_sample_rate: 0.70,
            batch_size: 16,
            continual_transfer: true,
        }
    }
}
