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

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Task formulation governing output layer interpretation and evaluation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskKind {
    MultiClassClassification,
    ContinuousRegression,
    KinematicControl,
    MultiPropertyQSAR,
}

impl TaskKind {
    pub fn primary_metric_label(self) -> &'static str {
        match self {
            Self::MultiClassClassification => "Val Accuracy (%)",
            Self::ContinuousRegression => "R² Valuation Fit (%)",
            Self::KinematicControl => "Collision-Free Score (%)",
            Self::MultiPropertyQSAR => "QSAR Concordance (%)",
        }
    }

    pub fn loss_label(self) -> &'static str {
        match self {
            Self::MultiClassClassification => "Cross-Entropy",
            Self::ContinuousRegression => "Huber / RMSE",
            Self::KinematicControl => "Control Cost J(u)",
            Self::MultiPropertyQSAR => "Multi-Task MAE",
        }
    }
}

/// Seven benchmark datasets spanning classical computer vision, botany, real-estate economics,
/// autonomous LiDAR navigation, and small-molecule QSAR cheminformatics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatasetId {
    D1ImageClassification,
    D2FacialRecognition,
    D3ObjectRecognition,
    D4PlantSpecies,
    D5HomeMarketValue,
    D6RoboticsNavigation,
    D7MolecularProperties,
}

impl DatasetId {
    pub const ALL: [DatasetId; 7] = [
        Self::D1ImageClassification,
        Self::D2FacialRecognition,
        Self::D3ObjectRecognition,
        Self::D4PlantSpecies,
        Self::D5HomeMarketValue,
        Self::D6RoboticsNavigation,
        Self::D7MolecularProperties,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::D1ImageClassification => "D1",
            Self::D2FacialRecognition => "D2",
            Self::D3ObjectRecognition => "D3",
            Self::D4PlantSpecies => "D4",
            Self::D5HomeMarketValue => "D5",
            Self::D6RoboticsNavigation => "D6",
            Self::D7MolecularProperties => "D7",
        }
    }

    pub fn from_code(code: &str) -> Result<Self, String> {
        let trimmed = code.trim();
        Self::ALL
            .into_iter()
            .find(|d| d.code().eq_ignore_ascii_case(trimmed))
            .ok_or_else(|| {
                let valid = Self::ALL
                    .into_iter()
                    .map(Self::code)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Unknown dataset code '{code}'. Valid codes (D1..D7): {valid}.")
            })
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::D1ImageClassification => "Image Classification (CipherGlyph 8x8)",
            Self::D2FacialRecognition => "Facial Recognition (EigenFace-3D Landmarks)",
            Self::D3ObjectRecognition => "Object Recognition (SceneBox RGB-D)",
            Self::D4PlantSpecies => "Plant Species Taxonomy (FloraMorph PNW)",
            Self::D5HomeMarketValue => "Home Market Value (Willamette Hedonic)",
            Self::D6RoboticsNavigation => "Robotics & Autonomous Nav (LiDAR-12)",
            Self::D7MolecularProperties => "Molecular Properties (ChemGraph QSAR)",
        }
    }

    pub fn domain_summary(self) -> &'static str {
        match self {
            Self::D1ImageClassification => {
                "16 pooled spatial features from noisy 8x8 glyph bitmaps across 6 visual classes."
            }
            Self::D2FacialRecognition => {
                "14 craniofacial geometric landmark ratios & pose-invariant descriptors across 6 identities."
            }
            Self::D3ObjectRecognition => {
                "12 RGB-D contour, depth-discontinuity & aspect descriptors localizing 5 object classes."
            }
            Self::D4PlantSpecies => {
                "10 botanical morphometrics (venation angle, stomatal density, petal/sepal ratio) for 6 taxa."
            }
            Self::D5HomeMarketValue => {
                "9 hedonic property attributes predicting continuous market valuation ($200k-$1.2M)."
            }
            Self::D6RoboticsNavigation => {
                "12-sector LiDAR obstacle rangefinder + goal bearing outputting [v_lin, omega_yaw, brake, clearance]."
            }
            Self::D7MolecularProperties => {
                "11 topological & electronic descriptors (MW, TPSA, logP, HBD/HBA, Wiener) -> [BBB, HOMO-LUMO, Sol]."
            }
        }
    }

    pub fn task_kind(self) -> TaskKind {
        match self {
            Self::D1ImageClassification
            | Self::D2FacialRecognition
            | Self::D3ObjectRecognition
            | Self::D4PlantSpecies => TaskKind::MultiClassClassification,
            Self::D5HomeMarketValue => TaskKind::ContinuousRegression,
            Self::D6RoboticsNavigation => TaskKind::KinematicControl,
            Self::D7MolecularProperties => TaskKind::MultiPropertyQSAR,
        }
    }

    pub fn input_dim(self) -> usize {
        match self {
            Self::D1ImageClassification => 16,
            Self::D2FacialRecognition => 14,
            Self::D3ObjectRecognition => 12,
            Self::D4PlantSpecies => 10,
            Self::D5HomeMarketValue => 9,
            Self::D6RoboticsNavigation => 12,
            Self::D7MolecularProperties => 11,
        }
    }

    pub fn output_dim(self) -> usize {
        match self {
            Self::D1ImageClassification => 6,
            Self::D2FacialRecognition => 6,
            Self::D3ObjectRecognition => 5,
            Self::D4PlantSpecies => 6,
            Self::D5HomeMarketValue => 1,
            Self::D6RoboticsNavigation => 4,
            Self::D7MolecularProperties => 3,
        }
    }

    pub fn feature_labels(self) -> &'static [&'static str] {
        match self {
            Self::D1ImageClassification => &[
                "Q00", "Q01", "Q02", "Q03", "Q10", "Q11", "Q12", "Q13", "Q20", "Q21", "Q22", "Q23",
                "Q30", "Q31", "Q32", "Q33",
            ],
            Self::D2FacialRecognition => &[
                "Interpupillary",
                "NasalBridge",
                "ZygomaticArch",
                "MandibleAngle",
                "BrowRidge",
                "PhiltrumRatio",
                "CanthalTilt",
                "JawSymmetry",
                "FacialIndex",
                "OrbitalDepth",
                "YawCos",
                "PitchSin",
                "Specularity",
                "ShadowGrad",
            ],
            Self::D3ObjectRecognition => &[
                "BBoxAspect",
                "DepthMean",
                "DepthEdge",
                "Convexity",
                "SpecAlbedo",
                "SurfaceNormX",
                "SurfaceNormY",
                "SurfaceNormZ",
                "CornerDensity",
                "ColorEntropy",
                "RadialSym",
                "OcclusionEst",
            ],
            Self::D4PlantSpecies => &[
                "LeafLength",
                "LeafWidth",
                "VenationAngle",
                "StomatalDens",
                "PetalLength",
                "SepalRatio",
                "SerrationIdx",
                "ChlReflect",
                "CuticleWax",
                "CanopyShade",
            ],
            Self::D5HomeMarketValue => &[
                "LivingSqFt",
                "LotAcres",
                "StructureAge",
                "BedBathScore",
                "TransitProx",
                "SolarCanopy",
                "SchoolIndex",
                "FloodSafety",
                "EnergyRating",
            ],
            Self::D6RoboticsNavigation => &[
                "Ray-90L",
                "Ray-60L",
                "Ray-30L",
                "Ray-10L",
                "Ray-0F",
                "Ray+10R",
                "Ray+30R",
                "Ray+60R",
                "Ray+90R",
                "GoalDist",
                "GoalSin",
                "CurSpeed",
            ],
            Self::D7MolecularProperties => &[
                "MolWeight",
                "TPSA",
                "cLogP",
                "HBondDonors",
                "HBondAccept",
                "AromaticRings",
                "RotBonds",
                "Sp3Fraction",
                "WienerTopo",
                "DipoleEst",
                "HalogenCount",
            ],
        }
    }

    pub fn output_labels(self) -> &'static [&'static str] {
        match self {
            Self::D1ImageClassification => &["Ring-0", "Spire-1", "Helix-2", "Cross-3", "Wave-4", "Star-5"],
            Self::D2FacialRecognition => &["Subj-A", "Subj-B", "Subj-C", "Subj-D", "Subj-E", "Subj-F"],
            Self::D3ObjectRecognition => &["Vessel", "Vehicle", "Biped", "Foliage", "Tool"],
            Self::D4PlantSpecies => &[
                "I. setosa",
                "I. versicolor",
                "I. virginica",
                "P. menziesii",
                "T. ovatum",
                "P. munitum",
            ],
            Self::D5HomeMarketValue => &["MarketValue ($M)"],
            Self::D6RoboticsNavigation => &["LinearVel", "AngularYaw", "BrakeForce", "SafeClear"],
            Self::D7MolecularProperties => &["BBB-Perm", "HOMO-LUMO", "AqSolubility"],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRecord {
    pub features: Vec<f32>,
    pub targets: Vec<f32>,
    pub class_index: usize,
    pub sample_tag: String,
}

#[derive(Debug, Clone)]
pub struct BuiltDataset {
    pub id: DatasetId,
    pub train: Vec<SampleRecord>,
    pub val: Vec<SampleRecord>,
}

impl BuiltDataset {
    /// Deterministically synthesizes structured, non-linear manifold data for the chosen benchmark.
    pub fn generate(id: DatasetId, seed: u64, noise_std: f32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ (id as u64 * 0x9E37_79B9));
        let in_dim = id.input_dim();
        let out_dim = id.output_dim();
        let n_total = 320usize;
        let n_train = 240usize;

        // Construct latent class prototypes or nonlinear teacher kernels so tasks reward hidden depth & topology.
        let mut prototypes = vec![vec![0.0f32; in_dim]; out_dim.max(4)];
        for (c, proto) in prototypes.iter_mut().enumerate() {
            for (j, val) in proto.iter_mut().enumerate() {
                let phase = (c as f32 * 1.37) + (j as f32 * 0.73);
                *val = phase.sin() * 0.85 + (phase * 1.9).cos() * 0.35;
            }
        }

        let mut all_samples = Vec::with_capacity(n_total);
        for idx in 0..n_total {
            match id.task_kind() {
                TaskKind::MultiClassClassification => {
                    let c = idx % out_dim;
                    // Multi-modal class clusters (XOR-like sub-clusters require nonlinear hidden representations)
                    let sub_mode = if (idx / out_dim) % 2 == 0 { 1.0f32 } else { -0.65f32 };
                    let mut x = vec![0.0f32; in_dim];
                    for j in 0..in_dim {
                        let base = prototypes[c][j];
                        let harmonic = ((c + 1) as f32 * (j + 1) as f32 * 0.41).cos() * 0.42 * sub_mode;
                        let eps = (rng.gen::<f32>() - 0.5) * 2.0 * (0.18 + noise_std);
                        x[j] = (base * sub_mode.signum() * 0.65 + base * 0.55 + harmonic + eps)
                            .clamp(-2.0, 2.0);
                    }
                    let mut y = vec![0.0f32; out_dim];
                    y[c] = 1.0;
                    let label = id.output_labels()[c];
                    all_samples.push(SampleRecord {
                        features: x,
                        targets: y,
                        class_index: c,
                        sample_tag: format!("{} #{}", label, idx + 1),
                    });
                }
                TaskKind::ContinuousRegression => {
                    // Willamette Valley hedonic home value in normalized [$0.15M .. $1.15M] -> [0..1]
                    let mut x = vec![0.0f32; in_dim];
                    for val in &mut x {
                        *val = (rng.gen::<f32>() * 2.0 - 1.0).clamp(-1.0, 1.0);
                    }
                    let sqft = x[0];
                    let lot = x[1];
                    let age = x[2];
                    let school = x[6];
                    let flood_safe = x[7];
                    let solar = x[5];
                    let raw = 0.52
                        + 0.22 * sqft
                        + 0.11 * lot
                        - 0.09 * age
                        + 0.14 * school
                        + 0.08 * (sqft * school)
                        + 0.06 * (flood_safe * solar).sin()
                        + (rng.gen::<f32>() - 0.5) * noise_std * 0.25;
                    let price_m = raw.clamp(0.12, 0.98);
                    all_samples.push(SampleRecord {
                        features: x,
                        targets: vec![price_m],
                        class_index: (price_m * 5.0) as usize,
                        sample_tag: format!("Est ${:.0}k", price_m * 1000.0),
                    });
                }
                TaskKind::KinematicControl => {
                    // 12-input LiDAR + goal vector -> [v_lin, yaw_rate, brake, safe_clearance]
                    let mut x = vec![0.0f32; in_dim];
                    for val in &mut x {
                        *val = rng.gen::<f32>();
                    }
                    let left_clear = (x[0] + x[1] + x[2]) / 3.0;
                    let front_clear = (x[3] + x[4] + x[5]) / 3.0;
                    let right_clear = (x[6] + x[7] + x[8]) / 3.0;
                    let goal_sin = x[10] * 2.0 - 1.0;
                    let noise = (rng.gen::<f32>() - 0.5) * noise_std * 0.2;

                    let v_lin = (front_clear * 0.85 + 0.1 + noise).clamp(0.0, 1.0);
                    let yaw = (0.5 + 0.35 * (right_clear - left_clear) + 0.25 * goal_sin + noise)
                        .clamp(0.0, 1.0);
                    let brake = ((1.0 - front_clear).powi(2) + noise).clamp(0.0, 1.0);
                    let clearance = front_clear.min(left_clear.max(right_clear)).clamp(0.0, 1.0);
                    all_samples.push(SampleRecord {
                        features: x.iter().map(|v| v * 2.0 - 1.0).collect(),
                        targets: vec![v_lin, yaw, brake, clearance],
                        class_index: if brake > 0.55 { 2 } else if yaw > 0.55 { 1 } else { 0 },
                        sample_tag: format!("NavFrame #{}", idx + 1),
                    });
                }
                TaskKind::MultiPropertyQSAR => {
                    // 11-descriptor ChemGraph -> [BBB permeability, HOMO-LUMO gap, Aqueous Solubility]
                    let mut x = vec![0.0f32; in_dim];
                    for val in &mut x {
                        *val = rng.gen::<f32>() * 2.0 - 1.0;
                    }
                    let mw = x[0];
                    let tpsa = x[1];
                    let clogp = x[2];
                    let hbd = x[3];
                    let arom = x[5];
                    let sp3 = x[7];
                    let noise = (rng.gen::<f32>() - 0.5) * noise_std * 0.2;

                    // Lipinski / Clark rules nonlinearly coupled
                    let bbb = (1.0 / (1.0 + (1.8 * tpsa + 0.9 * mw - 1.1 * clogp + 0.7 * hbd).exp()) + noise)
                        .clamp(0.02, 0.98);
                    let homo_lumo = (0.55 - 0.24 * arom + 0.19 * sp3 - 0.08 * (arom * clogp) + noise)
                        .clamp(0.05, 0.95);
                    let aq_sol = (0.52 - 0.28 * clogp + 0.18 * tpsa - 0.12 * mw + noise)
                        .clamp(0.05, 0.95);

                    all_samples.push(SampleRecord {
                        features: x,
                        targets: vec![bbb, homo_lumo, aq_sol],
                        class_index: if bbb > 0.5 { 1 } else { 0 },
                        sample_tag: format!("Mol-QSAR #{}", idx + 1),
                    });
                }
            }
        }

        all_samples.shuffle(&mut rng);
        let val = all_samples.split_off(n_train);
        Self {
            id,
            train: all_samples,
            val,
        }
    }
}
