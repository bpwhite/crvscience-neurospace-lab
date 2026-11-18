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

#![forbid(unsafe_code)]

use neurospace_lab::datasets::DatasetId;
use neurospace_lab::engine::{NeuralSpaceSim, NeuronRole, OptoClamp};
use neurospace_lab::icon::app_egui_icon;
use neurospace_lab::models::{ModelArch, OutputGoal, RenderMode};
use neurospace_lab::persistence::{LabStore, WorksetEnvelope};
use neurospace_lab::renderer_3d::classify_neuron_zone;

#[test]
fn all_fifteen_models_including_cortex_grey_matter_step_cleanly() {
    assert_eq!(ModelArch::ALL.len(), 15);
    for arch in ModelArch::ALL {
        let mut sim = NeuralSpaceSim::new(arch, DatasetId::D4PlantSpecies, 101);
        sim.params.plasticity_growth = 0.75;
        assert!(!sim.neurons.is_empty());
        assert!(!sim.synapses.is_empty());

        // Verify sparse flow sampling caps active tracers calmly
        let sampled_count = sim.synapses.iter().filter(|s| s.sampled_tracer).count();
        assert!(sampled_count <= 22);

        if arch == ModelArch::M13CerebralCortexGreyMatter {
            assert!(sim.neurons.iter().any(|n| n.role == NeuronRole::PyramidalL23));
            assert!(sim.neurons.iter().any(|n| n.role == NeuronRole::SpinyStellateL4));
            assert!(sim.neurons.iter().any(|n| n.role == NeuronRole::ThickPyramidalL5));
            assert!(sim.neurons.iter().any(|n| n.role == NeuronRole::CorticothalamicL6));
        }

        for _ in 0..95 {
            sim.step_simulation();
        }
        let sampled_after = sim.synapses.iter().filter(|s| s.sampled_tracer).count();
        assert!(sampled_after <= 22);
        assert!(sim.current_loss.is_finite());
        assert!(sim.primary_metric >= 0.0 && sim.primary_metric <= 100.0);
    }
}

#[test]
fn all_seven_datasets_generate_and_evaluate() {
    for ds in DatasetId::ALL {
        let mut sim = NeuralSpaceSim::new(ModelArch::M13CerebralCortexGreyMatter, ds, 2026);
        assert_eq!(sim.input_ids.len(), ds.input_dim());
        assert_eq!(sim.output_ids.len(), ds.output_dim());
        for _ in 0..16 {
            sim.step_simulation();
        }
        assert_eq!(sim.last_prediction.len(), ds.output_dim());
    }
}

#[test]
fn dataset_lock_and_model_lock_populate_comparison_ledger() {
    let mut sim = NeuralSpaceSim::new(ModelArch::M1PerceptronMlp, DatasetId::D5HomeMarketValue, 77);
    sim.dataset_locked = true;
    sim.step_simulation();
    sim.select_dataset(DatasetId::D1ImageClassification, 77);
    assert_eq!(sim.dataset_id, DatasetId::D5HomeMarketValue);

    sim.select_model(ModelArch::M13CerebralCortexGreyMatter);
    assert_eq!(sim.model, ModelArch::M13CerebralCortexGreyMatter);
    assert_eq!(sim.comparison_ledger.len(), 1);

    sim.dataset_locked = false;
    sim.model_locked = true;
    sim.step_simulation();
    sim.select_dataset(DatasetId::D6RoboticsNavigation, 77);
    assert_eq!(sim.dataset_id, DatasetId::D6RoboticsNavigation);
    assert_eq!(sim.comparison_ledger.len(), 2);
}

#[test]
fn sqlite_wal_and_sha256_workset_roundtrip() {
    let store = LabStore::open_in_memory().expect("in-memory sqlite");
    let mut sim = NeuralSpaceSim::new(
        ModelArch::M13CerebralCortexGreyMatter,
        DatasetId::D7MolecularProperties,
        99,
    );
    sim.goal = OutputGoal::MetabolicBioEconomy;
    for _ in 0..12 {
        sim.step_simulation();
    }
    let env = WorksetEnvelope::from_sim(&sim, RenderMode::HistologicalGreyMatter3D, 1.0);
    assert!(env.verify_digest());

    store.save_slot("slot_cortex", &env).expect("save slot");
    let loaded = store
        .load_slot("slot_cortex")
        .expect("query slot")
        .expect("present");
    assert!(loaded.verify_digest());
    assert_eq!(loaded.model, ModelArch::M13CerebralCortexGreyMatter);
    assert_eq!(loaded.dataset, DatasetId::D7MolecularProperties);
}


#[test]
fn interactive_ablation_tract_severing_and_optogenetic_clamping_work() {
    use neurospace_lab::engine::OptoClamp;
    let mut sim = NeuralSpaceSim::new(ModelArch::M13CerebralCortexGreyMatter, DatasetId::D4PlantSpecies, 2026);
    for _ in 0..20 {
        sim.step_simulation();
    }
    let severed = sim.sever_feedback_tracts();
    assert!(severed > 0);
    assert_eq!(sim.severed_count(), severed);

    let killed = sim.induce_focal_stroke();
    assert!(killed > 0);
    assert_eq!(sim.ablated_count(), killed);

    // Clamp a surviving hidden soma with ChR2 excitation
    let alive_idx = sim
        .neurons
        .iter()
        .position(|n| !n.ablated && n.layer >= 1)
        .expect("surviving soma");
    sim.set_neuron_clamp(alive_idx, OptoClamp::ChR2Excite);
    sim.step_simulation();
    assert!((sim.neurons[alive_idx].activation - sim.opto_clamp_drive).abs() < 1e-4);

    sim.heal_all_interventions();
    assert_eq!(sim.ablated_count(), 0);
    assert_eq!(sim.severed_count(), 0);
    assert_eq!(sim.clamped_count(), 0);
}

#[test]
fn m14_frontier_llm_top2_moe_gating_and_zone_isolation() {
    let mut sim = NeuralSpaceSim::new(
        ModelArch::M14FrontierLlmMoe,
        DatasetId::D7MolecularProperties,
        2026,
    );
    assert!(sim.neurons.iter().any(|n| n.role == NeuronRole::MoeRouter));
    assert!(!sim.neurons.iter().any(|n| n.role == NeuronRole::GeneratorUnit));

    let (_, l1_label, _) = classify_neuron_zone(ModelArch::M14FrontierLlmMoe, 1, NeuronRole::ExcitatoryHidden);
    let (_, l2_label, _) = classify_neuron_zone(ModelArch::M14FrontierLlmMoe, 2, NeuronRole::MoeRouter);
    let (_, l3_label, _) = classify_neuron_zone(ModelArch::M14FrontierLlmMoe, 3, NeuronRole::ExcitatoryHidden);
    assert_eq!(l1_label, "CAUSAL GQA + KV CACHE");
    assert_eq!(l2_label, "TOP-2 MoE ROUTER");
    assert_eq!(l3_label, "SwiGLU EXPERTS");

    // Preserve M4MinimaxGan's GENERATOR MANIFOLD zone name
    let (_, gan_label, _) = classify_neuron_zone(ModelArch::M4MinimaxGan, 1, NeuronRole::GeneratorUnit);
    assert_eq!(gan_label, "GENERATOR MANIFOLD");

    for _ in 0..10 {
        sim.step_simulation();
    }
    let feats = sim.active_sample.features.clone();
    let _ = sim.forward_sample(&feats, false);

    let active_experts = sim.moe_expert_gates.iter().filter(|&&g| g > 0.0).count();
    let inactive_experts = sim.moe_expert_gates.iter().filter(|&&g| g == 0.0).count();
    let gate_sum: f32 = sim.moe_expert_gates.iter().sum();
    assert_eq!(active_experts, 2);
    assert_eq!(inactive_experts, 2);
    assert!((gate_sum - 1.0).abs() < 1e-4);

    // Confirm inactive Layer-3 quadrant experts are suppressed to 0.0 activation
    for n in &sim.neurons {
        if n.layer == 3 && !n.ablated && n.opto_clamp == OptoClamp::None {
            let q = (if n.rest_pos.y >= 0.0 { 0 } else { 2 })
                + (if n.rest_pos.z >= 0.0 { 0 } else { 1 });
            if sim.moe_expert_gates[q] == 0.0 {
                assert_eq!(n.activation, 0.0);
            }
        }
    }
}

#[test]
fn m14_frontier_llm_masks_ablated_and_severed_quadrants_from_moe_routing() {
    let quadrant_of = |y: f32, z: f32| -> usize {
        (if y >= 0.0 { 0 } else { 2 }) + (if z >= 0.0 { 0 } else { 1 })
    };

    // Case 1: Q1, Q2, Q3 ablated; Q0 has negative router logit -> sole living Q0 gets gate 1.0
    let mut sim_ablate = NeuralSpaceSim::new(
        ModelArch::M14FrontierLlmMoe,
        DatasetId::D4PlantSpecies,
        123,
    );
    for n in &mut sim_ablate.neurons {
        if n.layer == 2 {
            n.bias = 1.0;
        } else if n.layer == 3 && quadrant_of(n.rest_pos.y, n.rest_pos.z) != 0 {
            n.ablated = true;
        }
    }
    for syn in &mut sim_ablate.synapses {
        if sim_ablate.neurons[syn.from].layer == 2
            && sim_ablate.neurons[syn.to].layer == 3
            && quadrant_of(
                sim_ablate.neurons[syn.to].rest_pos.y,
                sim_ablate.neurons[syn.to].rest_pos.z,
            ) == 0
        {
            syn.weight = -2.0;
        }
    }
    let feats = sim_ablate.active_sample.features.clone();
    let _ = sim_ablate.forward_sample(&feats, false);
    assert_eq!(sim_ablate.moe_expert_gates, [1.0, 0.0, 0.0, 0.0]);

    // Case 2: L2->L3 tracts severed to Q1, Q2, Q3; Q0 negative logit -> Q0 gets 1.0
    let mut sim_sever = NeuralSpaceSim::new(
        ModelArch::M14FrontierLlmMoe,
        DatasetId::D4PlantSpecies,
        123,
    );
    for n in &mut sim_sever.neurons {
        if n.layer == 2 {
            n.bias = 1.0;
        }
    }
    for syn in &mut sim_sever.synapses {
        if sim_sever.neurons[syn.from].layer == 2 && sim_sever.neurons[syn.to].layer == 3 {
            let q = quadrant_of(
                sim_sever.neurons[syn.to].rest_pos.y,
                sim_sever.neurons[syn.to].rest_pos.z,
            );
            if q == 0 {
                syn.weight = -2.0;
            } else {
                syn.severed = true;
            }
        }
    }
    let feats_sever = sim_sever.active_sample.features.clone();
    let _ = sim_sever.forward_sample(&feats_sever, false);
    assert_eq!(sim_sever.moe_expert_gates, [1.0, 0.0, 0.0, 0.0]);

    // Case 3: Two active quadrants (Q0, Q2) with negative logits; Q1/Q3 dead -> Top-2 strictly in {Q0, Q2}
    let mut sim_two = NeuralSpaceSim::new(
        ModelArch::M14FrontierLlmMoe,
        DatasetId::D4PlantSpecies,
        123,
    );
    for n in &mut sim_two.neurons {
        if n.layer == 2 {
            n.bias = 1.0;
        } else if n.layer == 3 {
            let q = quadrant_of(n.rest_pos.y, n.rest_pos.z);
            if q == 1 || q == 3 {
                n.ablated = true;
            }
        }
    }
    for syn in &mut sim_two.synapses {
        if sim_two.neurons[syn.from].layer == 2 && sim_two.neurons[syn.to].layer == 3 {
            let q = quadrant_of(
                sim_two.neurons[syn.to].rest_pos.y,
                sim_two.neurons[syn.to].rest_pos.z,
            );
            if q == 0 {
                syn.weight = -1.5;
            } else if q == 2 {
                syn.weight = -2.5;
            }
        }
    }
    let feats_two = sim_two.active_sample.features.clone();
    let _ = sim_two.forward_sample(&feats_two, false);
    assert_eq!(sim_two.moe_expert_gates[1], 0.0);
    assert_eq!(sim_two.moe_expert_gates[3], 0.0);
    assert!(sim_two.moe_expert_gates[0] > 0.0 && sim_two.moe_expert_gates[2] > 0.0);
    assert!(((sim_two.moe_expert_gates[0] + sim_two.moe_expert_gates[2]) - 1.0).abs() < 1e-5);

    // Case 4: All 4 quadrants ablated -> gates = [0.0; 4]
    let mut sim_none = sim_ablate;
    for n in &mut sim_none.neurons {
        if n.layer == 3 {
            n.ablated = true;
        }
    }
    let _ = sim_none.forward_sample(&feats, false);
    assert_eq!(sim_none.moe_expert_gates, [0.0; 4]);
}

#[test]
fn m3_sparse_transformer_attention_competition_suppresses_weak_keys() {
    let mut sim = NeuralSpaceSim::new(
        ModelArch::M3SparseTransformer,
        DatasetId::D4PlantSpecies,
        2026,
    );
    for _ in 0..10 {
        sim.step_simulation();
    }
    let feats = sim.active_sample.features.clone();
    let _ = sim.forward_sample(&feats, false);

    assert!(!sim.attn_weights.is_empty());
    assert!(sim.attn_max_weight > 0.0);
    assert!(sim.attn_suppressed_frac > 0.0 && sim.attn_suppressed_frac < 1.0);
    let zero_l1 = sim
        .neurons
        .iter()
        .filter(|n| n.layer == 1 && !n.ablated && n.activation == 0.0)
        .count();
    assert!(zero_l1 > 0);
}

#[test]
fn m10_predictive_coding_error_subtraction_and_feedback_sever() {
    let mut sim = NeuralSpaceSim::new(
        ModelArch::M10PredictiveCoding,
        DatasetId::D4PlantSpecies,
        2026,
    );
    for _ in 0..12 {
        sim.step_simulation();
    }
    let feats = sim.active_sample.features.clone();
    let _ = sim.forward_sample(&feats, false);
    assert!(sim.pc_mean_free_energy.is_finite());
    assert!(sim.pc_mean_free_energy > 0.0);

    let severed = sim.sever_feedback_tracts();
    assert!(severed > 0);
    let _ = sim.forward_sample(&feats, false);
    assert!(sim.pc_mean_free_energy.is_finite());
}

#[test]
fn workset_envelope_serializes_exact_network_and_keeps_legacy_compat() {
    let mut sim = NeuralSpaceSim::new(
        ModelArch::M13CerebralCortexGreyMatter,
        DatasetId::D4PlantSpecies,
        2026,
    );
    for _ in 0..30 {
        sim.step_simulation();
    }
    let target_n = sim
        .neurons
        .iter()
        .position(|n| !n.ablated && n.layer >= 1)
        .expect("hidden neuron");
    let clamp_n = sim
        .neurons
        .iter()
        .enumerate()
        .find(|&(i, n)| i != target_n && !n.ablated && n.layer >= 1)
        .map(|(i, _)| i)
        .expect("second hidden neuron");
    sim.toggle_ablate_neuron(target_n);
    sim.set_neuron_clamp(clamp_n, OptoClamp::ChR2Excite);
    sim.sever_feedback_tracts();
    sim.induce_focal_stroke();
    sim.opto_clamp_drive = 1.65;
    sim.synapses[0].weight = 1.777;
    assert!(!sim.lesion_events.is_empty());

    let env = WorksetEnvelope::from_sim(&sim, RenderMode::HistologicalGreyMatter3D, 0.85);
    assert!(env.verify_digest());
    assert_eq!(env.neurons.len(), sim.neurons.len());
    assert_eq!(env.synapses.len(), sim.synapses.len());
    assert!(!env.loss_history.is_empty());
    assert!(!env.metric_history.is_empty());
    assert_eq!(env.lesion_events, sim.lesion_events);
    assert_eq!(env.opto_clamp_drive, Some(sim.opto_clamp_drive));
    assert_eq!(env.metabolic_flux, Some(sim.metabolic_flux));
    assert_eq!(env.adversarial_margin, Some(sim.adversarial_margin));
    assert_eq!(env.last_prediction, sim.last_prediction);

    // File roundtrip + SQLite LabStore roundtrip
    let tmp_path = std::env::temp_dir().join("neurospace_test_workset.neuroworkset.json");
    env.save_to_file(&tmp_path).expect("save file");
    let loaded_file = WorksetEnvelope::load_from_file(&tmp_path).expect("load file");
    let _ = std::fs::remove_file(&tmp_path);

    let store = LabStore::open_in_memory().expect("in-memory sqlite");
    store.save_slot("quicksave_f5", &env).expect("save sqlite slot");
    let loaded_sqlite = store
        .load_slot("quicksave_f5")
        .expect("query sqlite slot")
        .expect("sqlite slot present");
    assert!(loaded_sqlite.verify_digest());

    for loaded_env in [&loaded_file, &loaded_sqlite] {
        let mut restored = NeuralSpaceSim::new(
            ModelArch::M1PerceptronMlp,
            DatasetId::D1ImageClassification,
            99,
        );
        for _ in 0..8 {
            restored.step_simulation();
        }
        restored.opto_clamp_drive = 0.77;
        restored.metabolic_flux = 2.41;
        restored.adversarial_margin = 12.34;
        restored.lesion_events = vec![(1, "StaleMarker".to_string())];

        restored
            .apply_workset_envelope(loaded_env)
            .expect("apply exact envelope");
        assert_eq!(restored.model, ModelArch::M13CerebralCortexGreyMatter);
        assert_eq!(restored.dataset_id, DatasetId::D4PlantSpecies);
        assert_eq!(restored.neurons.len(), sim.neurons.len());
        assert_eq!(restored.synapses.len(), sim.synapses.len());
        assert!((restored.synapses[0].weight - 1.777).abs() < 1e-6);
        assert!(restored.neurons[target_n].ablated);
        assert_eq!(restored.neurons[clamp_n].opto_clamp, OptoClamp::ChR2Excite);
        assert_eq!(restored.severed_count(), sim.severed_count());
        assert_eq!(restored.step_count, sim.step_count);
        assert_eq!(restored.loss_history, sim.loss_history);
        assert_eq!(restored.metric_history, sim.metric_history);
        assert_eq!(restored.pre_lesion_metric, sim.pre_lesion_metric);
        assert_eq!(restored.acute_lesion_metric, sim.acute_lesion_metric);
        assert_eq!(restored.lesion_events, sim.lesion_events);
        assert_eq!(restored.opto_clamp_drive.to_bits(), sim.opto_clamp_drive.to_bits());
        assert_eq!(restored.metabolic_flux.to_bits(), sim.metabolic_flux.to_bits());
        assert_eq!(restored.adversarial_margin.to_bits(), sim.adversarial_margin.to_bits());
        assert_eq!(restored.last_prediction.len(), sim.last_prediction.len());
        for (a, b) in restored.last_prediction.iter().zip(&sim.last_prediction) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (rest_n, orig_n) in restored.neurons.iter().zip(&sim.neurons) {
            assert_eq!(rest_n.activation.to_bits(), orig_n.activation.to_bits());
        }
    }

    // Legacy envelope compatibility (without neurons/synapses/R2.4 keys)
    let legacy_json_no_digest = r#"{
        "schema_version": "neurospace.workset.v1",
        "model": "M11CorticalColumnsHtm",
        "dataset": "D6RoboticsNavigation",
        "goal": "MaxTaskFidelity",
        "render_mode": "AutoAdaptive",
        "manual_organic_blend": 0.5,
        "dataset_locked": true,
        "model_locked": false,
        "params": {
            "learning_rate": 0.035,
            "momentum": 0.85,
            "prune_threshold": 0.045,
            "plasticity_growth": 0.35,
            "hidden_scale": 1.0,
            "input_noise": 0.08,
            "adversarial_eps": 0.12,
            "flow_sample_rate": 0.7,
            "batch_size": 16,
            "continual_transfer": true
        },
        "step_count": 42,
        "primary_metric": 78.5,
        "current_loss": 0.12,
        "comparison_ledger": [],
        "sha256_digest": ""
    }"#;
    let mut legacy_env: WorksetEnvelope =
        serde_json::from_str(legacy_json_no_digest).expect("parse legacy");
    legacy_env.sha256_digest = legacy_env.compute_digest();
    assert!(legacy_env.verify_digest());

    // Confirm none of the newly added fields are emitted when re-serializing a legacy envelope
    let mut legacy_no_digest = legacy_env.clone();
    legacy_no_digest.sha256_digest.clear();
    let legacy_canonical = serde_json::to_string(&legacy_no_digest).expect("canonical legacy");
    for omitted_key in [
        "lesion_events",
        "opto_clamp_drive",
        "metabolic_flux",
        "adversarial_margin",
        "last_prediction",
    ] {
        assert!(!legacy_canonical.contains(omitted_key), "unexpected key {omitted_key}");
    }

    let mut legacy_target = NeuralSpaceSim::new(
        ModelArch::M1PerceptronMlp,
        DatasetId::D1ImageClassification,
        7,
    );
    legacy_target.lesion_events = vec![(3, "StaleMarker".to_string())];
    legacy_target.opto_clamp_drive = 1.99;
    legacy_target.metabolic_flux = 2.49;
    legacy_target.adversarial_margin = 11.11;

    legacy_target
        .apply_workset_envelope(&legacy_env)
        .expect("apply legacy envelope");
    assert_eq!(legacy_target.model, ModelArch::M11CorticalColumnsHtm);
    assert_eq!(legacy_target.step_count, 42);
    assert!(legacy_target.lesion_events.is_empty());
    assert!((legacy_target.opto_clamp_drive - 1.10).abs() < 1e-6);
    assert!((legacy_target.metabolic_flux - 2.49).abs() > 1e-3);
    let expected_margin = (100.0 - 0.12_f32 * 38.0).clamp(10.0, 99.0);
    assert!((legacy_target.adversarial_margin - expected_margin).abs() < 1e-4);
}

#[test]
fn loss_metric_history_and_lesion_events_track_accurately() {
    let mut sim = NeuralSpaceSim::new(
        ModelArch::M13CerebralCortexGreyMatter,
        DatasetId::D4PlantSpecies,
        2026,
    );
    for _ in 0..30 {
        sim.step_simulation();
    }
    assert!(sim.loss_history.len() >= 6);
    assert!(sim.metric_history.len() >= 6);

    let killed = sim.induce_focal_stroke();
    assert!(killed > 0);
    assert!(sim.pre_lesion_metric.is_some());
    assert!(sim.acute_lesion_metric.is_some());
    assert!(sim.lesion_events.iter().any(|(_, tag)| tag == "Stroke"));

    sim.heal_all_interventions();
    assert!(sim.pre_lesion_metric.is_none());
    assert!(sim.acute_lesion_metric.is_none());
    assert!(sim.lesion_events.is_empty());
}

#[test]
fn app_icon_rgba_buffer_is_valid_64x64() {
    let app_icon = app_egui_icon();
    assert_eq!(app_icon.width, 64);
    assert_eq!(app_icon.height, 64);
    assert_eq!(app_icon.rgba.len(), 64 * 64 * 4);
    assert!(app_icon.rgba.iter().skip(3).step_by(4).any(|&a| a > 200));
}

#[test]
fn workset_load_from_file_surfaces_missing_corrupt_and_tampered_errors() {
    let missing = std::env::temp_dir().join("neurospace_does_not_exist_999.json");
    assert!(WorksetEnvelope::load_from_file(&missing).is_err());

    let corrupt = std::env::temp_dir().join("neurospace_corrupt_999.json");
    std::fs::write(&corrupt, "{ invalid json").expect("write corrupt");
    assert!(WorksetEnvelope::load_from_file(&corrupt).is_err());
    let _ = std::fs::remove_file(&corrupt);

    let sim = NeuralSpaceSim::new(ModelArch::M1PerceptronMlp, DatasetId::D1ImageClassification, 7);
    let mut env = WorksetEnvelope::from_sim(&sim, RenderMode::AutoAdaptive, 0.5);
    env.sha256_digest = "0000deadbeef".to_string();
    let tampered = std::env::temp_dir().join("neurospace_tampered_999.json");
    env.save_to_file(&tampered).expect("save tampered");
    let err = WorksetEnvelope::load_from_file(&tampered).expect_err("must reject bad sha256");
    assert!(err.contains("SHA-256"));
    let _ = std::fs::remove_file(&tampered);
}
