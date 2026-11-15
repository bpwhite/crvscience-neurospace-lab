#![forbid(unsafe_code)]

use neurospace_lab::datasets::DatasetId;
use neurospace_lab::engine::{NeuralSpaceSim, NeuronRole};
use neurospace_lab::models::{ModelArch, OutputGoal, RenderMode};
use neurospace_lab::persistence::{LabStore, WorksetEnvelope};

#[test]
fn all_thirteen_models_including_cortex_grey_matter_step_cleanly() {
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

        for _ in 0..24 {
            sim.step_simulation();
        }
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
