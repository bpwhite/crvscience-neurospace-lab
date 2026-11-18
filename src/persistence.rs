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
use crate::engine::{ComparisonRecord, NeuralSpaceSim, NeuronNode, SynapseEdge};
use crate::models::{HyperParams, ModelArch, OutputGoal, RenderMode};
use rusqlite::{params, Connection, Error as SqlError, Result as SqlResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

/// Portable cross-compatible snapshot envelope shared between the native Rust workstation
/// and the interactive browser observatory (`.neuroworkset.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorksetEnvelope {
    pub schema_version: String,
    pub model: ModelArch,
    pub dataset: DatasetId,
    pub goal: OutputGoal,
    pub render_mode: RenderMode,
    pub manual_organic_blend: f32,
    pub dataset_locked: bool,
    pub model_locked: bool,
    pub params: HyperParams,
    pub step_count: u64,
    pub primary_metric: f32,
    pub current_loss: f32,
    pub comparison_ledger: Vec<ComparisonRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub neurons: Vec<NeuronNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synapses: Vec<SynapseEdge>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_ids: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_ids: Vec<usize>,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub epoch_count: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub pruned_total: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub sprouted_total: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub neurogenesis_total: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub apoptosis_total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_lesion_metric: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acute_lesion_metric: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loss_history: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metric_history: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synapse_history: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lesion_events: Vec<(usize, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opto_clamp_drive: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metabolic_flux: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adversarial_margin: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub last_prediction: Vec<f32>,
    pub sha256_digest: String,
}

impl WorksetEnvelope {
    pub fn from_sim(
        sim: &NeuralSpaceSim,
        render_mode: RenderMode,
        manual_organic_blend: f32,
    ) -> Self {
        let mut env = Self {
            schema_version: "neurospace.workset.v1".to_string(),
            model: sim.model,
            dataset: sim.dataset_id,
            goal: sim.goal,
            render_mode,
            manual_organic_blend,
            dataset_locked: sim.dataset_locked,
            model_locked: sim.model_locked,
            params: sim.params.clone(),
            step_count: sim.step_count,
            primary_metric: sim.primary_metric,
            current_loss: sim.current_loss,
            comparison_ledger: sim.comparison_ledger.clone(),
            neurons: sim.neurons.clone(),
            synapses: sim.synapses.clone(),
            input_ids: sim.input_ids.clone(),
            output_ids: sim.output_ids.clone(),
            epoch_count: sim.epoch_count,
            pruned_total: sim.pruned_total,
            sprouted_total: sim.sprouted_total,
            neurogenesis_total: sim.neurogenesis_total,
            apoptosis_total: sim.apoptosis_total,
            pre_lesion_metric: sim.pre_lesion_metric,
            acute_lesion_metric: sim.acute_lesion_metric,
            loss_history: sim.loss_history.clone(),
            metric_history: sim.metric_history.clone(),
            synapse_history: sim.synapse_history.clone(),
            lesion_events: sim.lesion_events.clone(),
            opto_clamp_drive: Some(sim.opto_clamp_drive),
            metabolic_flux: Some(sim.metabolic_flux),
            adversarial_margin: Some(sim.adversarial_margin),
            last_prediction: sim.last_prediction.clone(),
            sha256_digest: String::new(),
        };
        env.sha256_digest = env.compute_digest();
        env
    }

    pub fn compute_digest(&self) -> String {
        let mut clone = self.clone();
        clone.sha256_digest.clear();
        let canonical = serde_json::to_string(&clone).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn verify_digest(&self) -> bool {
        !self.sha256_digest.is_empty() && self.sha256_digest == self.compute_digest()
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create directory '{}': {e}",
                        parent.display()
                    )
                })?;
            }
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize workset JSON: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write workset file '{}': {e}", path.display()))
    }

    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read workset file '{}': {e}", path.display()))?;
        let env: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("Invalid workset JSON in '{}': {e}", path.display()))?;
        if !env.verify_digest() {
            return Err(format!(
                "SHA-256 checksum verification failed for '{}'",
                path.display()
            ));
        }
        Ok(env)
    }
}

/// Local SQLite WAL persistence store for save slots (`F5`/`F9`) and cross-model/dataset runs.
pub struct LabStore {
    conn: Connection,
}

impl LabStore {
    pub fn open<P: AsRef<Path>>(path: P) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    pub fn open_in_memory() -> SqlResult<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS save_slots (
                slot_name TEXT PRIMARY KEY,
                envelope_json TEXT NOT NULL,
                sha256_hex TEXT NOT NULL,
                updated_epoch INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS comparison_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_code TEXT NOT NULL,
                dataset_code TEXT NOT NULL,
                goal_label TEXT NOT NULL,
                lock_context TEXT NOT NULL,
                primary_score REAL NOT NULL,
                loss REAL NOT NULL,
                neuron_count INTEGER NOT NULL,
                synapse_count INTEGER NOT NULL,
                metabolic_cost REAL NOT NULL
            );",
        )
    }

    pub fn save_slot(&self, slot_name: &str, env: &WorksetEnvelope) -> SqlResult<()> {
        let payload = serde_json::to_string_pretty(env)
            .map_err(|e| SqlError::ToSqlConversionFailure(Box::new(e)))?;
        self.conn.execute(
            "INSERT INTO save_slots (slot_name, envelope_json, sha256_hex, updated_epoch)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(slot_name) DO UPDATE SET
                envelope_json = excluded.envelope_json,
                sha256_hex = excluded.sha256_hex,
                updated_epoch = excluded.updated_epoch",
            params![slot_name, payload, env.sha256_digest, env.step_count as i64],
        )?;
        Ok(())
    }

    pub fn load_slot(&self, slot_name: &str) -> SqlResult<Option<WorksetEnvelope>> {
        let mut stmt = self
            .conn
            .prepare("SELECT envelope_json FROM save_slots WHERE slot_name = ?1")?;
        let mut rows = stmt.query(params![slot_name])?;
        if let Some(row) = rows.next()? {
            let json_str: String = row.get(0)?;
            let parsed: WorksetEnvelope = serde_json::from_str(&json_str).map_err(|e| {
                SqlError::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            Ok(Some(parsed))
        } else {
            Ok(None)
        }
    }

    pub fn append_comparison(&self, rec: &ComparisonRecord) -> SqlResult<()> {
        self.conn.execute(
            "INSERT INTO comparison_runs (
                model_code, dataset_code, goal_label, lock_context,
                primary_score, loss, neuron_count, synapse_count, metabolic_cost
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                rec.model.code(),
                rec.dataset.code(),
                rec.goal.label(),
                rec.lock_context,
                rec.primary_score,
                rec.loss,
                rec.neuron_count as i64,
                rec.synapse_count as i64,
                rec.metabolic_cost,
            ],
        )?;
        Ok(())
    }
}
