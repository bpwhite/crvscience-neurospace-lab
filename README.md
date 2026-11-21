# NeuroSpace Lab (`neurospace-lab` v1.6.0)

Native safe-Rust (`#![forbid(unsafe_code)]`) 3D Neural Network Topology, Structural Plasticity & Cerebral Cortex Grey-Matter Workstation built with `eframe`/`egui` `0.31`, `glam` `0.29`, bundled `rusqlite` `0.32` (WAL), and `clap` `4.5`, paired with a synchronized Three.js `r128` 3D browser observatory (`web/neurospace_lab_preview.html`).

- **Website**: [CRV Science Network](https://crvscience.com) · [CRV Science Public Apps](https://crvscience.com/apps)
- **License**: GNU General Public License v3.0 or later (`GPL-3.0-or-later`, see [LICENSE](LICENSE))

## What's New in v1.6.0

- **Calm, Sparse Macroscopic Flow Sampling**: Replaces dense per-synapse dart barrages with gentle, sine-faded flow wisps sampled across a configurable fraction of representative pathways (`~10%` default, capped at `<= 16–22` staggered waves with refractory rest intervals) so directional signal flow is clear without overwhelming stationary neurons.
- **Tier V (`M13`) — 6-Layered Cerebral Cortex Grey Matter (`L1–L6` 3D Cytoarchitecture)**:
  - Models canonical neocortical columnar laminar anatomy from **Pia Mater (`+Y`)** down to **Subcortical White Matter (`-Y`)**: `Thalamic Input -> L4 Spiny Stellate Granules -> L2/3 Association Pyramidals -> L5 Thick-Tufted Pyramidals -> L6 Corticothalamic Fusiforms & White-Matter Output`, plus `L1 Molecular` horizontal plexus, `GABAergic Basket/Martinotti` interneurons, and protoplasmic astrocytes.
  - **True 3D Anatomical Neuron Meshes**: Sculpted 3D conical/rounded pyramidal cell bodies, tapered ascending **3D Apical Dendrite Trunks** climbing up to arborize into **Layer I Apical Tufts**, curved **Basal Dendritic Skirts**, descending **Axon Hillocks**, and multipolar `L4` spiny stellate arbors.

## Build, Run & Test Locally

```bash
# Launch interactive 3D native Rust workstation
cargo run --release

# Run full unit & integration test suite
cargo test

# Headless CLI: sweep all 15 models on locked Plant Species (D4) dataset
cargo run --release -- bench-dataset --dataset D4 --steps 80

# Headless CLI: sweep all 7 datasets on locked Cerebral Cortex Grey Matter (M13)
cargo run --release -- bench-model --model M13 --steps 80
```
