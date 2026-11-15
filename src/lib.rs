#![forbid(unsafe_code)]
//! # NeuroSpace Lab (`neurospace-lab` v1.0.0)
//!
//! Native safe-Rust 3D Neural Network Topology, Structural Plasticity & Biocomputing Lab.
//! Features 12 architectures spanning Rosenblatt perceptrons to GANs, GANNs/NEAT, spiking LIF+STDP,
//! Liquid Time-Constant (LTC) networks, Hierarchical Predictive Coding, Thousand-Brains HTM columns,
//! and Connectomic Organoid-Astrocyte tripartite networks across 7 classical & scientific datasets.

pub mod datasets;
pub mod engine;
pub mod gui;
pub mod models;
pub mod persistence;
pub mod renderer_3d;
