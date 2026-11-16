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
