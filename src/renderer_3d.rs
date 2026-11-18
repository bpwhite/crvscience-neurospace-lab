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

use crate::engine::{NeuralSpaceSim, NeuronRole, OptoClamp};
use crate::models::{ModelArch, RenderMode};
use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, StrokeKind};
use glam::{Mat4, Vec3, Vec4};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct OrbitCamera3D {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub auto_orbit: bool,
}

impl Default for OrbitCamera3D {
    fn default() -> Self {
        Self {
            yaw: -0.52,
            pitch: 0.24,
            distance: 25.0,
            target: Vec3::ZERO,
            auto_orbit: true,
        }
    }
}

impl OrbitCamera3D {
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let cp = self.pitch.cos();
        let eye = self.target
            + Vec3::new(
                self.distance * cp * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * cp * self.yaw.cos(),
            );
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh_gl(42.0f32.to_radians(), aspect.max(0.25), 0.5, 160.0);
        proj * view
    }

    pub fn project(&self, vp: &Mat4, rect: Rect, world: Vec3) -> Option<(Pos2, f32)> {
        let clip: Vec4 = *vp * world.extend(1.0);
        if clip.w <= 0.2 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() > 1.35 || ndc.y.abs() > 1.35 {
            return None;
        }
        let sx = rect.center().x + ndc.x * (rect.width() * 0.5);
        let sy = rect.center().y - ndc.y * (rect.height() * 0.5);
        Some((Pos2::new(sx, sy), clip.w))
    }
}

/// Adaptive 3D Neural Space Renderer (v1.2.0):
/// - Calm, sparse representative flow wisps (`sampled_tracer` + sine-bell fade) instead of barrage darts
/// - Tier V Histological 3D Grey Matter mode: sculpted 3D pyramidal somata, ascending 3D apical trunks
///   arborizing into L1 apical tufts, basal dendritic skirts, spiny stellate stars, and Pia/White-Matter planes.
#[allow(clippy::too_many_arguments)]
pub fn draw_neural_space_3d(
    painter: &Painter,
    rect: Rect,
    sim: &NeuralSpaceSim,
    cam: &OrbitCamera3D,
    render_mode: RenderMode,
    manual_blend: f32,
    show_group_highlights: bool,
    hovered_out: &mut Option<usize>,
    pointer_pos: Option<Pos2>,
    selected_neuron: Option<usize>,
) {
    let morph = render_mode.resolve_organic_factor(sim.model, manual_blend);
    let organic = morph.clamp(0.0, 1.0);
    let histological = morph > 1.08;

    let bg_color = if histological {
        Color32::from_rgb(8, 14, 20)
    } else {
        lerp_color(
            Color32::from_rgb(9, 14, 25),
            Color32::from_rgb(6, 17, 23),
            organic,
        )
    };
    painter.rect_filled(rect, 6.0, bg_color);

    let aspect = rect.width() / rect.height().max(1.0);
    let vp = cam.view_proj(aspect);

    // 1. Reference Scaffold (Pia Mater + White Matter Lamina Planes in Histological Mode)
    if histological {
        draw_cortical_surface_planes(painter, rect, cam, &vp);
    } else if organic < 0.72 {
        let grid_alpha = ((1.0 - organic * 1.15).clamp(0.12, 0.85) * 40.0) as u8;
        let stroke = Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(70, 120, 180, grid_alpha));
        for g in -6..=6 {
            let gz = g as f32 * 2.0;
            if let (Some((p0, _)), Some((p1, _))) = (
                cam.project(&vp, rect, Vec3::new(-12.0, -5.8, gz)),
                cam.project(&vp, rect, Vec3::new(12.0, -5.8, gz)),
            ) {
                painter.line_segment([p0, p1], stroke);
            }
            let gx = g as f32 * 2.0;
            if let (Some((p0, _)), Some((p1, _))) = (
                cam.project(&vp, rect, Vec3::new(gx, -5.8, -12.0)),
                cam.project(&vp, rect, Vec3::new(gx, -5.8, 12.0)),
            ) {
                painter.line_segment([p0, p1], stroke);
            }
        }
    } else {
        let ring_stroke = Stroke::new(
            1.1_f32,
            Color32::from_rgba_unmultiplied(40, 190, 155, (organic * 36.0) as u8),
        );
        for r_idx in 1..=4 {
            let rad = r_idx as f32 * 3.1;
            let mut prev: Option<Pos2> = None;
            for seg in 0..=36 {
                let a = (seg as f32 / 36.0) * std::f32::consts::TAU;
                let pt = Vec3::new(a.cos() * rad * 1.35, -5.6, a.sin() * rad);
                if let Some((sp, _)) = cam.project(&vp, rect, pt) {
                    if let Some(p0) = prev {
                        painter.line_segment([p0, sp], ring_stroke);
                    }
                    prev = Some(sp);
                } else {
                    prev = None;
                }
            }
        }
    }

    // 2. 3D Group / Cortical Lamina Enclosures (wireframe boxes pre-synapse; pills deferred post-soma)
    let zone_pills = if show_group_highlights {
        draw_group_zone_boxes(painter, rect, sim, cam, &vp, organic)
    } else {
        Vec::new()
    };

    let mut projected: Vec<Option<(Pos2, f32)>> = Vec::with_capacity(sim.neurons.len());
    for n in &sim.neurons {
        projected.push(cam.project(&vp, rect, n.pos));
    }

    // 3. Braided Information River Channels: Compute inter-group confluence centroids so
    //    individual pathways bundle into organic river trunks & fan out into downstream deltas,
    //    with continuous downstream sinusoidal wave currents along each braided filament.
    let mut layer_centers: BTreeMap<usize, (Vec3, usize)> = BTreeMap::new();
    for n in &sim.neurons {
        let e = layer_centers.entry(n.layer).or_insert((Vec3::ZERO, 0));
        e.0 += n.pos;
        e.1 += 1;
    }
    let centroids: BTreeMap<usize, Vec3> = layer_centers
        .into_iter()
        .map(|(k, (sum, cnt))| (k, sum / cnt.max(1) as f32))
        .collect();

    let river_intensity = sim.params.flow_sample_rate.clamp(0.0, 1.0);
    let river_phase = sim.step_count as f32 * 0.045;
    let bundle_strength = if histological || organic > 0.45 { 0.68 } else { 0.34 };

    for (s_idx, syn) in sim.synapses.iter().enumerate() {
        if syn.severed || sim.neurons[syn.from].ablated || sim.neurons[syn.to].ablated {
            if let (Some((pf, _)), Some((pt, _))) = (projected[syn.from], projected[syn.to]) {
                painter.line_segment(
                    [pf, pt],
                    Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(100, 116, 139, 26)),
                );
            }
            continue;
        }
        let (Some((p_from, d_from)), Some((p_to, d_to))) =
            (projected[syn.from], projected[syn.to])
        else {
            continue;
        };
        let depth = (d_from + d_to) * 0.5;
        let persp_w = (20.0 / depth.max(8.0)).clamp(0.45, 1.45);
        let w_mag = syn.weight.abs().clamp(0.02, 2.4);

        // Non-sampled structural edges render as quiet, thin, low-alpha background filaments
        // so the <= 22 sampled tracer river wisps remain calm and legible.
        if !syn.sampled_tracer || river_intensity <= 0.01 {
            let bg_alpha = (11.0 + w_mag * 12.0).clamp(10.0, 26.0) as u8;
            let bg_width = (0.42_f32 + w_mag * 0.24) * persp_w;
            let bg_col = if syn.is_feedback {
                if syn.weight >= 0.0 {
                    Color32::from_rgba_unmultiplied(217, 159, 64, bg_alpha)
                } else {
                    Color32::from_rgba_unmultiplied(219, 112, 168, bg_alpha)
                }
            } else if syn.weight >= 0.0 {
                Color32::from_rgba_unmultiplied(72, 182, 210, bg_alpha)
            } else {
                Color32::from_rgba_unmultiplied(225, 108, 128, bg_alpha)
            };
            painter.line_segment([p_from, p_to], Stroke::new(bg_width, bg_col));
            continue;
        }

        let u_pos = sim.neurons[syn.from].pos;
        let v_pos = sim.neurons[syn.to].pos;
        let c_up = centroids.get(&sim.neurons[syn.from].layer).copied().unwrap_or(u_pos);
        let c_dn = centroids.get(&sim.neurons[syn.to].layer).copied().unwrap_or(v_pos);

        // Spatially separated Forward vs. Reverse-Feedback river corridors so bidirectional
        // streams form clear 3D counter-current loops instead of overlapping!
        let fb_sign = if syn.is_feedback { -1.0 } else { 1.0 };
        let fb_arch = if syn.is_feedback { Vec3::new(0.0, 0.85, -1.35) } else { Vec3::ZERO };
        let meander = Vec3::new(
            0.0,
            ((sim.neurons[syn.from].layer as f32 * 1.7).sin()) * 0.50 * fb_sign,
            ((sim.neurons[syn.to].layer as f32 * 2.3).cos()) * 0.70 * fb_sign,
        ) + fb_arch;
        let trunk_q1 = c_up.lerp(c_dn, 0.33) + meander + syn.curve_offset * 0.22;
        let trunk_q2 = c_up.lerp(c_dn, 0.67) + fb_arch * 0.8 - meander * 0.45 + syn.curve_offset * 0.22;
        let direct_q1 = u_pos.lerp(v_pos, 0.33) + syn.curve_offset * 0.35;
        let direct_q2 = u_pos.lerp(v_pos, 0.67) + syn.curve_offset * 0.35;

        let q1_w = direct_q1.lerp(trunk_q1, bundle_strength);
        let q2_w = direct_q2.lerp(trunk_q2, bundle_strength);

        let p_q1 = cam.project(&vp, rect, q1_w).map(|(p, _)| p).unwrap_or(p_from.lerp(p_to, 0.33));
        let p_q2 = cam.project(&vp, rect, q2_w).map(|(p, _)| p).unwrap_or(p_from.lerp(p_to, 0.67));

        let segs = 14usize;
        let fil_phase = (s_idx as f32) * 0.73;
        let mut prev_pt = p_from;

        for s in 1..=segs {
            let u0 = (s - 1) as f32 / segs as f32;
            let u1 = s as f32 / segs as f32;
            let umid = (u0 + u1) * 0.5;
            let cur_pt = cubic_bezier_2d(p_from, p_q1, p_q2, p_to, u1);

            // Smooth endpoint taper sin(pi * u) so river swells in channel without flashing somata
            let env = (umid * std::f32::consts::PI).sin().powf(0.75);
            let wave1 = (umid * 8.5 - river_phase + fil_phase).sin() * 0.5 + 0.5;
            let wave2 = (umid * 15.0 - river_phase * 1.45 + fil_phase * 1.7).sin() * 0.5 + 0.5;
            let current = (0.35 + 0.65 * (0.62 * wave1 + 0.38 * wave2)) * env * river_intensity;

            let base_alpha = 52.0 + w_mag * 58.0;
            let flow_alpha = current * (85.0 + w_mag * 75.0);
            let alpha = (base_alpha + flow_alpha).clamp(42.0, 245.0) as u8;
            let width = ((0.45 + w_mag * 0.65) + current * 1.15) * persp_w;

            let seg_col = if syn.is_feedback {
                if syn.weight >= 0.0 {
                    // Excitatory Top-Down / Recurrent Reverse Feedback: Warm Amber-Gold -> Heliotrope
                    lerp_color(
                        Color32::from_rgba_unmultiplied(245, 158, 11, alpha),
                        Color32::from_rgba_unmultiplied(192, 132, 252, alpha),
                        current.clamp(0.0, 1.0),
                    )
                } else {
                    // Inhibitory Reverse Feedback: Magenta-Coral
                    Color32::from_rgba_unmultiplied(244, 114, 182, alpha)
                }
            } else if syn.weight >= 0.0 {
                lerp_color(
                    Color32::from_rgba_unmultiplied(34, 197, 235, alpha),
                    Color32::from_rgba_unmultiplied(110, 231, 183, alpha),
                    (organic * 0.65 + current * 0.35).clamp(0.0, 1.0),
                )
            } else {
                Color32::from_rgba_unmultiplied(251, 113, 133, alpha)
            };
            painter.line_segment([prev_pt, cur_pt], Stroke::new(width, seg_col));
            prev_pt = cur_pt;
        }

        // Suspended laminar river droplets drifting smoothly along the <= 22 sampled tracer wisps
        if river_intensity > 0.05 && w_mag > 0.14 {
            let u_mote = ((river_phase * 0.14 + fil_phase * 0.21) % 1.0).clamp(0.06, 0.94);
            let mote_env = (u_mote * std::f32::consts::PI).sin();
            let mote_pt = cubic_bezier_2d(p_from, p_q1, p_q2, p_to, u_mote);
            let mote_alpha = (mote_env * river_intensity * 215.0).clamp(0.0, 245.0) as u8;
            let mote_col = if syn.weight >= 0.0 {
                Color32::from_rgba_unmultiplied(167, 243, 208, mote_alpha)
            } else {
                Color32::from_rgba_unmultiplied(253, 164, 175, mote_alpha)
            };
            painter.circle_filled(mote_pt, (2.2 * persp_w).clamp(1.2, 3.2), mote_col);
        }
    }

    // 4. Depth-sort Neurons back-to-front and render adaptive or 3D Histological Grey-Matter Neurons
    let mut order: Vec<(usize, Pos2, f32)> = projected
        .iter()
        .enumerate()
        .filter_map(|(idx, p)| p.map(|(pos, depth)| (idx, pos, depth)))
        .collect();
    order.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_hover: Option<(usize, f32)> = None;

    for (idx, screen_pos, depth) in order {
        let n = &sim.neurons[idx];
        let persp = (22.0 / depth.max(6.0)).clamp(0.42, 2.5);
        let base_r = 8.2 * n.soma_scale * persp;

        if let Some(ptr) = pointer_pos {
            let d = ptr.distance(screen_pos);
            if d <= base_r.max(10.0) * 1.35 && best_hover.is_none_or(|(_, bd)| d < bd) {
                best_hover = Some((idx, d));
            }
        }

        let act_norm = n.smooth_activation.clamp(0.0, 1.0);
        let core_color = if n.ablated {
            Color32::from_rgb(71, 85, 105)
        } else {
            role_color(n.role, act_norm, organic)
        };
        if n.opto_clamp == OptoClamp::ChR2Excite {
            painter.circle_stroke(screen_pos, base_r * 1.48, Stroke::new(2.2_f32, Color32::from_rgb(56, 189, 248)));
        } else if n.opto_clamp == OptoClamp::NpHRSilence {
            painter.circle_stroke(screen_pos, base_r * 1.48, Stroke::new(2.2_f32, Color32::from_rgb(192, 132, 252)));
        } else if n.ablated {
            painter.line_segment([screen_pos - egui::vec2(base_r*0.7, base_r*0.7), screen_pos + egui::vec2(base_r*0.7, base_r*0.7)], Stroke::new(1.8_f32, Color32::from_rgb(244, 63, 94)));
        }
        let is_selected = selected_neuron == Some(idx);

        if histological && n.role != NeuronRole::SensoryInput && n.role != NeuronRole::ReadoutOutput {
            draw_histological_neuron_3d(
                painter,
                rect,
                cam,
                &vp,
                n,
                screen_pos,
                base_r,
                persp,
                core_color,
                act_norm,
                is_selected,
            );
            continue;
        }

        if n.role == NeuronRole::AstrocyteGlia {
            let halo_r = base_r * (2.3 + (sim.step_count as f32 * 0.012 + n.phase_seed).sin() * 0.08);
            painter.circle_filled(
                screen_pos,
                halo_r,
                Color32::from_rgba_unmultiplied(45, 212, 191, 34),
            );
        }

        if organic < 0.28 {
            let half = base_r * 0.96;
            let outer_rect = Rect::from_center_size(screen_pos, egui::vec2(half * 2.0, half * 2.0));
            painter.rect_filled(outer_rect, 2.5, Color32::from_rgba_unmultiplied(12, 20, 36, 235));
            painter.rect_stroke(
                outer_rect,
                2.5,
                Stroke::new(
                    if is_selected { 2.6_f32 } else { 1.6_f32 },
                    if is_selected { Color32::WHITE } else { core_color },
                ),
                StrokeKind::Outside,
            );
            let core_half = half * (0.48 + act_norm * 0.36);
            let inner_rect = Rect::from_center_size(screen_pos, egui::vec2(core_half * 2.0, core_half * 2.0));
            painter.rect_filled(inner_rect, 1.5, core_color);
        } else {
            if organic > 0.55 && n.role != NeuronRole::SensoryInput {
                let branches = 5usize;
                let dend_stroke = Stroke::new(
                    (1.2 * persp).clamp(0.7, 2.0),
                    Color32::from_rgba_unmultiplied(core_color.r(), core_color.g(), core_color.b(), 140),
                );
                for b in 0..branches {
                    let ang = n.phase_seed
                        + (b as f32 / branches as f32) * std::f32::consts::TAU
                        + (sim.step_count as f32 * 0.03).sin() * 0.15;
                    let mid = Pos2::new(
                        screen_pos.x + ang.cos() * base_r * 1.45,
                        screen_pos.y + ang.sin() * base_r * 1.45,
                    );
                    let tip_a = Pos2::new(
                        mid.x + (ang + 0.35).cos() * base_r * 0.65 * organic,
                        mid.y + (ang + 0.35).sin() * base_r * 0.65 * organic,
                    );
                    let tip_b = Pos2::new(
                        mid.x + (ang - 0.35).cos() * base_r * 0.65 * organic,
                        mid.y + (ang - 0.35).sin() * base_r * 0.65 * organic,
                    );
                    painter.line_segment([screen_pos, mid], dend_stroke);
                    painter.line_segment([mid, tip_a], dend_stroke);
                    painter.line_segment([mid, tip_b], dend_stroke);
                }
            }

            painter.circle_filled(
                screen_pos,
                base_r * 1.05,
                Color32::from_rgba_unmultiplied(8, 15, 26, 225),
            );
            painter.circle_filled(
                screen_pos,
                base_r,
                Color32::from_rgba_unmultiplied(core_color.r(), core_color.g(), core_color.b(), 88),
            );
            painter.circle_stroke(
                screen_pos,
                base_r,
                Stroke::new(
                    if is_selected { 2.4_f32 } else { 1.5_f32 },
                    if is_selected { Color32::WHITE } else { core_color },
                ),
            );
            painter.circle_filled(
                screen_pos,
                base_r * (0.44 + act_norm * 0.34),
                Color32::from_rgb(
                    core_color.r().saturating_add(45),
                    core_color.g().saturating_add(45),
                    core_color.b().saturating_add(45),
                ),
            );
        }
    }

    // 5. Render 3D Lamina / Zone Pills AFTER synapses and somata with 2D AABB de-overlapping
    if show_group_highlights {
        draw_group_zone_pills(painter, rect, zone_pills);
    }

    *hovered_out = best_hover.map(|(idx, _)| idx);
}

/// Renders true 3D anatomical Grey-Matter Neurons projected in world coordinates:
/// - Pyramidal Neurons (L2/3, L5, or general hidden in Histological mode): 3D triangular/conical pyramidal
///   soma + ascending 3D Apical Dendrite Trunk rising to Layer I (`Y = +4.7`) + 3D Apical Tuft crown +
///   5 curved 3D Basal Dendrites + descending Axon Initial Segment!
/// - Spiny Stellate / Interneurons / Astrocytes: 3D multipolar/fusiform arbors.
#[allow(clippy::too_many_arguments)]
fn draw_histological_neuron_3d(
    painter: &Painter,
    rect: Rect,
    cam: &OrbitCamera3D,
    vp: &Mat4,
    n: &crate::engine::NeuronNode,
    screen_pos: Pos2,
    base_r: f32,
    persp: f32,
    core_color: Color32,
    act_norm: f32,
    is_selected: bool,
) {
    let arbor_col = Color32::from_rgba_unmultiplied(
        core_color.r(),
        core_color.g(),
        core_color.b(),
        185,
    );
    let tuft_col = Color32::from_rgba_unmultiplied(
        core_color.r().saturating_add(30),
        core_color.g().saturating_add(35),
        core_color.b().saturating_add(25),
        165,
    );

    let is_pyr = n.role.is_pyramidal()
        || matches!(n.role, NeuronRole::ExcitatoryHidden | NeuronRole::CorticalMiniColumn);

    if is_pyr {
        // 1. Ascending 3D Apical Dendritic Trunk toward Pia / Layer I (Y = +4.65)
        let tuft_y = (n.pos.y + if n.role == NeuronRole::ThickPyramidalL5 { 6.2 } else { 2.4 }).min(4.75);
        let apex_world = n.pos + Vec3::new(0.0, 0.55, 0.0);
        let trunk_top_world = Vec3::new(n.pos.x, tuft_y, n.pos.z);

        if let (Some((p_apex, _)), Some((p_top, _))) = (
            cam.project(vp, rect, apex_world),
            cam.project(vp, rect, trunk_top_world),
        ) {
            painter.line_segment(
                [p_apex, p_top],
                Stroke::new((2.1 * persp).clamp(1.1, 3.4), arbor_col),
            );

            // 3D Apical Tuft Crown fanning out in Layer I
            for branch in 0..4 {
                let a = n.phase_seed + (branch as f32 / 4.0) * std::f32::consts::TAU;
                let tip_world = trunk_top_world + Vec3::new(a.cos() * 1.15, 0.52, a.sin() * 1.15);
                if let Some((p_tip, _)) = cam.project(vp, rect, tip_world) {
                    painter.line_segment(
                        [p_top, p_tip],
                        Stroke::new((1.2 * persp).clamp(0.7, 2.0), tuft_col),
                    );
                }
            }
        }

        // 2. 3D Basal Dendritic Skirt (5 branches radiating outward & slightly downward from soma base)
        for b in 0..5 {
            let a = n.phase_seed + (b as f32 / 5.0) * std::f32::consts::TAU;
            let basal_mid = n.pos + Vec3::new(a.cos() * 0.85, -0.28, a.sin() * 0.85);
            let basal_tip = n.pos + Vec3::new((a + 0.25).cos() * 1.45, -0.52, (a + 0.25).sin() * 1.45);
            if let (Some((pm, _)), Some((pt, _))) = (
                cam.project(vp, rect, basal_mid),
                cam.project(vp, rect, basal_tip),
            ) {
                painter.line_segment([screen_pos, pm], Stroke::new((1.3 * persp).clamp(0.8, 2.1), arbor_col));
                painter.line_segment([pm, pt], Stroke::new((0.9 * persp).clamp(0.6, 1.6), arbor_col));
            }
        }

        // 3. Descending Axon Initial Segment (AIS)
        if let Some((p_ais, _)) = cam.project(vp, rect, n.pos + Vec3::new(0.0, -1.05, 0.0)) {
            painter.line_segment(
                [screen_pos, p_ais],
                Stroke::new(
                    (1.0 * persp).clamp(0.6, 1.5),
                    Color32::from_rgba_unmultiplied(250, 204, 21, 130),
                ),
            );
        }

        // 4. Sculpted Pyramidal Soma Silhouette (Conical Apex + Rounded Basal Perikaryon + Nucleus)
        let h_up = base_r * 1.45;
        let w_base = base_r * 0.98;
        let pts = vec![
            Pos2::new(screen_pos.x, screen_pos.y - h_up),
            Pos2::new(screen_pos.x + w_base, screen_pos.y + base_r * 0.72),
            Pos2::new(screen_pos.x - w_base, screen_pos.y + base_r * 0.72),
        ];
        painter.add(Shape::convex_polygon(
            pts,
            Color32::from_rgba_unmultiplied(core_color.r(), core_color.g(), core_color.b(), 155),
            Stroke::new(
                if is_selected { 2.4_f32 } else { 1.4_f32 },
                if is_selected { Color32::WHITE } else { core_color },
            ),
        ));
        painter.circle_filled(
            Pos2::new(screen_pos.x, screen_pos.y + base_r * 0.15),
            base_r * (0.38 + act_norm * 0.25),
            Color32::from_rgb(236, 253, 245),
        );
    } else {
        // Spiny Stellate (L4), Basket/Martinotti Interneuron, or Protoplasmic Astrocyte
        let rays = if n.role == NeuronRole::AstrocyteGlia { 8 } else { 6 };
        let reach = if n.role == NeuronRole::AstrocyteGlia { 1.55 } else { 1.15 };
        for r in 0..rays {
            let a = n.phase_seed + (r as f32 / rays as f32) * std::f32::consts::TAU;
            let z_tilt = ((r as f32) * 1.7).sin() * 0.65;
            let tip_w = n.pos + Vec3::new(a.cos() * reach, z_tilt * 0.7, a.sin() * reach);
            if let Some((pt, _)) = cam.project(vp, rect, tip_w) {
                painter.line_segment(
                    [screen_pos, pt],
                    Stroke::new((1.15 * persp).clamp(0.7, 1.9), arbor_col),
                );
            }
        }
        painter.circle_filled(
            screen_pos,
            base_r * 0.88,
            Color32::from_rgba_unmultiplied(core_color.r(), core_color.g(), core_color.b(), 140),
        );
        painter.circle_stroke(
            screen_pos,
            base_r * 0.88,
            Stroke::new(if is_selected { 2.3_f32 } else { 1.4_f32 }, core_color),
        );
        painter.circle_filled(
            screen_pos,
            base_r * 0.36,
            Color32::from_rgb(240, 249, 255),
        );
    }
}

fn draw_cortical_surface_planes(
    painter: &Painter,
    rect: Rect,
    cam: &OrbitCamera3D,
    vp: &Mat4,
) {
    // Pia Mater top boundary (Y = +5.3) & Subcortical White Matter boundary (Y = -4.6)
    for (y_plane, col) in [
        (5.3f32, Color32::from_rgba_unmultiplied(45, 212, 191, 46)),
        (-4.6f32, Color32::from_rgba_unmultiplied(148, 163, 184, 42)),
    ] {
        let corners = [
            Vec3::new(-7.5, y_plane, -5.5),
            Vec3::new(7.5, y_plane, -5.5),
            Vec3::new(7.5, y_plane, 5.5),
            Vec3::new(-7.5, y_plane, 5.5),
        ];
        for i in 0..4 {
            if let (Some((pa, _)), Some((pb, _))) = (
                cam.project(vp, rect, corners[i]),
                cam.project(vp, rect, corners[(i + 1) % 4]),
            ) {
                painter.line_segment([pa, pb], Stroke::new(1.2_f32, col));
            }
        }
    }
}

struct ZonePill {
    pos: Pos2,
    depth: f32,
    text: String,
    col: Color32,
}

fn draw_group_zone_boxes(
    painter: &Painter,
    rect: Rect,
    sim: &NeuralSpaceSim,
    cam: &OrbitCamera3D,
    vp: &Mat4,
    organic: f32,
) -> Vec<ZonePill> {
    let mut groups: BTreeMap<usize, (String, Color32, Vec3, Vec3, usize)> = BTreeMap::new();

    for n in &sim.neurons {
        let (key, label, col) = classify_neuron_zone(sim.model, n.layer, n.role);
        let entry = groups.entry(key).or_insert_with(|| {
            (
                label,
                col,
                n.pos - Vec3::splat(0.85),
                n.pos + Vec3::splat(0.85),
                0,
            )
        });
        entry.2 = entry.2.min(n.pos - Vec3::splat(0.85));
        entry.3 = entry.3.max(n.pos + Vec3::splat(0.85));
        entry.4 += 1;
    }

    let mut pills = Vec::with_capacity(groups.len());
    for (_k, (title, col, min_v, max_v, count)) in groups {
        let corners = [
            Vec3::new(min_v.x, min_v.y, min_v.z),
            Vec3::new(max_v.x, min_v.y, min_v.z),
            Vec3::new(max_v.x, max_v.y, min_v.z),
            Vec3::new(min_v.x, max_v.y, min_v.z),
            Vec3::new(min_v.x, min_v.y, max_v.z),
            Vec3::new(max_v.x, min_v.y, max_v.z),
            Vec3::new(max_v.x, max_v.y, max_v.z),
            Vec3::new(min_v.x, max_v.y, max_v.z),
        ];
        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];

        let wire_alpha = if organic > 0.65 { 46 } else { 72 };
        let stroke = Stroke::new(
            1.0_f32,
            Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), wire_alpha),
        );
        for (a, b) in edges {
            if let (Some((pa, _)), Some((pb, _))) = (
                cam.project(vp, rect, corners[a]),
                cam.project(vp, rect, corners[b]),
            ) {
                painter.line_segment([pa, pb], stroke);
            }
        }

        let top_anchor = Vec3::new(
            (min_v.x + max_v.x) * 0.5,
            max_v.y + 0.65,
            (min_v.z + max_v.z) * 0.5,
        );
        if let Some((label_pos, depth)) = cam.project(vp, rect, top_anchor) {
            pills.push(ZonePill {
                pos: label_pos,
                depth,
                text: format!("{} ({})", title, count),
                col,
            });
        }
    }
    pills
}

fn draw_group_zone_pills(painter: &Painter, rect: Rect, mut pills: Vec<ZonePill>) {
    // Sort front-to-back during layout so foreground labels stay close to their true 3D anchors
    pills.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));

    let mut placed: Vec<(Rect, ZonePill)> = Vec::with_capacity(pills.len());
    let pill_h = 18.0_f32;
    let pad = 6.0_f32;

    for mut pill in pills {
        let pill_w = (pill.text.len() as f32 * 6.5 + 14.0).clamp(84.0, 236.0);
        let min_x = rect.min.x + pill_w * 0.5 + pad;
        let max_x = (rect.max.x - pill_w * 0.5 - pad).max(min_x);
        let min_y = rect.min.y + pill_h * 0.5 + pad;
        let max_y = (rect.max.y - pill_h * 0.5 - pad).max(min_y);

        let anchor = Pos2::new(
            pill.pos.x.clamp(min_x, max_x),
            pill.pos.y.clamp(min_y, max_y),
        );
        let mut chosen = Rect::from_center_size(anchor, egui::vec2(pill_w, pill_h));

        if placed.iter().any(|(r, _)| r.intersects(chosen)) {
            let mut found = false;
            for step in 1..=14 {
                let dy = step as f32 * 21.0;
                for dir in [-1.0_f32, 1.0_f32] {
                    let cand_y = (anchor.y + dir * dy).clamp(min_y, max_y);
                    let cand = Rect::from_center_size(
                        Pos2::new(anchor.x, cand_y),
                        egui::vec2(pill_w, pill_h),
                    );
                    if !placed.iter().any(|(r, _)| r.intersects(cand)) {
                        chosen = cand;
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                for step in 1..=6 {
                    let dx = step as f32 * (pill_w * 0.52);
                    for dir in [-1.0_f32, 1.0_f32] {
                        let cand_x = (anchor.x + dir * dx).clamp(min_x, max_x);
                        let cand = Rect::from_center_size(
                            Pos2::new(cand_x, anchor.y),
                            egui::vec2(pill_w, pill_h),
                        );
                        if !placed.iter().any(|(r, _)| r.intersects(cand)) {
                            chosen = cand;
                            found = true;
                            break;
                        }
                    }
                    if found {
                        break;
                    }
                }
            }
        }
        pill.pos = chosen.center();
        placed.push((chosen, pill));
    }

    // Paint back-to-front so closer pills draw cleanly on top if viewport is extremely narrow
    placed.sort_by(|a, b| {
        b.1.depth
            .partial_cmp(&a.1.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (pill_rect, pill) in placed {
        painter.rect_filled(
            pill_rect,
            4.0,
            Color32::from_rgba_unmultiplied(10, 16, 30, 226),
        );
        painter.rect_stroke(
            pill_rect,
            4.0,
            Stroke::new(
                1.0_f32,
                Color32::from_rgba_unmultiplied(pill.col.r(), pill.col.g(), pill.col.b(), 175),
            ),
            StrokeKind::Outside,
        );
        painter.text(
            pill.pos,
            Align2::CENTER_CENTER,
            pill.text,
            FontId::monospace(10.2),
            pill.col,
        );
    }
}

pub fn classify_neuron_zone(model: ModelArch, layer: usize, role: NeuronRole) -> (usize, String, Color32) {
    if model == ModelArch::M13CerebralCortexGreyMatter {
        return match role {
            NeuronRole::SensoryInput => (0, "THALAMIC INPUT".to_string(), Color32::from_rgb(56, 189, 248)),
            NeuronRole::MolecularInterneuronL1 => (1, "L1 MOLECULAR PLEXUS".to_string(), Color32::from_rgb(167, 243, 208)),
            NeuronRole::PyramidalL23 => (2, "L2/3 PYRAMIDAL".to_string(), Color32::from_rgb(52, 211, 153)),
            NeuronRole::SpinyStellateL4 => (3, "L4 GRANULAR STELLATE".to_string(), Color32::from_rgb(56, 189, 248)),
            NeuronRole::ThickPyramidalL5 => (4, "L5 THICK-TUFTED PYR".to_string(), Color32::from_rgb(45, 212, 191)),
            NeuronRole::CorticothalamicL6 => (5, "L6 CORTICOTHALAMIC".to_string(), Color32::from_rgb(129, 140, 248)),
            NeuronRole::ReadoutOutput => (90, "WHITE-MATTER OUTPUT".to_string(), Color32::from_rgb(250, 204, 21)),
            _ => (6, "CORTICAL INTERNEURONS".to_string(), Color32::from_rgb(244, 114, 182)),
        };
    }

    if model == ModelArch::M14FrontierLlmMoe {
        return match (layer, role) {
            (_, NeuronRole::SensoryInput) => (0, "INPUT LAYER".to_string(), Color32::from_rgb(56, 189, 248)),
            (_, NeuronRole::ReadoutOutput) => (90, "OUTPUT READOUT".to_string(), Color32::from_rgb(250, 204, 21)),
            (1, _) => (1, "CAUSAL GQA + KV CACHE".to_string(), Color32::from_rgb(52, 211, 153)),
            (2, _) | (_, NeuronRole::MoeRouter) => {
                (2, "TOP-2 MoE ROUTER".to_string(), Color32::from_rgb(192, 132, 252))
            }
            (3, _) => (3, "SwiGLU EXPERTS".to_string(), Color32::from_rgb(45, 212, 191)),
            _ => (layer, format!("HIDDEN L{}", layer), Color32::from_rgb(52, 211, 153)),
        };
    }

    match role {
        NeuronRole::SensoryInput => (0, "INPUT LAYER".to_string(), Color32::from_rgb(56, 189, 248)),
        NeuronRole::ReadoutOutput => (90, "OUTPUT READOUT".to_string(), Color32::from_rgb(250, 204, 21)),
        NeuronRole::GeneratorUnit => (10, "GENERATOR MANIFOLD".to_string(), Color32::from_rgb(192, 132, 252)),
        NeuronRole::MoeRouter => (12, "TOP-2 MoE ROUTER".to_string(), Color32::from_rgb(192, 132, 252)),
        NeuronRole::DiscriminatorUnit => (20, "DISCRIMINATOR CRITIC".to_string(), Color32::from_rgb(251, 146, 60)),
        NeuronRole::PredictionErrorUnit => (25, "ERROR LAMINA".to_string(), Color32::from_rgb(251, 113, 133)),
        _ => match model {
            ModelArch::M7SpikingLifStdp if role == NeuronRole::InhibitoryInterneuron => {
                (22, "INHIBITORY RING".to_string(), Color32::from_rgb(244, 114, 182))
            }
            ModelArch::M8LiquidTimeConstant => {
                (15, "LIQUID ODE CORE".to_string(), Color32::from_rgb(52, 211, 153))
            }
            ModelArch::M9EchoStateReservoir => {
                (15, "CHAOTIC RESERVOIR".to_string(), Color32::from_rgb(167, 139, 250))
            }
            ModelArch::M10PredictiveCoding => {
                (12, "VALUE LAMINA".to_string(), Color32::from_rgb(52, 211, 153))
            }
            ModelArch::M12ConnectomicOrganoid => {
                (15, "ORGANOID + GLIA".to_string(), Color32::from_rgb(45, 212, 191))
            }
            _ => (
                layer,
                format!("HIDDEN L{}", layer),
                Color32::from_rgb(52, 211, 153),
            ),
        },
    }
}

fn cubic_bezier_2d(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
    let u = 1.0 - t;
    let uu = u * u;
    let uuu = uu * u;
    let tt = t * t;
    let ttt = tt * t;
    Pos2::new(
        uuu * p0.x + 3.0 * uu * t * p1.x + 3.0 * u * tt * p2.x + ttt * p3.x,
        uuu * p0.y + 3.0 * uu * t * p1.y + 3.0 * u * tt * p2.y + ttt * p3.y,
    )
}

#[allow(dead_code)]
fn quad_bezier_2d(p0: Pos2, p1: Pos2, p2: Pos2, t: f32) -> Pos2 {
    let omt = 1.0 - t;
    Pos2::new(
        omt * omt * p0.x + 2.0 * omt * t * p1.x + t * t * p2.x,
        omt * omt * p0.y + 2.0 * omt * t * p1.y + t * t * p2.y,
    )
}

fn role_color(role: NeuronRole, act_norm: f32, organic: f32) -> Color32 {
    let boost = (act_norm * 32.0) as u8;
    match role {
        NeuronRole::SensoryInput => Color32::from_rgb(56u8.saturating_add(boost), 189, 248),
        NeuronRole::ExcitatoryHidden | NeuronRole::PyramidalL23 => lerp_color(
            Color32::from_rgb(59u8.saturating_add(boost), 130, 246),
            Color32::from_rgb(16u8.saturating_add(boost), 185, 129),
            organic,
        ),
        NeuronRole::SpinyStellateL4 => Color32::from_rgb(56u8.saturating_add(boost), 205, 232),
        NeuronRole::ThickPyramidalL5 => Color32::from_rgb(20u8.saturating_add(boost), 212, 185),
        NeuronRole::CorticothalamicL6 => Color32::from_rgb(129u8.saturating_add(boost / 2), 140, 248),
        NeuronRole::MolecularInterneuronL1 => Color32::from_rgb(167, 243, 208),
        NeuronRole::InhibitoryInterneuron => Color32::from_rgb(236, 72u8.saturating_add(boost / 2), 153),
        NeuronRole::ReadoutOutput => Color32::from_rgb(250, 204, 21u8.saturating_add(boost)),
        NeuronRole::GeneratorUnit | NeuronRole::MoeRouter => {
            Color32::from_rgb(168, 85, 247u8.saturating_add(boost / 2))
        }
        NeuronRole::DiscriminatorUnit => Color32::from_rgb(249, 115, 22u8.saturating_add(boost)),
        NeuronRole::PredictionErrorUnit => Color32::from_rgb(251, 113, 133),
        NeuronRole::CorticalMiniColumn => Color32::from_rgb(45u8.saturating_add(boost), 212, 191),
        NeuronRole::AstrocyteGlia => Color32::from_rgb(134, 239, 172),
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    Color32::from_rgba_unmultiplied(
        (a.r() as f32 * inv + b.r() as f32 * t) as u8,
        (a.g() as f32 * inv + b.g() as f32 * t) as u8,
        (a.b() as f32 * inv + b.b() as f32 * t) as u8,
        (a.a() as f32 * inv + b.a() as f32 * t) as u8,
    )
}
