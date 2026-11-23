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
    pub vel_yaw: f32,
    pub vel_pitch: f32,
    pub vel_distance: f32,
    pub vel_target: Vec3,
    pub preset_goal: Option<(f32, f32, f32, Vec3)>,
}

impl Default for OrbitCamera3D {
    fn default() -> Self {
        Self {
            yaw: -0.52,
            pitch: 0.24,
            distance: 25.0,
            target: Vec3::ZERO,
            auto_orbit: true,
            vel_yaw: 0.0,
            vel_pitch: 0.0,
            vel_distance: 0.0,
            vel_target: Vec3::ZERO,
            preset_goal: None,
        }
    }
}

impl OrbitCamera3D {
    pub fn eye_position(&self) -> Vec3 {
        let cp = self.pitch.cos();
        self.target
            + Vec3::new(
                self.distance * cp * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * cp * self.yaw.cos(),
            )
    }

    pub fn screen_axes(&self) -> (Vec3, Vec3) {
        let eye = self.eye_position();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        (right, up)
    }

    pub fn apply_orbit_delta(&mut self, dx: f32, dy: f32) {
        self.auto_orbit = false;
        self.preset_goal = None;
        let dyaw = -dx * 0.008;
        let dpitch = dy * 0.008;
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-1.35, 1.35);
        self.vel_yaw = (self.vel_yaw * 0.4 + dyaw * 24.0).clamp(-6.0, 6.0);
        self.vel_pitch = (self.vel_pitch * 0.4 + dpitch * 24.0).clamp(-4.5, 4.5);
    }

    pub fn apply_pan_delta(&mut self, dx: f32, dy: f32, viewport_h: f32) {
        self.auto_orbit = false;
        self.preset_goal = None;
        let (right, up) = self.screen_axes();
        let world_per_px =
            2.0 * self.distance * (21.0f32.to_radians().tan()) / viewport_h.max(200.0);
        let shift = (-right * dx + up * dy) * world_per_px;
        self.target = (self.target + shift).clamp(Vec3::splat(-18.0), Vec3::splat(18.0));
        self.vel_target = (self.vel_target * 0.35 + shift * 22.0).clamp(Vec3::splat(-30.0), Vec3::splat(30.0));
    }

    pub fn apply_zoom_delta(&mut self, scroll_y: f32) {
        self.preset_goal = None;
        let dd = -scroll_y * 0.035;
        self.distance = (self.distance + dd).clamp(8.5, 54.0);
        self.vel_distance = (self.vel_distance * 0.4 + dd * 20.0).clamp(-45.0, 45.0);
    }

    pub fn set_preset(&mut self, yaw: f32, pitch: f32, distance: f32, target: Vec3, auto_orbit: bool) {
        self.auto_orbit = auto_orbit;
        self.vel_yaw = 0.0;
        self.vel_pitch = 0.0;
        self.vel_distance = 0.0;
        self.vel_target = Vec3::ZERO;
        self.preset_goal = Some((yaw, pitch.clamp(-1.35, 1.35), distance.clamp(8.5, 54.0), target));
    }

    pub fn reset(&mut self) {
        self.set_preset(-0.52, 0.24, 25.0, Vec3::ZERO, true);
    }

    /// Integrates exponential velocity damping and smooth camera preset transitions.
    /// Returns `true` while camera motion is still settling.
    pub fn damp_step(&mut self, dt: f32) -> bool {
        let dt = dt.clamp(0.001, 0.05);
        if let Some((gy, gp, gd, gt)) = self.preset_goal {
            let alpha = 1.0 - (-12.0 * dt).exp();
            let mut dy = gy - self.yaw;
            while dy > std::f32::consts::PI {
                dy -= std::f32::consts::TAU;
            }
            while dy < -std::f32::consts::PI {
                dy += std::f32::consts::TAU;
            }
            self.yaw += dy * alpha;
            self.pitch = (self.pitch + (gp - self.pitch) * alpha).clamp(-1.35, 1.35);
            self.distance = (self.distance + (gd - self.distance) * alpha).clamp(8.5, 54.0);
            self.target = self.target.lerp(gt, alpha);
            if dy.abs() < 1e-3
                && (gp - self.pitch).abs() < 1e-3
                && (gd - self.distance).abs() < 1e-2
                && self.target.distance(gt) < 1e-2
            {
                self.yaw = gy;
                self.pitch = gp;
                self.distance = gd;
                self.target = gt;
                self.preset_goal = None;
                return false;
            }
            return true;
        }

        let moving = self.vel_yaw.abs() > 1e-3
            || self.vel_pitch.abs() > 1e-3
            || self.vel_distance.abs() > 1e-2
            || self.vel_target.length_squared() > 1e-4;
        if !moving {
            self.vel_yaw = 0.0;
            self.vel_pitch = 0.0;
            self.vel_distance = 0.0;
            self.vel_target = Vec3::ZERO;
            return false;
        }

        let decay = (-11.0 * dt).exp();
        self.yaw += self.vel_yaw * dt;
        self.pitch = (self.pitch + self.vel_pitch * dt).clamp(-1.35, 1.35);
        self.distance = (self.distance + self.vel_distance * dt).clamp(8.5, 54.0);
        self.target =
            (self.target + self.vel_target * dt).clamp(Vec3::splat(-18.0), Vec3::splat(18.0));
        self.vel_yaw *= decay;
        self.vel_pitch *= decay;
        self.vel_distance *= decay;
        self.vel_target *= decay;
        true
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.eye_position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh_gl(42.0f32.to_radians(), aspect.max(0.25), 0.5, 160.0);
        proj * view
    }

    pub fn project(&self, vp: &Mat4, rect: Rect, world: Vec3) -> Option<(Pos2, f32)> {
        let clip: Vec4 = *vp * world.extend(1.0);
        if clip.w <= 0.25 || clip.w + clip.z < 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() > 1.25 || ndc.y.abs() > 1.25 {
            return None;
        }
        let sx = rect.center().x + ndc.x * (rect.width() * 0.5);
        let sy = rect.center().y - ndc.y * (rect.height() * 0.5);
        Some((Pos2::new(sx, sy), clip.w))
    }

    /// Clips a 3D world-space line segment in 4D homogeneous clip space against the near plane
    /// and padded frustum boundaries so edges entering/exiting the viewport never pop out.
    pub fn project_segment(
        &self,
        vp: &Mat4,
        rect: Rect,
        a: Vec3,
        b: Vec3,
    ) -> Option<((Pos2, f32), (Pos2, f32))> {
        let c0 = *vp * a.extend(1.0);
        let c1 = *vp * b.extend(1.0);
        let mut t0 = 0.0f32;
        let mut t1 = 1.0f32;

        let clip_planes: [Vec4; 6] = [
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 0.0, 1.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.12),
            Vec4::new(-1.0, 0.0, 0.0, 1.12),
            Vec4::new(0.0, 1.0, 0.0, 1.12),
            Vec4::new(0.0, -1.0, 0.0, 1.12),
        ];

        for (idx, plane) in clip_planes.iter().enumerate() {
            let bias = if idx == 0 { -0.25 } else { 0.0 };
            let d0 = c0.dot(*plane) + bias;
            let d1 = c1.dot(*plane) + bias;
            if d0 < 0.0 && d1 < 0.0 {
                return None;
            }
            if d0 < 0.0 {
                t0 = t0.max(d0 / (d0 - d1));
            } else if d1 < 0.0 {
                t1 = t1.min(d0 / (d0 - d1));
            }
            if t0 > t1 {
                return None;
            }
        }

        let ca = c0.lerp(c1, t0);
        let cb = c0.lerp(c1, t1);
        if ca.w <= 0.2 || cb.w <= 0.2 {
            return None;
        }
        let ndc_a = ca.truncate() / ca.w;
        let ndc_b = cb.truncate() / cb.w;
        let pa = Pos2::new(
            rect.center().x + ndc_a.x * (rect.width() * 0.5),
            rect.center().y - ndc_a.y * (rect.height() * 0.5),
        );
        let pb = Pos2::new(
            rect.center().x + ndc_b.x * (rect.width() * 0.5),
            rect.center().y - ndc_b.y * (rect.height() * 0.5),
        );
        Some(((pa, ca.w), (pb, cb.w)))
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
    let eye = cam.eye_position();

    // 1. Reference Scaffold (Pia Mater + White Matter Lamina Planes in Histological Mode)
    if histological {
        draw_cortical_surface_planes(painter, rect, cam, &vp);
    } else if organic < 0.72 {
        let grid_alpha = ((1.0 - organic * 1.15).clamp(0.12, 0.85) * 40.0) as u8;
        let stroke = Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(70, 120, 180, grid_alpha));
        for g in -6..=6 {
            let gz = g as f32 * 2.0;
            if let Some(((p0, _), (p1, _))) = cam.project_segment(
                &vp,
                rect,
                Vec3::new(-12.0, -5.8, gz),
                Vec3::new(12.0, -5.8, gz),
            ) {
                painter.line_segment([p0, p1], stroke);
            }
            let gx = g as f32 * 2.0;
            if let Some(((p0, _), (p1, _))) = cam.project_segment(
                &vp,
                rect,
                Vec3::new(gx, -5.8, -12.0),
                Vec3::new(gx, -5.8, 12.0),
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
            for seg in 0..36 {
                let a0 = (seg as f32 / 36.0) * std::f32::consts::TAU;
                let a1 = ((seg + 1) as f32 / 36.0) * std::f32::consts::TAU;
                let pt0 = Vec3::new(a0.cos() * rad * 1.35, -5.6, a0.sin() * rad);
                let pt1 = Vec3::new(a1.cos() * rad * 1.35, -5.6, a1.sin() * rad);
                if let Some(((p0, _), (p1, _))) = cam.project_segment(&vp, rect, pt0, pt1) {
                    painter.line_segment([p0, p1], ring_stroke);
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
    //    with continuous downstream sinusoidal wave currents evaluated in 3D prior to projection.
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
        let u_pos = sim.neurons[syn.from].pos;
        let v_pos = sim.neurons[syn.to].pos;

        if syn.severed || sim.neurons[syn.from].ablated || sim.neurons[syn.to].ablated {
            if let Some(((pf, df), (pt, dt))) = cam.project_segment(&vp, rect, u_pos, v_pos) {
                let col = apply_depth_fog(
                    Color32::from_rgba_unmultiplied(100, 116, 139, 26),
                    bg_color,
                    (df + dt) * 0.5,
                    cam.distance,
                );
                painter.line_segment([pf, pt], Stroke::new(0.6_f32, col));
            }
            continue;
        }
        let w_mag = syn.weight.abs().clamp(0.02, 2.4);

        // Non-sampled structural edges render as quiet, thin, low-alpha background filaments
        // so the <= 22 sampled tracer river wisps remain calm and legible.
        if !syn.sampled_tracer || river_intensity <= 0.01 {
            if let Some(((p_from, d_from), (p_to, d_to))) =
                cam.project_segment(&vp, rect, u_pos, v_pos)
            {
                let depth = (d_from + d_to) * 0.5;
                let persp_w = (20.0 / depth.max(8.0)).clamp(0.45, 1.45);
                let bg_alpha = (11.0 + w_mag * 12.0).clamp(10.0, 26.0) as u8;
                let bg_width = (0.42_f32 + w_mag * 0.24) * persp_w;
                let bg_raw = if syn.is_feedback {
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
                let bg_col = apply_depth_fog(bg_raw, bg_color, depth, cam.distance);
                painter.line_segment([p_from, p_to], Stroke::new(bg_width, bg_col));
            }
            continue;
        }

        let c_up = centroids.get(&sim.neurons[syn.from].layer).copied().unwrap_or(u_pos);
        let c_dn = centroids.get(&sim.neurons[syn.to].layer).copied().unwrap_or(v_pos);

        // Spatially separated Forward vs. Reverse-Feedback river corridors so bidirectional
        // streams form clear 3D counter-current loops instead of overlapping.
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

        let segs = 14usize;
        let fil_phase = (s_idx as f32) * 0.73;

        for s in 1..=segs {
            let u0 = (s - 1) as f32 / segs as f32;
            let u1 = s as f32 / segs as f32;
            let umid = (u0 + u1) * 0.5;
            // Evaluate 3D cubic Bezier in world space before perspective projection + near-plane clipping
            let w0 = cubic_bezier_3d(u_pos, q1_w, q2_w, v_pos, u0);
            let w1 = cubic_bezier_3d(u_pos, q1_w, q2_w, v_pos, u1);
            let Some(((p0, d0), (p1, d1))) = cam.project_segment(&vp, rect, w0, w1) else {
                continue;
            };
            let seg_depth = (d0 + d1) * 0.5;
            let seg_persp = (20.0 / seg_depth.max(8.0)).clamp(0.45, 1.45);

            // Smooth endpoint taper sin(pi * u) so river swells in channel without flashing somata
            let env = (umid * std::f32::consts::PI).sin().powf(0.75);
            let wave1 = (umid * 8.5 - river_phase + fil_phase).sin() * 0.5 + 0.5;
            let wave2 = (umid * 15.0 - river_phase * 1.45 + fil_phase * 1.7).sin() * 0.5 + 0.5;
            let current = (0.35 + 0.65 * (0.62 * wave1 + 0.38 * wave2)) * env * river_intensity;

            let base_alpha = 52.0 + w_mag * 58.0;
            let flow_alpha = current * (85.0 + w_mag * 75.0);
            let alpha = (base_alpha + flow_alpha).clamp(42.0, 245.0) as u8;
            let width = ((0.45 + w_mag * 0.65) + current * 1.15) * seg_persp;

            let seg_raw = if syn.is_feedback {
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
            let seg_col = apply_depth_fog(seg_raw, bg_color, seg_depth, cam.distance);
            painter.line_segment([p0, p1], Stroke::new(width, seg_col));
        }

        // Suspended laminar river droplets evaluated in 3D drifting smoothly along sampled wisps
        if river_intensity > 0.05 && w_mag > 0.14 {
            let u_mote = ((river_phase * 0.14 + fil_phase * 0.21) % 1.0).clamp(0.06, 0.94);
            let mote_env = (u_mote * std::f32::consts::PI).sin();
            let mote_world = cubic_bezier_3d(u_pos, q1_w, q2_w, v_pos, u_mote);
            if let Some((mote_pt, mote_depth)) = cam.project(&vp, rect, mote_world) {
                let mote_persp = (20.0 / mote_depth.max(8.0)).clamp(0.45, 1.45);
                let mote_alpha = (mote_env * river_intensity * 215.0).clamp(0.0, 245.0) as u8;
                let mote_raw = if syn.weight >= 0.0 {
                    Color32::from_rgba_unmultiplied(167, 243, 208, mote_alpha)
                } else {
                    Color32::from_rgba_unmultiplied(253, 164, 175, mote_alpha)
                };
                let mote_col = apply_depth_fog(mote_raw, bg_color, mote_depth, cam.distance);
                painter.circle_filled(mote_pt, (2.2 * mote_persp).clamp(1.2, 3.2), mote_col);
            }
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
        let raw_core = if n.ablated {
            Color32::from_rgb(71, 85, 105)
        } else {
            role_color(n.role, act_norm, organic)
        };
        let core_color = apply_depth_fog(raw_core, bg_color, depth, cam.distance);

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
                eye,
                n,
                screen_pos,
                base_r,
                persp,
                depth,
                core_color,
                bg_color,
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
                apply_depth_fog(
                    Color32::from_rgba_unmultiplied(45, 212, 191, 34),
                    bg_color,
                    depth,
                    cam.distance,
                ),
            );
        }

        if organic < 0.28 {
            draw_lit_octahedron_3d(
                painter,
                rect,
                cam,
                &vp,
                eye,
                n.pos,
                0.38 * n.soma_scale,
                n.phase_seed,
                raw_core,
                bg_color,
                act_norm,
                is_selected,
                screen_pos,
                base_r,
            );
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

/// Renders true 3D anatomical Grey-Matter Neurons projected in world coordinates with
/// directional Lambertian lighting, tapered 3D apical cylinder shafts, lit 3D pyramidal prisms,
/// and depth-fog attenuation.
#[allow(clippy::too_many_arguments)]
fn draw_histological_neuron_3d(
    painter: &Painter,
    rect: Rect,
    cam: &OrbitCamera3D,
    vp: &Mat4,
    eye: Vec3,
    n: &crate::engine::NeuronNode,
    screen_pos: Pos2,
    base_r: f32,
    persp: f32,
    depth: f32,
    core_color: Color32,
    bg_color: Color32,
    act_norm: f32,
    is_selected: bool,
) {
    let arbor_col = apply_depth_fog(
        Color32::from_rgba_unmultiplied(core_color.r(), core_color.g(), core_color.b(), 185),
        bg_color,
        depth,
        cam.distance,
    );
    let tuft_col = apply_depth_fog(
        Color32::from_rgba_unmultiplied(
            core_color.r().saturating_add(30),
            core_color.g().saturating_add(35),
            core_color.b().saturating_add(25),
            165,
        ),
        bg_color,
        depth,
        cam.distance,
    );

    let is_pyr = n.role.is_pyramidal()
        || matches!(n.role, NeuronRole::ExcitatoryHidden | NeuronRole::CorticalMiniColumn);

    if is_pyr {
        // 1. Ascending Lit Tapered 3D Apical Dendritic Trunk toward Pia / Layer I (Y = +4.65)
        let tuft_y = (n.pos.y + if n.role == NeuronRole::ThickPyramidalL5 { 6.2 } else { 2.4 }).min(4.75);
        let apex_world = n.pos + Vec3::new(0.0, 0.45 * n.soma_scale, 0.0);
        let trunk_top_world = Vec3::new(n.pos.x, tuft_y, n.pos.z);

        draw_lit_tapered_cylinder_3d(
            painter,
            rect,
            cam,
            vp,
            eye,
            apex_world,
            trunk_top_world,
            0.11 * n.soma_scale,
            0.04 * n.soma_scale,
            core_color,
            bg_color,
            act_norm,
        );

        // 3D Apical Tuft Crown fanning out in Layer I
        for branch in 0..4 {
            let a = n.phase_seed + (branch as f32 / 4.0) * std::f32::consts::TAU;
            let tip_world = trunk_top_world + Vec3::new(a.cos() * 1.15, 0.52, a.sin() * 1.15);
            if let Some(((p_top, _), (p_tip, _))) =
                cam.project_segment(vp, rect, trunk_top_world, tip_world)
            {
                painter.line_segment(
                    [p_top, p_tip],
                    Stroke::new((1.2 * persp).clamp(0.7, 2.0), tuft_col),
                );
            }
        }

        // 2. 3D Basal Dendritic Skirt (5 branches radiating outward & slightly downward from soma base)
        for b in 0..5 {
            let a = n.phase_seed + (b as f32 / 5.0) * std::f32::consts::TAU;
            let basal_mid = n.pos + Vec3::new(a.cos() * 0.85, -0.28, a.sin() * 0.85);
            let basal_tip = n.pos + Vec3::new((a + 0.25).cos() * 1.45, -0.52, (a + 0.25).sin() * 1.45);
            if let Some(((ps, _), (pm, _))) = cam.project_segment(vp, rect, n.pos, basal_mid) {
                painter.line_segment([ps, pm], Stroke::new((1.3 * persp).clamp(0.8, 2.1), arbor_col));
            }
            if let Some(((pm, _), (pt, _))) = cam.project_segment(vp, rect, basal_mid, basal_tip) {
                painter.line_segment([pm, pt], Stroke::new((0.9 * persp).clamp(0.6, 1.6), arbor_col));
            }
        }

        // 3. Descending Axon Initial Segment (AIS)
        if let Some(((ps, _), (p_ais, d_ais))) =
            cam.project_segment(vp, rect, n.pos, n.pos + Vec3::new(0.0, -1.05, 0.0))
        {
            let ais_col = apply_depth_fog(
                Color32::from_rgba_unmultiplied(250, 204, 21, 130),
                bg_color,
                d_ais,
                cam.distance,
            );
            painter.line_segment([ps, p_ais], Stroke::new((1.0 * persp).clamp(0.6, 1.5), ais_col));
        }

        // 4. Lit 3D Pyramidal Soma Prism + Active Nucleus
        draw_lit_pyramid_3d(
            painter,
            rect,
            cam,
            vp,
            eye,
            n.pos,
            0.46 * n.soma_scale,
            n.phase_seed,
            core_color,
            bg_color,
            act_norm,
            is_selected,
            screen_pos,
            base_r,
        );
        painter.circle_filled(
            screen_pos,
            base_r * (0.30 + act_norm * 0.22),
            apply_depth_fog(Color32::from_rgb(236, 253, 245), bg_color, depth, cam.distance),
        );
    } else {
        // Spiny Stellate (L4), Basket/Martinotti Interneuron, or Protoplasmic Astrocyte
        let rays = if n.role == NeuronRole::AstrocyteGlia { 8 } else { 6 };
        let reach = if n.role == NeuronRole::AstrocyteGlia { 1.55 } else { 1.15 };
        for r in 0..rays {
            let a = n.phase_seed + (r as f32 / rays as f32) * std::f32::consts::TAU;
            let z_tilt = ((r as f32) * 1.7).sin() * 0.65;
            let tip_w = n.pos + Vec3::new(a.cos() * reach, z_tilt * 0.7, a.sin() * reach);
            if let Some(((ps, _), (pt, _))) = cam.project_segment(vp, rect, n.pos, tip_w) {
                painter.line_segment(
                    [ps, pt],
                    Stroke::new((1.15 * persp).clamp(0.7, 1.9), arbor_col),
                );
            }
        }
        draw_lit_octahedron_3d(
            painter,
            rect,
            cam,
            vp,
            eye,
            n.pos,
            0.38 * n.soma_scale,
            n.phase_seed,
            core_color,
            bg_color,
            act_norm,
            is_selected,
            screen_pos,
            base_r,
        );
        painter.circle_filled(
            screen_pos,
            base_r * 0.30,
            apply_depth_fog(Color32::from_rgb(240, 249, 255), bg_color, depth, cam.distance),
        );
    }
}

fn apply_depth_fog(col: Color32, bg: Color32, depth: f32, cam_dist: f32) -> Color32 {
    let near_d = (cam_dist * 0.72).max(6.0);
    let span_d = (cam_dist * 1.45).max(14.0);
    let fog = ((depth - near_d) / span_d).clamp(0.0, 0.68);
    lerp_color(
        col,
        Color32::from_rgba_unmultiplied(bg.r(), bg.g(), bg.b(), col.a()),
        fog,
    )
}

fn shade_face_color(base: Color32, normal: Vec3, act_norm: f32, alpha: u8) -> Color32 {
    let light_dir = Vec3::new(0.42, 0.82, 0.39).normalize();
    let ndotl = normal.dot(light_dir).max(0.0);
    let rim = (1.0 - normal.y.abs()).powf(1.5) * 0.14;
    let lum = (0.36 + 0.64 * ndotl + rim + act_norm * 0.22).clamp(0.25, 1.28);
    Color32::from_rgba_unmultiplied(
        (base.r() as f32 * lum).clamp(0.0, 255.0) as u8,
        (base.g() as f32 * lum).clamp(0.0, 255.0) as u8,
        (base.b() as f32 * lum).clamp(0.0, 255.0) as u8,
        alpha,
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_lit_tapered_cylinder_3d(
    painter: &Painter,
    rect: Rect,
    cam: &OrbitCamera3D,
    vp: &Mat4,
    eye: Vec3,
    p0: Vec3,
    p1: Vec3,
    r0: f32,
    r1: f32,
    base_col: Color32,
    bg_color: Color32,
    act_norm: f32,
) {
    let axis = p1 - p0;
    let len = axis.length();
    if len < 1e-4 {
        return;
    }
    let dir = axis / len;
    let ref_up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = dir.cross(ref_up).normalize_or_zero();
    let v = dir.cross(u).normalize_or_zero();
    let mid_center = (p0 + p1) * 0.5;

    let mut ring0 = [Vec3::ZERO; 4];
    let mut ring1 = [Vec3::ZERO; 4];
    for k in 0..4 {
        let a = (k as f32 / 4.0) * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let offset = u * a.cos() + v * a.sin();
        ring0[k] = p0 + offset * r0;
        ring1[k] = p1 + offset * r1;
    }

    let mut faces: Vec<(f32, Vec<Pos2>, Color32)> = Vec::with_capacity(4);
    for k in 0..4 {
        let kn = (k + 1) % 4;
        let verts = [ring0[k], ring0[kn], ring1[kn], ring1[k]];
        let fc = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
        let mut normal = (verts[1] - verts[0]).cross(verts[3] - verts[0]).normalize_or_zero();
        if normal.dot(fc - mid_center) < 0.0 {
            normal = -normal;
        }
        if (eye - fc).dot(normal) <= -0.05 {
            continue;
        }
        let mut scr = Vec::with_capacity(4);
        let mut depth_sum = 0.0f32;
        let mut ok = true;
        for w in verts {
            if let Some((sp, d)) = cam.project(vp, rect, w) {
                scr.push(sp);
                depth_sum += d;
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            let f_depth = depth_sum * 0.25;
            let lit = shade_face_color(base_col, normal, act_norm, 200);
            let fogged = apply_depth_fog(lit, bg_color, f_depth, cam.distance);
            faces.push((f_depth, scr, fogged));
        }
    }

    if faces.is_empty() {
        if let Some(((sa, da), (sb, db))) = cam.project_segment(vp, rect, p0, p1) {
            let d = (da + db) * 0.5;
            let persp = (22.0 / d.max(6.0)).clamp(0.42, 2.5);
            painter.line_segment(
                [sa, sb],
                Stroke::new((2.0 * persp).clamp(1.0, 3.2), apply_depth_fog(base_col, bg_color, d, cam.distance)),
            );
        }
        return;
    }

    faces.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, pts, col) in faces {
        painter.add(Shape::convex_polygon(pts, col, Stroke::NONE));
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_lit_pyramid_3d(
    painter: &Painter,
    rect: Rect,
    cam: &OrbitCamera3D,
    vp: &Mat4,
    eye: Vec3,
    center: Vec3,
    radius: f32,
    phase: f32,
    base_col: Color32,
    bg_color: Color32,
    act_norm: f32,
    is_selected: bool,
    fallback_pos: Pos2,
    fallback_r: f32,
) {
    let apex = center + Vec3::new(0.0, radius * 1.18, 0.0);
    let base_y = center.y - radius * 0.55;
    let mut base = [Vec3::ZERO; 4];
    for k in 0..4 {
        let a = phase + (k as f32) * std::f32::consts::FRAC_PI_2;
        base[k] = Vec3::new(
            center.x + a.cos() * radius * 0.88,
            base_y,
            center.z + a.sin() * radius * 0.88,
        );
    }

    let mut tris: Vec<(f32, Vec<Pos2>, Color32)> = Vec::with_capacity(5);
    for k in 0..4 {
        let kn = (k + 1) % 4;
        let verts = [apex, base[k], base[kn]];
        let fc = (verts[0] + verts[1] + verts[2]) / 3.0;
        let mut normal = (verts[1] - verts[0]).cross(verts[2] - verts[0]).normalize_or_zero();
        if normal.dot(fc - center) < 0.0 {
            normal = -normal;
        }
        if (eye - fc).dot(normal) <= 0.0 {
            continue;
        }
        if let (Some((p0, d0)), Some((p1, d1)), Some((p2, d2))) = (
            cam.project(vp, rect, verts[0]),
            cam.project(vp, rect, verts[1]),
            cam.project(vp, rect, verts[2]),
        ) {
            let f_depth = (d0 + d1 + d2) / 3.0;
            let lit = shade_face_color(base_col, normal, act_norm, 224);
            let fogged = apply_depth_fog(lit, bg_color, f_depth, cam.distance);
            tris.push((f_depth, vec![p0, p1, p2], fogged));
        }
    }

    // Bottom quad
    let base_fc = (base[0] + base[1] + base[2] + base[3]) * 0.25;
    let base_norm = -Vec3::Y;
    if (eye - base_fc).dot(base_norm) > 0.0 {
        if let (Some((p0, d0)), Some((p1, d1)), Some((p2, d2)), Some((p3, d3))) = (
            cam.project(vp, rect, base[0]),
            cam.project(vp, rect, base[1]),
            cam.project(vp, rect, base[2]),
            cam.project(vp, rect, base[3]),
        ) {
            let f_depth = (d0 + d1 + d2 + d3) * 0.25;
            let lit = shade_face_color(base_col, base_norm, act_norm, 220);
            let fogged = apply_depth_fog(lit, bg_color, f_depth, cam.distance);
            tris.push((f_depth, vec![p0, p1, p2, p3], fogged));
        }
    }

    if tris.is_empty() {
        let h_up = fallback_r * 1.45;
        let w_base = fallback_r * 0.98;
        let pts = vec![
            Pos2::new(fallback_pos.x, fallback_pos.y - h_up),
            Pos2::new(fallback_pos.x + w_base, fallback_pos.y + fallback_r * 0.72),
            Pos2::new(fallback_pos.x - w_base, fallback_pos.y + fallback_r * 0.72),
        ];
        painter.add(Shape::convex_polygon(
            pts,
            Color32::from_rgba_unmultiplied(base_col.r(), base_col.g(), base_col.b(), 155),
            Stroke::new(
                if is_selected { 2.4_f32 } else { 1.4_f32 },
                if is_selected { Color32::WHITE } else { base_col },
            ),
        ));
        return;
    }

    tris.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let edge_stroke = Stroke::new(
        if is_selected { 1.8_f32 } else { 0.85_f32 },
        if is_selected {
            Color32::WHITE
        } else {
            Color32::from_rgba_unmultiplied(
                base_col.r().saturating_add(35),
                base_col.g().saturating_add(35),
                base_col.b().saturating_add(35),
                170,
            )
        },
    );
    for (_, pts, col) in tris {
        painter.add(Shape::convex_polygon(pts, col, edge_stroke));
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_lit_octahedron_3d(
    painter: &Painter,
    rect: Rect,
    cam: &OrbitCamera3D,
    vp: &Mat4,
    eye: Vec3,
    center: Vec3,
    radius: f32,
    phase: f32,
    base_col: Color32,
    bg_color: Color32,
    act_norm: f32,
    is_selected: bool,
    fallback_pos: Pos2,
    fallback_r: f32,
) {
    let c = phase.cos();
    let s = phase.sin();
    let rx = Vec3::new(c * radius, 0.0, s * radius);
    let rz = Vec3::new(-s * radius, 0.0, c * radius);
    let ry = Vec3::new(0.0, radius, 0.0);

    let verts = [
        center + ry,  // 0: top
        center - ry,  // 1: bottom
        center + rx,  // 2
        center + rz,  // 3
        center - rx,  // 4
        center - rz,  // 5
    ];
    let ring = [2usize, 3, 4, 5];
    let mut faces: Vec<(f32, Vec<Pos2>, Color32)> = Vec::with_capacity(8);

    for k in 0..4 {
        let kn = (k + 1) % 4;
        for pole in [0usize, 1usize] {
            let tri = [verts[pole], verts[ring[k]], verts[ring[kn]]];
            let fc = (tri[0] + tri[1] + tri[2]) / 3.0;
            let mut normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
            if normal.dot(fc - center) < 0.0 {
                normal = -normal;
            }
            if (eye - fc).dot(normal) <= 0.0 {
                continue;
            }
            if let (Some((p0, d0)), Some((p1, d1)), Some((p2, d2))) = (
                cam.project(vp, rect, tri[0]),
                cam.project(vp, rect, tri[1]),
                cam.project(vp, rect, tri[2]),
            ) {
                let f_depth = (d0 + d1 + d2) / 3.0;
                let lit = shade_face_color(base_col, normal, act_norm, 224);
                let fogged = apply_depth_fog(lit, bg_color, f_depth, cam.distance);
                faces.push((f_depth, vec![p0, p1, p2], fogged));
            }
        }
    }

    if faces.is_empty() {
        painter.circle_filled(
            fallback_pos,
            fallback_r * 0.88,
            Color32::from_rgba_unmultiplied(base_col.r(), base_col.g(), base_col.b(), 150),
        );
        painter.circle_stroke(
            fallback_pos,
            fallback_r * 0.88,
            Stroke::new(
                if is_selected { 2.3_f32 } else { 1.4_f32 },
                if is_selected { Color32::WHITE } else { base_col },
            ),
        );
        return;
    }

    faces.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let edge_stroke = Stroke::new(
        if is_selected { 1.8_f32 } else { 0.8_f32 },
        if is_selected { Color32::WHITE } else { base_col },
    );
    for (_, pts, col) in faces {
        painter.add(Shape::convex_polygon(pts, col, edge_stroke));
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
            if let Some(((pa, _), (pb, _))) =
                cam.project_segment(vp, rect, corners[i], corners[(i + 1) % 4])
            {
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
            if let Some(((pa, _), (pb, _))) =
                cam.project_segment(vp, rect, corners[a], corners[b])
            {
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

/// Evaluates a 3D cubic Bezier curve in world coordinates prior to perspective projection.
pub fn cubic_bezier_3d(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let u = 1.0 - t;
    let uu = u * u;
    let uuu = uu * u;
    let tt = t * t;
    let ttt = tt * t;
    p0 * uuu + p1 * (3.0 * uu * t) + p2 * (3.0 * u * tt) + p3 * ttt
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
