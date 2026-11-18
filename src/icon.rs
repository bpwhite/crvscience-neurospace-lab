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

/// Returns the 64x64 RGBA window/taskbar icon for NeuroSpace Lab.
pub fn app_egui_icon() -> egui::IconData {
    let size = 64u32;
    egui::IconData {
        rgba: render_cortex_icon_rgba(size),
        width: size,
        height: size,
    }
}

fn dist_seg(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let vx = bx - ax;
    let vy = by - ay;
    let wx = px - ax;
    let wy = py - ay;
    let len_sq = vx * vx + vy * vy;
    let t = if len_sq > 1e-6 {
        ((wx * vx + wy * vy) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = px - (ax + vx * t);
    let dy = py - (ay + vy * t);
    (dx * dx + dy * dy).sqrt()
}

fn sd_rounded_box(px: f32, py: f32, bx: f32, by: f32, r: f32) -> f32 {
    let qx = px.abs() - bx + r;
    let qy = py.abs() - by + r;
    let ox = qx.max(0.0);
    let oy = qy.max(0.0);
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0) - r
}

fn point_in_triangle(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    let e0x = bx - ax;
    let e0y = by - ay;
    let e1x = cx - bx;
    let e1y = cy - by;
    let e2x = ax - cx;
    let e2y = ay - cy;
    let p0x = px - ax;
    let p0y = py - ay;
    let p1x = px - bx;
    let p1y = py - by;
    let p2x = px - cx;
    let p2y = py - cy;

    let d0 = dist_seg(px, py, ax, ay, bx, by);
    let d1 = dist_seg(px, py, bx, by, cx, cy);
    let d2 = dist_seg(px, py, cx, cy, ax, ay);
    let min_d = d0.min(d1).min(d2);

    let c0 = e0x * p0y - e0y * p0x;
    let c1 = e1x * p1y - e1y * p1x;
    let c2 = e2x * p2y - e2y * p2x;
    let inside = (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0) || (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0);
    if inside {
        -min_d
    } else {
        min_d
    }
}

fn blend_rgba(dst: &mut [u8], sr: f32, sg: f32, sb: f32, sa: f32) {
    let a = sa.clamp(0.0, 1.0);
    if a <= 0.001 {
        return;
    }
    let da = dst[3] as f32 / 255.0;
    let out_a = a + da * (1.0 - a);
    if out_a <= 1e-5 {
        return;
    }
    let dr = dst[0] as f32;
    let dg = dst[1] as f32;
    let db = dst[2] as f32;
    dst[0] = ((sr * a + dr * da * (1.0 - a)) / out_a).clamp(0.0, 255.0) as u8;
    dst[1] = ((sg * a + dg * da * (1.0 - a)) / out_a).clamp(0.0, 255.0) as u8;
    dst[2] = ((sb * a + db * da * (1.0 - a)) / out_a).clamp(0.0, 255.0) as u8;
    dst[3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
}

/// Renders a crisp anti-aliased neural-cortex pyramidal soma & dendritic arbor badge.
pub fn render_cortex_icon_rgba(size: u32) -> Vec<u8> {
    let s = size as usize;
    let mut buf = vec![0u8; s * s * 4];
    let aa = 2.2 / size as f32;

    let branches: [((f32, f32), (f32, f32), f32, (f32, f32, f32)); 9] = [
        // Ascending apical trunk
        ((0.0, -0.04), (0.0, -0.44), 0.055, (56.0, 189.0, 248.0)),
        // L1 apical tuft crown
        ((0.0, -0.42), (-0.44, -0.64), 0.036, (52.0, 211.0, 153.0)),
        ((0.0, -0.42), (0.44, -0.64), 0.036, (52.0, 211.0, 153.0)),
        ((0.0, -0.42), (-0.22, -0.69), 0.030, (167.0, 243.0, 208.0)),
        ((0.0, -0.42), (0.22, -0.69), 0.030, (167.0, 243.0, 208.0)),
        // Basal dendritic skirt
        ((0.0, 0.30), (-0.54, 0.56), 0.038, (56.0, 189.0, 248.0)),
        ((0.0, 0.30), (0.54, 0.56), 0.038, (56.0, 189.0, 248.0)),
        ((0.0, 0.32), (-0.26, 0.65), 0.032, (129.0, 140.0, 248.0)),
        ((0.0, 0.32), (0.26, 0.65), 0.032, (129.0, 140.0, 248.0)),
    ];

    let nodes: [((f32, f32), f32, (f32, f32, f32)); 5] = [
        ((0.0, -0.42), 0.085, (56.0, 189.0, 248.0)),
        ((-0.44, -0.64), 0.080, (250.0, 204.0, 21.0)),
        ((0.44, -0.64), 0.080, (250.0, 204.0, 21.0)),
        ((-0.54, 0.56), 0.080, (192.0, 132.0, 252.0)),
        ((0.54, 0.56), 0.080, (192.0, 132.0, 252.0)),
    ];

    for y in 0..s {
        for x in 0..s {
            let px = ((x as f32 + 0.5) / size as f32) * 2.0 - 1.0;
            let py = ((y as f32 + 0.5) / size as f32) * 2.0 - 1.0;
            let idx = (y * s + x) * 4;
            let cell = &mut buf[idx..idx + 4];

            let box_d = sd_rounded_box(px, py, 0.90, 0.90, 0.28);
            let badge_cov = (1.0 - (box_d / aa).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            if badge_cov <= 0.001 {
                continue;
            }

            let r2 = (px * px + py * py).sqrt().clamp(0.0, 1.2);
            let bg_r = 9.0 + (1.0 - r2 * 0.7) * 8.0;
            let bg_g = 14.0 + (1.0 - r2 * 0.7) * 16.0;
            let bg_b = 25.0 + (1.0 - r2 * 0.7) * 26.0;
            blend_rgba(cell, bg_r, bg_g, bg_b, badge_cov);

            // Subtle cyan-emerald border ring
            let ring_d = (box_d + 0.045).abs() - 0.024;
            if ring_d < aa {
                let cov = (1.0 - (ring_d / aa).max(0.0)) * badge_cov * 0.85;
                blend_rgba(cell, 56.0, 189.0, 248.0, cov);
            }

            // Dendritic branches
            for &((ax, ay), (bx, by), w, (cr, cg, cb)) in &branches {
                let d = dist_seg(px, py, ax, ay, bx, by) - w;
                if d < aa {
                    let cov = (1.0 - (d / aa).max(0.0)) * badge_cov;
                    blend_rgba(cell, cr, cg, cb, cov);
                }
            }

            // Pyramidal soma silhouette
            let tri_d = point_in_triangle(px, py, 0.0, -0.10, 0.28, 0.36, -0.28, 0.36) - 0.025;
            if tri_d < aa {
                let cov = (1.0 - (tri_d / aa).max(0.0)) * badge_cov;
                let t = ((py + 0.10) / 0.48).clamp(0.0, 1.0);
                let sr = 56.0 * (1.0 - t) + 52.0 * t;
                let sg = 189.0 * (1.0 - t) + 211.0 * t;
                let sb = 248.0 * (1.0 - t) + 153.0 * t;
                blend_rgba(cell, sr, sg, sb, cov);
            }

            // Glowing nucleus inside soma
            let nuc_d = (px * px + (py - 0.17) * (py - 0.17)).sqrt() - 0.11;
            if nuc_d < aa {
                let cov = (1.0 - (nuc_d / aa).max(0.0)) * badge_cov;
                blend_rgba(cell, 236.0, 253.0, 245.0, cov);
            }

            // Synaptic terminal nodes
            for &((nx, ny), nr, (cr, cg, cb)) in &nodes {
                let nd = ((px - nx) * (px - nx) + (py - ny) * (py - ny)).sqrt() - nr;
                if nd < aa {
                    let cov = (1.0 - (nd / aa).max(0.0)) * badge_cov;
                    blend_rgba(cell, cr, cg, cb, cov);
                }
            }
        }
    }

    buf
}
