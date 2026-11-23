# NeuroSpace Lab User Guide (`v1.6.0`)

**NeuroSpace Lab** (`neurospace-lab`) is an interactive safe-Rust (`#![forbid(unsafe_code)]`) workstation for exploring 15 neural network architectures—from a 1958 single-layer perceptron (`M0`) to sparse Mixture-of-Experts transformers (`M14`) and 6-layered histological cerebral cortex grey matter (`M13`)—with real-time structural plasticity, causal lesion/optogenetic interventions, and genuine FGSM/PGD adversarial evaluation.

---

## 1. Installation & Building from Source

### Prerequisites
- **Rust toolchain** (`1.75+` recommended via `rustup`).
- **GPU driver** supporting Vulkan, DirectX 12, Metal, or OpenGL (`eframe`/`wgpu` backend).
- **SQLite** is statically bundled via `rusqlite` (`features = ["bundled"]`), requiring no external system SQLite library.

### Build & Run Commands

```bash
# Run interactive 3D GUI in optimized release mode
cargo run --release

# Run all unit and integration tests
CARGO_BUILD_JOBS=4 cargo test --all-targets

# Build standalone release binary
cargo build --release
```

On Windows, `src/main.rs` uses `#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]` and automatically attaches to the parent console (`AttachConsole`) whenever CLI arguments are passed.

---

## 2. GUI Layout & Workflow

The workstation window (`1440x900` default, `1024x680` minimum) is organized into four synchronized regions:

1. **Top HUD Bar**:
   - **Simulation Transport**: `Run / Pause`, `Step` (single tick), `Rebuild Topology`, and `xSpeed` (`1..=8` steps/frame).
   - **Experiment & Sweep Controls**: `Snapshot Run` pins the current state into the Comparison Ledger and SQLite `comparison_runs` table; `Sweep All 15 Models (Locked Dataset)` or `Sweep All 7 Datasets (Locked Model)` runs an incremental non-blocking benchmark with a live `ProgressBar` and `✖ Cancel` button.
   - **Workset I/O**: `Export Workset...` and `Import Workset...` save/restore SHA-256 verified `.neuroworkset.json` files.
   - **Live KPI Strip**: `Step`, `Somata`, `Synapses`, primary task metric (`Val Accuracy (%)`, `R² Valuation Fit (%)`, `Collision-Free Score (%)`, or `QSAR Concordance (%)`), `Adv Margin (ε=...)`, task loss, and active `Ablated | Severed` counts.

2. **Left Control Column (Sections 1–6)**:
   - **1. Neural Architecture (Tiers I–V)**: Select `M0`–`M14`, toggle `🔒 Lock Model`, or enable continual weight transfer across dataset switches.
   - **2. Benchmark & Scientific Dataset**: Select `D1`–`D7` or toggle `🔒 Lock Data`.
   - **3. Output Goal & Optimization Pressure**: Choose `MaxTaskFidelity`, `SparseEdgeEfficiency`, `AdversarialImmunity`, `FreeEnergyCalibration`, or `MetabolicBioEconomy`.
   - **4. Parameter Determinants & Flow Sample**: Tune Learning Rate `η`, Momentum `β`, Synaptic Prune `|w|`, Neurogenesis / Sprout rate, River Stream Intensity, Cortical Capacity (`0.5..=2.0`), and Adversarial `ε` (`0.0..=0.45`, dynamically recomputing `adversarial_margin` even while paused).
   - **5. Adaptive & Histological 3D Renderer**: Choose `RenderMode` (`AutoAdaptive`, `MechanisticLattice`, `HybridCyberManifold`, `OrganicCorticalArbor`, `HistologicalGreyMatter3D`), toggle Layer/Lamina zones, toggle Auto-Orbit (`A`), or click quick camera preset buttons (`1:Iso`, `2:Sag`, `3:Cor`, `4:Top`, `R:Reset`).
   - **6. Causal Lesion & Optogenetics Lab**: Switch `ProbeTool` (`Inspect`, `Ablate`, `ChR2 (+)`, `NpHR (-)`), tune `ChR2 Laser Drive` (`0.2..=1.8`), trigger `Sever Feedback`, `Sever Skip/Lat`, `Focal Stroke (25%)`, or `Heal All`, and monitor Pre/Acute/Now lesion recovery.

3. **Center 3D Neural Space Viewport**:
   - Displays 3D somata, lit pyramidal prisms (`M13` & histological modes), tapered 3D apical trunks, L1 apical tufts, basal skirts, background pathways, and calm braided 3D cubic-Bezier tracer wisps (`<= 22` sampled pathways).
   - Hovering or clicking any soma opens the **Soma Inspector Card** showing `Soma #ID`, role, layer, `(x, y, z)`, apical `Ca2+` burst level, activation, `Vm`, bias, utility, and backprop gradient.

4. **Bottom Lock Matrix & Telemetry Panel** (toggleable via `📊 Lock Matrix`):
   - **Left**: Live Sample Inspector (`pred` vs. `target` progress bars) and cumulative `+Sprouted | -Pruned | +Neurogenesis` counters.
   - **Center**: Interactive hoverable telemetry sparklines for Primary Metric and Loss, featuring vertical `lesion_events` tag markers (`Stroke`, `SeverFB`, `SeverLat`, `Ablate`), Pre/Acute horizontal reference lines with top-right legend, and bottom X-axis sample ticks (`#1`, midpoint, `#N`).
   - **Right**: Scrollable Comparison Ledger across locked model/dataset runs.

---

## 3. Interactive Controls & Keyboard Shortcuts

All single-key shortcuts (`1`–`4`, `R`, `A`, `N`, `.`, `I`, `X`, `E`, `S`, `Space`, `F5`, `F9`) require `!ctx.wants_keyboard_input()` and no modifier keys (`Ctrl`/`Cmd`/`Alt`), preventing interference with text fields or `Ctrl+S` export.

| Gesture / Key | Action |
|---|---|
| `Left-Drag` | Orbit camera (`yaw` / `pitch`) with smooth exponential damping |
| `Right-Drag` or `Middle-Drag` | Pan camera focal target in screen right/up plane |
| `Scroll Wheel` | Zoom camera distance (`8.5 .. 54.0`) with smooth damping |
| `1` | Camera Preset: Isometric 3/4 view |
| `2` | Camera Preset: Sagittal (`X-Y`) lateral profile |
| `3` | Camera Preset: Coronal (`Y-Z`) transverse profile |
| `4` | Camera Preset: Top-Down (`X-Z`) dorsal view |
| `R` | Reset camera target to origin, Isometric view, and resume Auto-Orbit |
| `A` | Toggle camera Auto-Orbit |
| `Space` | Play / Pause simulation |
| `N` or `.` | Pause and advance exactly 1 training step |
| `I` | Probe Tool: Inspect soma telemetry |
| `X` | Probe Tool: Focal Ablation (`Shift+Click` shortcut) |
| `E` | Probe Tool: Optogenetic ChR2 Excitation (`Alt+Click` shortcut) |
| `S` | Probe Tool: Optogenetic NpHR Photoinhibition |
| `F5` / `F9` | Save / Load SQLite WAL slot `'quicksave_f5'` |
| `Ctrl/Cmd + S` / `Ctrl/Cmd + O` | Export / Import `.neuroworkset.json` |
| `F1` | Toggle About, Shortcuts & Optogenetics Field Guide modal |
| `Esc` | Close `F1` modal if open; otherwise clear selected neuron |

---

## 4. CLI Reference

```bash
# Launch native GUI (default)
neurospace-lab gui

# Sweep all 15 models on locked D7 Molecular Properties for 80 steps
neurospace-lab bench-dataset --dataset D7 --steps 80

# Sweep all 7 datasets on locked M14 Frontier LLM for 60 steps
neurospace-lab bench-model --model M14 --steps 60

# Generate portable SHA-256 verified JSON workset to stdout
neurospace-lab export-sample-workset > sample.neuroworkset.json

# Render offscreen GPU screenshots for views 1, 2, or 3
neurospace-lab --headless-screenshot view1.png --screenshot-view 1
```

Valid dataset codes: `D1`, `D2`, `D3`, `D4`, `D5`, `D6`, `D7`.  
Valid model codes: `M0`, `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M7`, `M8`, `M9`, `M10`, `M11`, `M12`, `M13`, `M14`.

---

## 5. Scientific Features & Causal Protocols

### Optogenetics & Lesion Workflow
1. Train `M13` (`Cerebral Cortex Grey Matter`) or `M10` (`Predictive Coding`) for 40–60 steps until primary metric stabilizes.
2. Observe `Pre` baseline in Section 6 and the sparklines.
3. Click **`Sever Feedback`** (cuts `is_feedback` pathways, collapsing `L1` apical tuft `Ca2+` bursts in `M13` and top-down prediction subtraction in `M10`) or **`Focal Stroke (25%)`** (ablates a spatial cluster and sprouts compensatory bypass collaterals).
4. Use **`ChR2 (E)`** or **`NpHR (S)`** on individual somata to clamp depolarization/hyperpolarization in real time.
5. Click **`Heal All`** to restore intact circuitry and clear lesion markers.

### Persistence & Checksum Verification
- **SQLite WAL**: Stores `save_slots` and `comparison_runs` in `neurospace_lab.sqlite`. If the working directory is read-only, the app falls back to in-memory operation and reports status in the footer.
- **Portable Worksets (`.neuroworkset.json`)**: Captures exact `neurons`, `synapses`, ablations, clamps, severed tracts, histories, and metrics, guarded by canonical SHA-256 digests (`WorksetEnvelope::verify_digest()`).

---

## 6. Troubleshooting

- **Offscreen `--headless-screenshot` fails**: Ensure a Vulkan/DX12/Metal/OpenGL adapter is visible to `wgpu` (on headless Linux servers, verify Mesa/Vulkan ICD or `xvfb`/software adapter availability).
- **Hotkeys do not respond**: Verify no text box or modal input holds keyboard focus (`ctx.wants_keyboard_input()`) and no unintended modifier key (`Ctrl`/`Alt`) is held.
- **Workset import error (`SHA-256 checksum verification failed`)**: The `.neuroworkset.json` file was edited or corrupted after export; re-export from a clean session.
- **Database lock or permission notice**: When running from a non-writable directory, launch from a writable folder or use `Export Workset...` to save `.neuroworkset.json` directly.

---

Licensed under `GPL-3.0-or-later`. See [LICENSE](../LICENSE).
