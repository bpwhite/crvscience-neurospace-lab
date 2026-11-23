# NeuroSpace Lab — Scientific Methods & Numerical Formulation (`v1.6.0`)

**License:** GNU General Public License v3.0 or later (`GPL-3.0-or-later`, see [LICENSE](../LICENSE)).

---

## 1. Units, Normalization & State Variables

NeuroSpace Lab operates in normalized dimensionless biophysical and machine-learning coordinates so classical connectionist architectures (`M0`–`M6`, `M14`) and continuous dynamical/biological wetware (`M7`–`M13`) share a unified 3D spatial, energetic, and perturbation framework:

| Symbol | Quantity | Domain / Bounds | Module |
|---|---|---|---|
| `x` | Normalized input feature vector | `[-2.2, +2.2]^d_in` | `datasets.rs`, `engine.rs` |
| `w_{ij}` | Synaptic weight (`from = i -> to = j`) | `[-2.8, +2.8]` (`[-2.5, +2.5]` at init) | `engine.rs` |
| `b_i` | Somatic bias (`layer >= 1`) | `[-1.8, +1.8]` | `engine.rs` |
| `V_m,i` | Normalized somatic membrane potential | Unclamped subthreshold; reset `-0.22` | `engine.rs` |
| `τ_i` | Membrane time constant (`M7`: `[0.45, 1.00]`, `M8`: `[0.25, 1.90]`) | Dimensionless solver time units | `engine.rs` |
| `x_trace,i` | Exponential post-synaptic / pre-synaptic spike trace | `[0.0, 2.0]` | `engine.rs` |
| `Ca_i` | Apical dendritic calcium plateau (`apical_ca_burst`) | `[0.0, 1.0]` | `engine.rs` |
| `ε` (`adversarial_eps`) | `L_∞` input perturbation radius | `[0.0, 0.45]` | `models.rs`, `engine.rs` |
| `F_flux` | Metabolic ATP / spike flux proxy (`metabolic_flux`) | `[0.05, 2.50]` | `engine.rs` |

---

## 2. Frontier Backpropagation & Output Objectives

### 2.1 Synchronous Frontier Backpropagation

Reverse-mode sensitivity `δ_i = ∂L / ∂z_i` is propagated backward from readout neurons (`output_ids`) via synchronous hop-staged wavefronts (`backpropagate_from_outputs`) rather than unsorted in-place passes, preventing multi-pass triple accumulation and order dependence:

```text
Initial frontier (hop 0):
  δ_out[o] = ∂L / ∂z_o
  frontier[o] = δ_out[o],   all other nodes = 0

For hop = 0 .. max_hops - 1:
  next_frontier[i] = 0  for all i
  For each active synapse (i -> j):
    if is_feedback and hop > 1: continue
    edge_scale = 0.35   if (is_feedback or layer[i] >= layer[j]) else 1.0
    deriv[i]   = 1.0    if role[i] == SensoryInput
                 else max(1 - clamp(a_i, -1, 1)^2, 0.08)
    next_frontier[i] += frontier[j] * w_{ij} * edge_scale * deriv[i]
  grad[i] = clamp(grad[i] + next_frontier[i], -3.0, +3.0)
  frontier = next_frontier
```

Notice `deriv[i]` evaluates `1 - a_i^2` directly on post-activation `a_i` (avoiding double `tanh(tanh(z))` squashing) and preserves unit sensitivity (`1.0`) at `SensoryInput` nodes for exact input gradients `∇_x L`.

### 2.2 Output Goals (`OutputGoal`)

Let `L_task` denote multiclass cross-entropy (`-Σ y_c ln p_c`) for classification (`D1`–`D4`) or mean squared error (`(1/d_out) Σ (p_c - y_c)^2`) with sigmoid readouts for regression/control/QSAR (`D5`–`D7`):

1. **`MaxTaskFidelity`**: `l1_reg = 0.0008`, `l2_reg = 0.0`.
2. **`SparseEdgeEfficiency`**: Strong `L1` sparsity penalty `l1_reg = 0.0045` and `1.45x` structural pruning threshold multiplier.
3. **`MetabolicBioEconomy`**: `l1_reg = 0.0035`, `1.45x` pruning threshold, plus somatic activity penalty `+0.015 * tanh(a_i)` in bias updates to minimize firing energy `F_flux`.
4. **`AdversarialImmunity`**: Inner-loop FGSM/PGD perturbation during training (`x_adv = clip(x + α sign(∇_x L), x - ε, x + ε)`) combined with `L2` Lipschitz weight regularization `l2_reg = 0.0022`.
5. **`FreeEnergyCalibration`**: Variational Free-Energy objective combining inaccuracy, Gaussian weight-prior complexity (`l2_reg = 0.0028`), and softmax maximum-entropy calibration (`λ_H = 0.06`):

```text
F_ELBO = L_task(p, y) - λ_H * H(p) + (1/2) * λ_KL * mean(w^2)
H(p)   = - Σ_k p_k * ln(p_k)
∂F_ELBO / ∂z_k = (p_k - y_k) + λ_H * p_k * (ln(p_k) + H(p))
```

The live variational free energy is tracked in `pc_mean_free_energy`.

---

## 3. Spiking LIF Dynamics & Asymmetric Trace STDP (`M7SpikingLifStdp`)

### 3.1 Exponential Leaky Integrate-and-Fire Membrane Integration

Each pass (`dt = M7_LIF_DT_PASS = 0.20`) integrates the exact zero-order-hold exponential solution to the leaky membrane ODE `τ_i dV_i/dt = -V_i + R_m I_i` (`R_m = 1.65`, firing threshold `V_th = 0.55`, reset `V_reset = -0.22`):

```text
leak_i     = exp(-dt / max(τ_i, 0.05))
V_m,i(t+dt) = V_m,i(t) * leak_i + 1.65 * (1 - leak_i) * (I_syn,i + b_i)

If V_m,i(t+dt) > 0.55:
  s_i = 1,   V_m,i <- -0.22,   x_trace,i <- min(x_trace,i + 1.0, 2.0),   a_i = 1.0
Else:
  s_i = 0,   a_i = clamp(0.75 * max(V_m,i, 0) + 0.35 * x_trace,i, 0, 1)
```

Under zero synaptic and bias drive (`I_syn,i + b_i = 0`), subthreshold voltage decays strictly as `V_m,i(t + k·dt) = V_m,i(t) * exp(-k·dt / τ_i)`.

### 3.2 Asymmetric Trace-Based STDP & Dale's Law

Each neuron maintains an exponential spike trace decayed every pass before spike emission (`τ_trace = 0.45`):

```text
x_trace,i <- x_trace,i * exp(-dt / τ_trace)

Δw_STDP(i -> j) = A+ * x_trace,i * s_j - A- * x_trace,j * s_i
with A+ = 1.0,  A- = 0.85
```

- **Pre-before-post (`Δt > 0`)**: `x_trace,i > 0` when post fires (`s_j = 1, s_i = 0`) produces `Δw_STDP > 0` (LTP potentiation).
- **Post-before-pre (`Δt < 0`)**: `x_trace,j > 0` when pre fires (`s_i = 1, s_j = 0`) produces `Δw_STDP < 0` (LTD depression).
- **Dale's Law**: Outgoing weights from `InhibitoryInterneuron` somata are constrained strictly non-positive (`w <= -0.02`).

---

## 4. Six-Layered Cerebral Cortex & Two-Compartment Pyramidal Integration (`M13`)

`M13CerebralCortexGreyMatter` organizes somata across laminar cortical depths from **Pia Mater (`Y = +5.3`)** down to **Subcortical White Matter (`Y = -4.6 .. -5.4`)**:

```text
Y = +4.6   L1 Molecular Plexus (Cajal-Retzius / Horizontal; apical tuft target)
Y = +2.4   L2/3 Supragranular Association Pyramidals (apical trunk -> L1)
Y = +0.2   L4 Internal Granular Spiny Stellate (Thalamocortical recipients)
Y = -1.9   L5 Infragranular Thick-Tufted Pyramidals (long apical trunk -> L1)
Y = -3.7   L6 Corticothalamic Fusiforms (corticothalamic feedback -> L4)
Y = -5.4   Thalamic Input Manifold (left) & White-Matter Efferent Readout (right)
```

Following Larkum's two-compartment pyramidal coincidence mechanism, `forward_sample_impl` separates bottom-up/perisomatic currents (`!syn.is_feedback -> basal_cur`) from top-down Layer I / corticothalamic feedback currents (`syn.is_feedback -> apical_cur`):

```text
basal_drive  = basal_cur[i] + b_i
apical_drive = apical_cur[i]

V_m,i   <- 0.55 * V_m,i + 0.85 * basal_drive
bAP_i    = 0.35 + 0.65 * tanh(min(|V_m,i|, 1.5))
coinc_i  = tanh(|apical_drive| * bAP_i)
Ca_i    <- clamp(0.80 * Ca_i + 0.35 * coinc_i, 0, 1)

ca_gain  = 1 + 0.35 * Ca_i
ap_mod   = 0.32 * tanh(apical_drive) * (0.40 + 0.60 * Ca_i)
a_i      = tanh((V_m,i + ap_mod) * ca_gain)
```

Severing reverse feedback tracts (`sever_feedback_tracts()`) zeroes `apical_cur`, causing `Ca_i` to decay exponentially while leaving basal feedforward signaling intact.

---

## 5. Additional Tier Solvers (`M3`, `M8`, `M10`, `M14`)

- **`M3SparseTransformer`**: Per-head scaled softmax (`1.8 * drive`) followed by top-`k` (`k = max(h_len / 2, 1)`) competitive key pruning and renormalization.
- **`M8LiquidTimeConstant`**: Continuous ODE forward Euler integration (`dt = 0.28`):
  ```text
  σ = 1 / (1 + exp(-drive))
  dx/dt = -(1 / τ_i + σ) * x_i + σ * tanh(drive)
  ```
- **`M10PredictiveCoding`**: Value vs. prediction-error units subtracting top-down predictions (`e = sensory_cur - pred_cur + 0.25 b`) and reporting mean squared free energy `mean(e^2)`.
- **`M14FrontierLlmMoe`**: Causal GQA (`L1` upper) + persistent KV-cache slots (`L1` lower) feeding a Top-2 sparse router (`L2`) over 4 SwiGLU expert quadrants (`L3`), masking ablated or severed quadrants before softmax normalization.

---

## 6. Genuine FGSM / PGD Adversarial Evaluation

In `evaluate_validation_core()`, validation samples are evaluated on both clean inputs (`primary_metric`) and adversarial `L_∞(x, ε)` perturbations generated from true backpropagated input gradients `g = ∇_x L(θ, x, y)`:

```text
FGSM probe:
  x_fgsm = x + ε * sign(∇_x L(θ, x, y))

Monotone-nested fixed-step PGD probe (α = 0.05, up to 8 steps + boundary step):
  x^(0) = x
  x^(t+1) = clip_{[x - ε, x + ε]}( x^(t) + α * sign(∇_x L(θ, x^(t), y)) )

Worst-case sample evaluation over {x, x_fgsm, x^(1), ..., x^(K)}:
  minimizes (0.70 * score_adv + 0.30 * soft_margin_pct)
```

Because `PGD` with fixed step `α = 0.05` yields strictly nested search prefixes as `ε` grows, `adversarial_margin` behaves physically and monotonically across `ε ∈ [0.00, 0.40]`, while `evaluate_validation_core()` snapshots and restores all neuron dynamical states (`V_m`, `Ca_i`, `spike_trace`, `activation`) so repeated evaluation is side-effect-free.

---

## 7. 3D Projection, Homogeneous Clipping & Lighting

- **Homogeneous Clip-Space Line Clipping (`project_segment`)**: Clips 3D world segments against `w >= 0.25`, `w + z >= 0`, and padded frustum bounds `|x| <= 1.12 w`, `|y| <= 1.12 w` prior to perspective division, eliminating viewport edge pop-out.
- **3D Cubic Beziers (`cubic_bezier_3d`)**: Evaluates `B(u) = (1-u)^3 P0 + 3(1-u)^2 u Q1 + 3(1-u) u^2 Q2 + u^3 P3` directly in world coordinates before projecting each sub-segment.
- **Lit 3D Geometry & Depth Fog**: Histological pyramidal somata, tapered apical shafts, and mechanistic octahedra are shaded using normalized directional Lambertian lighting (`L_dir = normalize(0.42, 0.82, 0.39)`), back-face culled against the camera eye vector, depth-sorted back-to-front, and blended with exponential-linear depth fog (`clamp((depth - 0.72 d_cam) / (1.45 d_cam), 0, 0.68)`).

---

## 8. Boundary Guards & Integrity Checks

- **Weight & Bias Bounds**: Synaptic weights clamped to `[-2.8, +2.8]`; biases clamped to `[-1.8, +1.8]`; gradients clamped to `[-3.0, +3.0]`.
- **Ablation Safeguard**: `toggle_ablate_neuron` preserves at least one surviving `SensoryInput` and one surviving `ReadoutOutput` soma.
- **Workset Integrity**: `WorksetEnvelope` validates synapse endpoint bounds (`from < n_neurons && to < n_neurons`) and verifies canonical `SHA-256` digests before restoring states.
- **Memory Safety**: `#![forbid(unsafe_code)]` across `src/lib.rs` and `tests/integration_suite.rs` (only `AttachConsole` in `src/main.rs` on Windows CLI startup uses `#[allow(unsafe_code)]`).

---

## 9. Peer-Reviewed & Technical References

1. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408.
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
3. Bi, G., & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464–10472.
4. Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological models of synaptic plasticity based on spike timing. *Biological Cybernetics*, 98(6), 459–478.
5. Larkum, M. E., Zhu, J. J., & Sakmann, B. (1999). A new cellular mechanism for coupling inputs arriving at different cortical layers. *Nature*, 398, 338–341.
6. Larkum, M. (2013). A cellular mechanism for cortical associations: an organizing principle for the cerebral cortex. *Trends in Neurosciences*, 36(3), 141–151.
7. Douglas, R. J., & Martin, K. A. C. (2004). Neuronal circuits of the neocortex. *Annual Review of Neuroscience*, 27, 419–451.
8. Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.
9. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
10. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR 2015*.
11. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR 2018*.
12. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid time-constant networks. *AAAI 2021*, 35(9), 7657–7666.
13. Hawkins, J., & Ahmad, S. (2016). Why neurons have thousands of synapses, a theory of sequence memory in neocortex. *Frontiers in Neural Circuits*, 10, 23.
14. Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR 2017*.
15. White, B. P. (2026). *NeuroSpace Lab: 3D Neural Network Topology, Structural Plasticity, and Six-Layered Cerebral Cortex Simulation in Rust* (`APP-PREPRINT-19`, DOI: `10.5281/zenodo.24263468`) & *The CRV Science App Network* (`APP-PREPRINT-01`, DOI: `10.5281/zenodo.24222398`).

---

*Copyright (C) 2026 CRV Science Network (`https://crvscience.com`). Licensed under GNU General Public License v3.0 or later (`GPL-3.0-or-later`).*
