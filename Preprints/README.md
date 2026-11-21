# Preprints for NeuroSpace Lab (`crvscience-neurospace-lab`)

**Author:** Bryan P. White (ORCID: [0000-0003-4191-3511](https://orcid.org/0000-0003-4191-3511))  
**Hub:** https://crvscience.com/apps

This directory contains the self-contained LaTeX source (`.tex`) and single-column PDF (`.pdf`) editions for both:

1. **Companion Application Preprint (`APP-PREPRINT-19`)**
   - **Title:** NeuroSpace Lab: 3D Neural Network Topology, Structural Plasticity, and Six-Layered Cerebral Cortex Simulation in Rust
   - **Files:** `neurospace-lab-3d-neural-cortex.tex`, `neurospace-lab-3d-neural-cortex.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24263468
   - **Figshare Mirror:** https://figshare.com/articles/preprint/NeuroSpace_Lab_3D_Neural_Network_Topology_Structural_Plasticity_and_Six-Layered_Cerebral_Cortex_Simulation_in_Safe_Rust/35016232
   - **Web Preprint:** https://crvscience.com/apps/preprints/neurospace-lab-3d-neural-cortex

2. **CRV Science Public Apps Network Overview (`APP-PREPRINT-01`)**
   - **Title:** The CRV Science App Network: Twenty Open-Source Rust Programs for Science, Data, and Simulation
   - **Files:** `crv-science-public-apps-network.tex`, `crv-science-public-apps-network.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222398
   - **Figshare Mirror:** https://figshare.com/articles/preprint/The_CRV_Science_App_Network_Seventeen_Open-Source_Rust_Programs_for_Science_Data_and_Simulation/35013736
   - **Web Preprint:** https://crvscience.com/apps/preprints/crv-science-public-apps-network

## Compiling Standalone LaTeX

Both `.tex` files inline their bibliography so each compiles standalone with standard `pdflatex`:

```bash
pdflatex neurospace-lab-3d-neural-cortex.tex
pdflatex crv-science-public-apps-network.tex
```
