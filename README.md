<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=13&duration=3000&pause=1000&color=E85D24&center=true&vCenter=true&width=600&lines=First+embodied+lobster+nervous+system+emulation;Hodgkin-Huxley+%C3%97+MuJoCo+%C3%97+Brian2;No+RL.+No+training.+Pure+biology." alt="Typing SVG" />

# 🦞 NeuroLobster

### *World's first embodied emulation of the stomatogastric nervous system*

<p align="center">
  <img src="https://img.shields.io/badge/Status-Phase_1_Complete-4ade80?style=flat-square&labelColor=0a0a0a" />
  <img src="https://img.shields.io/badge/License-MIT-E85D24?style=flat-square&labelColor=0a0a0a" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0a0a0a" />
  <img src="https://img.shields.io/badge/Brian2-2.5.4-orange?style=flat-square&labelColor=0a0a0a" />
  <img src="https://img.shields.io/badge/MuJoCo-3.6-00B4D8?style=flat-square&labelColor=0a0a0a" />
  <img src="https://img.shields.io/badge/Open_Science-yes-1d9e75?style=flat-square&labelColor=0a0a0a" />
</p>

<p align="center">
  <a href="#overview">Overview</a> · 
  <a href="#demo">Demo</a> · 
  <a href="#quickstart">Quick Start</a> · 
  <a href="#architecture">Architecture</a> · 
  <a href="#roadmap">Roadmap</a> · 
  <a href="#references">References</a>
</p>

---

> *"The lobster doesn't know it's in a computer. Its stomach still churns at 1Hz. Its legs still move in the same metachronal wave they've used for 400 million years."*

</div>

---

## Overview

**NeuroLobster** is a closed-loop brain-body simulation of *Homarus americanus* — the American lobster. We implement the **stomatogastric ganglion (STG)** using conductance-based Hodgkin-Huxley neurons validated against Marder Lab experimental data, and connect them to a physically simulated body running in **MuJoCo 3.6**.

No reinforcement learning. No reward functions. No scripted behavior. **Pure biological circuit dynamics.**

The same circuit that has been driving lobster stomachs for 400 million years now drives a simulated body in real time on a MacBook Pro.

```
STG Neurons (Brian2) → CPG Oscillator → Motor Commands → MuJoCo Body → Proprioception → STG
        ↑_______________________________________________________________|
                         Closed sensorimotor loop · 15ms sync
```

---

## Demo

<div align="center">

| MuJoCo Simulation | Neural Activity | 3D Brain Point Cloud |
|:-----------------:|:---------------:|:--------------------:|
| Lobster walking, turning, reversing | AB/PD · LP · PY live traces | STG · CoG · VNC · Brain |
| 8 legs · metachronal wave | Pyloric rhythm ~1Hz | 317 neurons · rotating |

> 📹 *Demo video coming soon — [watch on Twitter](https://twitter.com)*

</div>

---

## Quick Start

```bash
# 1. Create environment
conda create -n neurolobster python=3.11 -y
conda activate neurolobster

# 2. Install dependencies
pip install "brian2==2.5.4" "numpy<2.0" mujoco matplotlib scipy trimesh pycollada

# 3. Run full simulation (STG + MuJoCo + Brain visualization)
mjpython neurolobster_full.py

# 4. Run STG only (neural dynamics, no body)
python stg_pyloric.py
```

**Requirements:** macOS 12+ or Linux · Python 3.11 · MuJoCo 3.6 · 8GB RAM

---

## Architecture

```
neurolobster/
│
├── 🧠 Neural
│   ├── stg_pyloric.py          # Hodgkin-Huxley STG (AB/PD, LP, PY)
│   └── stg_find.py             # Parameter search utility
│
├── 🦞 Simulation
│   ├── neurolobster_full.py    # Main: STG + CPG + MuJoCo closed loop
│   ├── neurolobster_sim.py     # CPG-only locomotion
│   └── lobster.xml             # MuJoCo body definition
│
├── 👁️ Visualization
│   └── neurolobster_brain3d.py # 3D neural point cloud + wireframe
│
├── 🗂️ Assets
│   ├── lobster.obj             # 3D body mesh (Homarus americanus)
│   ├── lambert1_albedo.png     # PBR texture — color
│   ├── lambert1_normal.png     # PBR texture — normals
│   ├── lambert1_roughness.jpeg # PBR texture — roughness
│   └── lambert1_AO.jpeg        # Ambient occlusion
│
└── 🌐 Web
    └── neurolab.html           # Lab website (Three.js + WebGL)
```

---

## Neural Model

The STG pyloric circuit implements three identified neurons with full conductance-based Hodgkin-Huxley dynamics:

| Neuron | Role | Firing Rate | Ion Channels |
|--------|------|-------------|--------------|
| **AB/PD** | Pacemaker | ~127 Hz | Na⁺, K⁺, H (Ih), Leak |
| **LP** | Lateral pyloric | ~5 Hz | Na⁺, K⁺, Leak |
| **PY** | Pyloric | ~1 Hz | Na⁺, K⁺, Leak |

**Synaptic inhibition matrix** (from Marder & Bucher 2007):

```
         AB    LP    PY
AB  [  0.0  -0.04 -0.03 ]   ← AB inhibits LP and PY
LP  [ -0.02   0.0 -0.03 ]   ← LP inhibits PY and AB
PY  [ -0.01   0.0   0.0 ]   ← PY provides feedback to AB
```

**Validated parameters:** `w_fwd=0.04 mS/cm²` · `tau_off=80ms` · `I_app=[14.0, 15.5, 15.2] μA/cm²`

---

## Body Simulation

```xml
<!-- lobster.xml — MuJoCo model summary -->
Joints:     17  (8 hip + 8 knee + 1 freejoint)
Actuators:  16  (motor-driven, gear=1-2)
Sensors:    8   (proprioceptive joint position)
Mesh:       Homarus americanus · 454KB OBJ · PBR textures
Physics:    timestep=2ms · gravity=-9.81 · viscosity=0.01
```

**CPG metachronal wave phase offsets:**
```python
phases = [0.00, 0.25, 0.50, 0.75,   # Left legs L1-L4
          0.50, 0.75, 0.00, 0.25]    # Right legs R1-R4 (antiphase)
```

---

## Emergent Behaviors

All behaviors emerge from circuit dynamics — zero behavioral code:

| Behavior | Mechanism |
|----------|-----------|
| **Forward walk** | Symmetric CPG · metachronal wave |
| **Backward walk** | Reversed phase offsets |
| **Turn left** | Right hemisphere amplitude × 1.6 |
| **Turn right** | Left hemisphere amplitude × 1.6 |

---

## Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Neural | ![Brian2](https://img.shields.io/badge/Brian2-2.5.4-E85D24?style=flat-square) | Spiking neural network simulation |
| Physics | ![MuJoCo](https://img.shields.io/badge/MuJoCo-3.6-00B4D8?style=flat-square) | Real-time physics engine |
| Math | ![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy) | Numerical integration |
| Visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c?style=flat-square) | Neural activity plots |
| 3D Web | ![Three.js](https://img.shields.io/badge/Three.js-r128-black?style=flat-square&logo=three.js) | Brain point cloud |
| Mesh | ![Trimesh](https://img.shields.io/badge/Trimesh-latest-4a4a4a?style=flat-square) | 3D model processing |
| Language | ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white) | Primary language |

</div>

---

## Roadmap

```
2025 Q3  ████████████████████  ✅  Project initialization · literature review
2025 Q4  ████████████████████  ✅  HH neuron model · STG pyloric rhythm verified
2026 Q1  ████████████████████  ✅  MuJoCo body · CPG locomotion · closed loop
2026 Q2  ████████░░░░░░░░░░░░  🔄  Phase 2: Full 25-neuron STG · gastric mill
2026 Q3  ░░░░░░░░░░░░░░░░░░░░  📅  Phase 3: CoG integration · olfactory navigation
2027     ░░░░░░░░░░░░░░░░░░░░  📅  Cancer borealis full connectome
```

**Phase 2 targets:**
- [ ] Complete 25-neuron STG (pyloric + gastric mill)
- [ ] Neuromodulation switching (dopamine / serotonin)
- [ ] Feeding rhythm embodied alongside locomotion

---

## References

```bibtex
@article{marder2007,
  author  = {Marder, E. and Bucher, D.},
  title   = {Understanding circuit dynamics using the stomatogastric nervous system},
  journal = {Neuron},
  year    = {2007}
}

@article{prinz2004,
  author  = {Prinz, A.A. and Bucher, D. and Marder, E.},
  title   = {Similar network activity from disparate circuit parameters},
  journal = {Nature Neuroscience},
  year    = {2004}
}

@article{ayers2010,
  author  = {Ayers, J. and Rulkov, N. and Knudsen, D.},
  title   = {Controlling Synchronized Oscillations in a Lobster CPG},
  journal = {Neurocomputing},
  year    = {2010}
}
```

---

## License & Citation

```
MIT License · © 2025–2026 NeuroLobster Lab · Open Science
```

If you use this work in research, please cite:
```bibtex
@software{neurolobster2025,
  author  = {0xNickdev and NeuroLobster Lab},
  title   = {NeuroLobster: Embodied STG Emulation of Homarus americanus},
  year    = {2025},
  url     = {https://github.com/0xNickdev/neurolobster}
}
```

---

<div align="center">

**NeuroLobster Lab** · Open Science · MIT License

*The lobster is running. Come look.*

[![Twitter](https://img.shields.io/badge/Twitter-@0xNickdev-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com)
[![GitHub](https://img.shields.io/badge/GitHub-0xNickdev-181717?style=flat-square&logo=github)](https://github.com/0xNickdev/neurolobster)

</div>
