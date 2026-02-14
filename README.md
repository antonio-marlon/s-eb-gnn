# S-EB-GNN-Q: Quantum-Inspired Semantic Resource Allocation for 6G

[![Watch Demo](https://img.shields.io/badge/▶️_Watch_Demo-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=7Ng696Rku24)
[![GitHub Clones](https://img.shields.io/badge/283+_clones_in_14_days-blue)](https://github.com/antonio-marlon/s-eb-gnn)

> 📈 **283+ clones in 14 days**  
> 💬 *“Well aligned with AI-native wireless systems.”* — **Prof. Merouane Debbah**, Center Director, 6G Research Center  
> 💰 **Pro Bundle**: US$ 70 (first 10 buyers) → [Get it now](https://ko-fi.com/s/4a88e99001)

Lightweight JAX implementation of **quantum-inspired semantic resource allocation** for THz/RIS-enabled 6G networks. Achieves **negative energy states** (e.g., −6.62) under semantic prioritization (Critical > Video > IoT).

---

## 🔬 Key Features

- **Physics-based THz channel modeling** (path loss, blockage)
- **Reconfigurable Intelligent Surfaces (RIS)** support
- **Quantum-inspired semantic refinement** via graph kernels
- **Semantic prioritization** (Critical > Video > IoT)
- **Energy-based optimization** with negative energy convergence
- **Zero-shot inference** (no retraining required)
- **Pure JAX + Equinox** (<250 lines core logic)
- **MIT License** — free for research and commercial use

---

## 📊 Benchmark vs Baselines (v1.1)

| Metric             | S-EB-GNN-Q | WMMSE     | Heuristic |
|--------------------|------------|-----------|-----------|
| Final Energy       | **−6.62**  | +0.01     | +0.10     |
| Semantic Efficiency| **0.97**   | 0.00      | 1.99      |
| Latency (ms)       | 187.1      | 279.7     | 299.8     |

### 🔍 Interpretation
- **S-EB-GNN-Q**: achieves **balanced fairness** (0.97 ≈ 1.0) while minimizing energy.
- **WMMSE**: collapses to critical-only allocation → poor fairness.
- **Heuristic**: over-prioritizes critical users (efficiency = 1.99), risking starvation of IoT/Video traffic.

→ **Only S-EB-GNN-Q combines energy efficiency, semantic awareness, and fairness.**

---

## ▶️ Quick Start

```bash
git clone https://github.com/antonio-marlon/s-eb-gnn.git
cd s-eb-gnn
pip install jax equinox matplotlib
