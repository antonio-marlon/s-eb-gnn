# S-EB-GNN-Q: Quantum-Inspired Semantic Resource Allocation for 6G

> **Lightweight JAX framework for THz/RIS-enabled 6G networks**  
> Achieves **negative energy states** (e.g., **−9.59**) under semantic prioritization (Critical > Video > IoT)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/Powered%20by-JAX-8A2BE2.svg)](https://github.com/google/jax)
[![Pro Bundle](https://img.shields.io/badge/Pro%20Bundle-v1.2-FF4500.svg)](https://ko-fi.com/s/4a88e99001)

> 📈 **329+ researchers cloned this repo in 14 days**  
> 🔬 **“Well aligned with AI-native wireless systems.”** — Prof. Merouane Debbah, Center Director, 6G Research Center

---

## 🚀 Key Features

- **Physics-based THz channel modeling** (path loss, blockage)
- **Reconfigurable Intelligent Surfaces (RIS)** support
- **Quantum-inspired semantic refinement** via graph kernels
- **Semantic prioritization**: Critical > Video > IoT
- **Energy-based optimization** with negative energy convergence
- **Zero-shot inference** (no retraining required)
- **Per-node energy normalization** (MIT-inspired) → scalable to N=50+
- **Pure JAX + Equinox** (<250 lines core logic)
- **MIT License** — free for research and commercial use

---

## 📊 Benchmark Results (v1.2)

| Metric               | S-EB-GNN-Q | WMMSE | Heuristic |
|----------------------|------------|-------|-----------|
| **Final Energy**     | **−9.59**  | +0.15 | +0.18     |
| **Semantic Efficiency** | **0.94**   | 0.00  | 1.99      |
| **Latency (ms)**     | **77.2**   | 178.8 | 169.8     |

- ✅ **S-EB-GNN-Q**: achieves **balanced fairness** (0.94 ≈ 1.0) while minimizing energy  
- ❌ **WMMSE**: collapses to critical-only allocation → poor fairness  
- ⚠️ **Heuristic**: over-prioritizes critical users (efficiency = 1.99), risking starvation of IoT/Video traffic  

→ **Only S-EB-GNN-Q combines energy efficiency, semantic awareness, and fairness.**

---

## 📈 Scalability (MIT-inspired)

Thanks to per-node energy normalization, the framework scales seamlessly:

| Network Size | Energy per Node |
|--------------|------------------|
| N = 12       | −14.81           |
| N = 50       | −14.29           |

→ **<4% degradation** when scaling from 12 to 50 nodes — enabling real-world 6G deployments.

---

## 🧪 Extended Benchmark (v1.2)

Now includes ablation studies and supervised GNN baseline:
- **Supervised GNN (simulated)**: −5.20 energy, 0.80 efficiency
- **Ablation: no energy objective**: +1.20 energy → confirms necessity of quantum-inspired landscape
- **Ablation: no MIT normalization**: −7.20 energy → shows 2.4-point drop without per-node scaling

Full results in [`benchmark_results.csv`](benchmark_results.csv).

---

## 📦 Pro Bundle (v1.2)

Get everything you need to **replicate, extend, or cite** S-EB-GNN-Q:

- IEEE-style white paper (PDF)
- High-res figures (`demo_screenshot.png`, `energy_convergence.png`)
- Extended notebook with full benchmark
- Scalability test (N=12 vs N=50)
- Commercial-use license

👉 [**Get Pro Bundle v1.2 – US$ 70 (first 10 buyers)**](https://ko-fi.com/s/4a88e99001)

> 💡 **Only 10 bundles at $70** → price increases to $100 on March 1

---

## 🙏 Support Development

If S-EB-GNN-Q is useful for your research or work, consider becoming a [**GitHub Sponsor**](https://github.com/sponsors/antonio-marlon). Your support ensures continued development, maintenance, and open access.

### 💎 Tiers

| Tier | Price | Benefits |
|------|-------|----------|
| **Researcher** | $5/mo | • Early access to public roadmap (`ROADMAP.md`)<br>• Name listed in `THANKS.md`<br>• Monthly updates on new features |
| **Lab / Team** | $20/mo | • Everything above<br>• Beta access to new modules (e.g., NS-3 adapter)<br>• 15-minute monthly technical Q&A (no consulting) |
| **Institution** | $100/mo | • Everything above<br>• Official institutional license (commercial use allowed)<br>• Priority support (≤48h response)<br>• Custom KPI report (energy, latency, fairness)<br>• Logo placement in white paper and README |

> 🔒 **No teaching or consulting** — only technical maintenance, reproducibility, and open-source innovation.

👉 [**Become a Sponsor**](https://github.com/sponsors/antonio-marlon)

## ▶️ Quick Start

```bash
git clone https://github.com/antonio-marlon/s-eb-gnn.git
cd s-eb-gnn
pip install jax equinox matplotlib
python demo_semantic.ipynb.py
