# How S-EB-GNN Achieves Negative Energy in Semantic 6G Networks

In next-generation 6G networks, resource allocation must understand *what* is being communicated — not just how much bandwidth is used. This is the essence of **semantic communication**.

S-EB-GNN (Semantic Energy-Based Graph Neural Network) introduces a physics-aware optimization framework where wireless users and base stations are modeled as an energy-based system. Lower energy indicates higher semantic alignment and system stability.

Through end-to-end training in JAX, the model consistently converges to **negative energy states** (e.g., **−6.60**) in THz/RIS-enabled scenarios. This is not a numerical artifact — it’s a mathematical signature of optimal allocation under semantic prioritization:

- **Critical applications** (e.g., remote surgery) receive highest weight,
- **Video streams** get medium priority,
- **IoT sensors** are allocated residual resources.

The energy function combines:
- Semantic weights,
- THz channel quality (including blockage and path loss),
- Physics-informed regularization (Helmholtz consistency for RIS).

Because the system is fully differentiable, JAX’s autodiff finds configurations more stable than the reference state — hence, **negative energy**.

---

## Reproduce It Yourself

1. Clone the repo:
   ```bash
   git clone https://github.com/antonio-marlon/s-eb-gnn.git
