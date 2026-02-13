# %% [markdown]
# # S-EB-GNN-Q: Quantum-Inspired Semantic Resource Allocation for 6G
# Benchmarking against WMMSE baseline. Achieves negative energy states (e.g., −6.60) under semantic prioritization.

# %%
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
from s_eb_gnn import (
    SEBGNN, create_thz_adjacency, add_ris_to_features,
    create_quantum_semantic_adjacency, solve_allocation, normalize
)
from baselines import wmmse_baseline
import matplotlib.pyplot as plt
import time
import csv

# Initial setup
key = jax.random.PRNGKey(42)

# Network parameters
N = 12  # 10 users + 2 RIS elements
D = 8
ris_nodes = jnp.array([10, 11])

# Random 2D positions (meters)
pos = jax.random.uniform(key, (N, 2)) * 100
distances = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :])**2, axis=-1))

# Frequencies (GHz)
freqs = jnp.where(jnp.arange(N) < 5, 3.5, 140.0)
freqs = freqs.at[ris_nodes].set(140.0)

# User types: 0=IoT, 1=Video, 2=Critical
user_types = jnp.array([0]*5 + [1]*3 + [2]*2 + [0]*2)
assert len(user_types) == N

# Blocked links (30%)
k1, k2 = jax.random.split(key)
blocked = jax.random.bernoulli(k1, 0.3, (N, N))
blocked = (blocked + blocked.T) > 0

# Initial node features
x = jax.random.normal(k2, (N, D))

# RIS phase shifts
phase_shifts = jnp.array([0.8, 1.5])
x = add_ris_to_features(x, ris_nodes, phase_shifts)

# %%
# Create physical adjacency (THz + blockage)
adj_thz = create_thz_adjacency(distances, freqs, blocked)

# Apply quantum-inspired semantic layer
adj_semantic = create_quantum_semantic_adjacency(
    adj_thz, user_types, x, base_weights=[0.5, 1.0, 5.0]
)

# %%
# Initialize model
model_key, train_key = jax.random.split(key)
model = SEBGNN(depth=3, dim=D, semantic_weights=[0.5, 1.0, 5.0], key=model_key)

# === BENCHMARKING FUNCTION ===
def semantic_efficiency(alloc, user_types):
    critical_mask = (user_types == 2)
    non_critical_mask = (user_types != 2)

    # Filter out near-zero allocations
    def safe_norms(vecs):
        norms = jnp.linalg.norm(vecs, axis=1)
        return norms[norms > 1e-3]  # ignore negligible allocations

    crit_norms = safe_norms(alloc[critical_mask])
    non_norms = safe_norms(alloc[non_critical_mask])

    if len(crit_norms) == 0 or len(non_norms) == 0:
        return jnp.array(0.0)

    mean_crit = jnp.mean(crit_norms)
    mean_non = jnp.mean(non_norms)

    return mean_crit / jnp.maximum(mean_non, 1e-6)

def time_allocation(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    latency_ms = (time.time() - start) * 1000
    return result, latency_ms

# === RUN BENCHMARK ===
# S-EB-GNN-Q
alloc_ours, latency_ours = time_allocation(
    solve_allocation, model, x, adj_semantic, user_types, steps=50, lr=0.1
)
energy_ours = model(alloc_ours, adj_semantic, user_types)
eff_ours = semantic_efficiency(alloc_ours, user_types)

# WMMSE baseline
alloc_wmmse, latency_wmmse = time_allocation(
    wmmse_baseline, adj_thz, user_types, max_iter=10, power_budget=1.0
)
energy_wmmse = model(alloc_wmmse, adj_semantic, user_types)
eff_wmmse = semantic_efficiency(alloc_wmmse, user_types)

# === PRINT RESULTS ===
print("="*50)
print("📊 BENCHMARK: S-EB-GNN-Q vs WMMSE")
print("="*50)
print(f"{'Metric':<20} {'S-EB-GNN-Q':<15} {'WMMSE':<15}")
print("-"*50)
print(f"{'Final Energy':<20} {energy_ours:<15.2f} {energy_wmmse:<15.2f}")
print(f"{'Semantic Eff.':<20} {eff_ours:<15.2f} {eff_wmmse:<15.2f}")
print(f"{'Latency (ms)':<20} {latency_ours:<15.1f} {latency_wmmse:<15.1f}")
print("="*50)

# Save to CSV
with open('benchmark_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'S-EB-GNN-Q', 'WMMSE'])
    writer.writerow(['Final Energy', energy_ours.item(), energy_wmmse.item()])
    writer.writerow(['Semantic Efficiency', eff_ours.item(), eff_wmmse.item()])
    writer.writerow(['Latency (ms)', latency_ours, latency_wmmse])

print("💾 Results saved to benchmark_results.csv")

# Use our allocation for figures
alloc = alloc_ours
final_energy = energy_ours

# %%
# === FIGURE 1: Main results ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im1 = axes[0].imshow(adj_thz, cmap='viridis', vmin=0, vmax=1)
axes[0].set_title('Physical Adjacency\n(THz + Blockage)')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(adj_semantic, cmap='plasma', vmin=0, vmax=jnp.max(adj_semantic))
axes[1].set_title('Quantum Semantic Adjacency')
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

colors = ['green', 'orange', 'red']
for i in range(N):
    ut = int(user_types[i])
    label = None
    if i == 0: label = 'IoT'
    elif i == 5: label = 'Video'
    elif i == 8: label = 'Critical'
    axes[2].scatter(alloc[i, 0], alloc[i, 1], c=colors[ut], label=label, s=80)
axes[2].set_title(f'Final Allocation\n(Energy: {final_energy:.2f})')
axes[2].legend()

plt.tight_layout()
plt.savefig("demo_screenshot.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# === FIGURE 2: Energy convergence ===
def track_energy(model, x, adj, user_types, steps=50, lr=0.1):
    energies = []
    current_x = x
    def energy_fn(xi):
        return model(xi, adj, user_types)
    for step in range(steps + 1):
        e = energy_fn(current_x)
        energies.append(e)
        if step < steps:
            grad = jax.grad(energy_fn)(current_x)
            current_x = current_x - lr * normalize(grad)
    return jnp.array(energies)

energy_history = track_energy(model, x, adj_semantic, user_types, steps=50, lr=0.1)

plt.figure(figsize=(8, 4))
plt.plot(energy_history, marker='o', markersize=3, linewidth=2, color='purple')
plt.title('Energy Convergence During Optimization')
plt.xlabel('Optimization Step')
plt.ylabel('Total Energy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig("energy_convergence.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"📉 Final energy after convergence: {energy_history[-1]:.2f}")
