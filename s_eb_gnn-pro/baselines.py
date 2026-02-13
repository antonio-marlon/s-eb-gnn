# baselines.py
import jax
import jax.numpy as jnp


def wmmse_baseline(adj_matrix, user_types, max_iter=10, power_budget=1.0):
    """
    Simplified WMMSE for vector resource allocation in graph-based wireless networks.

    Args:
        adj_matrix: (N, N) adjacency matrix (channel gains)
        user_types: (N,) semantic types (0=IoT, 1=Video, 2=Critical)
        max_iter: number of WMMSE iterations
        power_budget: total power constraint

    Returns:
        alloc: (N, D) resource allocation (D=8)
    """
    N = adj_matrix.shape[0]
    D = 8  # match your feature dimension

    # Semantic weights (higher = more critical)
    semantic_weights = jnp.array([0.5, 1.0, 5.0])
    weights = semantic_weights[user_types]  # (N,)

    # Initialize random allocation
    key = jax.random.PRNGKey(0)
    alloc = jax.random.normal(key, (N, D)) * 0.1

    # Normalize initial power
    alloc = alloc * jnp.sqrt(power_budget / (jnp.sum(alloc ** 2) + 1e-8))

    for _ in range(max_iter):
        # Step 1: Compute interference (simplified)
        interference = adj_matrix @ (alloc ** 2)  # (N, D)
        noise = 1e-3

        # Step 2: MMSE weights (higher weight → more resources)
        mmse_weights = weights[:, None] / (interference + noise)  # (N, D)

        # Step 3: Update allocation
        alloc = mmse_weights * alloc

        # Step 4: Enforce total power budget
        total_power = jnp.sum(alloc ** 2)
        if total_power > 0:
            alloc = alloc * jnp.sqrt(power_budget / (total_power + 1e-8))

    return alloc


def heuristic_scheduler(adj_matrix, user_types, power_budget=1.0, min_alloc=0.05):
    """
    Robust heuristic scheduler with semantic priority and minimum allocation.
    - Critical (2) > Video (1) > IoT (0)
    - Guarantees min_alloc per user to avoid zero-energy collapse
    """
    N = adj_matrix.shape[0]
    D = 8

    # Semantic priorities
    priorities = jnp.array([0.5, 1.0, 5.0])
    weights = priorities[user_types]  # (N,)

    # Ensure minimum allocation
    base_alloc = jnp.full((N, D), min_alloc)

    # Compute extra power to distribute
    base_power = jnp.sum(base_alloc ** 2)  # total power used by min alloc
    remaining_power = jnp.maximum(power_budget - base_power, 0.0)

    if remaining_power > 0:
        # Distribute remaining power proportionally to priorities
        total_weight = jnp.sum(weights)
        extra_per_user = weights / total_weight * remaining_power
        # Convert to per-dimension allocation
        extra_dim = jnp.sqrt(extra_per_user[:, None] / D)
        base_alloc = base_alloc + extra_dim

    return base_alloc
