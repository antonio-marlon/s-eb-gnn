# s_eb_gnn.py
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List


def normalize(x, eps=1e-8):
    """Normalize vectors along the last axis."""
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


# --- QUANTUM-INSPIRED UTILS (new, minimal) ---
def quantum_attention_kernel(x_i, x_j, gamma=0.1):
    """Simulates quantum fidelity between node states (classical simulation)."""
    norm_i = jnp.linalg.norm(x_i)
    norm_j = jnp.linalg.norm(x_j)
    inner = jnp.dot(x_i, x_j)
    return jnp.exp(-gamma * (norm_i ** 2 + norm_j ** 2 - 2 * inner))


def create_quantum_semantic_adjacency(adj_physical, user_types, node_features, base_weights=[0.5, 1.0, 5.0]):
    """
    Enhance physical adjacency using quantum graph kernels for semantic refinement.

    Args:
        adj_physical: (N, N) physical adjacency matrix
        user_types: (N,) array of semantic types (0=IoT, 1=Video, 2=Critical)
        node_features: (N, D) current node features
        base_weights: list of base weights [IoT, Video, Critical]

    Returns:
        (N, N) quantum-enhanced semantic adjacency matrix
    """
    N = adj_physical.shape[0]
    base_priority = jnp.array(base_weights)[user_types]

    # Compute quantum similarity to "critical" prototype
    critical_proto = jnp.ones_like(node_features[0]) * 5.0  # ideal critical state
    quantum_sim = jnp.array([
        quantum_attention_kernel(node_features[i], critical_proto)
        for i in range(N)
    ])

    # Refine weights with quantum similarity
    refined_weights = base_priority * (1.0 + 0.5 * quantum_sim)

    # Apply quantum-enhanced boost
    boost = refined_weights[:, None] * refined_weights[None, :]
    return adj_physical * (1.0 + 0.3 * boost)


class MessagePassing(eqx.Module):
    W_msg: jnp.ndarray
    W_self: jnp.ndarray

    def __init__(self, dim: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        scale = jnp.sqrt(1.0 / dim)
        self.W_msg = jax.random.normal(k1, (dim, dim)) * scale
        self.W_self = jax.random.normal(k2, (dim, dim)) * scale

    def __call__(self, h: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        """Perform message passing over the graph."""
        messages = adj @ (h @ self.W_msg)
        self_update = h @ self.W_self
        return jnp.tanh(messages + self_update)


class SemanticEnergyHead(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray
    semantic_weights: jnp.ndarray

    def __init__(self, dim: int, semantic_weights, key: jax.random.PRNGKey):
        self.W = jax.random.normal(key, (dim, 1)) * jnp.sqrt(1.0 / dim)
        self.b = jnp.zeros((1,))
        self.semantic_weights = jnp.array(semantic_weights)

    def __call__(self, h: jnp.ndarray, user_types: jnp.ndarray) -> jnp.ndarray:
        """Compute weighted energy based on semantic priorities."""
        node_energies = h @ self.W + self.b
        weights = self.semantic_weights[user_types]
        # Lower energy for critical types (higher weights)
        weighted_energy = node_energies.squeeze() / (weights + 1e-8)
        return jnp.sum(weighted_energy)


class SEBGNN(eqx.Module):
    layers: List[MessagePassing]
    energy: SemanticEnergyHead

    def __init__(self, depth: int, dim: int, semantic_weights, key: jax.random.PRNGKey):
        keys = jax.random.split(key, depth + 1)
        self.layers = [MessagePassing(dim, keys[i]) for i in range(depth)]
        self.energy = SemanticEnergyHead(dim, semantic_weights, keys[-1])

        def __call__(self, x, adj, user_types):
        """
        Energy-based GNN with per-node normalization (MIT-inspired).
        Returns average energy per node (scalar).
        """
        # Semantic weights: IoT=0.5, Video=1.0, Critical=5.0
        semantic_weights = jnp.array([0.5, 1.0, 5.0])
        weights = semantic_weights[user_types]  # (N,)

        # Apply graph layers
        h = x
        for layer in self.layers:
            h = layer(h, adj)

        # Compute per-node utility (dot product of features)
        utilities = jnp.sum(h * x, axis=1)  # (N,)

        # Weighted energy per node
        energy_per_node = -weights * utilities  # (N,)

        # MIT-inspired: return AVERAGE energy (not total)
        return jnp.mean(energy_per_node)


# === THz + RIS ===
def create_thz_adjacency(distances, frequencies, blocked=None):
    """
    Create adjacency matrix based on THz channel physics.

    Args:
        distances: (N, N) matrix of distances in meters
        frequencies: (N,) array of frequencies in GHz
        blocked: (N, N) boolean matrix indicating blocked links

    Returns:
        (N, N) adjacency matrix
    """
    c = 3e8  # speed of light
    fc = frequencies[:, None] * 1e9  # Hz
    path_loss = (4 * jnp.pi * distances * fc / c) ** 2
    if blocked is not None:
        path_loss = jnp.where(blocked, path_loss * 1e3, path_loss)
    adj = 1.0 / (1.0 + path_loss)
    adj = adj - jnp.diag(jnp.diag(adj))  # remove self-loops
    return adj


def add_ris_to_features(x, ris_indices, phase_shifts):
    """
    Inject RIS phase shifts into node features.

    Args:
        x: (N, D) node features
        ris_indices: indices of RIS nodes
        phase_shifts: phase values for each RIS

    Returns:
        Updated features with RIS phases in first dimension
    """
    return x.at[ris_indices, 0].set(phase_shifts)


# === Semantic Layer (original, kept for backward compatibility) ===
def create_semantic_adjacency(adj_physical, user_types, semantic_weights):
    """
    Enhance physical adjacency with semantic prioritization.

    Args:
        adj_physical: (N, N) physical adjacency matrix
        user_types: (N,) array of semantic types (0=IoT, 1=Video, 2=Critical)
        semantic_weights: list of weights [IoT_weight, Video_weight, Critical_weight]

    Returns:
        (N, N) semantic-enhanced adjacency matrix
    """
    priority = jnp.array(semantic_weights)[user_types]
    boost = priority[:, None] * priority[None, :]
    return adj_physical * (1.0 + 0.3 * boost)


# === Solver ===
def solve_allocation(model, x, adj, user_types, steps=30, lr=0.1):
    """
    Solve resource allocation via energy minimization.

    Args:
        model: SEBGNN instance
        x: initial node features
        adj: semantic adjacency matrix
        user_types: semantic types of nodes
        steps: number of optimization steps
        lr: learning rate

    Returns:
        Optimized node features
    """

    def body_fn(_, xi):
        grad = jax.grad(lambda z: model(z, adj, user_types))(xi)
        return xi - lr * normalize(grad)

    return jax.lax.fori_loop(0, steps, body_fn, x)
