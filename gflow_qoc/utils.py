from scipy.optimize import minimize
import numpy as np
import re
from sympy import sympify, sqrt
from itertools import combinations
from torch_geometric.data import Data
import torch
import sys


def parse_args(argv):
    """
    Parses command-line args of the form key=value into a dict.
    Example: python driver.py n_a=2 foo=bar  -> {"n_a": "2", "foo": "bar"}
    """
    out = {}
    for a in argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def all_perfect_matchings(edge_list):
    # Get total number of unique nodes (ignoring color)
    node_set = set()
    for (u, v), _ in edge_list:
        node_set.update([u, v])

    n = len(node_set)
    if n % 2 != 0:
        return []

    perfect_matchings = []

    # Try all combinations of n/2 edges
    for edge_combo in combinations(edge_list, n // 2):
        used_nodes = set()
        valid = True
        for (u, v), _ in edge_combo:
            if u in used_nodes or v in used_nodes:
                valid = False
                break
            used_nodes.update([u, v])
        if valid:
            perfect_matchings.append(edge_combo)

    return perfect_matchings

def matching_to_state(matching, state, weights, num_nodes):
    n_nodes = max(max(u, v) for ((u, v), _) in state) + 1
    #print(n_nodes)
    #print(state)
    #print(weights)
    colors = [-1] * n_nodes  # placeholder for node colors

    weight_product = 1.0
    for edge in matching:
        (u, v), (cu, cv) = edge

        # Assign colors
        colors[u] = cu
        colors[v] = cv

        # Find the exact match index in state using full edge match
        for i, edge_state in enumerate(state):
            if edge == edge_state:
                weight_product *= weights[i]
                break
        else:
            raise ValueError(f"Edge {edge} not found in original state!")

    return weight_product, tuple(colors)

def basis_index(color_tuple, num_colors):
    return int("".join(str(c) for c in color_tuple), num_colors)


def _infer_num_nodes_from_state_dimension(state_dimension, num_colors):
    """Infer ``num_nodes`` from an exact ``num_colors**num_nodes`` dimension."""
    state_dimension = int(state_dimension)
    num_colors = int(num_colors)

    if state_dimension <= 0:
        raise ValueError("State dimension must be positive.")
    if num_colors < 2:
        raise ValueError("num_colors must be at least 2 to infer num_nodes.")

    num_nodes = 0
    dimension = 1
    while dimension < state_dimension:
        dimension *= num_colors
        num_nodes += 1

    if dimension != state_dimension:
        raise ValueError(
            f"target_state length {state_dimension} is not an exact power of "
            f"num_colors={num_colors}."
        )

    return num_nodes


def build_normalized_state(state, weights, num_colors):
    matchings = all_perfect_matchings(state)
    n_nodes = max(max(u, v) for ((u, v), _) in state) + 1
    vec_len = num_colors**n_nodes #Size of hilbert space spanned by nodes and modes.
    state_vector = np.zeros(vec_len, dtype=complex)

    for pm in matchings:
        amp, color_state = matching_to_state(pm, state, weights, n_nodes)
        idx = basis_index(color_state, num_colors)
        state_vector[idx] += amp

    # Normalize
    norm = np.linalg.norm(state_vector)
    if norm == 0:
        raise ValueError("Resulting state vector has zero norm.")
    return state_vector / norm

def state_vector_to_dirac_notation(state_vector, n_nodes, num_colors, tol=1e-16):
    terms = []
    for i, amplitude in enumerate(state_vector):
        if abs(amplitude) > tol:
            # Convert index i to base `num_colors` with `n_nodes` digits
            digits = []
            val = i
            for _ in range(n_nodes):
                digits.append(str(val % num_colors))
                val //= num_colors
            basis_state = ''.join(reversed(digits)).rjust(n_nodes, '0')
            amp_str = f"{amplitude:.6g}"
            terms.append(f"{amp_str}|{basis_state}⟩")
    return " + ".join(terms)


def compute_fidelity(weights, state, target_state, num_colors):
    normalized = build_normalized_state(state, weights, num_colors)
    overlap = np.vdot(target_state, normalized)
    return float(np.abs(overlap)**2)

def fidelity_objective(weights, state, target_state, num_colors, alpha=0):
    try:
        fidelity = compute_fidelity(weights, state, target_state, num_colors)
        return -fidelity # we minimize, so negate
    except Exception as e:
        #print("Error during fidelity evaluation:", e)
        return 0.1 # penalty for failure. 

def fidelity_l1_objective(weights, state, target_state, num_colors, alpha):
    try:
        fidelity = compute_fidelity(weights, state, target_state, num_colors)
        penalty = alpha * np.sum(np.abs(weights))
        return -fidelity + penalty # we minimize, so negate
    except Exception as e:
        #print("Error during fidelity evaluation:", e)
        return len(state)*10  # penalty for failure. 
        #  I use len state to ensure that the L1 regularization does not break the clauses for invalid states, 
        # since weights are bound from -1 to 1, the L1 norm is at most len(state). 

def reward_fidelity(target_state, state, num_colors, pruning, alpha=0.1):
    """This reward function computes the optimal weights and fidelities. Returns squared of fidelity if the state is valid."""
    # Initial weights
    init_weights = np.random.uniform(-1, 1, len(state))

    if target_state.ndim != 1:
        raise ValueError("target_state must be a one-dimensional state vector.")

    target_num_nodes = _infer_num_nodes_from_state_dimension(target_state.size, num_colors)
    state_nodes = {node for ((u, v), _) in state for node in (u, v)}
    if state_nodes != set(range(target_num_nodes)):
        return 0.0, init_weights

    # Optimize
    if pruning:
        result = minimize(
            fidelity_objective,#fidelity_l1_objective,
            init_weights,
            args=(state, target_state, num_colors, alpha),
            method='L-BFGS-B',
            bounds=[(-1, 1)] * len(state),
            options={'disp': True}
        )
        # Optimized weights
        opt_weights = np.asarray(result.x)
        l1_term = alpha*np.sum(np.abs(opt_weights))
        opt_fidelity = l1_term-result.fun
    else:
        result = minimize(
            fidelity_objective,
            init_weights,
            args=(state, target_state, num_colors),
            method='L-BFGS-B',
            bounds=[(-1, 1)] * len(state),
            options={'disp': True}
        )
        # Optimized weights
        opt_weights = np.asarray(result.x)
        opt_fidelity = -result.fun

    if opt_fidelity < 0.0:
        return 0, opt_weights#opt_fidelity
    else:
        return opt_fidelity**2, opt_weights#np.exp(opt_fidelity/2)#**2
    #return opt_fidelity

def opt_fidelity(target_state, state, num_colors):
    """This reward function computes the optimal weights and fidelities. Returns squared of fidelity if the state is valid."""
    # Initial weights
    init_weights = np.random.uniform(-1, 1, len(state))

    # Optimize
    result = minimize(
        fidelity_objective,
        init_weights,
        args=(state, target_state, num_colors),
        method='L-BFGS-B',
        bounds=[(-1, 1)] * len(state),
        options={'disp': True}
    )
    # Optimized weights
    #opt_weights = np.asarray(result.x)
    opt_fidelity = -result.fun

    if opt_fidelity < 0.0:
        return 0#opt_fidelity
    else:
        return opt_fidelity#np.exp(opt_fidelity/2)#**2
    #return opt_fidelity

def prune_state_by_weight(state, weights, weight_eps=1e-2, keep_at_least=4):
    """
    Keeps edges whose |weight| > weight_eps. Ensures at least keep_at_least edges remain
    by keeping the largest-|w| edges if threshold prunes too much.
    """
    weights = np.asarray(weights)

    absw = np.abs(weights)
    keep = absw > weight_eps

    # if keep.sum() < keep_at_least and len(weights) > 0:
    #     # keep the top-|w| edges
    #     top_idx = np.argsort(absw)[::-1][:keep_at_least]
    #     keep = np.zeros_like(keep, dtype=bool)
    #     keep[top_idx] = True

    pruned_state = [e for e, k in zip(state, keep) if k]
    pruned_weights = weights[keep]
    return pruned_state, pruned_weights, keep

def parse_dirac_expression(expr, num_nodes, num_colors ):
    """
    Parse expressions like '1|0000⟩ + 1|1111⟩' or '1/sqrt(2)|0000⟩' into a normalized NumPy vector.
    """
    expr = expr.replace("⟩", "").replace(" ", "")
    terms = re.findall(r'([^\+]+?\|[0-9]+)', expr) #re.findall(r'([^\+]+?\|[01]+)', expr) #Bug! This assumes binary! Solved.

    if not terms:
        raise ValueError("Invalid expression or format.")

    amplitudes = {}
    for term in terms:
        match = re.match(r'([^|]+)\|([0-9]+)', term)
        if not match:
            raise ValueError(f"Invalid term: {term}")
        coef_str, basis = match.groups()
        coef_val = complex(sympify(coef_str).evalf())
        amplitudes[basis] = coef_val
    dim = num_colors ** num_nodes
    vec = np.zeros(dim, dtype=complex)

    for digits, amplitude in amplitudes.items():
        if len(digits) != num_nodes:
            raise ValueError(f"Basis state length {len(digits)} does not match num_nodes={num_nodes}")
        if any(int(d) >= num_colors for d in digits):
            raise ValueError(f"Digit in basis state {digits} exceeds allowed num_colors={num_colors}")
        index = int(digits, base=num_colors)
        vec[index] = amplitude

    # Normalize
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-norm state.")
    return vec / norm


def state_to_data(state, num_nodes, num_colors):
    """
    Convert the current state (e.g., a set of selected FEATURE_KEYS) into a PyG Data object.
    `state` is assumed to be a set of FEATURE_KEYS currently active (edges with color pairs).
    """
    node_features = torch.eye(num_nodes)  # one-hot for node IDs, or use learned embeddings

    # Extract edges and their color pairs
    edge_list = []
    edge_features = []
    for ((u, v), (c1, c2)) in state:
        edge_list.append([u, v])
        #edge_list.append([v, u])  # Undirected
        edge_features.append([c1, c2])
        #edge_features.append([c2, c1])  # Make edge_attr symmetric if needed

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # shape [2, num_edges]
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    # print("Edge index shape:", edge_index.shape)
    # print("Edge attr shape:", edge_attr.shape)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

#Functions for pruning based on Logical clauses.

def target_support_from_vector(target_state, tol=1e-12):
    """
    Return set of basis indices whose amplitude magnitude > tol.
    """
    target_state = np.asarray(target_state)
    return set(np.where(np.abs(target_state) > tol)[0].tolist())


def pm_counts_by_basis_index(state, num_colors):
    """
    Count how many perfect matchings lead to each basis index (vertex-coloring outcome).

    Returns:
        counts: dict[int -> int]
        num_nodes: int
    """
    if len(state) == 0:
        return {}, 0

    matchings = all_perfect_matchings(state)
    n_nodes = max(max(u, v) for ((u, v), _) in state) + 1

    # Use weights=1 so matching_to_state works but amplitude doesn't matter for counting.
    ones = np.ones(len(state), dtype=float)

    counts = {}
    for pm in matchings:
        _, color_state = matching_to_state(pm, state, ones, n_nodes)
        idx = basis_index(color_state, num_colors)
        counts[idx] = counts.get(idx, 0) + 1

    return counts, n_nodes


def satisfies_clauses(state, target_support, num_colors, num_nodes=None):
    """
    Implements the two logic clauses via PM enumeration:

      S: For every idx in target_support, there exists >= 1 perfect matching producing idx.
      C: For any idx not in target_support, it is forbidden to have exactly 1 PM producing idx. This Clause doesn't work.

    Returns:
        bool
    """
    if num_nodes is not None:
        state_nodes = {node for ((u, v), _) in state for node in (u, v)}
        if state_nodes != set(range(num_nodes)):
            return False

    counts, _ = pm_counts_by_basis_index(state, num_colors)

    # S clause
    for idx in target_support:
        if counts.get(idx, 0) < 1:
            return False

    # C clause
    # for idx, c in counts.items():
    #     if idx not in target_support and c == 1:
    #         return False

    return True

def prune_state_by_logic(
    state,
    target_state,
    num_colors,
    weights=None,
    order="increasing_abs_weight",
    support_tol=1e-12,
    max_passes=10,
):
    """
    Greedily remove edges if the remaining graph still satisfies Logical clauses (S & C).

    Args:
        state: list of edges like [((u,v),(cu,cv)), ...]
        target_state: complex vector of length num_colors**num_nodes
        num_colors: int
        weights: optional np array aligned with state (used only to decide removal order)
        order:
            - "increasing_abs_weight" (default): try remove smallest-|w| edges first
            - "original": try in original order
        support_tol: threshold for target support extraction
        max_passes: number of times to repeat the whole greedy sweep (usually 1 is enough;
                    >1 helps if deletion of one edge enables others)

    Returns:
        pruned_state, pruned_weights, keep_mask (mask over original edges)
    """
    state0 = list(state)
    n0 = len(state0)

    if weights is not None:
        weights0 = np.asarray(weights, dtype=float)
        if len(weights0) != n0:
            raise ValueError(f"weights length {len(weights0)} != number of edges {n0}")
    else:
        weights0 = None

    if n0 == 0:
        return [], (np.asarray([]) if weights0 is not None else None), np.zeros(0, dtype=bool)

    target_state = np.asarray(target_state)
    if target_state.ndim != 1:
        raise ValueError("target_state must be a one-dimensional state vector.")
    target_num_nodes = _infer_num_nodes_from_state_dimension(target_state.size, num_colors)

    target_support = target_support_from_vector(target_state, tol=support_tol)

    # If the current state already violates clauses, pruning can't fix that reliably.
    # Return original to avoid surprises.
    if not satisfies_clauses(state0, target_support, num_colors, num_nodes=target_num_nodes):
        keep = np.ones(n0, dtype=bool)
        return state0, (weights0.copy() if weights0 is not None else None), keep

    # Decide removal order over ORIGINAL indices
    orig_indices = list(range(n0))
    if order == "original" or weights0 is None:
        removal_order = orig_indices
    elif order == "increasing_abs_weight":
        removal_order = sorted(orig_indices, key=lambda i: abs(weights0[i]))
    else:
        raise ValueError(f"Unknown order='{order}'")

    # Maintain a live list + mapping to original indices so we can build keep_mask at end
    cur_state = list(state0)
    cur_weights = weights0.copy() if weights0 is not None else None
    cur_orig = list(orig_indices)  # cur_state[k] came from original index cur_orig[k]

    for _pass in range(max_passes):
        changed = False

        # Build position map (orig_idx -> current position)
        pos_of = {oi: k for k, oi in enumerate(cur_orig)}

        for oi in removal_order:
            if oi not in pos_of:
                continue  # already removed

            k = pos_of[oi]

            # Try removing edge k
            cand_state = cur_state[:k] + cur_state[k+1:]

            # If removing makes it impossible to have any PMs at all, clauses will fail anyway.
            if not satisfies_clauses(cand_state, target_support, num_colors, num_nodes=target_num_nodes):
                continue

            # Accept removal
            cur_state = cand_state
            if cur_weights is not None:
                cur_weights = np.concatenate([cur_weights[:k], cur_weights[k+1:]])
            removed_oi = cur_orig[k]
            cur_orig = cur_orig[:k] + cur_orig[k+1:]

            changed = True

            # Update mapping cheaply: rebuild (graphs are small; this is fine)
            pos_of = {oi2: kk for kk, oi2 in enumerate(cur_orig)}

        if not changed:
            break

    keep_mask = np.zeros(n0, dtype=bool)
    for oi in cur_orig:
        keep_mask[oi] = True

    return cur_state, cur_weights, keep_mask

#Function for count rates (based on PyTheus), need to double check, PyTheus functions are not well documented.
def build_unnormalized_state(state, weights, num_colors):
    """
    Build the (generally unnormalized) post-selected state vector by summing
    amplitudes from all perfect matchings.

    Returns:
        state_vector (np.ndarray complex) of length num_colors**num_nodes
    """
    matchings = all_perfect_matchings(state)
    if len(state) == 0:
        return np.zeros(0, dtype=complex)

    n_nodes = max(max(u, v) for ((u, v), _) in state) + 1
    vec_len = num_colors ** n_nodes
    state_vector = np.zeros(vec_len, dtype=complex)

    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(state):
        raise ValueError(f"weights length {len(weights)} != number of edges {len(state)}")

    for pm in matchings:
        amp, color_state = matching_to_state(pm, state, weights, n_nodes)
        idx = basis_index(color_state, num_colors)
        state_vector[idx] += amp

    return state_vector

def count_rate_to_target(
    state,
    weights,
    target_state,
    num_colors,
    mode="pytheus_new",   # "pytheus_new", "pytheus_old", "raw_overlap", "total_rate"
    tol=1e-15,
):
    """
    Target-conditioned count rate, PyTheus-style.

    Let psi be the unnormalized post-selected state vector built from perfect matchings.
    Let t be the (normalized) target state vector.

    Modes:
      - "raw_overlap":  |<t|psi>|^2
      - "pytheus_new":  |<t|psi>|^2 / (1 + ||psi||)^2    (matches PyTheus new_loss branch)
      - "pytheus_old":  |<t|psi>|^2 / (1 + ||psi||^2)    (closer to their symbolic 1/(1+norm) form)
      - "total_rate":   ||psi||^2  (not target-conditioned; included for completeness)

    Returns:
      float
    """
    psi = build_unnormalized_state(state, weights, num_colors)
    if psi.size == 0:
        return 0.0

    target_state = np.asarray(target_state, dtype=complex)
    if target_state.shape != psi.shape:
        raise ValueError(
            f"target_state shape {target_state.shape} != psi shape {psi.shape}. "
            "Check num_colors/num_nodes consistency."
        )

    # Normalize target (PyTheus uses a normalized target vector) :contentReference[oaicite:2]{index=2}
    t_norm = np.linalg.norm(target_state)
    if t_norm <= tol:
        raise ValueError("target_state has near-zero norm; cannot define target-conditioned count rate.")
    t = target_state / t_norm

    # Target overlap on the UNnormalized produced state
    overlap = np.vdot(t, psi)              # <t|psi>
    raw_overlap = float((overlap.conjugate() * overlap).real)  # |<t|psi>|^2

    if mode == "raw_overlap":
        return raw_overlap

    psi_norm = float(np.linalg.norm(psi))
    psi_norm2 = float(np.vdot(psi, psi).real)

    if mode == "pytheus_new":
        # PyTheus new_loss uses state/(1+||state||) then |dot|^2 :contentReference[oaicite:3]{index=3}
        denom = (1.0 + psi_norm) ** 2
        return raw_overlap / max(denom, tol)

    if mode == "pytheus_old":
        # Closer to their older "1/(1+norm)" form where norm is typically ||psi||^2 :contentReference[oaicite:4]{index=4}
        denom = 1.0 + psi_norm2
        return raw_overlap / max(denom, tol)

    if mode == "total_rate":
        return psi_norm2

    raise ValueError(f"Unknown mode='{mode}'.")