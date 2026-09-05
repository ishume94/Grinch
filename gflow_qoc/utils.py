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
        # Restore these two lines only together with fidelity_l1_objective above.
        # l1_term = alpha*np.sum(np.abs(opt_weights))
        # opt_fidelity = l1_term-result.fun
        opt_fidelity = -result.fun
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
        amplitudes[basis] = amplitudes.get(basis, 0) + coef_val
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
    Check target support via perfect-matching enumeration:

      S: For every idx in target_support, there exists >= 1 perfect matching producing idx.

    The unwanted-basis clause C remains disabled: even a single unwanted
    matching can have a sufficiently small amplitude after weight optimization.
    Support is necessary, but does not by itself guarantee target fidelity.

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


def _pruning_matching_data(state, num_colors, num_nodes):
    """Compile matching edge indices and outcomes once for a fixed node universe."""
    incident = [[] for _ in range(num_nodes)]
    for index, ((u, v), _) in enumerate(state):
        if u != v:
            incident[u].append(index)
            incident[v].append(index)

    matching_edges = []
    matching_basis = []
    colors = [-1] * num_nodes

    def visit(unmatched, chosen):
        if not unmatched:
            matching_edges.append(tuple(chosen))
            matching_basis.append(basis_index(tuple(colors), num_colors))
            return
        node = min(unmatched)
        for index in incident[node]:
            (u, v), (cu, cv) = state[index]
            if u in unmatched and v in unmatched:
                colors[u], colors[v] = cu, cv
                visit(unmatched.difference((u, v)), chosen + [index])

    if num_nodes > 0 and num_nodes % 2 == 0:
        visit(set(range(num_nodes)), [])
    return (
        np.asarray(matching_edges, dtype=int).reshape(-1, max(1, num_nodes // 2)),
        np.asarray(matching_basis, dtype=int),
    )

def prune_state_by_logic(
    state,
    target_state,
    num_colors,
    weights=None,
    order="increasing_abs_weight",
    support_tol=1e-12,
    max_passes=10,
    fidelity_threshold=0.95,
    reoptimize=True,
):
    """
    Remove edges while preserving target support and, with weights, fidelity.

    Repeated copies of the same colored edge are invalid repeated actions, not
    parallel sources. Keep their first occurrence and its weight before pruning,
    including when the canonical graph fails the initial support/fidelity check.
    The unwanted-basis clause stays disabled; the target-support check already
    protects edges needed by a target matching even if they also create intruders.

    Args:
        state: list of edges like [((u,v),(cu,cv)), ...]
        target_state: normalized complex vector of length num_colors**num_nodes
        num_colors: int
        weights: optional np array aligned with state. If omitted, retain legacy
                 support-only pruning, which does not guarantee target fidelity.
        order:
            - "increasing_abs_weight" (default): try remove smallest-|w| edges first
            - "original": try in original order
        support_tol: threshold for target support extraction
        max_passes: number of times to repeat the whole greedy sweep (usually 1 is enough;
                    >1 helps if deletion of one edge enables others)
        fidelity_threshold: minimum physical fidelity (not squared reward) to accept.
        reoptimize: if inherited weights fail the fidelity test, try bounded
                    optimization initialized from the surviving weights. Accept
                    only after verifying the resulting physical fidelity. Invalid
                    starting graphs/weights are returned without optimization.

    Returns:
        pruned_state, pruned_weights, keep_mask (mask over original edges)
    """
    input_state = list(state)
    n_input = len(input_state)

    if weights is not None:
        input_weights = np.asarray(weights, dtype=float)
        if input_weights.shape != (n_input,):
            raise ValueError(f"weights must have shape ({n_input},), got {input_weights.shape}")
    else:
        input_weights = None

    if n_input == 0:
        return [], (np.asarray([]) if weights is not None else None), np.zeros(0, dtype=bool)

    first_indices = []
    seen = set()
    for index, edge in enumerate(input_state):
        key = (tuple(edge[0]), tuple(edge[1]))
        if key not in seen:
            seen.add(key)
            first_indices.append(index)
    first_indices = np.asarray(first_indices, dtype=int)
    state0 = [input_state[index] for index in first_indices]
    weights0 = input_weights[first_indices].copy() if input_weights is not None else None
    n0 = len(state0)
    keep = np.ones(n0, dtype=bool)

    def result_for(kept, live_weights):
        keep_mask = np.zeros(n_input, dtype=bool)
        keep_mask[first_indices[kept]] = True
        pruned_weights = live_weights[kept].copy() if live_weights is not None else None
        return [edge for edge, retained in zip(state0, kept) if retained], pruned_weights, keep_mask

    target_state = np.asarray(target_state)
    if target_state.ndim != 1:
        raise ValueError("target_state must be a one-dimensional state vector.")
    target_num_nodes = _infer_num_nodes_from_state_dimension(target_state.size, num_colors)
    if not np.isfinite(fidelity_threshold) or not 0 <= fidelity_threshold <= 1:
        raise ValueError("fidelity_threshold must be finite and between 0 and 1.")
    target_support = target_support_from_vector(target_state, tol=support_tol)
    state_nodes = {node for ((u, v), _) in state0 for node in (u, v)}
    if state_nodes != set(range(target_num_nodes)):
        return result_for(keep, weights0)

    pm_edges, pm_basis = _pruning_matching_data(state0, num_colors, target_num_nodes)
    if not target_support or not target_support.issubset(set(pm_basis)):
        return result_for(keep, weights0)

    if order not in ("original", "increasing_abs_weight"):
        raise ValueError(f"Unknown order='{order}'")

    # Deletions can only remove existing PMs. Reuse their incidence and basis
    # indices for both support checks and every numerical objective evaluation.
    unique_basis, basis_rows = np.unique(pm_basis, return_inverse=True)
    compact_target = target_state[unique_basis]

    def matching_fidelity(local_weights, local_edges, local_basis):
        if not np.all(np.isfinite(local_weights)):
            return np.nan
        amplitudes = np.zeros(len(unique_basis), dtype=complex)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            np.add.at(amplitudes, local_basis, np.prod(local_weights[local_edges], axis=1))
            norm = np.linalg.norm(amplitudes)
            if not np.isfinite(norm) or norm == 0:
                return np.nan
            return float(abs(np.vdot(compact_target, amplitudes / norm)) ** 2)

    live_pm = np.ones(len(pm_basis), dtype=bool)
    if weights0 is not None:
        initial_fidelity = matching_fidelity(weights0, pm_edges, basis_rows)
        if not np.isfinite(initial_fidelity) or initial_fidelity < fidelity_threshold:
            return result_for(keep, weights0)

    for _pass in range(max_passes):
        changed = False
        removal_order = np.flatnonzero(keep).tolist()
        if order == "increasing_abs_weight" and weights0 is not None:
            removal_order.sort(key=lambda index: abs(weights0[index]))

        for index in removal_order:
            candidate_pm = live_pm & ~np.any(pm_edges == index, axis=1)
            # This rejects a target-critical edge without optimizing weights,
            # regardless of whether it also participates in unwanted matchings.
            if not target_support.issubset(set(pm_basis[candidate_pm])):
                continue

            candidate_keep = keep.copy()
            candidate_keep[index] = False
            if weights0 is not None:
                candidate_weights = weights0[candidate_keep].copy()
                positions = np.full(n0, -1, dtype=int)
                positions[candidate_keep] = np.arange(candidate_keep.sum())
                local_edges = positions[pm_edges[candidate_pm]]
                local_basis = basis_rows[candidate_pm]
                candidate_fidelity = matching_fidelity(candidate_weights, local_edges, local_basis)

                if not np.isfinite(candidate_fidelity) or candidate_fidelity < fidelity_threshold:
                    if not reoptimize:
                        continue

                    def objective(local_weights):
                        fidelity = matching_fidelity(local_weights, local_edges, local_basis)
                        return -fidelity if np.isfinite(fidelity) else 0.1

                    try:
                        optimized = minimize(
                            objective, candidate_weights, method='L-BFGS-B',
                            bounds=[(-1, 1)] * len(candidate_weights),
                        )
                        optimized_weights = np.asarray(optimized.x, dtype=float)
                        if optimized_weights.shape != candidate_weights.shape:
                            continue
                        candidate_fidelity = matching_fidelity(optimized_weights, local_edges, local_basis)
                    except (ValueError, RuntimeError, FloatingPointError):
                        continue
                    if not np.isfinite(candidate_fidelity) or candidate_fidelity < fidelity_threshold:
                        continue
                    candidate_weights = optimized_weights

                weights0[candidate_keep] = candidate_weights

            keep = candidate_keep
            live_pm = candidate_pm
            changed = True

        if not changed:
            break

    return result_for(keep, weights0)

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
