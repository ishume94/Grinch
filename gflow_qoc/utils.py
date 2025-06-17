from scipy.optimize import minimize
import numpy as np
import re
from sympy import sympify, sqrt
from itertools import combinations

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

def fidelity_objective(weights, state, target_state, num_colors):
    try:
        normalized = build_normalized_state(state, weights, num_colors)
        overlap = np.vdot(target_state, normalized)
        fidelity = abs(overlap)**2
        return -fidelity  # we minimize, so negate
    except Exception as e:
        #print("Error during fidelity evaluation:", e)
        return 0.1  # penalty for failure

def reward_fidelity(target_state, state, num_colors):
    """This reward function computes the optimal weights and fidelities. Returns squared of fidelity if the state is valid."""
    # Initial weights
    init_weights = np.random.uniform(0, 1, len(state))

    # Optimize
    result = minimize(
        fidelity_objective,
        init_weights,
        args=(state, target_state, num_colors),
        method='L-BFGS-B',
        bounds=[(0, 1)] * len(state),
        options={'disp': True}
    )
    # Optimized weights
    opt_weights = result.x
    opt_fidelity = -result.fun

    if opt_fidelity < 0:
        return 0#opt_fidelity
    else:
        return np.exp(opt_fidelity/2)#**2
    #return opt_fidelity

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