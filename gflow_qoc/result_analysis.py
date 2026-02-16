import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import torch
import re
from collections import Counter
from pathlib import Path
from matplotlib import cm
from matplotlib.animation import FFMpegWriter, ImageMagickWriter, PillowWriter
from .utils import *
from .gflow_utils import *

def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Draws a MultiGraph with multi-edges (multiple edges between nodes) and labels the edges
    based on an attribute (e.g., 'w' for weight, 'colors'). Each edge has two colors in this example.
    Based on the example https://networkx.org/documentation/stable/auto_examples/drawing/plot_multigraphs.html
    """
    # Define color mapping for two colors per edge
    color_mapping = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow', 4: 'purple'}
    #Tried to plot each edge with 2 colors but could not do it. It is better to print the label.
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # ordered_nodes = sorted(G.nodes())
    # pos = nx.circular_layout(G)  # You can change the layout if needed
    # pos = {n: pos[n] for n in ordered_nodes}
    # Define node layout manually (clockwise order)
    ordered_nodes = sorted(G.nodes())  # You can customize this order
    n = len(ordered_nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)[::-1]  # Clockwise
    pos = {
        node: (np.cos(theta), np.sin(theta))
        for node, theta in zip(ordered_nodes, angles)
    }
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    # Prepare edge labels
    labels = {
        tuple(edge): f"{edge}, {attr_name}:{attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    #labels
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )
def plot_graph(edge_vec, filename="graph.svg", n_ancilla=0):
    state = list(edge_vec)
    if state:
        total_nodes = max(max(int(n1), int(n2)) for ((n1, n2), _colors) in state) + 1
    else:
        total_nodes = 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    outer_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        facecolor="white",
        edgecolor="#424242",
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(outer_circle)

    node_angles = np.linspace(0, 2 * np.pi, total_nodes, endpoint=False)[::-1]
    node_positions = {
        node_idx: np.array([0.72 * np.cos(theta), 0.72 * np.sin(theta)])
        for node_idx, theta in enumerate(node_angles)
    }

    pair_counts = {}
    for ((n1, n2), _colors) in state:
        pair_key = (int(n1), int(n2))
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    pair_seen = {}
    for ((n1, n2), (c1, c2)) in state:
        n1 = int(n1)
        n2 = int(n2)
        p1 = node_positions.get(n1)
        p2 = node_positions.get(n2)
        if p1 is None or p2 is None:
            continue

        pair_key = (n1, n2)
        edge_idx = pair_seen.get(pair_key, 0)
        pair_seen[pair_key] = edge_idx + 1
        total_for_pair = pair_counts.get(pair_key, 1)
        offset = (edge_idx - 0.5 * (total_for_pair - 1)) * 0.10

        edge_vec_np = p2 - p1
        norm = np.linalg.norm(edge_vec_np)
        if norm > 1e-9:
            perp = np.array([-edge_vec_np[1], edge_vec_np[0]]) / norm
        else:
            perp = np.zeros(2)

        p1o = p1 + offset * perp
        p2o = p2 + offset * perp
        pmid = 0.5 * (p1o + p2o)

        ax.plot(
            [p1o[0], pmid[0]],
            [p1o[1], pmid[1]],
            color=_mode_color(c1),
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [pmid[0], p2o[0]],
            [pmid[1], p2o[1]],
            color=_mode_color(c2),
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )

    anc = int(max(0, min(int(n_ancilla), total_nodes)))
    first_anc_idx = total_nodes - anc

    for node_idx, node_pos in node_positions.items():
        node_fill = "#B0B4BB" if node_idx >= first_anc_idx else "#f5f5f5"
        ax.add_patch(
            plt.Circle(
                (node_pos[0], node_pos[1]),
                0.12,
                facecolor=node_fill,
                edgecolor="black",
                linewidth=0.8,
                zorder=2,
            )
        )
        ax.text(
            node_pos[0],
            node_pos[1],
            str(node_idx),
            fontsize=12,
            ha="center",
            va="center",
            zorder=3,
        )

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)

    output_path = Path(filename)
    if output_path.suffix.lower() in {".svg", ".png"}:
        base = output_path.with_suffix("")
    else:
        base = output_path

    fig.savefig(str(base) + ".svg", format="svg", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def histo_fidelity(fidelities, filename):
    """
    Plots a histogram of fidelities for a list of states with fidelities.
    """
    if len(fidelities) > 0 and isinstance(fidelities[0], (tuple, list)):
        fidelities = [x[0] for x in fidelities]

    plt.figure(figsize=(8, 5))
    plt.xlim(0, 1) 
    plt.hist(fidelities, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Reward Fidelity')
    plt.ylabel('Number of States')
    plt.title('Histogram of Reward Fidelities')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=600)

def plot_loss_curve(figure, losses_A, title=""):
    filename = f"{figure}_loss.svg"
    plt.figure(figsize=(10,5))
    plt.plot(losses_A, color="black")
    plt.savefig(filename, format='svg', dpi=600)

def plot_rewards(rewards):

    # Best-so-far (cumulative max)
    best = []
    cur = None
    for r in rewards:
        cur = r if cur is None or r > cur else cur
        best.append(cur)

    x = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(x, best, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Best reward")
    plt.ylim(0, 1)
    plt.savefig("rewards_progress.svg", format="svg", dpi=600, bbox_inches="tight")
    plt.savefig("rewards_progress.png", format="png", dpi=600, bbox_inches="tight")


def _state_key(state):
    return tuple(state)


def _extract_episode_from_path(path):
    match = re.search(r"_ep(\d+)\.pth$", str(path))
    if match:
        return int(match.group(1))
    return -1


def _snapshot_run_key(path):
    match = re.search(r"(.+)_ep\d+\.pth$", Path(path).name)
    if match:
        return match.group(1)
    return Path(path).stem


def _infer_num_nodes_num_colors(FEATURE_KEYS):
    if not FEATURE_KEYS:
        raise ValueError("FEATURE_KEYS is empty; cannot infer num_nodes/num_colors.")
    max_node = max(max(n1, n2) for ((n1, n2), _) in FEATURE_KEYS)
    max_color = max(max(c1, c2) for (_, (c1, c2)) in FEATURE_KEYS)
    return int(max_node + 1), int(max_color + 1)


def _format_node_label(node_idx, total_nodes, n_ancilla):
    anc = int(max(0, n_ancilla))
    if anc == 0:
        return str(int(node_idx))

    first_anc_idx = int(total_nodes) - anc
    if node_idx >= first_anc_idx:
        return f"n_a{int(node_idx - first_anc_idx)}"
    return str(int(node_idx))


def _format_feature_key(feature_key, total_nodes, n_ancilla):
    (n1, n2), (c1, c2) = feature_key
    n1_label = _format_node_label(n1, total_nodes, n_ancilla)
    n2_label = _format_node_label(n2, total_nodes, n_ancilla)
    return f"{n1_label}:{c1}->{n2_label}:{c2}"


def _format_state_for_label(state, total_nodes, n_ancilla, per_line=2):
    if not state:
        return "∅"

    edge_texts = [
        _format_feature_key(edge, total_nodes=total_nodes, n_ancilla=n_ancilla)
        for edge in state
    ]

    line_width = max(1, int(per_line))
    lines = []
    for i in range(0, len(edge_texts), line_width):
        lines.append(", ".join(edge_texts[i:i + line_width]))
    return "\n".join(lines)


def _mode_color(mode):
    palette = {
        0: "#1f77b4",
        1: "#d62728",
        2: "#2ca02c",
        3: "#ff7f0e",
        4: "#9467bd",
    }
    mode_int = int(mode)
    if mode_int in palette:
        return palette[mode_int]
    return cm.tab20(mode_int % 20)


def _draw_state_inset(
    ax,
    center_xy,
    state,
    state_name,
    total_nodes,
    n_ancilla,
    border_color,
    radius_x_data=0.28,
    radius_y_data=0.28,
    show_state_name=True,
):
    x, y = center_xy
    if total_nodes is None or int(total_nodes) <= 0:
        return

    inset = ax.inset_axes(
        [x - radius_x_data, y - radius_y_data, 2 * radius_x_data, 2 * radius_y_data],
        transform=ax.transData,
        zorder=6,
    )
    inset.set_xlim(-1.1, 1.1)
    inset.set_ylim(-1.1, 1.1)
    inset.set_aspect("equal")
    inset.axis("off")

    outer_circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        facecolor="white",
        edgecolor=border_color,
        linewidth=1.4,
        zorder=0,
    )
    inset.add_patch(outer_circle)

    n_nodes = int(total_nodes)
    node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)[::-1]
    node_positions = {
        node_idx: np.array([0.72 * np.cos(theta), 0.72 * np.sin(theta)])
        for node_idx, theta in enumerate(node_angles)
    }

    pair_counts = {}
    for ((n1, n2), _colors) in state:
        pair_key = (int(n1), int(n2))
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    pair_seen = {}
    for ((n1, n2), (c1, c2)) in state:
        n1 = int(n1)
        n2 = int(n2)
        p1 = node_positions.get(n1)
        p2 = node_positions.get(n2)
        if p1 is None or p2 is None:
            continue

        pair_key = (n1, n2)
        edge_idx = pair_seen.get(pair_key, 0)
        pair_seen[pair_key] = edge_idx + 1
        total_for_pair = pair_counts.get(pair_key, 1)
        offset = (edge_idx - 0.5 * (total_for_pair - 1)) * 0.10

        edge_vec = p2 - p1
        norm = np.linalg.norm(edge_vec)
        if norm > 1e-9:
            perp = np.array([-edge_vec[1], edge_vec[0]]) / norm
        else:
            perp = np.zeros(2)

        p1o = p1 + offset * perp
        p2o = p2 + offset * perp
        pmid = 0.5 * (p1o + p2o)

        color1 = _mode_color(c1)
        color2 = _mode_color(c2)
        inset.plot(
            [p1o[0], pmid[0]],
            [p1o[1], pmid[1]],
            color=color1,
            linewidth=1.6,
            solid_capstyle="round",
            zorder=1,
        )
        inset.plot(
            [pmid[0], p2o[0]],
            [pmid[1], p2o[1]],
            color=color2,
            linewidth=1.6,
            solid_capstyle="round",
            zorder=1,
        )

    anc = int(max(0, min(int(n_ancilla), n_nodes)))
    first_anc_idx = n_nodes - anc
    for node_idx, node_pos in node_positions.items():
        node_fill = "#B0B4BB" if node_idx >= first_anc_idx else "#f5f5f5"
        inset.add_patch(
            plt.Circle(
                (node_pos[0], node_pos[1]),
                0.14,
                facecolor=node_fill,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
        )
        inset.text(
            node_pos[0],
            node_pos[1],
            str(node_idx),
            fontsize=4.6,
            ha="center",
            va="center",
            zorder=4,
        )

    if show_state_name and state_name:
        inset.text(
            0.0,
            0.98,
            state_name,
            fontsize=4.6,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=5,
        )


def load_tb_snapshot_paths(snapshot_dir, snapshot_glob="*TB_snapshot_ep*.pth"):
    snapshot_path = Path(snapshot_dir)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    paths = list(snapshot_path.glob(snapshot_glob))
    if not paths:
        paths = list(snapshot_path.glob("*_snapshot_ep*.pth"))
    if not paths:
        raise FileNotFoundError(
            f"No snapshots matched '{snapshot_glob}' (or fallback '*_snapshot_ep*.pth') in {snapshot_dir}"
        )

    grouped = {}
    for p in paths:
        grouped.setdefault(_snapshot_run_key(p), []).append(p)

    def run_score(paths_in_run):
        max_ep = max(_extract_episode_from_path(p) for p in paths_in_run)
        return (max_ep, len(paths_in_run))

    selected_run = max(grouped.keys(), key=lambda key: run_score(grouped[key]))
    selected_paths = sorted(grouped[selected_run], key=_extract_episode_from_path)
    return [str(p) for p in selected_paths]


def select_tb_states_for_visualization(
    sampled_states,
    rewards,
    selection_scores=None,
    top_k=3,
    mid_k=3,
    bottom_k=3,
    random_seed=0,
):
    if len(sampled_states) != len(rewards):
        raise ValueError("sampled_states and rewards must have the same length.")
    if selection_scores is None:
        selection_scores = rewards
    if len(sampled_states) != len(selection_scores):
        raise ValueError("sampled_states and selection_scores must have the same length.")

    best_by_state = {}
    for idx, (state, reward, score) in enumerate(zip(sampled_states, rewards, selection_scores)):
        reward_value = float(reward) if reward is not None else float("-inf")
        score_value = float(score) if score is not None else float("-inf")
        key = _state_key(state)
        if key not in best_by_state:
            best_by_state[key] = {
                "state": list(state),
                "reward": reward_value,
                "score": score_value,
                "episode_idx": idx,
            }
            continue

        entry = best_by_state[key]
        if score_value > entry["score"]:
            entry["score"] = score_value
            entry["episode_idx"] = idx
        if reward_value > entry["reward"]:
            entry["reward"] = reward_value

    if not best_by_state:
        raise ValueError("No sampled states were provided.")

    records = sorted(best_by_state.values(), key=lambda x: x["score"])

    worst = records[: min(bottom_k, len(records))]
    best = list(reversed(records[max(0, len(records) - top_k) :]))

    used = {_state_key(r["state"]) for r in worst + best}
    middle_pool = [r for r in records if _state_key(r["state"]) not in used]
    rng = np.random.default_rng(random_seed)
    if len(middle_pool) > mid_k:
        chosen_idx = rng.choice(len(middle_pool), size=mid_k, replace=False)
        middle = [middle_pool[i] for i in chosen_idx]
    else:
        middle = middle_pool
    middle = sorted(middle, key=lambda x: x["score"], reverse=True)

    selected = []
    for rec in best:
        item = dict(rec)
        item["category"] = "top"
        selected.append(item)
    for rec in middle:
        item = dict(rec)
        item["category"] = "mid"
        selected.append(item)
    for rec in worst:
        item = dict(rec)
        item["category"] = "worst"
        selected.append(item)
    return selected


def _build_tb_model(
    model_kind,
    FEATURE_KEYS,
    model_kwargs=None,
):
    kwargs = dict(model_kwargs or {})
    kind = str(model_kind).lower()
    if kind in {"tb", "mlp"}:
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        return TBModel(hidden_dim, FEATURE_KEYS)
    if kind in {"embtb", "emb_tb"}:
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        n_emb = int(kwargs.get("n_emb", 16))
        return embTBModel(hidden_dim, FEATURE_KEYS, n_emb=n_emb)
    if kind == "gin":
        node_feat_dim = int(kwargs.get("node_feat_dim", kwargs.get("num_nodes", 0)))
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        return GIN_TBModel(node_feat_dim, hidden_dim, FEATURE_KEYS)
    if kind == "gine":
        node_feat_dim = int(kwargs.get("node_feat_dim", kwargs.get("num_nodes", 0)))
        edge_feat_dim = int(kwargs.get("edge_feat_dim", 2))
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        return GINE_TBModel(node_feat_dim, edge_feat_dim, hidden_dim, FEATURE_KEYS)
    if kind == "gat":
        node_feat_dim = int(kwargs.get("node_feat_dim", kwargs.get("num_nodes", 0)))
        edge_feat_dim = int(kwargs.get("edge_feat_dim", 2))
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        num_layers = int(kwargs.get("num_layers", kwargs.get("gat_layers", 3)))
        heads = int(kwargs.get("heads", kwargs.get("gat_heads", 4)))
        dropout = float(kwargs.get("dropout", 0.0))
        return GAT_TBModel(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            FEATURE_KEYS=FEATURE_KEYS,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
    if kind == "transformer":
        node_feat_dim = int(kwargs.get("node_feat_dim", kwargs.get("num_nodes", 0)))
        edge_feat_dim = int(kwargs.get("edge_feat_dim", 2))
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        num_layers = int(kwargs.get("num_layers", kwargs.get("tf_layers", 3)))
        heads = int(kwargs.get("heads", kwargs.get("tf_heads", 4)))
        dropout = float(kwargs.get("dropout", 0.0))
        return Transformer_TBModel(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            FEATURE_KEYS=FEATURE_KEYS,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_kind '{model_kind}'.")


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _forward_policy_logits(model, state, FEATURE_KEYS, num_nodes, num_colors):
    device = _model_device(model)
    with torch.no_grad():
        try:
            x = state_to_tensor(state, FEATURE_KEYS).to(device)
            p_f, _ = model(x, FEATURE_KEYS)
        except TypeError:
            data = state_to_data(state, num_nodes, num_colors)
            if not hasattr(data, "batch") or data.batch is None:
                data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            data = data.to(device)
            p_f, _ = model(data)

    logits = torch.as_tensor(p_f, device=device).reshape(-1)
    return logits


def _forward_action_probs(model, state, FEATURE_KEYS, target_expr, num_nodes, num_colors):
    logits = _forward_policy_logits(model, state, FEATURE_KEYS, num_nodes, num_colors)
    mask = calculate_forward_mask_from_state(state, target_expr, FEATURE_KEYS).to(logits.device)
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
    probs = torch.softmax(masked_logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    return probs.detach().cpu().numpy()


def _build_reduced_tb_structure(
    selected_records,
    compressed_depth=2,
    total_nodes=None,
    n_ancilla=0,
):
    graph = nx.DiGraph()
    root_id = "s0"
    root_state = []
    root_state_repr = _format_state_for_label(
        root_state,
        total_nodes=total_nodes,
        n_ancilla=n_ancilla,
    )
    graph.add_node(
        root_id,
        node_type="root",
        label="s0",
        depth=0,
        category="root",
        state=root_state,
        state_repr=root_state_repr,
    )

    edge_specs = {}
    traj_infos = []
    positions = {}
    n_rows = len(selected_records)
    root_y = 0.5 * (n_rows - 1) if n_rows > 0 else 0.0
    positions[root_id] = (0.0, root_y)

    final_x = float(compressed_depth + 1)

    for traj_idx, record in enumerate(selected_records):
        trajectory = list(record["state"])
        traj_len = len(trajectory)
        prefix_steps = min(max(int(compressed_depth), 0), max(traj_len - 1, 0))
        y = float(n_rows - 1 - traj_idx)

        prev_node = root_id
        prev_len = 0

        for step in range(1, prefix_steps + 1):
            node_id = f"traj{traj_idx}_p{step}"
            step_label = f"|s{step}|"
            prefix_state = trajectory[:step]
            prefix_state_repr = _format_state_for_label(
                prefix_state,
                total_nodes=total_nodes,
                n_ancilla=n_ancilla,
            )
            graph.add_node(
                node_id,
                node_type="prefix",
                label=step_label,
                depth=step,
                category=record["category"],
                traj_idx=traj_idx,
                state=prefix_state,
                state_repr=prefix_state_repr,
            )
            positions[node_id] = (float(step), y)
            edge = (prev_node, node_id)
            graph.add_edge(*edge, traj_idx=traj_idx, edge_kind="primary")
            edge_specs[edge] = {
                "start_state": trajectory[:prev_len],
                "action_seq": trajectory[prev_len:step],
            }
            prev_node = node_id
            prev_len = step

        final_node = f"traj{traj_idx}_final"
        final_state_repr = _format_state_for_label(
            trajectory,
            total_nodes=total_nodes,
            n_ancilla=n_ancilla,
        )
        graph.add_node(
            final_node,
            node_type="final",
            label="final state",
            depth=compressed_depth + 1,
            category=record["category"],
            reward=float(record["reward"]),
            traj_idx=traj_idx,
            state=trajectory,
            state_repr=final_state_repr,
        )
        positions[final_node] = (final_x, y)
        edge = (prev_node, final_node)
        graph.add_edge(*edge, traj_idx=traj_idx, edge_kind="primary")
        edge_specs[edge] = {
            "start_state": trajectory[:prev_len],
            "action_seq": trajectory[prev_len:],
        }

        traj_infos.append(
            {
                "traj_idx": traj_idx,
                "category": record["category"],
                "reward": float(record["reward"]),
                "final_node": final_node,
                "trajectory": trajectory,
            }
        )

    def _add_compatibility_edges(source_nodes, target_nodes):
        for source_node_id, source_state, source_traj_idx in source_nodes:
            for target_node_id, target_state, target_traj_idx in target_nodes:
                edge = (source_node_id, target_node_id)
                if edge in edge_specs:
                    continue

                completion_seq = _completion_action_sequence(source_state, target_state)
                if completion_seq is None:
                    continue

                graph.add_edge(
                    *edge,
                    traj_idx=target_traj_idx,
                    source_traj_idx=source_traj_idx,
                    target_traj_idx=target_traj_idx,
                    edge_kind="compat",
                )
                edge_specs[edge] = {
                    "start_state": list(source_state),
                    "action_seq": completion_seq,
                }

    displayed_depth = int(max(int(compressed_depth), 0))
    depth_to_prefix_nodes = {}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "prefix":
            continue
        depth = int(attrs.get("depth", -1))
        depth_to_prefix_nodes.setdefault(depth, []).append(
            (node_id, list(attrs.get("state", [])), attrs.get("traj_idx"))
        )

    # Add compatibility arrows for every adjacent displayed prefix-layer pair.
    # This includes |s1|->|s2|, |s2|->|s3|, ... when those layers are present.
    for depth in range(2, displayed_depth + 1):
        prev_nodes = depth_to_prefix_nodes.get(depth - 1, [])
        curr_nodes = depth_to_prefix_nodes.get(depth, [])
        _add_compatibility_edges(prev_nodes, curr_nodes)

    # Keep compatibility arrows for n -> final (already requested earlier).
    n_nodes = depth_to_prefix_nodes.get(displayed_depth, [])
    final_nodes = [
        (info["final_node"], list(info["trajectory"]), info["traj_idx"])
        for info in traj_infos
    ]
    _add_compatibility_edges(n_nodes, final_nodes)

    return {
        "graph": graph,
        "positions": positions,
        "edge_specs": edge_specs,
        "traj_infos": traj_infos,
        "final_x": final_x,
        "total_nodes": total_nodes,
        "n_ancilla": n_ancilla,
    }


def _compute_probs_for_structure(
    model,
    structure,
    FEATURE_KEYS,
    target_expr,
    num_nodes,
    num_colors,
):
    action_to_idx = {action: i for i, action in enumerate(FEATURE_KEYS)}
    probs_cache = {}

    def get_probs(state):
        key = _state_key(state)
        if key not in probs_cache:
            probs_cache[key] = _forward_action_probs(
                model,
                state,
                FEATURE_KEYS,
                target_expr,
                num_nodes,
                num_colors,
            )
        return probs_cache[key]

    def segment_probability(start_state, action_seq):
        state = list(start_state)
        log_prob = 0.0
        for action in action_seq:
            idx = action_to_idx.get(action)
            if idx is None:
                return 0.0
            probs = get_probs(state)
            prob = float(probs[idx])
            if (not np.isfinite(prob)) or prob <= 0:
                return 0.0
            log_prob += np.log(max(prob, 1e-30))
            state.append(action)
        return float(np.exp(log_prob))

    edge_probs = {}
    for edge, spec in structure["edge_specs"].items():
        edge_probs[edge] = segment_probability(spec["start_state"], spec["action_seq"])

    traj_probs = {}
    for info in structure["traj_infos"]:
        traj_probs[info["traj_idx"]] = segment_probability([], info["trajectory"])
    return edge_probs, traj_probs


def _compute_trajectory_probabilities_for_states(
    model,
    states,
    FEATURE_KEYS,
    target_expr,
    num_nodes,
    num_colors,
):
    action_to_idx = {action: i for i, action in enumerate(FEATURE_KEYS)}
    probs_cache = {}

    def get_probs(state):
        key = _state_key(state)
        if key not in probs_cache:
            probs_cache[key] = _forward_action_probs(
                model,
                state,
                FEATURE_KEYS,
                target_expr,
                num_nodes,
                num_colors,
            )
        return probs_cache[key]

    def trajectory_probability(action_seq):
        state = []
        log_prob = 0.0
        for action in action_seq:
            idx = action_to_idx.get(action)
            if idx is None:
                return 0.0
            probs = get_probs(state)
            prob = float(probs[idx])
            if (not np.isfinite(prob)) or prob <= 0:
                return 0.0
            log_prob += np.log(max(prob, 1e-30))
            state.append(action)
        return float(np.exp(log_prob))

    return [trajectory_probability(list(s)) for s in states]


def _completion_action_sequence(prefix_state, final_state):
    """
    Build one valid completion sequence from prefix_state to final_state.
    Compatibility is multiset-based; completion keeps final_state order
    for actions not already present in the prefix multiset.
    """
    remaining_prefix = Counter(prefix_state)
    completion = []
    for action in final_state:
        if remaining_prefix[action] > 0:
            remaining_prefix[action] -= 1
        else:
            completion.append(action)

    if any(v > 0 for v in remaining_prefix.values()):
        return None
    return completion


def _draw_tb_structure(
    ax,
    structure,
    edge_probs,
    traj_probs,
    title,
    edge_norm=None,
    show_edge_prob_labels=True,
):
    graph = structure["graph"]
    pos = structure["positions"]
    traj_infos = structure["traj_infos"]
    final_x = structure["final_x"]
    total_nodes = structure.get("total_nodes")
    n_ancilla = structure.get("n_ancilla", 0)

    ax.clear()
    ax.set_title(title)
    ax.axis("off")
    state_node_size = 2600.0

    y_values = [coord[1] for coord in pos.values()]
    ax.set_xlim(-1.2, final_x + 1.7)
    if y_values:
        ax.set_ylim(min(y_values) - 0.75, max(y_values) + 0.90)

    category_colors = {
        "top": "#2E7D32",
        "mid": "#F9A825",
        "worst": "#C62828",
        "root": "#1565C0",
    }

    node_colors = []
    node_sizes = []
    node_border_colors = []
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "")
        category = attrs.get("category", "mid")
        border_color = category_colors.get(category, "#424242")
        node_border_colors.append(border_color)

        if node_type == "root":
            node_colors.append(category_colors["root"])
        elif node_type == "prefix":
            node_colors.append("#ECEFF1")
        else:
            node_colors.append("#FAFAFA")
        node_sizes.append(state_node_size)

    edge_list = list(graph.edges())
    edge_values = [float(edge_probs.get(edge, 0.0)) for edge in edge_list]
    if edge_norm is None:
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
    else:
        norm = edge_norm
    edge_prob_map = {edge: float(edge_probs.get(edge, 0.0)) for edge in edge_list}
    primary_edges = []
    compat_edges = []
    for edge in edge_list:
        edge_kind = graph.get_edge_data(*edge).get("edge_kind", "primary")
        if edge_kind == "compat":
            compat_edges.append(edge)
        else:
            primary_edges.append(edge)

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.2,
        edgecolors=node_border_colors,
    )
    if primary_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=primary_edges,
            edge_color=[cm.Reds(norm(edge_prob_map[e])) for e in primary_edges],
            width=[1.2 + 5.0 * edge_prob_map[e] for e in primary_edges],
            node_size=state_node_size,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            min_source_margin=10,
            min_target_margin=12,
            style="solid",
        )
    if compat_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=compat_edges,
            edge_color=[cm.Reds(norm(edge_prob_map[e])) for e in compat_edges],
            width=[1.0 + 4.0 * edge_prob_map[e] for e in compat_edges],
            node_size=state_node_size,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=11,
            min_source_margin=10,
            min_target_margin=12,
            style="dashed",
            alpha=0.9,
        )
    if show_edge_prob_labels:
        primary_edge_labels = {edge: f"{edge_prob_map[edge]:.2e}" for edge in primary_edges}
        if primary_edge_labels:
            primary_label_artists = nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=primary_edge_labels,
                ax=ax,
                font_size=6,
                rotate=False,
                label_pos=0.50,
                bbox={"alpha": 0.6, "pad": 0.2, "fc": "white", "ec": "none"},
            )
            for txt in primary_label_artists.values():
                txt.set_zorder(9)

        compat_edge_labels = {edge: f"{edge_prob_map[edge]:.2e}" for edge in compat_edges}
        if compat_edge_labels:
            compat_label_artists = nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=compat_edge_labels,
                ax=ax,
                font_size=5.5,
                rotate=False,
                label_pos=0.40,
                bbox={"alpha": 0.65, "pad": 0.15, "fc": "white", "ec": "none"},
            )
            for txt in compat_label_artists.values():
                txt.set_zorder(9)

    def _marker_radius_in_data(center_xy, marker_size):
        radius_points = float(np.sqrt(max(marker_size, 1e-12) / np.pi))
        radius_pixels = radius_points * (ax.figure.dpi / 72.0)
        center_px = ax.transData.transform(center_xy)
        x_plus = ax.transData.inverted().transform((center_px[0] + radius_pixels, center_px[1]))
        y_plus = ax.transData.inverted().transform((center_px[0], center_px[1] + radius_pixels))
        radius_x = max(abs(float(x_plus[0]) - float(center_xy[0])), 1e-6)
        radius_y = max(abs(float(y_plus[1]) - float(center_xy[1])), 1e-6)
        return radius_x, radius_y

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "")
        category = attrs.get("category", "mid")
        border_color = category_colors.get(category, "#424242")
        state = attrs.get("state", [])
        state_name = attrs.get("label", "")
        center_xy = pos[node_id]
        inset_radius_x_data, inset_radius_y_data = _marker_radius_in_data(center_xy, state_node_size)
        _draw_state_inset(
            ax=ax,
            center_xy=center_xy,
            state=state,
            state_name=state_name,
            total_nodes=total_nodes,
            n_ancilla=n_ancilla,
            border_color=border_color,
            radius_x_data=inset_radius_x_data,
            radius_y_data=inset_radius_y_data,
            show_state_name=False,
        )

        if node_type == "final":
            traj_idx = attrs.get("traj_idx")
            reward = float(attrs.get("reward", 0.0))
            traj_prob = float(traj_probs.get(traj_idx, 0.0))
            ax.text(
                center_xy[0] + inset_radius_x_data + 0.08,
                center_xy[1],
                f"R={reward:.3f}\nP={traj_prob:.2e}",
                ha="left",
                va="center",
                fontsize=8,
                linespacing=1.2,
                bbox={"alpha": 0.7, "pad": 0.2, "fc": "white", "ec": "none"},
                zorder=7,
            )

    if y_values:
        column_label_y = max(y_values) + 0.60
        root_x = pos.get("s0", (None, None))[0]
        if root_x is not None:
            ax.text(
                root_x,
                column_label_y,
                "s0",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                bbox={"alpha": 0.8, "pad": 0.25, "fc": "white", "ec": "none"},
                zorder=8,
            )

        prefix_columns = {}
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") != "prefix":
                continue
            label = attrs.get("label", "")
            x = pos.get(node_id, (None, None))[0]
            if x is None or not label:
                continue
            prefix_columns.setdefault(label, x)

        for label, x in sorted(prefix_columns.items(), key=lambda item: item[1]):
            ax.text(
                x,
                column_label_y,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                bbox={"alpha": 0.8, "pad": 0.25, "fc": "white", "ec": "none"},
                zorder=8,
            )

        ax.text(
            final_x,
            column_label_y,
            "final state",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"alpha": 0.8, "pad": 0.25, "fc": "white", "ec": "none"},
            zorder=8,
        )


def _add_probability_colorbar(fig, ax, norm):
    sm = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        shrink=0.55,
        aspect=30,
    )
    cbar.set_label("Edge probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return cbar


def plot_tb_state_space_dynamics(
    sampled_states,
    rewards,
    FEATURE_KEYS,
    target_expr,
    snapshot_dir,
    output_prefix="tb_state_space",
    snapshot_glob="*TB_snapshot_ep*.pth",
    top_k=3,
    mid_k=3,
    bottom_k=3,
    random_seed=0,
    compressed_depth=1,
    n_ancilla=0,
    final_episode=None,
    rank_by_final_probability=False,
    show_edge_prob_labels=True,
    fps=8,
    frame_step=1,
    max_frames=None,
    animation_format="gif",
    animation_dpi=140,
):
    snapshot_paths = load_tb_snapshot_paths(
        snapshot_dir=snapshot_dir,
        snapshot_glob=snapshot_glob,
    )
    if not snapshot_paths:
        raise ValueError(f"No snapshots found in {snapshot_dir}.")

    frame_step = max(1, int(frame_step))
    animation_snapshot_paths = list(snapshot_paths[::frame_step])
    if animation_snapshot_paths[-1] != snapshot_paths[-1]:
        animation_snapshot_paths.append(snapshot_paths[-1])

    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0 when provided.")
        if len(animation_snapshot_paths) > max_frames:
            idx = np.linspace(
                0,
                len(animation_snapshot_paths) - 1,
                num=max_frames,
                dtype=int,
            )
            idx = sorted(set(int(i) for i in idx.tolist()))
            if idx[0] != 0:
                idx = [0] + idx
            if idx[-1] != len(animation_snapshot_paths) - 1:
                idx.append(len(animation_snapshot_paths) - 1)
            animation_snapshot_paths = [animation_snapshot_paths[i] for i in idx]

    num_nodes, num_colors = _infer_num_nodes_num_colors(FEATURE_KEYS)
    first_checkpoint = torch.load(snapshot_paths[0], map_location="cpu")
    model_kind = first_checkpoint.get("model_kind")
    model_kwargs = dict(first_checkpoint.get("model_kwargs", {}))
    if not model_kind:
        raise ValueError(
            "Snapshot metadata is missing model_kind. Re-run training with updated snapshot support."
        )
    model_kwargs.setdefault("node_feat_dim", num_nodes)

    model = _build_tb_model(
        model_kind=model_kind,
        FEATURE_KEYS=FEATURE_KEYS,
        model_kwargs=model_kwargs,
    )
    model.eval()

    selection_scores = None
    if rank_by_final_probability:
        final_checkpoint = None
        for checkpoint_path in reversed(snapshot_paths):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if checkpoint.get("model_kind", model_kind) != model_kind:
                continue
            final_checkpoint = checkpoint
            break
        if final_checkpoint is None:
            raise ValueError(
                f"No checkpoints with model_kind='{model_kind}' found in {snapshot_dir}."
            )
        model.load_state_dict(final_checkpoint["model_state_dict"])
        model.eval()
        selection_scores = _compute_trajectory_probabilities_for_states(
            model=model,
            states=sampled_states,
            FEATURE_KEYS=FEATURE_KEYS,
            target_expr=target_expr,
            num_nodes=num_nodes,
            num_colors=num_colors,
        )

    selected_records = select_tb_states_for_visualization(
        sampled_states=sampled_states,
        rewards=rewards,
        selection_scores=selection_scores,
        top_k=top_k,
        mid_k=mid_k,
        bottom_k=bottom_k,
        random_seed=random_seed,
    )

    structure = _build_reduced_tb_structure(
        selected_records=selected_records,
        compressed_depth=compressed_depth,
        total_nodes=num_nodes,
        n_ancilla=n_ancilla,
    )

    def _load_frame_probs(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("model_kind", model_kind) != model_kind:
            return None
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        episode = int(checkpoint.get("episode", _extract_episode_from_path(checkpoint_path)))
        edge_probs, traj_probs = _compute_probs_for_structure(
            model=model,
            structure=structure,
            FEATURE_KEYS=FEATURE_KEYS,
            target_expr=target_expr,
            num_nodes=num_nodes,
            num_colors=num_colors,
        )
        return {
            "episode": episode,
            "edge_probs": edge_probs,
            "traj_probs": traj_probs,
        }

    first_frame_path = None
    first_frame = None
    for checkpoint_path in animation_snapshot_paths:
        payload = _load_frame_probs(checkpoint_path)
        if payload is not None:
            first_frame_path = checkpoint_path
            first_frame = payload
            break

    if first_frame is None:
        raise ValueError(
            f"No checkpoints with model_kind='{model_kind}' found in animation selection for {snapshot_dir}."
        )

    last_frame_path = None
    last_frame = None
    for checkpoint_path in reversed(animation_snapshot_paths):
        payload = _load_frame_probs(checkpoint_path)
        if payload is not None:
            last_frame_path = checkpoint_path
            last_frame = payload
            break

    if last_frame is None:
        raise ValueError(
            f"No checkpoints with model_kind='{model_kind}' found in animation selection for {snapshot_dir}."
        )

    # Keep a fixed probability color scale across all frames and runs.
    edge_norm = plt.Normalize(vmin=0.0, vmax=1.0)

    n_selected = len(selected_records)
    # Give each trajectory row enough vertical room so state circles do not overlap.
    fig_height = max(7.2, 0.82 * n_selected + 2.8)

    start_plot_path = f"{output_prefix}_start.png"
    end_plot_path = f"{output_prefix}_end.png"

    fig, ax = plt.subplots(figsize=(14, fig_height))
    _draw_tb_structure(
        ax=ax,
        structure=structure,
        edge_probs=first_frame["edge_probs"],
        traj_probs=first_frame["traj_probs"],
        title=f"Reduced TB State Space (Start, episode {first_frame['episode']})",
        edge_norm=edge_norm,
        show_edge_prob_labels=show_edge_prob_labels,
    )
    _add_probability_colorbar(fig=fig, ax=ax, norm=edge_norm)
    fig.savefig(start_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, fig_height))
    _draw_tb_structure(
        ax=ax,
        structure=structure,
        edge_probs=last_frame["edge_probs"],
        traj_probs=last_frame["traj_probs"],
        title=(
            f"Reduced TB State Space (End, episode "
            f"{int(final_episode) if final_episode is not None else last_frame['episode']})"
        ),
        edge_norm=edge_norm,
        show_edge_prob_labels=show_edge_prob_labels,
    )
    _add_probability_colorbar(fig=fig, ax=ax, norm=edge_norm)
    fig.savefig(end_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    def _choose_writer(fmt, fps_value):
        fmt_norm = str(fmt).strip().lower()
        if fmt_norm == "mp4":
            if FFMpegWriter.isAvailable():
                return FFMpegWriter(fps=fps_value), "mp4"
            print("[warning] FFMpegWriter not available; falling back to GIF.")
            fmt_norm = "gif"

        if fmt_norm == "gif":
            if ImageMagickWriter.isAvailable():
                return ImageMagickWriter(fps=fps_value), "gif"
            print(
                "[warning] ImageMagickWriter not available; using PillowWriter. "
                "Long GIFs may consume large RAM."
            )
            return PillowWriter(fps=fps_value), "gif"

        raise ValueError("animation_format must be either 'gif' or 'mp4'.")

    writer, animation_ext = _choose_writer(animation_format, fps)
    animation_path = f"{output_prefix}_evolution.{animation_ext}"

    fig, ax = plt.subplots(figsize=(14, fig_height))
    rendered_episodes = []
    with writer.saving(fig, animation_path, dpi=int(animation_dpi)):
        _draw_tb_structure(
            ax=ax,
            structure=structure,
            edge_probs=first_frame["edge_probs"],
            traj_probs=first_frame["traj_probs"],
            title=f"Reduced TB State Space (episode {first_frame['episode']})",
            edge_norm=edge_norm,
            show_edge_prob_labels=show_edge_prob_labels,
        )
        _add_probability_colorbar(fig=fig, ax=ax, norm=edge_norm)
        writer.grab_frame()
        rendered_episodes.append(first_frame["episode"])

        for checkpoint_path in animation_snapshot_paths:
            if checkpoint_path == first_frame_path:
                continue
            payload = _load_frame_probs(checkpoint_path)
            if payload is None:
                continue

            episode_label = payload["episode"]
            if final_episode is not None and checkpoint_path == last_frame_path:
                episode_label = int(final_episode)

            _draw_tb_structure(
                ax=ax,
                structure=structure,
                edge_probs=payload["edge_probs"],
                traj_probs=payload["traj_probs"],
                title=f"Reduced TB State Space (episode {episode_label})",
                edge_norm=edge_norm,
                show_edge_prob_labels=show_edge_prob_labels,
            )
            writer.grab_frame()
            rendered_episodes.append(payload["episode"])
    plt.close(fig)

    return {
        "start_plot": start_plot_path,
        "end_plot": end_plot_path,
        "animation": animation_path,
        "selected_states": selected_records,
        "episodes": rendered_episodes,
        "snapshot_paths": animation_snapshot_paths,
        "all_snapshot_paths": snapshot_paths,
    }
