from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from .result_analysis import _draw_state_inset, _mode_color
from .utils import opt_fidelity, reward_fidelity

try:
    import seaborn as sns
except ImportError:
    sns = None


def _apply_style():
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def canonical_state_key(state, sort_edges=True):
    """Return a hashable representation of an optical graph state."""
    edges = [
        ((int(n1), int(n2)), (int(c1), int(c2)))
        for ((n1, n2), (c1, c2)) in state
    ]
    if sort_edges:
        edges = sorted(edges)
    return tuple(edges)


def state_key_to_label(state_key, max_edges=5):
    pieces = [
        f"{n1}{c1}-{n2}{c2}"
        for ((n1, n2), (c1, c2)) in list(state_key)[:max_edges]
    ]
    label = ", ".join(pieces)
    if len(state_key) > max_edges:
        label += f", +{len(state_key) - max_edges}"
    return label or "empty"


def aggregate_sampled_states(sampled_states, metric_values=None, sort_edges=True):
    if metric_values is None:
        metric_values = [np.nan] * len(sampled_states)
    if len(sampled_states) != len(metric_values):
        raise ValueError("sampled_states and metric_values must have the same length.")

    records = {}
    for idx, (state, metric) in enumerate(zip(sampled_states, metric_values), start=1):
        key = canonical_state_key(state, sort_edges=sort_edges)
        metric_value = float(metric) if metric is not None else np.nan
        if key not in records:
            records[key] = {
                "state": list(key),
                "key": key,
                "label": state_key_to_label(key),
                "count": 0,
                "first_iteration": idx,
                "last_iteration": idx,
                "metrics": [],
            }
        rec = records[key]
        rec["count"] += 1
        rec["last_iteration"] = idx
        rec["metrics"].append(metric_value)

    for rec in records.values():
        arr = np.asarray(rec["metrics"], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            rec["metric_mean"] = float(np.mean(finite))
            rec["metric_max"] = float(np.max(finite))
            rec["metric_min"] = float(np.min(finite))
            rec["metric_last"] = float(arr[np.flatnonzero(np.isfinite(arr))[-1]])
        else:
            rec["metric_mean"] = np.nan
            rec["metric_max"] = np.nan
            rec["metric_min"] = np.nan
            rec["metric_last"] = np.nan
    return list(records.values())


def state_feature_matrix(states, feature_keys=None, sort_edges=True):
    state_keys = [canonical_state_key(state, sort_edges=sort_edges) for state in states]
    if feature_keys is None:
        features = sorted({edge for key in state_keys for edge in key})
    else:
        features = [
            ((int(n1), int(n2)), (int(c1), int(c2)))
            for ((n1, n2), (c1, c2)) in feature_keys
        ]

    feature_index = {feature: idx for idx, feature in enumerate(features)}
    matrix = np.zeros((len(state_keys), len(features)), dtype=float)
    for row_idx, key in enumerate(state_keys):
        counts = Counter(key)
        for edge, count in counts.items():
            col_idx = feature_index.get(edge)
            if col_idx is not None:
                matrix[row_idx, col_idx] = float(count)
    return matrix, features


def _project_features(features, method="pca", random_state=0):
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if features.shape[0] == 0:
        raise ValueError("No states were provided.")
    if features.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    if features.shape[1] == 1:
        return np.column_stack([features[:, 0], np.zeros(features.shape[0])])
    method = str(method).lower()
    if features.shape[1] == 2 and method in {"auto", "identity"}:
        return features.astype(float)

    if method not in {"auto", "pca", "identity"}:
        print(f"[warning] Unknown projection method '{method}'; falling back to PCA.")

    centered = features.astype(float) - np.mean(features, axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return coords


def _save_figure(fig, filename, dpi=300):
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.with_suffix("") if output_path.suffix else output_path
    paths = []
    for suffix in (".svg", ".png"):
        path = str(base) + suffix
        fig.savefig(path, format=suffix[1:], dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _save_state_graph_figure(
    state,
    filename,
    total_nodes,
    n_ancilla=0,
    border_color="#333333",
    dpi=300,
    edge_weights=None,
):
    if edge_weights is not None:
        state = list(state)
        edge_weights = np.asarray(edge_weights)
        if edge_weights.ndim != 1 or len(edge_weights) != len(state) or not np.isrealobj(edge_weights):
            raise ValueError("edge_weights must contain one finite real weight per edge.")
        try:
            edge_weights = np.asarray(edge_weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge_weights must contain one finite real weight per edge.") from exc
        if not np.all(np.isfinite(edge_weights)):
            raise ValueError("edge_weights must contain one finite real weight per edge.")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")

    n_nodes = int(total_nodes)
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            1.0,
            facecolor="white",
            edgecolor=border_color,
            linewidth=2.8,
            zorder=0,
        )
    )

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
    for state_edge_idx, ((n1, n2), (c1, c2)) in enumerate(state):
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
        offset = (edge_idx - 0.5 * (total_for_pair - 1)) * 0.12

        edge_vec = p2 - p1
        norm = np.linalg.norm(edge_vec)
        perp = np.array([-edge_vec[1], edge_vec[0]]) / norm if norm > 1e-9 else np.zeros(2)

        p1o = p1 + offset * perp
        p2o = p2 + offset * perp
        pmid = 0.5 * (p1o + p2o)
        ax.plot(
            [p1o[0], pmid[0]],
            [p1o[1], pmid[1]],
            color=_mode_color(c1),
            linewidth=4.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [pmid[0], p2o[0]],
            [pmid[1], p2o[1]],
            color=_mode_color(c2),
            linewidth=4.0,
            solid_capstyle="round",
            zorder=1,
        )
        if edge_weights is not None and edge_weights[state_edge_idx] < 0:
            ax.plot(
                [pmid[0]],
                [pmid[1]],
                marker="D",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="None",
                zorder=2,
            )

    anc = int(max(0, min(int(n_ancilla), n_nodes)))
    first_anc_idx = n_nodes - anc
    for node_idx, node_pos in node_positions.items():
        node_fill = "#B0B4BB" if node_idx >= first_anc_idx else "#f5f5f5"
        ax.add_patch(
            plt.Circle(
                (node_pos[0], node_pos[1]),
                0.16,
                facecolor=node_fill,
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
            )
        )
        ax.text(
            node_pos[0],
            node_pos[1],
            str(node_idx),
            fontsize=20,
            ha="center",
            va="center",
            zorder=4,
        )

    return _save_figure(fig, filename, dpi=dpi)


def _ranking_windows(sampled_states, metric_values, top_n, num_windows, ranking_mode, sort_edges):
    highest = not str(ranking_mode).lower().startswith("lowest")
    cumulative = "over_all" in str(ranking_mode).lower() or "over all" in str(ranking_mode).lower()

    n_samples = len(sampled_states)
    edges = np.linspace(0, n_samples, min(num_windows, n_samples) + 1, dtype=int)
    windows = [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]
    window_centers = [0.5 * (start + end) + 0.5 for start, end in windows]

    per_window = []
    overall_best = {}
    for start, end in windows:
        scores = {}
        sampled_here = set()
        for state, metric in zip(sampled_states[start:end], metric_values[start:end]):
            key = canonical_state_key(state, sort_edges=sort_edges)
            sampled_here.add(key)
            value = float(metric) if metric is not None else np.nan
            if not np.isfinite(value):
                continue
            if key not in scores:
                scores[key] = value
            elif highest:
                scores[key] = max(scores[key], value)
            else:
                scores[key] = min(scores[key], value)

        if cumulative:
            for key, value in scores.items():
                if key not in overall_best:
                    overall_best[key] = value
                elif highest:
                    overall_best[key] = max(overall_best[key], value)
                else:
                    overall_best[key] = min(overall_best[key], value)
            rank_scores = dict(overall_best)
        else:
            rank_scores = scores

        ordered = sorted(rank_scores.items(), key=lambda item: item[1], reverse=highest)
        ranks = {key: rank + 1 for rank, (key, _value) in enumerate(ordered)}
        per_window.append(
            {
                "start": start,
                "end": end,
                "ranks": ranks,
                "sampled": sampled_here,
                "scores": rank_scores,
            }
        )

    final_scores = defaultdict(lambda: -np.inf if highest else np.inf)
    for window in per_window:
        for key, value in window["scores"].items():
            if highest:
                final_scores[key] = max(final_scores[key], value)
            else:
                final_scores[key] = min(final_scores[key], value)

    tracked = sorted(final_scores.items(), key=lambda item: item[1], reverse=highest)[:top_n]
    tracked_keys = [key for key, _value in tracked]
    return window_centers, per_window, tracked_keys, highest, cumulative


def _first_iteration_by_state(sampled_states, sort_edges=True):
    first_seen = {}
    for idx, state in enumerate(sampled_states, start=1):
        key = canonical_state_key(state, sort_edges=sort_edges)
        if key not in first_seen:
            first_seen[key] = idx
    return first_seen


def _build_callout_state_lookup(callout_states, sort_edges=True):
    if not callout_states:
        return {}
    return {
        canonical_state_key(state, sort_edges=sort_edges): list(state)
        for state in callout_states
    }


def _best_pruned_match(state_key, pruned_lookup):
    if not pruned_lookup:
        return None

    state_edges = set(state_key)
    exact = pruned_lookup.get(state_key)
    if exact is not None:
        return exact

    candidates = []
    for pruned_key, pruned_state in pruned_lookup.items():
        pruned_edges = set(pruned_key)
        if pruned_edges.issubset(state_edges):
            candidates.append((len(pruned_edges), pruned_state))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def iterative_fidelity_prune_state(state, target_state, num_colors, threshold=0.99, sort_edges=True):
    """Greedily remove edges while reoptimized fidelity remains above threshold."""
    if target_state is None:
        return list(state), None

    current = list(canonical_state_key(state, sort_edges=sort_edges))
    if len(current) <= 1:
        return current, None

    cache = {}

    def fidelity_for(candidate):
        key = canonical_state_key(candidate, sort_edges=sort_edges)
        if key not in cache:
            try:
                cache[key] = float(opt_fidelity(target_state, list(key), int(num_colors)))
            except Exception:
                cache[key] = float("-inf")
        return cache[key]

    current_fidelity = fidelity_for(current)
    changed = True
    while changed and len(current) > 1:
        changed = False
        best_candidate = None
        best_fidelity = float("-inf")

        for edge_idx in range(len(current)):
            candidate = current[:edge_idx] + current[edge_idx + 1 :]
            candidate_fidelity = fidelity_for(candidate)
            if candidate_fidelity >= float(threshold) and candidate_fidelity > best_fidelity:
                best_candidate = candidate
                best_fidelity = candidate_fidelity

        if best_candidate is not None:
            current = best_candidate
            current_fidelity = best_fidelity
            changed = True

    return current, current_fidelity


def plot_sample_ranking(
    sampled_states,
    metric_values,
    filename="sample_ranking.svg",
    top_n=20,
    num_windows=50,
    ranking_mode="highest_over_all",
    sort_edges=True,
    cmap="viridis",
    callout_states=None,
    total_nodes=None,
    n_ancilla=0,
    top_graph_count=3,
    callout_label="Reward",
    metric_label="Metric",
    target_state=None,
    num_colors=None,
    prune_callouts=True,
    prune_threshold=0.99,
    save_callout_graphs=True,
    sample_marker_mode="interval",
):
    """Plot a bump chart of top final states over sampled training windows."""
    if len(sampled_states) != len(metric_values):
        raise ValueError("sampled_states and metric_values must have the same length.")
    if not sampled_states:
        raise ValueError("No sampled states were provided.")

    _apply_style()
    top_n = int(top_n)
    n_samples = len(sampled_states)
    window_centers, per_window, tracked_keys, highest, cumulative = _ranking_windows(
        sampled_states,
        metric_values,
        top_n,
        max(1, int(num_windows)),
        ranking_mode,
        sort_edges,
    )

    fig_height = max(5.2, 0.24 * max(10, len(tracked_keys)))
    fig, ax = plt.subplots(figsize=(13.0, fig_height))
    first_seen = _first_iteration_by_state(sampled_states, sort_edges=sort_edges)
    marker_mode = str(sample_marker_mode).lower().replace("-", "_")
    if marker_mode not in {"interval", "first"}:
        raise ValueError("sample_marker_mode must be 'interval' or 'first'.")
    first_values = np.asarray([first_seen.get(key, 1) for key in tracked_keys], dtype=float)
    color_min = float(np.min(first_values)) if first_values.size else 1.0
    color_max = float(np.max(first_values)) if first_values.size else float(n_samples)
    if color_min == color_max:
        color_max = color_min + 1.0
    norm = colors.Normalize(vmin=color_min, vmax=color_max)
    palette = cm.get_cmap(cmap)
    pruned_lookup = _build_callout_state_lookup(callout_states, sort_edges=sort_edges)
    endpoint_by_key = {}
    output_paths = []
    final_score_by_key = {}
    for window in per_window:
        for key, score in window["scores"].items():
            if not np.isfinite(score):
                continue
            if key not in final_score_by_key:
                final_score_by_key[key] = score
            elif highest:
                final_score_by_key[key] = max(final_score_by_key[key], score)
            else:
                final_score_by_key[key] = min(final_score_by_key[key], score)

    for idx, key in enumerate(tracked_keys):
        xs = []
        ys = []
        sampled_x = []
        sampled_y = []
        for center, window in zip(window_centers, per_window):
            rank = window["ranks"].get(key)
            if rank is None or rank > top_n:
                xs.append(np.nan)
                ys.append(np.nan)
                continue
            xs.append(center)
            ys.append(rank)
            if marker_mode == "interval" and key in window["sampled"]:
                sampled_x.append(center)
                sampled_y.append(rank)
            elif marker_mode == "first":
                first_iteration = first_seen.get(key)
                if first_iteration is not None and key in window["sampled"]:
                    if window["start"] < first_iteration <= window["end"]:
                        sampled_x.append(first_iteration)
                        sampled_y.append(rank)
        color = palette(norm(first_seen.get(key, 1)))
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9)
        if sampled_x:
            ax.scatter(sampled_x, sampled_y, color=color, s=22, edgecolor="white", linewidth=0.5, zorder=3)
        last_finite = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
        if last_finite:
            endpoint_by_key[key] = last_finite[-1]

    callout_keys = [key for key in tracked_keys[: max(0, int(top_graph_count))] if key in endpoint_by_key]
    if callout_keys:
        inferred_nodes = 1
        for key in tracked_keys:
            if key:
                inferred_nodes = max(inferred_nodes, max(max(edge[0]) for edge in key) + 1)
        total_nodes = inferred_nodes if total_nodes is None else int(total_nodes)

        graph_x = n_samples * 1.125
        graph_y_max = min(top_n - 2.0, 2.2 + 5.0 * (len(callout_keys) - 1))
        graph_ys = np.linspace(2.2, graph_y_max, num=len(callout_keys))
        for callout_idx, (key, graph_y) in enumerate(zip(callout_keys, graph_ys), start=1):
            endpoint_x, endpoint_y = endpoint_by_key[key]
            color = palette(norm(first_seen.get(key, 1)))
            state_to_draw = _best_pruned_match(key, pruned_lookup) or list(key)
            if prune_callouts and target_state is not None and num_colors is not None:
                state_to_draw, _pruned_fidelity = iterative_fidelity_prune_state(
                    state_to_draw,
                    target_state=target_state,
                    num_colors=num_colors,
                    threshold=prune_threshold,
                    sort_edges=sort_edges,
                )
            edge_weights = None
            if state_to_draw and target_state is not None and num_colors is not None:
                optimized_reward, edge_weights = reward_fidelity(target_state, state_to_draw, int(num_colors), pruning=False)
                if not np.isfinite(optimized_reward) or optimized_reward <= 0:
                    edge_weights = None
            _draw_state_inset(
                ax=ax,
                center_xy=(graph_x, graph_y),
                state=state_to_draw,
                state_name=None,
                total_nodes=total_nodes,
                n_ancilla=n_ancilla,
                border_color=color,
                radius_x_data=n_samples * 0.120,
                radius_y_data=2.15,
                show_state_name=False,
                edge_weights=edge_weights,
            )
            score = final_score_by_key.get(key)
            if score is not None and np.isfinite(score):
                ax.text(
                    graph_x + n_samples * 0.112,
                    graph_y,
                    f"{callout_label}={score:.2f}",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    color="#222222",
                    zorder=7,
                )
            ax.annotate(
                "",
                xy=(graph_x - n_samples * 0.060, graph_y),
                xytext=(endpoint_x, endpoint_y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3, shrinkA=0, shrinkB=4),
                zorder=5,
            )

            if save_callout_graphs:
                output_base = Path(filename).with_suffix("")
                graph_filename = output_base.parent / f"{output_base.name}_top{callout_idx}_pruned_graph"
                output_paths.extend(
                    _save_state_graph_figure(
                        state_to_draw,
                        filename=graph_filename,
                        total_nodes=total_nodes,
                        n_ancilla=n_ancilla,
                        border_color=color,
                    )
                )

    ax.set_ylim(top_n + 0.5, 0.5)
    ax.set_xlim(1, n_samples * (1.39 if callout_keys else 1.02))
    tick_step = 20000 if n_samples >= 80000 else max(1, int(np.ceil(n_samples / 5 / 1000) * 1000))
    x_ticks = np.arange(tick_step, n_samples + 1, tick_step)
    if len(x_ticks) == 0 or x_ticks[-1] != n_samples:
        x_ticks = np.append(x_ticks, n_samples)
    ax.set_xticks(x_ticks)
    ax.set_yticks(range(1, top_n + 1))
    ax.set_xlabel("Sample iteration")
    ax.set_ylabel("Rank")
    direction = "highest" if highest else "lowest"
    ax.set_title(f"Sample Ranking ({direction} {metric_label.lower()})")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=palette), ax=ax, pad=0.012)
    cbar.set_label("First sampled iteration")
    output_paths.extend(_save_figure(fig, filename))
    return output_paths


def plot_state_projection(
    sampled_states,
    metric_values=None,
    feature_keys=None,
    filename="state_projection.svg",
    method="pca",
    style="hex",
    color_by="metric_mean",
    gridsize=25,
    sort_edges=True,
    random_state=0,
    top_label_count=0,
    cmap="viridis",
    metric_label="Metric",
    show_samples=False,
):
    """Project unique final states into 2D and render a scatter or hexbin view."""
    if not sampled_states:
        raise ValueError("No sampled states were provided.")

    _apply_style()
    records = aggregate_sampled_states(
        sampled_states,
        metric_values=metric_values,
        sort_edges=sort_edges,
    )
    unique_states = [rec["state"] for rec in records]
    features, _feature_list = state_feature_matrix(
        unique_states,
        feature_keys=feature_keys,
        sort_edges=sort_edges,
    )
    coords = np.asarray(_project_features(features, method=method, random_state=random_state), dtype=float)

    for rec, xy in zip(records, coords):
        rec["x"] = float(xy[0])
        rec["y"] = float(xy[1])

    if color_by == "count":
        values = np.asarray([rec["count"] for rec in records], dtype=float)
        color_label = "Sample count"
    elif color_by in {"last_iteration", "first_iteration"}:
        values = np.asarray([rec[color_by] for rec in records], dtype=float)
        color_label = color_by.replace("_", " ").title()
    else:
        values = np.asarray([rec.get(color_by, rec["metric_mean"]) for rec in records], dtype=float)
        color_label = metric_label if color_by == "metric_mean" else color_by.replace("_", " ").title()

    x = np.asarray([rec["x"] for rec in records], dtype=float)
    y = np.asarray([rec["y"] for rec in records], dtype=float)
    counts = np.asarray([rec["count"] for rec in records], dtype=float)
    sizes = 20.0 + 95.0 * np.sqrt(counts / max(1.0, counts.max()))

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    style = str(style).lower()
    finite = np.isfinite(values)

    if style == "hex":
        if np.any(finite):
            hb = ax.hexbin(
                x[finite],
                y[finite],
                C=values[finite],
                reduce_C_function=np.nanmean,
                gridsize=int(gridsize),
                cmap=cmap,
                mincnt=1,
                linewidths=0.35,
                edgecolors="white",
            )
            cbar = fig.colorbar(hb, ax=ax, pad=0.01)
            cbar.set_label(f"Mean {color_label}")
        if show_samples:
            ax.scatter(x, y, s=10, color="black", alpha=0.18, linewidths=0)
    elif style == "scatter":
        norm = None
        if np.any(finite):
            norm = colors.Normalize(vmin=np.nanmin(values[finite]), vmax=np.nanmax(values[finite]))
        sc = ax.scatter(
            x,
            y,
            c=values,
            s=sizes,
            cmap=cmap,
            norm=norm,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label(color_label)
    else:
        raise ValueError("style must be 'hex' or 'scatter'.")

    label_count = max(0, int(top_label_count))
    if label_count:
        label_records = sorted(
            records,
            key=lambda rec: (
                np.nan_to_num(rec.get("metric_max", np.nan), nan=-np.inf),
                rec["count"],
            ),
            reverse=True,
        )[:label_count]
        for rec in label_records:
            ax.annotate(
                state_key_to_label(rec["key"], max_edges=3),
                (rec["x"], rec["y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#dddddd", alpha=0.72),
            )

    ax.set_title(f"State Projection ({method.upper()}, {len(records)} unique states)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=0.7)
    return _save_figure(fig, filename)
