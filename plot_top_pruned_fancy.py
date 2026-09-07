"""Optimize and plot the best distinct graphs from a saved pruned_states file.

Usage:
    python plot_top_pruned_fancy.py
    python plot_top_pruned_fancy.py data_dir=results output_dir=plots seed=666

Settings are read from data_dir/driver.py without executing it; key=value
arguments override them. Only saved pruned graphs are evaluated, with no further
edge removal. Each distinct graph gets one random-start L-BFGS-B optimization.
The top 10 achieved fidelities, edge counts, and negative edges are printed and
saved alongside the aligned weights and SVG/PNG graph figures. Negative weights
are marked by white, black-outlined diamonds at the edge midpoints.
The smallest graph (fewest edges, then highest fidelity) is also reported and
plotted using its already optimized weights.
An additional graph is selected by fewest edges, then fewest negative weights,
then highest fidelity, using those same optimization results.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from gflow_qoc.fancy_plots import (
    _apply_style,
    _save_state_graph_figure,
    canonical_state_key,
)
from gflow_qoc.utils import (
    _infer_num_nodes_from_state_dimension,
    compute_fidelity,
    fidelity_objective,
    parse_dirac_expression,
)
from plot_results_fancy import (
    CONFIG as FANCY_CONFIG,
    _coerce_value,
    _infer_total_nodes,
    _load_driver_config,
    _load_pickle,
)


CONFIG = {
    key: FANCY_CONFIG[key]
    for key in (
        "data_dir", "output_dir", "fig_name", "num_nodes", "num_colors",
        "n_a", "c_a", "target_expr",
    )
}
CONFIG.update({"driver_file": "driver.py", "top_n": 10, "seed": None})


def _build_config(argv):
    overrides = {}
    for arg in argv[1:]:
        if "=" not in arg:
            raise ValueError(f"Arguments must use key=value syntax, got: {arg}")
        key, value = arg.split("=", 1)
        key = key.strip()
        if key not in CONFIG:
            raise KeyError(f"Unknown option '{key}'. Valid options: {', '.join(sorted(CONFIG))}")
        overrides[key] = _coerce_value(value.strip())

    config = dict(CONFIG)
    driver_path = Path(overrides.get("driver_file", config["driver_file"]))
    if not driver_path.is_absolute():
        driver_path = Path(overrides.get("data_dir", config["data_dir"])) / driver_path
    driver_config = _load_driver_config(driver_path)
    if driver_config:
        print(f"Loaded plot config from: {driver_path}")
        config.update(driver_config)
    else:
        print(f"[warning] No settings loaded from {driver_path}; using defaults and CLI overrides.")
    config.update(overrides)
    if int(config["top_n"]) < 1:
        raise ValueError("top_n must be at least 1.")
    if int(config["num_nodes"]) < 1 or int(config["n_a"]) < 0:
        raise ValueError("num_nodes must be positive and n_a must be nonnegative.")
    if int(config["num_colors"]) < 2:
        raise ValueError("num_colors must be at least 2.")
    return config


def _optimize_state(state, target_state, num_colors, rng):
    """Use the opt_fidelity objective, retaining the weights and solver status."""
    total_nodes = _infer_num_nodes_from_state_dimension(target_state.size, num_colors)
    nodes = {node for endpoints, _colors in state for node in endpoints}
    if nodes != set(range(total_nodes)):
        raise ValueError("Graph does not cover exactly the target nodes.")

    result = minimize(
        fidelity_objective,
        rng.uniform(-1, 1, len(state)),
        args=(state, target_state, num_colors),
        method="L-BFGS-B",
        bounds=[(-1, 1)] * len(state),
    )
    weights = np.asarray(result.x, dtype=float)
    # Evaluate the retained weights directly: the objective uses a penalty when
    # a graph has no normalized state, which is not a physical fidelity.
    fidelity = compute_fidelity(weights, state, target_state, num_colors)
    if not np.all(np.isfinite(weights)) or not np.isfinite(fidelity):
        raise ValueError("Optimization produced nonfinite weights or fidelity.")
    return {
        "state": state,
        "weights": weights.tolist(),
        "negative_edges": [
            {"edge_index": index, "edge": edge, "weight": float(weight)}
            for index, (edge, weight) in enumerate(zip(state, weights), start=1)
            if weight < 0
        ],
        "num_edges": len(state),
        "fidelity": float(fidelity),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def _rank_pruned_states(pruned_states, target_state, num_colors, seed=None):
    # Canonicalize before optimizing so the saved weights stay in edge order.
    unique_states = dict.fromkeys(canonical_state_key(state) for state in pruned_states)
    print(f"  saved pruned states={len(pruned_states)}, distinct graphs={len(unique_states)}")
    rng = np.random.default_rng(seed)
    records = []
    for index, key in enumerate(unique_states, start=1):
        try:
            record = _optimize_state(list(key), target_state, num_colors, rng)
        except ValueError as exc:
            print(f"[warning] Skipping pruned graph {index}: {exc}", flush=True)
            continue
        if not record["optimizer_success"]:
            print(
                f"[warning] Pruned graph {index}: {record['optimizer_message']}; "
                "retaining achieved fidelity.",
                flush=True,
            )
        records.append(record)
        if index % 100 == 0 or index == len(unique_states):
            print(f"  optimized {index}/{len(unique_states)} distinct pruned graphs", flush=True)
    if not records:
        raise ValueError("No pruned graph produced a valid normalized state.")
    records.sort(key=lambda record: (-record["fidelity"], record["num_edges"]))
    return records


def _print_solution(record, label):
    print(
        f"{label}: edges={record['num_edges']}, "
        f"negative_edges={len(record['negative_edges'])}, fidelity={record['fidelity']:.12f}"
    )
    if record["negative_edges"]:
        print("  Negative edges:")
        for negative in record["negative_edges"]:
            print(
                f"    edge {negative['edge_index']}: {negative['edge']}, "
                f"weight={negative['weight']:.12g}"
            )
    else:
        print("  Negative edges: none.")


def _save_solution(record, filename, total_nodes, n_ancilla):
    paths = _save_state_graph_figure(
        record["state"],
        filename=filename,
        total_nodes=total_nodes,
        n_ancilla=n_ancilla,
        edge_weights=record["weights"],
    )
    record["graph_files"] = paths
    return paths


def main(argv=None):
    config = _build_config(sys.argv if argv is None else argv)
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    fig_name = str(config["fig_name"])
    pruned_path = data_dir / f"{fig_name}_pruned_states.p"
    if not pruned_path.exists():
        raise FileNotFoundError(f"Missing pruned states file: {pruned_path}")
    pruned_states = _load_pickle(pruned_path)
    if len(pruned_states) == 0:
        raise ValueError(f"No pruned states found in {pruned_path}")

    total_nodes = int(config["num_nodes"]) + int(config["n_a"])
    inferred_nodes = _infer_total_nodes(pruned_states)
    if inferred_nodes > total_nodes:
        config["n_a"] = inferred_nodes - int(config["num_nodes"])
        total_nodes = inferred_nodes
        print(f"[info] Inferred n_a={config['n_a']} from pruned graphs.")
    num_colors = int(config["num_colors"])
    target_state = parse_dirac_expression(str(config["target_expr"]), total_nodes, num_colors)

    print(f"Optimizing saved pruned graphs from: {pruned_path}", flush=True)
    records = _rank_pruned_states(pruned_states, target_state, num_colors, config["seed"])
    top_records = records[:int(config["top_n"])]
    smallest_best = dict(min(records, key=lambda record: (record["num_edges"], -record["fidelity"])))
    smallest_fewest_negative = dict(min(
        records,
        key=lambda record: (record["num_edges"], len(record["negative_edges"]), -record["fidelity"]),
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    print(f"\nTop {len(top_records)} distinct pruned graphs by optimized fidelity:")
    print("Edge notation: ((node1, node2), (color1, color2)); edge indices start at 1.")
    outputs = []
    for rank, record in enumerate(top_records, start=1):
        record["rank"] = rank
        record["selection"] = "top"
        _print_solution(record, f"Rank {rank}")
        outputs.extend(_save_solution(
            record,
            filename=output_dir / f"{fig_name}_top_pruned_{rank:02d}_graph",
            total_nodes=total_nodes,
            n_ancilla=int(config["n_a"]),
        ))

    smallest_best["selection"] = "smallest_best"
    _print_solution(smallest_best, "\nSmallest best pruned state (fewest edges, then highest fidelity)")
    outputs.extend(_save_solution(
        smallest_best,
        filename=output_dir / f"{fig_name}_smallest_best_pruned_fancy_graph",
        total_nodes=total_nodes,
        n_ancilla=int(config["n_a"]),
    ))

    smallest_fewest_negative["selection"] = "smallest_fewest_negative"
    _print_solution(
        smallest_fewest_negative,
        "\nPruned state with fewest edges, then fewest negative edges, then highest fidelity",
    )
    outputs.extend(_save_solution(
        smallest_fewest_negative,
        filename=output_dir / f"{fig_name}_smallest_fewest_negative_pruned_fancy_graph",
        total_nodes=total_nodes,
        n_ancilla=int(config["n_a"]),
    ))

    csv_path = output_dir / f"{fig_name}_top_pruned_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["selection", "rank", "num_edges", "fidelity", "optimizer_success", "negative_edges"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for record in [*top_records, smallest_best, smallest_fewest_negative]:
            writer.writerow({**record, "negative_edges": json.dumps(record["negative_edges"])})
    json_path = output_dir / f"{fig_name}_top_pruned_results.json"
    with json_path.open("w") as handle:
        json.dump(
            {
                "config": config, "results": top_records, "smallest_best": smallest_best,
                "smallest_fewest_negative": smallest_fewest_negative,
            },
            handle, indent=2, allow_nan=False,
        )
        handle.write("\n")
    outputs.extend([str(csv_path), str(json_path)])
    print("\nSaved:")
    for path in outputs:
        print(f"  {path}")
    return top_records


if __name__ == "__main__":
    main()
