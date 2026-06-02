import ast
import pickle
import sys
from pathlib import Path

from gflow_qoc.fancy_plots import (
    canonical_state_key,
    plot_sample_ranking,
    plot_state_projection,
)
from gflow_qoc.gflow_utils import build_feature_keys
from gflow_qoc.utils import opt_fidelity, parse_dirac_expression


CONFIG = {
    "data_dir": ".",
    "output_dir": ".",
    "fig_name": "32_GHZ",
    "num_nodes": 4,
    "num_colors": 2,
    "n_a": 0,
    "c_a": None,
    "target_expr": "1|0000> + 1|1111>",
    "metric_source": "rewards",  # rewards, fidelity, or auto
    "ranking_top_n": 20,
    "ranking_windows": 60,
    "ranking_mode": "highest_over_all",
    "ranking_callout_graphs": 3,
    "callout_prune": True,
    "callout_prune_threshold": 0.99,
    "save_callout_graphs": True,
    "projection_method": "pca",  # pca or identity
    "projection_gridsize": 24,
    "projection_labels": 0,
    "make_scatter": True,
}


DRIVER_CONFIG_KEYS = {
    "target_expr",
    "n_a",
    "c_a",
    "num_nodes",
    "num_nodes_",
    "num_colors",
    "fig_name",
}


def _coerce_value(value):
    if value in {"None", "none", "null", ""}:
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
            return value


def _parse_cli_overrides(argv):
    overrides = {}
    for arg in argv[1:]:
        if "=" not in arg:
            raise ValueError(f"Arguments must use key=value syntax, got: {arg}")
        key, value = arg.split("=", 1)
        key = key.strip()
        if key not in CONFIG:
            valid = ", ".join(sorted(CONFIG))
            raise KeyError(f"Unknown option '{key}'. Valid options: {valid}")
        overrides[key] = _coerce_value(value.strip())
    return overrides


def _literal_from_node(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        pass

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"int", "float", "str"} and len(node.args) == 1:
            value = _literal_from_node(node.args[0])
            if value is None:
                return None
            try:
                return {"int": int, "float": float, "str": str}[node.func.id](value)
            except Exception:
                return None

        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if len(node.args) >= 2:
                return _literal_from_node(node.args[1])
            return None

    return None


def _driver_config_from_assign(target, value_node):
    if not isinstance(target, ast.Name):
        return None, None
    name = target.id
    if name not in DRIVER_CONFIG_KEYS:
        return None, None
    key = "num_nodes" if name == "num_nodes_" else name
    return key, _literal_from_node(value_node)


def _load_driver_config(driver_path):
    if not driver_path.exists():
        return {}

    try:
        tree = ast.parse(driver_path.read_text())
    except Exception as exc:
        print(f"[warning] Could not parse driver config from {driver_path}: {exc}")
        return {}

    driver_config = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple) and isinstance(node.value, (ast.Tuple, ast.List)):
                    for sub_target, sub_value in zip(target.elts, node.value.elts):
                        key, value = _driver_config_from_assign(sub_target, sub_value)
                        if key is not None and value is not None:
                            driver_config[key] = _coerce_value(str(value)) if isinstance(value, str) else value
                    continue

                key, value = _driver_config_from_assign(target, node.value)
                if key is not None and value is not None:
                    driver_config[key] = _coerce_value(str(value)) if isinstance(value, str) else value

        elif isinstance(node, ast.AnnAssign):
            key, value = _driver_config_from_assign(node.target, node.value)
            if key is not None and value is not None:
                driver_config[key] = _coerce_value(str(value)) if isinstance(value, str) else value

    return driver_config


def _build_config(argv):
    cli_overrides = _parse_cli_overrides(argv)
    config = dict(CONFIG)

    if "data_dir" in cli_overrides:
        config["data_dir"] = cli_overrides["data_dir"]

    driver_path = Path(config["data_dir"]) / "driver.py"
    driver_config = _load_driver_config(driver_path)
    if driver_config:
        print(f"Loaded plot config from: {driver_path}")
        config.update(driver_config)

    config.update(cli_overrides)
    return config


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_optional_pickle(path):
    if not path.exists():
        return None
    return _load_pickle(path)


def _load_metrics(config, sampled_states):
    data_dir = Path(config["data_dir"])
    fig_name = str(config["fig_name"])
    rewards_path = data_dir / f"{fig_name}_rewards.p"
    metric_source = str(config["metric_source"]).lower()

    if metric_source in {"auto", "rewards"} and rewards_path.exists():
        rewards = _load_pickle(rewards_path)
        if len(rewards) != len(sampled_states):
            raise ValueError(
                f"Rewards length ({len(rewards)}) does not match sampled states ({len(sampled_states)})."
            )
        return [float(x) for x in rewards], "reward"

    if metric_source == "rewards":
        raise FileNotFoundError(f"Missing rewards file: {rewards_path}")

    if not config["target_expr"]:
        raise ValueError("target_expr is required when metric_source=fidelity or rewards are unavailable.")

    total_nodes = int(config["num_nodes"]) + int(config["n_a"])
    target_state = parse_dirac_expression(
        str(config["target_expr"]),
        int(total_nodes),
        int(config["num_colors"]),
    )

    metric_by_state = {}
    metrics = []
    for idx, state in enumerate(sampled_states, start=1):
        key = canonical_state_key(state)
        if key not in metric_by_state:
            metric_by_state[key] = opt_fidelity(target_state, list(key), int(config["num_colors"]))
        metrics.append(metric_by_state[key])
        if idx % 1000 == 0:
            print(f"  computed fidelities for {idx}/{len(sampled_states)} samples")
    return metrics, "fidelity"


def _infer_total_nodes(sampled_states):
    max_node = -1
    for state in sampled_states:
        for (n1, n2), _colors in state:
            max_node = max(max_node, int(n1), int(n2))
    return max_node + 1


def _target_state_from_config(config):
    if not config["target_expr"]:
        return None

    total_nodes = int(config["num_nodes"]) + int(config["n_a"])
    try:
        return parse_dirac_expression(
            str(config["target_expr"]),
            int(total_nodes),
            int(config["num_colors"]),
        )
    except Exception as exc:
        print(f"[warning] Could not parse target_expr for callout pruning: {exc}")
        return None


def main():
    config = _build_config(sys.argv)

    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    fig_name = str(config["fig_name"])
    sampled_path = data_dir / f"{fig_name}_sampled_graphs.p"
    pruned_path = data_dir / f"{fig_name}_pruned_states.p"

    if not sampled_path.exists():
        raise FileNotFoundError(f"Missing sampled graphs: {sampled_path}")

    sampled_states = _load_pickle(sampled_path)
    if not sampled_states:
        raise ValueError(f"No sampled states found in {sampled_path}")
    pruned_states = _load_optional_pickle(pruned_path)
    if not pruned_states:
        pruned_states = None

    inferred_total_nodes = _infer_total_nodes(sampled_states)
    configured_total_nodes = int(config["num_nodes"]) + int(config["n_a"])
    if inferred_total_nodes > configured_total_nodes:
        inferred_n_a = inferred_total_nodes - int(config["num_nodes"])
        print(
            f"[info] Inferred n_a={inferred_n_a} from sampled graphs "
            f"(driver/config gave total_nodes={configured_total_nodes})."
        )
        config["n_a"] = inferred_n_a

    metrics, metric_name = _load_metrics(config, sampled_states)
    target_state = _target_state_from_config(config)
    feature_keys, total_nodes = build_feature_keys(
        num_nodes=int(config["num_nodes"]),
        num_colors=int(config["num_colors"]),
        n_a=int(config["n_a"]),
        c_a=config["c_a"],
        verbose=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating fancy saved-result plots:")
    print(f"  data_dir={data_dir}")
    print(f"  fig_name={fig_name}")
    print(f"  samples={len(sampled_states)}")
    print(f"  unique states={len({canonical_state_key(state) for state in sampled_states})}")
    print(f"  metric={metric_name}")
    print(f"  total_nodes={total_nodes}, features={len(feature_keys)}")

    outputs = []
    plot_metric_label = "Reward" if metric_name == "reward" else metric_name.title()

    ranking_prefix = output_dir / f"{fig_name}_sample_ranking"
    outputs.extend(
        plot_sample_ranking(
            sampled_states=sampled_states,
            metric_values=metrics,
            filename=ranking_prefix,
            top_n=int(config["ranking_top_n"]),
            num_windows=int(config["ranking_windows"]),
            ranking_mode=str(config["ranking_mode"]),
            callout_states=pruned_states,
            total_nodes=total_nodes,
            n_ancilla=int(config["n_a"]),
            top_graph_count=int(config["ranking_callout_graphs"]),
            callout_label="R" if metric_name == "reward" else plot_metric_label,
            metric_label=plot_metric_label,
            target_state=target_state,
            num_colors=int(config["num_colors"]),
            prune_callouts=bool(config["callout_prune"]),
            prune_threshold=float(config["callout_prune_threshold"]),
            save_callout_graphs=bool(config["save_callout_graphs"]),
        )
    )

    hex_prefix = output_dir / f"{fig_name}_state_projection_hex"
    outputs.extend(
        plot_state_projection(
            sampled_states=sampled_states,
            metric_values=metrics,
            feature_keys=feature_keys,
            filename=hex_prefix,
            method=str(config["projection_method"]),
            style="hex",
            color_by="metric_mean",
            gridsize=int(config["projection_gridsize"]),
            top_label_count=int(config["projection_labels"]),
            metric_label=plot_metric_label,
        )
    )

    if bool(config["make_scatter"]):
        scatter_prefix = output_dir / f"{fig_name}_state_projection_scatter"
        outputs.extend(
            plot_state_projection(
                sampled_states=sampled_states,
                metric_values=metrics,
                feature_keys=feature_keys,
                filename=scatter_prefix,
                method=str(config["projection_method"]),
                style="scatter",
                color_by="last_iteration",
                gridsize=int(config["projection_gridsize"]),
                top_label_count=int(config["projection_labels"]),
                metric_label=plot_metric_label,
            )
        )

    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
