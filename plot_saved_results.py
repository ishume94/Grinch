import pickle
from pathlib import Path

import torch

from gflow_qoc.result_analysis import *
from gflow_qoc.gflow_utils import *
from gflow_qoc.utils import *

num_nodes, num_colors = 4, 2  # Optical paths before ancillas.
n_a = 0
c_a = None  # Optional: number of ancilla colors (0..c_a-1). None keeps ancilla fixed to color 0.
target_expr = "1|0000⟩ + 1|1111⟩"
fig_name = "32_GHZ"
seed = 666
n_episodes = 1000

# Animation controls for TB state-space rendering.
animation_format = "mp4"  # "mp4" (recommended) or "gif"
animation_fps = 24
animation_frame_step = 1   # Use >1 to skip snapshots (e.g., 5 or 10)
animation_max_frames = None  # e.g., 600 to cap total frames
animation_dpi = 120


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _find_checkpoint(fig_name):
    candidates = []
    root = Path(".")

    for pattern in [f"{fig_name}*TBmodel.pth", f"{fig_name}*model.pth"]:
        candidates.extend(root.glob(pattern))

    fixed_names = [
        "TransformerTBmodel.pth",
        "GATTBmodel.pth",
        "GINETBmodel.pth",
        "GINTBmodel.pth",
        "TBmodel.pth",
    ]
    for name in fixed_names:
        path = root / name
        if path.exists():
            candidates.append(path)

    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _load_losses(checkpoint_path):
    if checkpoint_path is None:
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"[warning] Could not read checkpoint '{checkpoint_path}': {exc}")
        return None

    if isinstance(checkpoint, dict):
        losses = checkpoint.get("loss")
        if isinstance(losses, (list, tuple)):
            return [float(x) for x in losses]
    return None


def main():
    sampled_path = Path(f"{fig_name}_sampled_graphs.p")
    rewards_path = Path(f"{fig_name}_rewards.p")
    pruned_path = Path(f"{fig_name}_pruned_states.p")
    snapshot_dir = Path(f"{fig_name}_tb_snapshots")

    if not sampled_path.exists():
        raise FileNotFoundError(f"Missing sampled graphs: {sampled_path}")
    sampled_states = _load_pickle(sampled_path)
    if not sampled_states:
        raise ValueError(f"No sampled states found in {sampled_path}")

    rewards = _load_pickle(rewards_path) if rewards_path.exists() else None
    if rewards is None:
        print(f"[warning] Missing rewards file: {rewards_path}. Reward plots may be skipped.")

    pruned_states = _load_pickle(pruned_path) if pruned_path.exists() else None
    if pruned_states is None:
        print(f"[info] Missing pruned states file: {pruned_path}. Pruned-state plots will be skipped.")

    if not target_expr:
        raise ValueError("target_expr is empty. Set it at the top of plot_saved_results.py.")

    total_nodes = num_nodes + n_a
    target_state = parse_dirac_expression(target_expr, int(total_nodes), int(num_colors))

    print("Regenerating plots with:")
    print(f"  fig_name={fig_name}")
    print(
        f"  num_nodes={num_nodes} (optical), total_nodes={total_nodes}, "
        f"num_colors={num_colors}, n_a={n_a}, "
        f"c_a={(1 if c_a is None else c_a)}"
    )
    print(f"  target_expr={target_expr}")
    print(f"  sampled_states={len(sampled_states)}")
    if rewards is not None:
        print(f"  rewards={len(rewards)}")

    state_fids = [(opt_fidelity(target_state, state, int(num_colors)), state) for state in sampled_states]
    state_fids.sort(key=lambda x: x[0], reverse=True)
    ordered_states = [state for _fid, state in state_fids]

    print(f"Fidelity for best sampled state: {state_fids[0][0]:.6f}")

    plot_graph(ordered_states[0], filename=f"{fig_name}_best_state.svg", n_ancilla=n_a)
    histo_fidelity(state_fids, filename=f"{fig_name}_fidelity_all_states_histogram.svg")

    checkpoint_path = _find_checkpoint(fig_name)
    losses = _load_losses(checkpoint_path)
    if losses:
        print(f"Loaded {len(losses)} loss points from checkpoint: {checkpoint_path}")
        plot_loss_curve(fig_name, losses, title="Trajectory Balance Loss")
    else:
        print("[info] Could not load loss history; skipping loss curve.")

    if rewards is not None and len(rewards) > 0:
        plot_rewards(rewards)
    else:
        print("[info] Rewards are missing/empty; skipping rewards_progress plot.")

    if snapshot_dir.exists() and snapshot_dir.is_dir():
        if rewards is not None and len(rewards) == len(sampled_states):
            feature_keys, _ = build_feature_keys(
                num_nodes=num_nodes,
                num_colors=num_colors,
                n_a=n_a,
                c_a=c_a,
                verbose=False,
            )
            try:
                tb_outputs = plot_tb_state_space_dynamics(
                    sampled_states=sampled_states,
                    rewards=rewards,
                    FEATURE_KEYS=feature_keys,
                    target_expr=target_expr,
                    snapshot_dir=str(snapshot_dir),
                    output_prefix=f"{fig_name}_tb_state_space",
                    top_k=3,
                    mid_k=3,
                    bottom_k=3,
                    random_seed=seed,
                    compressed_depth=3,
                    n_ancilla=n_a,
                    final_episode=n_episodes,
                    rank_by_final_probability=False,
                    show_edge_prob_labels=False,
                    fps=animation_fps,
                    frame_step=animation_frame_step,
                    max_frames=animation_max_frames,
                    animation_format=animation_format,
                    animation_dpi=animation_dpi,
                )
                print("TB state space start plot:", tb_outputs["start_plot"])
                print("TB state space end plot:", tb_outputs["end_plot"])
                print("TB state space animation:", tb_outputs["animation"])
            except Exception as exc:
                print(f"[warning] Could not generate TB state-space plots: {exc}")
        else:
            print("[info] Missing/invalid rewards for TB snapshots; skipping state-space plots.")
    else:
        print(f"[info] Snapshot folder not found: {snapshot_dir}. Skipping state-space plots.")

    if pruned_states:
        pruned_fids = [(opt_fidelity(target_state, state, int(num_colors)), state) for state in pruned_states]
        pruned_fids.sort(key=lambda x: x[0], reverse=True)
        ordered_pruned_states = [state for _fid, state in pruned_fids]

        best_pruned_fid = pruned_fids[0][0]
        print(f"Fidelity for best pruned state: {best_pruned_fid:.6f}")

        smallest_best_fid, smallest_best_state = min(
            pruned_fids,
            key=lambda x: (len(x[1]), -x[0]),
        )
        print("Smallest best pruned state:")
        print(f"  edges={len(smallest_best_state)}, fidelity={smallest_best_fid:.6f}")

        plot_graph(
            smallest_best_state,
            filename=f"{fig_name}_smallest_best_pruned_state.svg",
            n_ancilla=n_a,
        )
        plot_graph(
            ordered_pruned_states[0],
            filename=f"{fig_name}_best_pruned_state.svg",
            n_ancilla=n_a,
        )
        histo_fidelity(pruned_fids, filename=f"{fig_name}_fidelity_pruned_states_histogram.svg")
    else:
        print("[info] No pruned states available; skipping pruned-state plots.")


if __name__ == "__main__":
    main()
