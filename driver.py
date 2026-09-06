from gflow_qoc.utils import *
from gflow_qoc.gflow_utils import *
from gflow_qoc.result_analysis import *
from gflow_qoc.training import *
import itertools
import time
import pickle

# Define number of nodes and colors
num_nodes, num_colors = 4, 2 #4, 2 for (4,2)-GHZ. 6, 3 for (6,3)-GHZ. 3, 2 for (3,2)-GHZ
# If the number of optical paths is odd, you will require an ancilla qubit. 
args = parse_args(sys.argv) # Pass CLI args as key=value, e.g. n_a=1 c_a=2
q_gate = "--q-gate" in sys.argv[1:]
a_edges = "--a-edges" in sys.argv[1:]
n_a = int(args.get("n_a", "0"))  # default: 0 ancillas
c_a = args.get("c_a")  # Optional: number of ancilla colors (0..c_a-1)
FEATURE_KEYS, num_nodes = build_feature_keys(
    num_nodes=num_nodes,
    num_colors=num_colors,
    n_a=n_a,
    c_a=c_a,
    verbose=True,
)
if a_edges and n_a > 1:
    ancilla_colors = range(int(c_a) if c_a is not None else 1)
    FEATURE_KEYS.extend(
        ((a1, a2), (c1, c2))
        for a1, a2 in itertools.combinations(range(num_nodes - n_a, num_nodes), 2)
        for c1, c2 in itertools.product(ancilla_colors, repeat=2)
    )
    print("Size of feature keys w/ancilla-ancilla edges = {}".format(len(FEATURE_KEYS)))
if q_gate:
    num_gate_nodes = num_nodes - n_a
    if num_gate_nodes % 2 != 0:
        raise ValueError("Quantum-gate mode requires an even number of non-ancilla nodes.")
    num_input_nodes = num_gate_nodes // 2
    FEATURE_KEYS = [
        key for key in FEATURE_KEYS
        if not (key[0][0] < num_input_nodes and key[0][1] < num_input_nodes)
    ]
    print(f"Quantum-gate mode: input nodes are 0 through {num_input_nodes - 1}")
    print("Size of feature keys w/q-gate restriction = {}".format(len(FEATURE_KEYS)))
# Fixed hyperparameters.
pruning = True
n_hid_units = 128 #512 for MLP, 128 for GIN/GAT/Transformer
edge_feat_dim = 2 #Dimension of edge features, 2 for pair of colors
n_episodes = 1000
learning_rate = 1e-3
logZ_lr_mult = 10.0
decay_rate = 1.00
beta = 1.0
beta_curve = False
beta_start = 1e-3
beta_warmup_steps = 1000
update_freq = 10 #Update every episode
seed = 666
max_edges = 4
#target_expr = "1|0000⟩ + 1|1110⟩" #(3,2)-GHZ with ancilla |\psi⟩ = 1/sqrt(2) (|000⟩ + |111⟩) tensor |0⟩
target_expr = "1|0000⟩ + 1|1111⟩" #(4,2)-GHZ
#target_expr = "1|000000⟩ + 1|010100⟩ + 1|101100⟩ + 1|111000⟩" # CNOT gate with two |0⟩ ancillas; run with --q-gate n_a=2
#target_expr = "1|000000⟩ + 1|111111⟩+ 1|222222⟩" #(6,3)-GHZ
#target_expr = "1|0120⟩ + 1|0210⟩ + 1|1020⟩ + 1|1200⟩ + 1|2010⟩ + 1|2100⟩" # |D(3,(1,1,1))⟩tensor|0⟩ 

target_state = parse_dirac_expression(target_expr, num_nodes, num_colors)
fig_name = "42_GHZ"
plot_tb_state_space = False # Set to True to save TB state space snapshots during training (can be large, e.g., 1000 episodes x 10 snapshots = 10,000 images)
tb_snapshot_dir = f"{fig_name}_tb_snapshots"

print("For all experiments, our hyperparameters will be:")
print("    + n_hid_units={}".format(n_hid_units))
print("    + n_episodes={}".format(n_episodes))
print("    + learning_rate={}".format(learning_rate))
print("    + logZ_lr_mult={}".format(logZ_lr_mult))
print("    + decay_rate={}".format(decay_rate))
print("    + beta={}".format(beta))
print("    + beta_curve={}".format(beta_curve))
print("    + beta_start={}".format(beta_start))
print("    + beta_warmup_steps={}".format(beta_warmup_steps))
print("    + update_freq={}".format(update_freq))
print("    + seed={}".format(seed))
print("    + max_edges={}".format(max_edges))
print("    + Target state:", state_vector_to_dirac_notation(target_state, num_nodes, num_colors))
print("    + Pruning:", pruning)
print("    + plot_tb_state_space={}".format(plot_tb_state_space))
t0 = time.time()

#For TBModel, uncomment the following lines: It requires the model beforehand!!!
#model = TBModel(n_hid_units, FEATURE_KEYS)
#model = embTBModel(n_hid_units, FEATURE_KEYS, n_emb=16)
#sampled_states, losses, logZs, rewards, pruned_states = TB_train(num_colors, num_nodes, FEATURE_KEYS,
#                                           target_expr, target_state, model, 
#                                           n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, pruning, logZ_lr_mult=logZ_lr_mult,
#                                           save_snapshot_history=plot_tb_state_space, snapshot_every=10, snapshot_dir=tb_snapshot_dir)

#sampled_states, losses, logZs, rewards, pruned_states = GIN_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, pruning, logZ_lr_mult=logZ_lr_mult, save_snapshot_history=plot_tb_state_space, snapshot_every=10, snapshot_dir=tb_snapshot_dir)
#sampled_states, losses, logZs, rewards, pruned_states = GINE_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim, pruning, logZ_lr_mult=logZ_lr_mult, save_snapshot_history=plot_tb_state_space, snapshot_every=10, snapshot_dir=tb_snapshot_dir)
#sampled_states, losses, logZs, rewards, pruned_states = GAT_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim, pruning, logZ_lr_mult=logZ_lr_mult, save_snapshot_history=plot_tb_state_space, snapshot_every=10, snapshot_dir=tb_snapshot_dir)
sampled_states, losses, logZs, rewards, pruned_states = Transformer_TB_train(
    num_colors,
    num_nodes,
    FEATURE_KEYS,
    target_expr,
    target_state,
    n_episodes,
    learning_rate,
    decay_rate,
    seed,
    update_freq,
    max_edges,
    n_hid_units,
    edge_feat_dim,
    pruning,
    logZ_lr_mult=logZ_lr_mult,
    beta=beta,
    beta_curve=beta_curve,
    beta_start=beta_start,
    beta_warmup_steps=beta_warmup_steps,
    save_snapshot_history=plot_tb_state_space,
    snapshot_every=10,
    snapshot_dir=tb_snapshot_dir,
)

with open(fig_name + "_sampled_graphs.p", 'wb') as f:
    pickle.dump(sampled_states, f, pickle.HIGHEST_PROTOCOL)

with open(fig_name + "_rewards.p", 'wb') as f:
    pickle.dump(rewards, f, pickle.HIGHEST_PROTOCOL)

with open(fig_name + "_pruned_states.p", 'wb') as f:
    pickle.dump(pruned_states, f, pickle.HIGHEST_PROTOCOL)

t1 = time.time()
print(f"Training time: {t1 - t0:.2f} seconds")

#Postprocessing section! Can be done separately.
#Recompute fidelities and plot best states

t2 = time.time()
state_fids = [(opt_fidelity(target_state, s, num_colors), s) for s in sampled_states]

# Sort by fidelity (descending)
state_fids.sort(key=lambda x: x[0], reverse=True)
ordered_states = [s for fid, s in state_fids]

# Print best + top 20 without recomputing
print("Fidelity for the best state:", state_fids[0][0])

print("Fidelity for the best 20 states:")
for fid, _s in state_fids[:min(20, len(state_fids))]:
    print("Fidelity:", fid)

t3 = time.time()
print(f"Sorting time: {t3 - t2:.2f} seconds")

plot_graph(ordered_states[0], filename=fig_name + "_best_state.svg", n_ancilla=n_a)
histo_fidelity(state_fids, filename=fig_name + "_fidelity_all_states_histogram.svg")
plot_loss_curve(fig_name, losses, title="Trajectory Balance Loss")
plot_rewards(rewards)

if plot_tb_state_space:
    tb_plot_outputs = plot_tb_state_space_dynamics(
        sampled_states=sampled_states,
        rewards=rewards,
        FEATURE_KEYS=FEATURE_KEYS,
        target_expr=target_expr,
        snapshot_dir=tb_snapshot_dir,
        output_prefix=fig_name + "_tb_state_space",
        top_k=3,
        mid_k=3,
        bottom_k=3,
        random_seed=seed,
        compressed_depth=3,
        n_ancilla=n_a,
        final_episode=n_episodes,
        rank_by_final_probability=False,
        show_edge_prob_labels=False,
        fps=24,
        frame_step=1,
        max_frames=None,
        animation_format="mp4",
        animation_dpi=120,
    )
    print("TB state space start plot:", tb_plot_outputs["start_plot"])
    print("TB state space end plot:", tb_plot_outputs["end_plot"])
    print("TB state space animation:", tb_plot_outputs["animation"])

t4 = time.time()
print(f"Plotting time: {t4 - t3:.2f} seconds")

if pruning and pruned_states:
    pruned_fids = [(opt_fidelity(target_state, s, num_colors), s) for s in pruned_states]

    # Sort by fidelity (descending) for your existing "best pruned state"
    pruned_fids.sort(key=lambda x: x[0], reverse=True)
    ordered_pruned_states = [s for fid, s in pruned_fids]

    print("Fidelity for the best pruned state:", pruned_fids[0][0])

    print("Fidelity for the best 20 pruned states:")
    for fid, _s in pruned_fids[:min(20, len(pruned_fids))]:
        print("Fidelity:", fid)

    #Smallest pruned state with the highest fidelity
    smallest_best_fid, smallest_best_state = min(
        pruned_fids,
        key=lambda x: (len(x[1]), -x[0])
    )

    print("\nSmallest pruned state (min edges, then max fidelity):")
    print("  #edges:", len(smallest_best_state))
    print("  fidelity:", smallest_best_fid)

    tp = time.time()
    print(f"Pruned states sorting time: {tp - t4:.2f} seconds")
    plot_graph(smallest_best_state, filename=fig_name + "_smallest_best_pruned_state.svg", n_ancilla=n_a)
    plot_graph(ordered_pruned_states[0], filename=fig_name + "_best_pruned_state.svg", n_ancilla=n_a)
    histo_fidelity(pruned_fids, filename=fig_name + "_fidelity_pruned_states_histogram.svg")
