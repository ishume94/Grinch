from gflow_qoc.utils import *
from gflow_qoc.gflow_utils import *
from gflow_qoc.result_analysis import *
from gflow_qoc.training import *
import time
import pickle
import itertools
from itertools import combinations

# Define number of nodes and colors
num_nodes, num_colors = 3, 2 #4, 2 for (4,2)-GHZ. 6, 3 for (6,3)-GHZ. 3, 2 for (3,2)-GHZ
# If the number of optical paths is odd, you will require an ancilla qubit. 
args = parse_args(sys.argv) # Pass in the command line the number of ancillas as n_a=Ancilla nodes
n_a = int(args.get("n_a", "0"))  # default: 0 ancillas
print(f"Number of optical paths: {num_nodes}, Number of modes: {num_colors}")

node_pairs = list(itertools.combinations(range(num_nodes), 2))
color_pairs = list(itertools.product(range(num_colors), repeat=2))
FEATURE_KEYS = [((n1, n2), (c1, c2)) for (n1, n2) in node_pairs for (c1, c2) in color_pairs]
print("Size of feature keys = {}".format(len(FEATURE_KEYS)))

if n_a > 0:
    print(f"Number of ancilla nodes: {n_a}")
    ancilla_color = 0
    ancilla_nodes = range(num_nodes, num_nodes + n_a)

    ancilla_feature_keys = [((i, a), (c, ancilla_color))
                            for a in ancilla_nodes
                            for i in range(num_nodes)
                            for c in range(num_colors)]

    FEATURE_KEYS.extend(ancilla_feature_keys)
    print(f"Added ancilla features = {len(ancilla_feature_keys)}")
    #print(f"Ancilla features = {ancilla_feature_keys}") 
    print("Size of feature keys w/ancilla= {}".format(len(FEATURE_KEYS)))
    num_nodes = num_nodes + n_a #Update number of nodes
    print(f"Updated number of nodes including ancillas: {num_nodes}")
# Fixed hyperparameters.
pruning = True
n_hid_units = 128 #512 for MLP, 128 for GIN/GAT/Transformer
edge_feat_dim = 2 #Dimension of edge features, 2 for pair of colors
n_episodes = 20000
learning_rate = 1e-3
decay_rate = 1.00
update_freq = 10 #Update every episode
seed = 666
max_edges = 6 #4 for (4,2)-GHZ and 9 for (6,3)-GHZ
target_expr = "1|0000⟩ + 1|1110⟩" #(3,2)-GHZ with ancilla |\psi⟩ = 1/sqrt(2) (|000⟩ + |111⟩) tensor |0⟩
#target_expr = "1|0000⟩ + 1|1111⟩" #(4,2)-GHZ
#target_expr = "1|000000⟩ + 1|111111⟩+ 1|222222⟩" #(6,3)-GHZ
#target_expr = "1|0120⟩ + 1|0210⟩ + 1|1020⟩ + 1|1200⟩ + 1|2010⟩ + 1|2100⟩" # |D(3,(1,1,1))⟩tensor|0⟩ 

target_state = parse_dirac_expression(target_expr, num_nodes, num_colors)
fig_name = "3D_GHZ"

print("For all experiments, our hyperparameters will be:")
print("    + n_hid_units={}".format(n_hid_units))
print("    + n_episodes={}".format(n_episodes))
print("    + learning_rate={}".format(learning_rate))
print("    + decay_rate={}".format(decay_rate))
print("    + update_freq={}".format(update_freq))
print("    + seed={}".format(seed))
print("    + max_edges={}".format(max_edges))
print("    + Target state:", state_vector_to_dirac_notation(target_state, num_nodes, num_colors))
print("    + Pruning:", pruning)
t0 = time.time()

#For TBModel, uncomment the following lines: It requires the model beforehand!!!
#model = TBModel(n_hid_units, FEATURE_KEYS)
#model = embTBModel(n_hid_units, FEATURE_KEYS, n_emb=16)
#sampled_states, losses, logZs, rewards, pruned_states = TB_train(num_colors, num_nodes, FEATURE_KEYS,
#                                           target_expr, target_state, model, 
#                                           n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, pruning)

#sampled_states, losses, logZs, rewards, pruned_states = GIN_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, pruning)
#sampled_states, losses, logZs, rewards, pruned_states = GINE_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim, pruning)
#sampled_states, losses, logZs, rewards, pruned_states = GAT_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim, pruning)
sampled_states, losses, logZs, rewards, pruned_states = Transformer_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim, pruning)

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
if pruning:
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

    t4 = time.time()
    print(f"Pruned states sorting time: {t4 - t3:.2f} seconds")
    plot_graph(smallest_best_state, filename=fig_name + "_smallest_best_pruned_state.svg")
    plot_graph(ordered_pruned_states[0], filename=fig_name + "_best_pruned_state.svg")
    histo_fidelity(pruned_fids, filename=fig_name + "_fidelity_pruned_states_histogram.svg")

tp = time.time()
plot_graph(ordered_states[0], filename=fig_name + "_best_state.svg")
histo_fidelity(state_fids, filename=fig_name + "_fidelity_all_states_histogram.svg")
plot_loss_curve(fig_name, losses, title="Trajectory Balance Loss")
plot_rewards(rewards)
t5 = time.time()
print(f"Plotting time: {t5 - tp:.2f} seconds")
