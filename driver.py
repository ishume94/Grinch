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
t0 = time.time()

#For TBModel, uncomment the following lines: It requires the model beforehand!!!
#model = TBModel(n_hid_units, FEATURE_KEYS)
#model = embTBModel(n_hid_units, FEATURE_KEYS)
# sampled_states, losses, logZs = TB_train(num_colors, num_nodes, FEATURE_KEYS,
#                                           target_expr, target_state, model, 
#                                           n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges)

sampled_states, losses, logZs, rewards = GIN_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units)
sampled_states, losses, logZs, rewards = GINE_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim)
sampled_states, losses, logZs, rewards = GAT_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim)
sampled_states, losses, logZs, rewards = Transformer_TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges, n_hid_units, edge_feat_dim)



with open(fig_name + "_sampled_graphs.p", 'wb') as f:
    pickle.dump(sampled_states, f, pickle.HIGHEST_PROTOCOL)

with open(fig_name + "_rewards.p", 'wb') as f:
    pickle.dump(rewards, f, pickle.HIGHEST_PROTOCOL)

t1 = time.time()
print(f"Training time: {t1 - t0:.2f} seconds")

t2 = time.time()
ordered_states = sorted(sampled_states, key=lambda i: reward_fidelity(target_state,i, num_colors), reverse=True)
print("Fidelity reward for the best state:", reward_fidelity(target_state, ordered_states[0], num_colors))
print("Fidelity reward for the best 20 states:")
for i in range(min(20, len(ordered_states))):
    print("Fidelity reward:", reward_fidelity(target_state, ordered_states[i], num_colors))

t3 = time.time()
print(f"Sorting time: {t3 - t2:.2f} seconds")

t4 = time.time()
plot_graph(ordered_states[0])
histo_fidelity(target_state, ordered_states, num_colors)
plot_loss_curve(fig_name, losses, title="Trajectory Balance Loss")
plot_rewards(rewards)
t5 = time.time()
print(f"Plotting time: {t5 - t4:.2f} seconds")
