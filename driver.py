from gflow_qoc.utils import *
from gflow_qoc.gflow_utils import *
from gflow_qoc.result_analysis import *
from gflow_qoc.training import *
import time

import itertools
from itertools import combinations

# Define number of nodes and colors
num_nodes, num_colors = 6, 3 #4, 2 for 4D-GHZ. 6, 3 for 6D-GHZ

print(f"Number of optical paths: {num_nodes}, Number of modes: {num_colors}")
node_pairs = list(itertools.combinations(range(num_nodes), 2))
color_pairs = list(itertools.product(range(num_colors), repeat=2))
FEATURE_KEYS = [((n1, n2), (c1, c2)) for (n1, n2) in node_pairs for (c1, c2) in color_pairs]

# Fixed hyperparameters.
n_hid_units = 512
n_episodes = 200
learning_rate = 1e-2
decay_rate = 1.00
update_freq = 10 #Update every episode
seed = 666
max_edges = 9
#target_expr = "1|0000⟩ + 1|1111⟩" #4D-GHZ
target_expr = "1|000000⟩ + 1|111111⟩+ 1|222222⟩" #6D-GHZ
target_state = parse_dirac_expression(target_expr, num_nodes, num_colors)

print("For all experiments, our hyperparameters will be:")
print("    + n_hid_units={}".format(n_hid_units))
print("    + n_episodes={}".format(n_episodes))
print("    + learning_rate={}".format(learning_rate))
print("    + decay_rate={}".format(decay_rate))
print("    + update_freq={}".format(update_freq))
print("    + seed={}".format(seed))
print("    + max_edges={}".format(max_edges))
#print("    + target state={}".format(target_state))
print("    + Target state:", state_vector_to_dirac_notation(target_state, num_nodes, num_colors))
t0 = time.time()

model = TBModel(n_hid_units, FEATURE_KEYS)
#model = embTBModel(n_hid_units, FEATURE_KEYS)

sampled_states, losses, logZs = TB_train(num_colors, num_nodes, FEATURE_KEYS,
                                          target_expr, target_state, model, 
                                          n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges)

ordered_states = sorted(sampled_states, key=lambda i: reward_fidelity(target_state,i, num_colors), reverse=True)
print(reward_fidelity(target_state,ordered_states[0], num_colors))

t1 = time.time()
print(f"Training time: {t1 - t0:.2f} seconds")

plot_graph(ordered_states[0])
histo_fidelity(target_state, ordered_states, num_colors)
